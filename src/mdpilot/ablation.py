"""Verifier ablation: three arms judging one and the same simulation.

The question is what the *verifier* contributes — the deterministic gates,
the LLM, or the two together — and the only fair way to ask it is to hold
everything else fixed. An arm that ran its own campaign would drive its own
simulation (the LLM's `extend` lengths and CV revisions shape the trajectory),
and two arms would be comparing two different runs. So the arms do not run
anything. One campaign on disk is the simulation; each round's report is
recomputed once from the per-round bias snapshots; and each arm is a
verifier that reads that report — its own *view* of it — and returns a
verdict. Identical simulation, bias design and seed are then true by
construction, and `Comparison` still asserts it, loudly, on the stored config
and the round table, so two campaigns that merely look alike cannot be
compared by accident.

The three arms:

    gates_only      a deterministic rule over the precision gate and the
                    falsifier block; no LLM
    lm_only         the LLM reads the raw statistics with no tolerances, no
                    fired flags, no gate verdicts, and no rules naming any
                    threshold in its prompt; it judges on its own
    lm_plus_gates   the current design: the LLM reads the full report with
                    flags and states, and the loop's stop refusal applies

A verdict per round is one of `accept` (the campaign is done), `continue`,
`revise` (the biased coordinate is wrong or insufficient). Each arm's
`final_delta_g` is the answer at the round it first accepted, or at the last
round when it never did — which is what makes an early acceptance visible as
a wrong number rather than a missing one. Ground truth for whether that
number is wrong belongs to the fault-injection suite (docs/falsifiers.md
Step 5); this module records, it does not score.
"""

from __future__ import annotations

import copy
import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

from mdpilot.diagnostics.report import report_field
from mdpilot.memory import store
from mdpilot.observables import ObservableSpec
from mdpilot.orchestrator import loop as _loop
from mdpilot.orchestrator.scientist import Decision, DecisionRefused, decide

Arm = Literal["gates_only", "lm_only", "lm_plus_gates"]
ARMS: tuple[Arm, ...] = ("gates_only", "lm_only", "lm_plus_gates")
Verdict = Literal["accept", "continue", "revise"]

_FS_PER_NS = 1_000_000.0

# What the LLM in `lm_only` must not see. Verdicts are booleans derived by
# comparing a statistic against a threshold; tolerances are the thresholds.
# Everything else — every magnitude, every input — stays.
_PRECISION_VERDICTS = frozenset({
    "gate", "fes_converged", "well_sampled", "plateau_reached", "exploring",
    "barrier_crossed",
})
_PRECISION_TOLERANCES = frozenset({"min_recrossings"})
_FALSIFIER_SHOWN = ("name", "statistic", "magnitude", "inputs")
_PRIOR_HIDDEN = frozenset({
    "precision_gate", "not_refuted", "falsifiers_refuted",
    "falsifiers_not_evaluable", "plateau_reached",
})

DecideFn = Callable[..., Decision]


class ArmsNotComparable(ValueError):
    """The arms of one comparison differ in something other than the verifier."""


@dataclass(frozen=True)
class ArmConfig:
    arm: Arm
    work_dir: Path
    model: str = "claude-sonnet-4-6"

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(f"unknown arm {self.arm!r}; expected one of {ARMS}")
        object.__setattr__(self, "work_dir", Path(self.work_dir))


@dataclass(frozen=True)
class Comparison:
    """A set of arms over one simulation. Refuses to exist otherwise."""

    arms: tuple[ArmConfig, ...]

    def __post_init__(self) -> None:
        if not self.arms:
            raise ArmsNotComparable("a comparison needs at least one arm")
        assert_arms_comparable(self.arms)

    @property
    def work_dir(self) -> Path:
        return self.arms[0].work_dir


def _round_identity(row: store.RoundRow) -> dict[str, Any]:
    return {
        "round_index": row.round_index,
        "n_steps": row.n_steps,
        "phase": "metad" if row.plumed_dat_path is not None else "vanilla",
        "decision": row.decision,
        "metad_proposal": row.metad_proposal,
    }


def assert_arms_comparable(arms: Sequence[ArmConfig]) -> None:
    """Every arm must read the same simulation: same stored campaign config
    (seed, system, engine, thresholds, bias shape — the whole locked set,
    legacy defaults applied) and the same round table. The arm name and the
    model are the verifier; nothing else may differ."""
    first = arms[0]
    reference_config = store.get_campaign_config(first.work_dir)
    if reference_config is None:
        raise ArmsNotComparable(f"{first.work_dir} has no campaign config")
    reference_rounds = [_round_identity(r) for r in store.list_rounds(first.work_dir)]
    for other in arms[1:]:
        config = store.get_campaign_config(other.work_dir)
        if config is None:
            raise ArmsNotComparable(f"{other.work_dir} has no campaign config")
        diffs = store.config_differences(reference_config, config)
        if diffs:
            named = ", ".join(f"{k}: {a!r} != {b!r}" for k, (a, b) in diffs.items())
            raise ArmsNotComparable(
                f"arm {other.arm!r} ({other.work_dir}) is not the same simulation as "
                f"arm {first.arm!r} ({first.work_dir}); config differs in {named}"
            )
        rounds = [_round_identity(r) for r in store.list_rounds(other.work_dir)]
        if rounds != reference_rounds:
            raise ArmsNotComparable(
                f"arm {other.arm!r} ({other.work_dir}) has a different round table "
                f"from arm {first.arm!r} ({first.work_dir}); the arms must judge "
                f"the same trajectory"
            )
        if other.model != first.model:
            raise ArmsNotComparable(
                f"arms {first.arm!r} and {other.arm!r} name different models "
                f"({first.model!r} vs {other.model!r}); one comparison, one model"
            )


# --------------------------------------------------------------------------
# views
# --------------------------------------------------------------------------

def raw_view(report: dict[str, Any]) -> dict[str, Any]:
    """The `lm_only` view: statistics without verdicts or tolerances.

    Precision keeps every number and loses every boolean that was derived
    by comparing one against a threshold, plus the threshold itself. Each
    falsifier keeps its name, statistic, magnitude and inputs, and loses its
    tolerance, state, fired flag and note. The summary goes entirely.
    """
    view = copy.deepcopy(report)
    precision = view.get("precision")
    if isinstance(precision, dict):
        for key in _PRECISION_VERDICTS | _PRECISION_TOLERANCES:
            precision.pop(key, None)
    correctness = view.get("correctness")
    if isinstance(correctness, dict):
        correctness.pop("summary", None)
        falsifiers = correctness.get("falsifiers") or {}
        correctness["falsifiers"] = {
            name: {k: f[k] for k in _FALSIFIER_SHOWN if k in f}
            for name, f in falsifiers.items()
        }
    return view


def raw_prior(summary: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in summary.items() if k not in _PRIOR_HIDDEN}


def view_for(arm: Arm, report: dict[str, Any]) -> dict[str, Any]:
    return raw_view(report) if arm == "lm_only" else report


# --------------------------------------------------------------------------
# verdicts
# --------------------------------------------------------------------------

def gates_only_verdict(
    report: dict[str, Any], *, task_expectation: str | None
) -> tuple[Verdict, str]:
    """The deterministic arm. Biased: accept on precision gate AND
    `not_refuted`; revise on any refuted falsifier; else continue. Unbiased:
    accept on the gate when the task needs no transition; revise (a pivot is
    needed) when it does and the system is pinned; else continue."""
    phase = report.get("phase", "vanilla")
    gate = report_field(report, "gate")
    if phase == "metad":
        summary = (report.get("correctness") or {}).get("summary") or {}
        if gate is True and summary.get("not_refuted"):
            return "accept", "precision.gate and correctness.summary.not_refuted"
        if summary.get("refuted"):
            return "revise", f"refuted: {summary['refuted']}"
        return "continue", (
            f"gate={gate!r}, not_evaluable={summary.get('not_evaluable')}"
        )
    exploring = report_field(report, "exploring")
    if task_expectation is None:
        if gate is True:
            return "accept", "precision.gate on a pure convergence task"
        return "continue", f"gate={gate!r}"
    if exploring is False:
        return "revise", "pinned in one basin against a task that needs a transition"
    return "continue", f"gate={gate!r}, exploring={exploring!r}"


_DECISION_TO_VERDICT: dict[str, Verdict] = {
    "stop": "accept",
    "extend": "continue",
    "switch_to_metad": "revise",
    "switch_cv": "revise",
    "add_cv": "revise",
}


# --------------------------------------------------------------------------
# replaying the campaign's reports
# --------------------------------------------------------------------------

def _snapshot_names(rounds_dir: Path, index: int) -> list[Path]:
    """The bias files snapshotted beside round `index`'s checkpoint."""
    stem = f"round_{index:03d}."
    return [
        p for p in rounds_dir.iterdir()
        if p.name.startswith(stem)
        and (p.name[len(stem):].startswith("hills") or p.name[len(stem):] in ("colvar", "plumed.dat"))
    ]


def replay_report(
    work_dir: Path,
    row: store.RoundRow,
    rows: Sequence[store.RoundRow],
    config: dict[str, Any],
    scratch: Path,
) -> dict[str, Any]:
    """Recompute round `row`'s three-block report from what is on disk.

    Read-only with respect to the campaign: the per-round bias snapshot is
    copied into `scratch` and the observable files are re-derived there. A
    biased round without a snapshot cannot be replayed — the live COLVAR has
    since grown past it — and says so.
    """
    work_dir = Path(work_dir)
    rounds_dir = work_dir / "rounds"
    index = row.round_index
    dcd = rounds_dir / f"round_{index:03d}{row.dcd_path.suffix}"
    topology = next(
        (p for p in (work_dir / "topology.pdb", work_dir / "cache" / "topology.pdb") if p.exists()),
        None,
    )
    if topology is None:
        raise FileNotFoundError(f"{work_dir}: no topology.pdb to replay against")

    spec = config.get("system_spec") or {}
    ensemble = spec.get("ensemble") or {}
    temperature_k = float(ensemble.get("temperature_k", 300.0))
    timestep_fs = float(ensemble.get("timestep_fs", 2.0))
    frame_ps = config["report_interval_steps"] * timestep_fs / 1000.0
    observable = (
        ObservableSpec.from_dict(config["observable"]) if config.get("observable") else None
    )
    thresholds = config.get("state_thresholds")
    state_thresholds = tuple(thresholds) if thresholds else None
    prior = [r for r in rows if r.round_index < index]
    active = _loop._active_cvs(prior)
    switches_used = sum(1 for r in prior if r.decision in ("switch_cv", "add_cv"))

    replay_rounds = scratch / "rounds"
    replay_rounds.mkdir(parents=True, exist_ok=True)
    for r in prior:
        src = rounds_dir / f"round_{r.round_index:03d}.obs.npy"
        if src.exists():
            shutil.copy2(src, replay_rounds / src.name)

    plumed_dat: Path | None = None
    if row.plumed_dat_path is not None:
        snapshots = _snapshot_names(rounds_dir, index)
        if not any(p.name.endswith(".colvar") for p in snapshots):
            raise FileNotFoundError(
                f"{work_dir}: round {index} is biased but has no COLVAR/HILLS snapshot; "
                f"campaigns recorded before snapshots existed cannot be replayed per round"
            )
        bias_dir = scratch / f"bias_{index:03d}"
        bias_dir.mkdir(parents=True, exist_ok=True)
        stem = f"round_{index:03d}."
        for p in snapshots:
            name = p.name[len(stem):]
            target = name.upper() if name.startswith("hills") else name
            if name.startswith("hills"):
                # `hills.psi` -> `HILLS.psi`: the prefix is upper-cased, the label is not.
                target = "HILLS" + name[len("hills"):]
            elif name == "colvar":
                target = "COLVAR"
            shutil.copy2(p, bias_dir / target)
        plumed_dat = bias_dir / "plumed.dat"

    return _loop._round_report(
        dcd, topology,
        plumed_dat_path=plumed_dat,
        temperature_k=temperature_k,
        fes_dir=scratch / f"fes_{index:03d}",
        min_recrossings=int(config.get("min_recrossings", 1)),
        rounds_dir=replay_rounds,
        round_index=index,
        state_thresholds=state_thresholds,
        observable=observable,
        cv_switches_used=switches_used,
        max_cv_switches=int(config.get("max_cv_switches", 2)),
        active_cvs=active,
        frame_ps=frame_ps,
        absence_tolerance_ns=float(config.get("absence_tolerance_ns", 8.0)),
    )


# --------------------------------------------------------------------------
# the comparison
# --------------------------------------------------------------------------

@dataclass
class _ArmState:
    config: ArmConfig
    ledger: list[str] = field(default_factory=list)
    priors: list[dict[str, Any]] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    accepted_at: dict[str, Any] | None = None
    first_non_continue: dict[str, Any] | None = None


def run_comparison(
    comparison: Comparison,
    *,
    out_dir: Path,
    decide_fn: DecideFn = decide,
    report_fn: Callable[..., dict[str, Any]] = replay_report,
    max_cv_switches: int = 2,
) -> dict[str, Any]:
    """Replay the campaign once; let every arm judge every round; write the
    per-round verdicts and a summary. Returns the summary."""
    work_dir = comparison.work_dir
    config = store.get_campaign_config(work_dir)
    assert config is not None
    rows = store.list_rounds(work_dir)
    task_expectation = config.get("task_expectation")
    spec = config.get("system_spec") or {}
    timestep_fs = float((spec.get("ensemble") or {}).get("timestep_fs", 2.0))
    steps_per_ns = _FS_PER_NS / timestep_fs

    out_dir = Path(out_dir)
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)
    states = {a.arm: _ArmState(a) for a in comparison.arms}
    gates_first_fire: dict[str, Any] | None = None
    biased_ns = 0.0

    with tempfile.TemporaryDirectory() as scratch_root:
        for row in rows:
            phase = "metad" if row.plumed_dat_path is not None else "vanilla"
            if phase == "metad":
                biased_ns += row.n_steps / steps_per_ns
            report = report_fn(work_dir, row, rows, config, Path(scratch_root) / f"r{row.round_index}")
            (out_dir / "reports" / f"round_{row.round_index:03d}.json").write_text(
                json.dumps(report, indent=2, default=str)
            )
            delta_g = report_field(report, "delta_g_low_minus_high_kj_per_mol")
            summary = (report.get("correctness") or {}).get("summary") or {}
            if gates_first_fire is None and summary.get("refuted"):
                gates_first_fire = {
                    "round": row.round_index, "biased_ns": biased_ns,
                    "refuted": summary["refuted"],
                }
            switches_used = sum(
                1 for r in rows if r.round_index < row.round_index
                and r.decision in ("switch_cv", "add_cv")
            )
            for state in states.values():
                verdict, llm_verdict, reason, note = _judge(
                    state, report, phase=phase, task_expectation=task_expectation,
                    allow_cv_switch=phase == "metad" and switches_used < max_cv_switches,
                    decide_fn=decide_fn,
                )
                entry = {
                    "round": row.round_index, "phase": phase, "biased_ns": biased_ns,
                    "verdict": verdict, "verdict_llm": llm_verdict, "reason": reason,
                    "override": note, "delta_g_low_minus_high_kj_per_mol": delta_g,
                    "precision_gate": report_field(report, "gate"),
                    "refuted": summary.get("refuted"),
                    "not_evaluable": summary.get("not_evaluable"),
                }
                state.rows.append(entry)
                # Detection latency is measured over the biased phase only: the
                # vanilla round's `revise` is the pivot the campaign was designed
                # to make, not a verifier noticing a broken answer.
                if (
                    state.first_non_continue is None
                    and verdict != "continue"
                    and phase == "metad"
                ):
                    state.first_non_continue = {
                        "round": row.round_index, "biased_ns": biased_ns, "verdict": verdict,
                    }
                if state.accepted_at is None and verdict == "accept":
                    state.accepted_at = {
                        "round": row.round_index, "biased_ns": biased_ns,
                        "delta_g_low_minus_high_kj_per_mol": delta_g,
                    }

    result: dict[str, Any] = {
        "work_dir": str(work_dir),
        "n_rounds": len(rows),
        "biased_ns": biased_ns,
        "model": comparison.arms[0].model,
        "gates_first_fire": gates_first_fire,
        "arms": {},
    }
    for arm, state in states.items():
        arm_dir = out_dir / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        (arm_dir / "verdicts.jsonl").write_text(
            "".join(json.dumps(r, default=str) + "\n" for r in state.rows)
        )
        last = state.rows[-1] if state.rows else None
        result["arms"][arm] = {
            "accepted": state.accepted_at is not None,
            "accepted_at": state.accepted_at,
            # First biased round the arm stopped saying `continue`, with the
            # biased ns at that point: the arm's detection latency.
            "first_non_continue": state.first_non_continue,
            "final_delta_g_low_minus_high_kj_per_mol": (
                state.accepted_at["delta_g_low_minus_high_kj_per_mol"]
                if state.accepted_at is not None
                else (last or {}).get("delta_g_low_minus_high_kj_per_mol")
            ),
            "verdicts": [r["verdict"] for r in state.rows],
        }
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def _judge(
    state: _ArmState,
    report: dict[str, Any],
    *,
    phase: str,
    task_expectation: str | None,
    allow_cv_switch: bool,
    decide_fn: DecideFn,
) -> tuple[Verdict, Verdict | None, str, str | None]:
    """One arm's verdict on one round: (verdict, the LLM's own verdict before
    any override, reason, override note)."""
    arm = state.config.arm
    if arm == "gates_only":
        verdict, reason = gates_only_verdict(report, task_expectation=task_expectation)
        return verdict, None, reason, None

    view = view_for(arm, report)
    priors = [raw_prior(p) for p in state.priors] if arm == "lm_only" else list(state.priors)
    note: str | None = None
    try:
        decision = decide_fn(
            view,
            prior_round_summaries=priors,
            hypothesis_ledger=list(state.ledger),
            task_expectation=task_expectation,
            phase=phase,
            allow_cv_switch=allow_cv_switch,
            max_extra_ns=None,
            validate_proposal=None,
            view="raw" if arm == "lm_only" else "gated",
            model=state.config.model,
        )
    except DecisionRefused as refused:
        decision, note = _loop._refuse_decision(refused)
    llm_verdict = _DECISION_TO_VERDICT[decision.decision]
    verdict = llm_verdict
    if arm == "lm_plus_gates" and phase == "metad":
        # The current design includes the loop's refusal of a stop the data
        # do not permit; the LLM's own verdict is kept alongside so the two
        # can be told apart.
        decision, stop_note = _loop._refuse_premature_stop(decision, report, None)
        note = note or stop_note
        verdict = _DECISION_TO_VERDICT[decision.decision]
    if decision.ledger_note:
        state.ledger.append(f"R{len(state.priors) + 1}: {decision.ledger_note}")
    state.priors.append(_prior_for(report, phase, decision))
    return verdict, llm_verdict, decision.reason, note


def _prior_for(report: dict[str, Any], phase: str, decision: Decision) -> dict[str, Any]:
    """The compact prior the loop would have built for this round."""
    fake = _loop.RoundResult(
        index=0, n_steps=0, dcd_path=Path("."), summary_path=Path("."),
        report=report, decision=decision,
        plumed_dat_path=Path(".") if phase == "metad" else None,
    )
    prior = _loop._compact_prior(fake)
    prior.pop("n_steps", None)
    prior.pop("round_index", None)
    return prior


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin CLI
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--work-dir", type=Path, required=True, help="a finished campaign")
    parser.add_argument("--arms", default=",".join(ARMS),
                        help="comma-separated subset of " + ",".join(ARMS))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="claude-sonnet-4-6")
    args = parser.parse_args(argv)
    arms = tuple(
        ArmConfig(arm=a.strip(), work_dir=args.work_dir, model=args.model)  # type: ignore[arg-type]
        for a in args.arms.split(",") if a.strip()
    )
    result = run_comparison(Comparison(arms), out_dir=args.out)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
