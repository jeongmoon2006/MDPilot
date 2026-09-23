"""Fault injection: campaigns that are wrong by construction, and the score.

The falsifiers (docs/falsifiers.md) can only refute. The one way to learn
whether they refute the right things is to hand them campaigns whose answer
is known to be wrong and count how often each verifier arm declares one done
anyway. That count is the **false-accept rate**, the headline metric; the
biased nanoseconds until an arm first stops saying `continue` is the
**detection latency**, the secondary one.

Each `Fault` is a way of breaking a campaign built from an ordinary task
file, together with the falsifier(s) expected to fire on it and why the
answer it produces is not the intended one. The registry is fixed in code
and every entry names its expected falsifiers up front, so a fault that
none of them catches is a gap in the set, not a surprise to explain later.

Broken campaigns are driven by a **scripted policy**, not the LLM: pivot to
the fault's coordinate at the first opportunity, then extend to the budget.
The verifier arms (`mdpilot.ablation`) then replay the finished campaign,
so every arm judges a trajectory no verifier shaped — the same separation
the ablation rests on.

Nothing here runs MD by itself; `benchmarks/run_faults.py` does, from a task
file. This module holds the registry, the driver, and the scorer, all of
which are unit-tested without an engine.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from mdpilot.ablation import ARMS
from mdpilot.diagnostics import falsifiers as fz
from mdpilot.orchestrator.scientist import Decision, MetadProposal

FAULT_FILE = "fault.json"
ABLATION_DIR = "ablation"


@dataclass(frozen=True)
class Fault:
    name: str
    description: str
    # Why the answer this campaign produces is not the intended quantity.
    ground_truth: str
    # The coordinate the scripted policy pivots to.
    cv: MetadProposal
    # `run_campaign` keyword overrides applied on top of the task file.
    overrides: dict[str, Any] = field(default_factory=dict)
    # Falsifiers expected to refute on this fault, and ones expected to stay
    # silent (recorded so an unexpected fire is visible too).
    expected_to_fire: tuple[str, ...] = ()
    expected_silent: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        unknown = set(self.expected_to_fire) | set(self.expected_silent)
        unknown -= set(fz.FALSIFIER_NAMES)
        if unknown:
            raise ValueError(f"fault {self.name!r} names unknown falsifier(s) {sorted(unknown)}")
        if not self.expected_to_fire:
            raise ValueError(
                f"fault {self.name!r} registers no falsifier expected to fire; a fault "
                f"the set cannot catch is a finding, and has to be written as one"
            )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["cv"] = self.cv.to_dict()
        return d


_CONTACTS = MetadProposal(cv_type="contacts", selections=("name CA",), label="q_ca")
_RMSD = MetadProposal(cv_type="rmsd", selections=("name CA",), label="rmsd_ca")

# The registry. Written against a folding task whose observable is the
# native-contact fraction with states at 0.3 / 0.7 and a configured 0.8 nm
# wall (`benchmarks/tasks/cln025_contacts.yaml`); the overrides are what make
# each campaign wrong, and are locked into that campaign's config so a resume
# cannot quietly repair it.
FAULTS: dict[str, Fault] = {
    f.name: f
    for f in (
        Fault(
            name="rmsd_no_configured_wall",
            description="Bias CA-RMSD to the native structure with no configured upper wall.",
            ground_truth=(
                "RMSD-to-native is unbounded above (F6): the bias drives the walker into "
                "an ever-larger disordered space and it does not return, so the surface "
                "is a one-way excursion, not F(Q). The only bound left is the one derived "
                "from the box, which bounds the artifact, not the science."
            ),
            cv=_RMSD,
            overrides={"cv_upper_wall_nm": None},
            expected_to_fire=("occupancy_invariance", "time_window_invariance"),
            expected_silent=("seed_invariance",),
        ),
        Fault(
            name="wall_inside_transition_region",
            description="Bias CA-RMSD with the upper wall at 0.3 nm, inside the unfolding path.",
            ground_truth=(
                "The wall sits where the task's unfolded state begins, so the walker "
                "leans on it instead of sampling that state; frames the wall acts on "
                "carry its penalty as if it were physics, and the unfolded population "
                "is set by the wall, not by the force field."
            ),
            cv=_RMSD,
            overrides={"cv_upper_wall_nm": 0.3},
            expected_to_fire=("bias_accounting_invariance", "state_definition_invariance"),
        ),
        Fault(
            name="thresholds_inside_one_basin",
            description="States defined at Q = 0.75 / 0.85, both inside the folded basin.",
            ground_truth=(
                "The two 'states' are the same basin: thermal fluctuation crosses both "
                "thresholds, the recrossing count measures jiggle, and ΔG between them "
                "is not a folding free energy under any estimator."
            ),
            cv=_CONTACTS,
            overrides={"state_thresholds": (0.75, 0.85)},
            expected_to_fire=("state_definition_invariance",),
        ),
        Fault(
            name="truncated_budget",
            description="Ordinary contacts bias, biased budget cut to 2 ns.",
            ground_truth=(
                "Two nanoseconds cannot converge a surface whose round trip costs "
                "5-10 ns; whatever ΔG the last round reports is a snapshot of a "
                "surface still moving."
            ),
            cv=_CONTACTS,
            overrides={"max_biased_ns": 2.0},
            expected_to_fire=("time_window_invariance",),
        ),
        Fault(
            name="walker_pinned_weak_bias",
            description="Contacts bias depositing a hill every 50 000 steps instead of every 500.",
            ground_truth=(
                "The bias builds a hundred times too slowly to fill the folded basin "
                "within the budget; the walker never leaves it, the surface stops "
                "changing because nothing new is sampled, and ΔG is undefined or "
                "reports a single basin."
            ),
            cv=_CONTACTS,
            overrides={"bias_pace": 50_000},
            expected_to_fire=("occupancy_invariance",),
        ),
        Fault(
            name="gamma_too_small",
            description="Contacts bias with well-tempered bias factor 1.5 instead of 10.",
            ground_truth=(
                "γ = 1.5 caps the bias at ~0.5 kT above the surface; a folding barrier "
                "of several kT is never flattened, the walker stays folded, and the "
                "surface it reports is the bottom of one well."
            ),
            cv=_CONTACTS,
            overrides={"bias_factor": 1.5},
            expected_to_fire=("occupancy_invariance",),
        ),
    )
}


# --------------------------------------------------------------------------
# the scripted policy
# --------------------------------------------------------------------------

def scripted_policy(fault: Fault, *, extend_ns: float) -> Callable[..., Decision]:
    """A decision function that produces the fault and nothing else.

    Vanilla round: pivot to the fault's coordinate. Biased round: extend by
    `extend_ns`, whatever the report says — this policy is the *cause* of a
    broken campaign, not a verifier of it, and the loop's budget cap is what
    ends the run. It never stops and never revises, so the finished campaign
    is one uninterrupted bias on one coordinate for every arm to judge.
    """

    def decide(report: dict[str, Any], *, phase: str = "vanilla", **_: Any) -> Decision:
        if phase == "vanilla":
            return Decision(
                decision="switch_to_metad",
                reason=f"fault {fault.name!r}: scripted pivot to {fault.cv.label}",
                extra_ns=None,
                metad_proposal=fault.cv,
            )
        return Decision(
            decision="extend", reason="scripted: run to the budget", extra_ns=extend_ns,
        )

    return decide


# --------------------------------------------------------------------------
# the score
# --------------------------------------------------------------------------

def write_fault_record(work_dir: Path, fault: Fault) -> Path:
    path = Path(work_dir) / FAULT_FILE
    path.write_text(json.dumps(fault.to_dict(), indent=2))
    return path


def _first_ns(reports: Sequence[dict[str, Any]], name: str, states: set[str]) -> float | None:
    """Biased ns at the first round where falsifier `name` is in one of `states`."""
    for r in reports:
        f = ((r.get("correctness") or {}).get("falsifiers") or {}).get(name)
        if f and f.get("state") in states:
            return r["_biased_ns"]
    return None


def score(fault_dirs: Sequence[Path]) -> dict[str, Any]:
    """Read every finished fault campaign's record and its ablation output;
    return the false-accept rate and latency per arm, and per fault whether
    the expected falsifiers fired and when.

    A fault directory holds `fault.json` (from `write_fault_record`) and
    `ablation/summary.json` + `ablation/<arm>/verdicts.jsonl` +
    `ablation/reports/round_NNN.json` (from `mdpilot.ablation`).
    """
    per_fault: dict[str, Any] = {}
    per_arm: dict[str, dict[str, Any]] = {
        arm: {"n_faults": 0, "false_accepts": 0, "accepted": [], "latency_ns": {}} for arm in ARMS
    }
    for d in fault_dirs:
        d = Path(d)
        fault = json.loads((d / FAULT_FILE).read_text())
        ab = d / ABLATION_DIR
        summary = json.loads((ab / "summary.json").read_text())
        # Reports in round order, each tagged with its cumulative biased ns
        # from the gates arm's verdict rows (identical for every arm).
        reports: list[dict[str, Any]] = []
        ns_by_round: dict[int, float] = {}
        for arm_dir in (ab / arm for arm in ARMS):
            rows_path = arm_dir / "verdicts.jsonl"
            if rows_path.exists():
                for line in rows_path.read_text().splitlines():
                    row = json.loads(line)
                    ns_by_round.setdefault(row["round"], row["biased_ns"])
                break
        for p in sorted((ab / "reports").glob("round_*.json")):
            r = json.loads(p.read_text())
            index = int(p.stem.split("_")[1])
            r["_biased_ns"] = ns_by_round.get(index, 0.0)
            reports.append(r)

        expected = list(fault["expected_to_fire"])
        fired = {n: _first_ns(reports, n, {fz.REFUTED}) for n in expected}
        blocked = {n: _first_ns(reports, n, {fz.REFUTED, fz.NOT_EVALUABLE}) for n in expected}
        all_refuted = sorted({
            n for r in reports
            for n, f in ((r.get("correctness") or {}).get("falsifiers") or {}).items()
            if f.get("state") == fz.REFUTED
        })
        arms_out: dict[str, Any] = {}
        for arm, a in summary["arms"].items():
            accepted = bool(a["accepted"])
            first = a.get("first_non_continue") or {}
            arms_out[arm] = {
                "accepted": accepted,
                "accepted_at_ns": (a.get("accepted_at") or {}).get("biased_ns"),
                "final_delta_g_low_minus_high_kj_per_mol": a.get("final_delta_g_low_minus_high_kj_per_mol"),
                "first_non_continue_ns": first.get("biased_ns"),
                "first_non_continue_verdict": first.get("verdict"),
            }
            stats = per_arm.setdefault(
                arm, {"n_faults": 0, "false_accepts": 0, "accepted": [], "latency_ns": {}}
            )
            stats["n_faults"] += 1
            if accepted:
                stats["false_accepts"] += 1
                stats["accepted"].append(fault["name"])
            stats["latency_ns"][fault["name"]] = first.get("biased_ns")
        per_fault[fault["name"]] = {
            "expected_to_fire": expected,
            "fired_at_ns": fired,
            "blocked_at_ns": blocked,
            "detected": any(v is not None for v in fired.values()),
            "unexpected_fires": [n for n in all_refuted if n not in expected],
            "expected_silent_but_fired": [
                n for n in fault.get("expected_silent", []) if n in all_refuted
            ],
            "biased_ns": summary.get("biased_ns"),
            "arms": arms_out,
        }
    for stats in per_arm.values():
        n = stats["n_faults"]
        stats["false_accept_rate"] = (stats["false_accepts"] / n) if n else None
    return {"arms": per_arm, "faults": per_fault}
