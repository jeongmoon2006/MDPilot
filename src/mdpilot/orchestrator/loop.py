"""Mechanical campaign loop: simulate → diagnose → persist → decide → repeat.

This is the state machine described in `docs/architecture.md`. The LLM is
called exactly once per round (in `scientist.decide`); everything else is
deterministic Python.

Engine independence (M3): the loop talks to MD engines exclusively through
the `MDAdapter` Protocol. Swap the adapter to change engines; nothing in
this file or in `scientist.py` needs to know.

Persistence (M2): per-campaign SQLite at `<work_dir>/state.db` is the
source of truth for completed rounds; the adapter's checkpoint captures
the state needed to continue. Commit order is `save_checkpoint` THEN
`store.append_round` — a crash between leaves the checkpoint dangling
(harmless) and the round absent, so restart re-runs it. The per-round
JSON file is kept alongside SQLite for human inspection.

Metadynamics pivot (M4): when `decide()` returns `switch_to_metad`, the
campaign does not end — it pivots in place. The proposed CV is resolved
against the topology, a bias is sized deterministically from the just-run
(vanilla) trajectory, a `plumed.dat` is written, and a biased adapter is
constructed from the same `SystemSpec`. Subsequent rounds run under that
bias and record its `plumed_dat_path`. The metaD phase starts from the
cached minimized state (the vanilla checkpoint is not portable across the
added `PlumedForce`); SIGMA is still sized from the vanilla basin sampling,
which is the relevant width. Warm-starting metaD from the vanilla endpoint
is a possible follow-up.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Literal, cast

import mdtraj as md
import numpy as np

from mdpilot.adapters.base import MDAdapter
from mdpilot.adapters.openmm_adapter import OpenMMAdapter
from mdpilot.adapters.plumed_writer import (
    ContactsCV,
    ParallelBias,
    PlumedInput,
    TorsionCV,
    enable_restart,
)
from mdpilot.adapters.system_spec import SystemSpec
from mdpilot.diagnostics.free_energy import (
    delta_g_kj_per_mol,
    load_colvar,
    metad_report,
    reweighted_profile,
    write_fes,
)
from mdpilot.diagnostics.report import make_report
from mdpilot import preflight
from mdpilot.observables import (
    COLVAR_OBSERVABLE_LABEL,
    ObservableSpec,
    campaign_observable,
    colvar_to_observable_factor,
    observable_cv_proposal,
)
from mdpilot.memory import store
from mdpilot.orchestrator.scientist import (
    Decision,
    DecisionRefused,
    MetadProposal,
    UnresolvableProposal,
    decide,
)
from mdpilot.sampling.bias_designer import design_bias, design_upper_wall
from mdpilot.sampling.cv_designer import CVProposal, design_cv

_FS_PER_NS = 1_000_000.0
_PLUMED_DAT_NAME = "plumed.dat"  # canonical bias artifact (the OpenMM adapter also writes here)
# PLUMED's own outputs, alongside plumed.dat. These names must match the
# defaults `_build_plumed_input` leaves on MetadynamicsBias.hills_file /
# ParallelBias.hills_prefix and PlumedInput.colvar_file — they are what the
# rendered FILE= directives point at. One CV deposits into `HILLS`; a parallel
# bias on several deposits into `HILLS.<label>`, one file per CV.
_HILLS_NAME = "HILLS"
_COLVAR_NAME = "COLVAR"
_STATE_NAME = "state.xml"   # the walker's State, snapshotted beside each checkpoint

StopReason = Literal[
    "scientist_said_stop",
    "max_rounds_reached",
    "switch_to_metad_requested",
    "biased_budget_exhausted",
]

# Given a rendered plumed.dat string, build an adapter that runs biased MD.
BiasedAdapterFactory = Callable[[str], MDAdapter]

# Optional observer, called as the campaign progresses. Purely for watching —
# a progress bar, a log file, an HPC monitor. It cannot influence the campaign,
# and `_emit` swallows anything it raises: a long biased run must not die
# because something that was only watching it threw.
CampaignEvent = Callable[[str, dict[str, Any]], None]


def _emit(on_event: CampaignEvent | None, name: str, **payload: Any) -> None:
    if on_event is None:
        return
    try:
        on_event(name, payload)
    except Exception:  # noqa: BLE001 - an observer must never break the run
        pass


def steps_per_ns_for(adapter: MDAdapter) -> int:
    """Steps in one nanosecond at this engine's timestep (500_000 at 2 fs).

    Read from the adapter rather than hardcoded. `extra_ns`, `max_extra_ns`
    and `max_biased_ns` are all stated in nanoseconds and converted here, so a
    constant that merely happened to match both adapters would have made every
    round the wrong length on an engine at a different timestep — with
    `extra_ns` quietly no longer meaning nanoseconds anywhere in the campaign
    record.

    Public because `benchmarks/run_cln025.py` states its budget in nanoseconds
    too, and a second copy of the conversion is exactly the drift this exists
    to remove.
    """
    return max(int(round(_FS_PER_NS / adapter.timestep_fs)), 1)


@dataclass(frozen=True)
class RoundResult:
    index: int
    n_steps: int
    dcd_path: Path
    summary_path: Path
    report: dict[str, Any]
    decision: Decision
    plumed_dat_path: Path | None = None


@dataclass(frozen=True)
class CampaignResult:
    work_dir: Path
    rounds: tuple[RoundResult, ...]
    stop_reason: StopReason


def run_campaign(
    work_dir: Path,
    *,
    adapter: MDAdapter | None = None,
    initial_steps: int = 25_000,         # 50 ps default at 2 fs
    max_rounds: int = 10,
    max_extra_ns: float = 2.0,
    max_biased_ns: float | None = None,
    min_recrossings: int = 1,
    state_thresholds: tuple[float, float] | None = None,
    max_cv_switches: int = 2,
    cv_upper_wall_nm: float | None = None,
    bias_pace: int | None = None,
    bias_factor: float | None = None,
    observable: ObservableSpec | None = None,
    description: str | None = None,
    on_event: CampaignEvent | None = None,
    seed: int = 42,
    report_interval_steps: int = 500,    # 1 ps/frame at 2 fs
    equilibration_steps: int = 0,
    task_expectation: str | None = None,
    biased_adapter_factory: BiasedAdapterFactory | None = None,
) -> CampaignResult:
    """Run the closed loop, resuming from `work_dir/state.db` if it exists.

    `adapter` defaults to `OpenMMAdapter(work_dir=work_dir, seed=seed)`.
    Pass a different `MDAdapter` to run through another engine.

    `state_thresholds` is `(low, high)` on the campaign observable — the two
    states the *task* defines, as positions on that coordinate rather than as
    roles. For a folding campaign that is (native, extended) on CA-RMSD; for a
    binding campaign it is (bound, unbound) on a distance, and nothing here
    changes. When given, biased-round recrossings are counted there rather than
    between the two deepest basins of the current free-energy surface. The
    surface-derived band moves as the bias fills, vanishes when fewer than two
    basins resolve, and collapses once the barrier is filled, so a count taken
    against it means something different every round (F7, F9). A fixed band on
    the coordinate the task defines its states on is comparable across rounds
    and across a change of biased CV.

    `max_cv_switches` is how many times the scientist may revise the biased
    coordinates within one campaign — by replacing the set (`switch_cv`) or by
    adding one coordinate and biasing all of them in parallel (`add_cv`, up
    to three). The biased phase starts on one coordinate and escalates only
    when the evidence says one cannot carry the transition. While revisions
    remain both actions are offered in the biased action space; once spent
    they are dropped from the tool schema, so a further revision is
    unrepresentable rather than emitted and then refused. Counted across
    resumes from the persisted rounds.

    `max_rounds`, `max_extra_ns`, `max_biased_ns` and `max_cv_switches` are
    loop-control bounds and may differ between invocations. Everything that
    defines what the campaign *is* — seed, initial_steps,
    report_interval_steps, equilibration_steps, the system spec, the engine,
    task_expectation, cv_upper_wall_nm, state_thresholds, min_recrossings and
    the bias shape — is locked at first init and a mismatch on resume raises.
    The thermostat temperature and timestep live on `spec.ensemble` and so are
    covered by the system-spec half of that lock; they used to be adapter class
    constants covered only by the engine-name lock, which would have stopped
    covering them the moment they became settable.

    `max_biased_ns` caps *cumulative* simulation time in the metadynamics
    phase, counting across rounds and across resumes (it is recomputed from
    the persisted biased rounds, so a restart cannot reset the meter). The
    round that would exceed it is shortened to land on the budget and the
    campaign then ends with `biased_budget_exhausted`; a remainder shorter than
    one trajectory frame is not run, since it could produce nothing to
    diagnose. The vanilla phase
    is not counted. Left at None the biased phase is bounded only by
    `max_rounds`. A compute budget stated only in `task_expectation` is
    advisory — the model can read it and still ask for more — so anything
    running unattended wants this set.

    `task_expectation` is free-form campaign-level guidance threaded into
    every `decide()` call — what the trajectory must accomplish, the
    characteristic timescale, the compute budget. Required for campaigns
    where the scientist may need to choose `switch_to_metad`; otherwise the
    LLM has no basis to judge "the budget cannot reach the transition."

    `observable` is the coordinate every round is judged on — the vanilla
    convergence bundle summarizes it and `state_thresholds` are positions on
    it. Left at None it is CA-RMSD to the campaign topology in Angstrom, the
    M1 observable. It defines what the campaign measures, so it locks with the
    rest of the config.

    `bias_pace` and `bias_factor` override the METAD deposition stride and the
    well-tempered gamma that `sampling.bias_designer` would otherwise pick.
    Left at None the designer's own defaults apply, so there is one source of
    truth for them rather than a second copy in this signature. Both are
    biased-phase physics and lock with the rest of the campaign config.
    gamma=10 flattens barriers up to ~gamma*kT; a surface deeper than that
    wants a larger one.

    `description` is prose about the system, used only by the pre-flight
    checks — it is compared against the structure that was actually fetched.
    It is deliberately *not* part of the locked config: editing prose must not
    stop a campaign resuming.

    `on_event` is an optional observer called with `(name, payload)` as the
    campaign progresses — `campaign_start`, `round_start`, `simulated`,
    `report`, `decision`, `override`, `pivot`, `campaign_end`. It cannot change
    what happens and anything it raises is swallowed, because a multi-hour
    biased run must not die because something watching it threw.

    `biased_adapter_factory` builds the adapter used after a `switch_to_metad`
    pivot from a rendered plumed.dat string. Defaults to an `OpenMMAdapter`
    over the same `SystemSpec` with `plumed_input` set. Inject to run the
    biased phase through a different engine (or a fake, in tests).
    """
    # A biased phase without task states would fall back to counting
    # recrossings between the two deepest basins of the current surface, which
    # F7 and F9 showed is not comparable between rounds — it migrates, vanishes
    # when fewer than two basins resolve, and inflates when the barrier fills.
    # `task_expectation` is the sole input gating `switch_to_metad`, so it is
    # exactly the predicate for "this campaign can reach a biased phase".
    # Refused here, before any MD is paid for, rather than at the pivot.
    if task_expectation is not None and state_thresholds is None:
        raise ValueError(
            "run_campaign: task_expectation is set, so this campaign can pivot "
            "to metadynamics, but state_thresholds is None. A biased phase "
            "needs the task's own state definitions to count recrossings "
            "against; without them the count is taken between whichever two "
            "basins are currently deepest, which means something different "
            "every round. Pass state_thresholds=(low, high) on the campaign "
            "observable."
        )
    if state_thresholds is not None:
        low, high = state_thresholds
        if not high > low:
            raise ValueError(
                f"run_campaign: state_thresholds must be (low, high) on the "
                f"campaign observable with high > low; got {state_thresholds!r}. "
                f"`count_recrossings` silently returns 0 for an inverted band, "
                f"so a swapped pair would read as a run that never crossed."
            )

    # A round shorter than one frame interval writes no frames — a 0-byte DCD
    # mdtraj cannot open — so it would be paid for and then fail to be
    # diagnosed. Extend rounds are floored in `_extend_steps`; the opening
    # round is the caller's number, so it is checked here, before any MD.
    if initial_steps < report_interval_steps:
        raise ValueError(
            f"run_campaign: initial_steps={initial_steps} is shorter than "
            f"report_interval_steps={report_interval_steps}, so the opening "
            f"round would write no trajectory frames and could not be diagnosed."
        )

    work_dir = Path(work_dir)
    rounds_dir = work_dir / "rounds"
    rounds_dir.mkdir(parents=True, exist_ok=True)
    plumed_dat_path = work_dir / _PLUMED_DAT_NAME

    if adapter is None:
        adapter = OpenMMAdapter(work_dir=work_dir, seed=seed)

    base_spec = adapter.spec
    # Captured once, from the vanilla adapter. The biased adapter is built by
    # `_default_biased_factory` over the same engine (and an injected factory
    # is documented as engine-matching, since the CV's atom indices are
    # resolved against this adapter's topology), so one reading holds for the
    # whole campaign.
    steps_per_ns = steps_per_ns_for(adapter)
    temperature_k = adapter.temperature_k

    if biased_adapter_factory is None:
        biased_adapter_factory = _default_biased_factory(
            adapter, work_dir=work_dir, seed=seed, spec=base_spec
        )

    config = {
        "seed": seed,
        "initial_steps": initial_steps,
        "report_interval_steps": report_interval_steps,
        "equilibration_steps": equilibration_steps,
        "system_spec": base_spec.to_dict(),
        # Engine identity is physics-bound: checkpoints are engine-specific
        # binary formats, and a swapped adapter would feed an OpenMM .chk to
        # `grompp -t` (or the reverse) instead of failing the config guard.
        "engine": type(adapter).__name__,
        # task_expectation is the only input gating `switch_to_metad`. Resuming
        # without it silently reverts an enhanced-sampling campaign to a plain
        # convergence task, so it locks with the rest.
        "task_expectation": task_expectation,
        # The wall bounds the CV the bias acts on, so it is biased-phase physics
        # rather than a loop-control bound: changing it on resume would join two
        # halves of a campaign sampled under different Hamiltonians.
        "cv_upper_wall_nm": cv_upper_wall_nm,
        # The band every biased-round recrossing count is taken against.
        # Resuming with a different one would splice two definitions of
        # "a transition" into a single campaign's history. Listed as a list
        # rather than a tuple so the JSON round-trip is stable.
        "state_thresholds": (
            list(state_thresholds) if state_thresholds is not None else None
        ),
        # The other half of that same definition: `state_thresholds` says where
        # the states are, `min_recrossings` says how many transitions between
        # them count as done. It is the threshold `fes_converged` compares the
        # count against, and `_refuse_premature_stop` reads that verdict to
        # decide whether the scientist may stop — so resuming with a different
        # one changes when the campaign is allowed to end, retroactively, for
        # rounds already judged under the old value.
        "min_recrossings": min_recrossings,
    }
    # Bias shape is biased-phase physics: resuming under a different gamma or
    # deposition stride would join two differently-filled surfaces into one
    # `sum_hills` integration. Added to the lock only when set, for the same
    # reason `SystemSpec.to_dict` omits a default ensemble — an unconditional
    # key would make every campaign recorded before this existed unresumable.
    # Omitted when default for the same reason `SystemSpec.to_dict` omits a
    # default ensemble: an unconditional key would strand campaigns recorded
    # before observables were declarable.
    if observable is not None and observable != ObservableSpec.ca_rmsd_angstrom():
        config["observable"] = observable.to_dict()
    if bias_pace is not None:
        config["bias_pace"] = bias_pace
    if bias_factor is not None:
        config["bias_factor"] = bias_factor
    store.init_campaign(work_dir, config)
    prior_rows = store.list_rounds(work_dir)
    rounds: list[RoundResult] = [_row_to_result(r) for r in prior_rows]

    if rounds and rounds[-1].decision.decision == "stop":
        return _finish(on_event, work_dir, rounds, "scientist_said_stop")

    if (
        rounds
        and rounds[-1].decision.decision == "switch_to_metad"
        and rounds[-1].plumed_dat_path is not None
    ):
        # Second pivot from an already-biased round: the in-process loop below
        # refuses this and ends cleanly for human inspection. Re-invoking must
        # not quietly perform the pivot the live run declined — the resume
        # branches would otherwise rebuild a bias from a *biased* trajectory's
        # spread, which is not the width anything was measured against.
        return _finish(on_event, work_dir, rounds, "switch_to_metad_requested")

    # Build the vanilla engine state up front. This is idempotent and cheap on
    # resume (rebuilds from cache), and it guarantees the cached System +
    # minimized state + topology exist for a biased adapter to reuse.
    adapter.prepare()
    adapter.start()

    # Pre-flight, on the starting structure, before a single step is
    # integrated. These compare across the one seam no schema can check: what
    # the task file *says* against what was actually built, and the
    # observable's own magnitude against the bands it will be judged in. A
    # campaign that cannot reach its own done criterion should cost seconds,
    # not the forty minutes it once did.
    preflight.check_residue_count(adapter.topology_path, description)
    reference = md.load(str(adapter.topology_path))
    first_value, observable_name = campaign_observable(
        reference, adapter.topology_path, observable
    )
    preflight.check_observable_scale(
        float(first_value[0]), state_thresholds, observable_name
    )
    _emit(
        on_event, "preflight_ok",
        residues=sum(1 for r in reference.topology.residues if r.is_protein),
        observable=observable_name,
        first_value=float(first_value[0]),
        state_thresholds=list(state_thresholds) if state_thresholds else None,
    )
    # Every CV proposal is resolved against this same structure the moment it
    # arrives, so a selection the topology cannot resolve goes back to the
    # model rather than being committed as a switch nothing can build.
    validate_proposal = _proposal_validator(reference)

    ledger_notes: list[store.LedgerNote] = list(store.list_ledger_notes(work_dir))

    in_metad = False
    current_plumed_dat: Path | None = None
    # The coordinates currently biased, replayed from the persisted decisions.
    active_cvs: list[MetadProposal] = _active_cvs(prior_rows)
    last = prior_rows[-1] if prior_rows else None

    if last is None:
        if equilibration_steps > 0:
            adapter.run_steps(equilibration_steps)
        start_round = 1
        n_steps = initial_steps
    elif last.decision in ("switch_to_metad", "switch_cv", "add_cv"):
        # Revision-resume: the round that redefined the biased coordinates is
        # persisted, but no round has run under the new bias yet. Rebuild it
        # exactly as the live loop would have — from the proposal(s) and that
        # round's trajectory — and warm-start from the walker's snapshotted
        # State when there is one. Ordered *before* the generic biased branch
        # below: a `switch_cv`/`add_cv` round is itself biased, so
        # `plumed_dat_path is not None` would match first and resume on the
        # coordinates the scientist just revised away from.
        if last.metad_proposal is None:
            raise RuntimeError(
                f"round {last.round_index} decided {last.decision} but stored "
                f"no metad_proposal; cannot build the bias"
            )
        wall_notes: list[str] = []
        adapter = _revise_bias(
            last.decision,
            previous=_active_cvs(prior_rows[:-1]),
            proposal=MetadProposal.from_dict(last.metad_proposal),
            source_trajectory=last.dcd_path,
            topology_path=adapter.topology_path,
            factory=biased_adapter_factory,
            plumed_dat_path=plumed_dat_path,
            temperature_k=temperature_k,
            cv_upper_wall_nm=cv_upper_wall_nm,
            bias_pace=bias_pace,
            bias_factor=bias_factor,
            observable=observable,
            warm_state=_read_state_snapshot(rounds_dir, last.round_index),
            notes=wall_notes,
        )
        in_metad = True
        current_plumed_dat = plumed_dat_path
        start_round = last.round_index + 1
        n_steps = initial_steps
        # The live revision writes these *after* `append_round`, so a crash
        # between the two loses them. Re-derived here; skipped for any the
        # live run did get written, since the ledger is append-only.
        _record_notes(
            work_dir, ledger_notes, last.round_index, wall_notes, skip_existing=True
        )
    elif last.plumed_dat_path is not None:
        # Mid-metaD-phase resume: rebuild the biased adapter from the persisted
        # plumed.dat and continue from that round's (biased) checkpoint. The
        # deposited bias is the other half of that resume point — restore the
        # snapshot paired with the checkpoint, then turn RESTART on so METAD
        # reads it back instead of backing it up and refilling from zero.
        _restore_bias_state(
            last.plumed_dat_path.parent, rounds_dir, last.round_index,
            _bias_state_files(active_cvs),
        )
        adapter = biased_adapter_factory(
            enable_restart(last.plumed_dat_path.read_text())
        )
        adapter.prepare()
        adapter.start()
        _require_checkpoint(last)
        adapter.load_checkpoint(last.checkpoint_path)  # type: ignore[arg-type]
        in_metad = True
        current_plumed_dat = last.plumed_dat_path
        start_round = last.round_index + 1
        n_steps = _extend_steps(
            last.extra_ns, max_extra_ns, steps_per_ns, report_interval_steps
        )
    else:
        # Vanilla resume.
        _require_checkpoint(last)
        adapter.load_checkpoint(last.checkpoint_path)  # type: ignore[arg-type]
        start_round = last.round_index + 1
        n_steps = _extend_steps(
            last.extra_ns, max_extra_ns, steps_per_ns, report_interval_steps
        )

    # Cumulative biased steps already spent, so a resume continues the meter
    # rather than restarting it.
    biased_steps_run = sum(
        r.n_steps for r in prior_rows if r.plumed_dat_path is not None
    )
    # Same reasoning as the biased-step meter: recomputed from disk so a
    # restart cannot buy the scientist a second allowance of CV switches.
    cv_switches_used = sum(
        1 for r in prior_rows if r.decision in ("switch_cv", "add_cv")
    )
    biased_step_budget = (
        int(max_biased_ns * steps_per_ns) if max_biased_ns is not None else None
    )

    _emit(
        on_event,
        "campaign_start",
        work_dir=str(work_dir),
        engine=type(adapter).__name__,
        forcefield=base_spec.forcefield,
        temperature_k=temperature_k,
        padding_nm=base_spec.padding_nm,
        timestep_fs=adapter.timestep_fs,
        observable=(observable or ObservableSpec.ca_rmsd_angstrom()).name,
        resuming_from_round=len(prior_rows),
        start_round=start_round,
        max_rounds=max_rounds,
        phase="metad" if in_metad else "vanilla",
    )

    for round_idx in range(start_round, max_rounds + 1):
        if in_metad and biased_step_budget is not None:
            remaining = biased_step_budget - biased_steps_run
            if remaining < report_interval_steps:
                # Spent, or less than one trajectory frame left — see the
                # `initial_steps` guard above for why that is not worth running.
                return _finish(on_event, work_dir, rounds, "biased_budget_exhausted")
            # Shorten the round that would overshoot rather than skipping it —
            # a partial round still deposits hills and still gets diagnosed.
            n_steps = min(n_steps, remaining)

        dcd = rounds_dir / f"round_{round_idx:03d}{adapter.trajectory_extension}"
        _emit(
            on_event, "round_start",
            round_index=round_idx, n_steps=n_steps,
            ns=n_steps / steps_per_ns, phase="metad" if in_metad else "vanilla",
        )
        started = time.monotonic()
        adapter.run_steps(
            n_steps,
            trajectory_path=dcd,
            report_interval_steps=report_interval_steps,
        )
        _emit(
            on_event, "simulated",
            round_index=round_idx, seconds=time.monotonic() - started,
            trajectory=str(dcd),
        )
        if in_metad:
            biased_steps_run += n_steps
        # Checkpoint first: everything after this line can fail on something
        # external (a `plumed sum_hills` that is not on PATH, a transient
        # Anthropic outage) and the MD is already paid for. Without a
        # checkpoint, restart has no resume point and re-runs the round. A
        # checkpoint with no matching row is the documented harmless case (D4).
        ckpt = adapter.save_checkpoint(rounds_dir / f"round_{round_idx:03d}.chk")
        _write_state_snapshot(adapter, rounds_dir, round_idx)
        if current_plumed_dat is not None:
            _snapshot_bias_state(
                current_plumed_dat.parent, rounds_dir, round_idx,
                _bias_state_files(active_cvs),
            )
        report = _round_report(
            dcd,
            adapter.topology_path,
            plumed_dat_path=current_plumed_dat,
            temperature_k=temperature_k,
            fes_dir=rounds_dir / f"round_{round_idx:03d}_fes",
            min_recrossings=min_recrossings,
            rounds_dir=rounds_dir,
            round_index=round_idx,
            state_thresholds=state_thresholds,
            observable=observable,
            cv_switches_used=cv_switches_used,
            max_cv_switches=max_cv_switches,
            active_cvs=active_cvs,
        )
        _emit(on_event, "report", round_index=round_idx, report=report)
        prior_summaries = [_compact_prior(r) for r in rounds]
        override_note: str | None = None
        try:
            decision = decide(
                report,
                prior_round_summaries=prior_summaries,
                hypothesis_ledger=[f"R{n.round_index}: {n.text}" for n in ledger_notes],
                task_expectation=task_expectation,
                phase="metad" if in_metad else "vanilla",
                allow_cv_switch=in_metad and cv_switches_used < max_cv_switches,
                max_extra_ns=max_extra_ns,
                validate_proposal=validate_proposal,
            )
        except DecisionRefused as refused:
            decision, override_note = _refuse_decision(refused)

        if in_metad:
            remaining = (
                biased_step_budget - biased_steps_run
                if biased_step_budget is not None
                else None
            )
            # At most one of the two fires: a refused proposal is already an
            # extend, and only a `stop` can be premature.
            decision, stop_note = _refuse_premature_stop(decision, report, remaining)
            override_note = override_note or stop_note
        if override_note:
            _emit(on_event, "override", round_index=round_idx, note=override_note)

        _emit(
            on_event, "decision",
            round_index=round_idx, decision=decision.decision,
            reason=decision.reason, extra_ns=decision.extra_ns,
            metad_proposal=(
                decision.metad_proposal.to_dict() if decision.metad_proposal else None
            ),
        )
        summary_path = rounds_dir / f"round_{round_idx:03d}.json"
        _persist_round_json(
            summary_path, round_idx, n_steps, dcd, report, decision, current_plumed_dat
        )
        store.append_round(
            work_dir,
            round_index=round_idx,
            n_steps=n_steps,
            dcd_path=dcd,
            checkpoint_path=ckpt,
            report=report,
            decision=decision.decision,
            reason=decision.reason,
            extra_ns=decision.extra_ns,
            metad_proposal=(
                decision.metad_proposal.to_dict() if decision.metad_proposal else None
            ),
            plumed_dat_path=current_plumed_dat,
        )
        for note in (decision.ledger_note, override_note):
            if not note:
                continue
            store.append_ledger_note(work_dir, round_index=round_idx, text=note)
            ledger_notes.append(
                store.LedgerNote(round_index=round_idx, text=note)
            )
        rounds.append(
            RoundResult(
                round_idx, n_steps, dcd, summary_path, report, decision, current_plumed_dat
            )
        )

        if decision.decision == "stop":
            return _finish(on_event, work_dir, rounds, "scientist_said_stop")

        if decision.decision == "switch_to_metad":
            if in_metad:
                # Already biased: a second switch is out of scope. End cleanly
                # so a human can inspect rather than rebuild the bias in a loop.
                return _finish(
                    on_event, work_dir, rounds, "switch_to_metad_requested"
                )
            assert decision.metad_proposal is not None  # guaranteed by the parser

        if decision.decision in ("switch_to_metad", "switch_cv", "add_cv"):
            assert decision.metad_proposal is not None  # guaranteed by the parser
            wall_notes: list[str] = []
            # A replacement or added CV is sized on *this* round's trajectory,
            # which for a revision was run under the outgoing bias. That spread
            # is inflated by the bias that drove the walker across the old
            # coordinate, which is what `_SIGMA_CEILINGS` in bias_designer
            # exists to catch. The walker itself carries over: the new adapter
            # warm-starts from this adapter's State rather than the cache.
            adapter = _revise_bias(
                decision.decision,
                previous=active_cvs,
                proposal=decision.metad_proposal,
                source_trajectory=dcd,
                topology_path=adapter.topology_path,
                factory=biased_adapter_factory,
                plumed_dat_path=plumed_dat_path,
                temperature_k=temperature_k,
                cv_upper_wall_nm=cv_upper_wall_nm,
                bias_pace=bias_pace,
                bias_factor=bias_factor,
                observable=observable,
                warm_state=_export_state(adapter),
                notes=wall_notes,
            )
            active_cvs = _revised_cvs(decision.decision, active_cvs, decision.metad_proposal)
            if decision.decision != "switch_to_metad":
                cv_switches_used += 1
            in_metad = True
            current_plumed_dat = plumed_dat_path
            n_steps = initial_steps
            _record_notes(work_dir, ledger_notes, round_idx, wall_notes)
            _emit(
                on_event, "pivot",
                round_index=round_idx, kind=decision.decision,
                plumed_dat=str(plumed_dat_path),
                cv=decision.metad_proposal.to_dict(),
                biased_cvs=[p.label for p in active_cvs],
            )
            continue

        n_steps = _extend_steps(
            decision.extra_ns, max_extra_ns, steps_per_ns, report_interval_steps
        )

    return _finish(on_event, work_dir, rounds, "max_rounds_reached")


def _finish(
    on_event: CampaignEvent | None,
    work_dir: Path,
    rounds: list[RoundResult],
    stop_reason: StopReason,
) -> CampaignResult:
    """Single exit point, so every `return` reports the outcome."""
    _emit(
        on_event, "campaign_end",
        stop_reason=stop_reason, n_rounds=len(rounds),
        biased_rounds=sum(1 for r in rounds if r.plumed_dat_path is not None),
    )
    return CampaignResult(work_dir, tuple(rounds), stop_reason)


def _default_biased_factory(
    adapter: MDAdapter,
    *,
    work_dir: Path,
    seed: int,
    spec: SystemSpec,
) -> BiasedAdapterFactory:
    """Biased adapter over the *same engine* that ran the vanilla phase.

    Engine-matching is a correctness requirement, not a preference. The CV's
    atom indices are resolved against the vanilla adapter's topology, and each
    engine builds a different system from the same `SystemSpec`: `gmx solvate`
    and OpenMM's Modeller place different numbers of waters, and pdb2gmx and
    Modeller name and order hydrogens differently. Handing those indices to
    another engine's system biases whichever atoms happen to sit at those
    positions — wrong physics, no error anywhere. (This is not an off-by-one:
    the 0-based to PLUMED 1-based conversion in `plumed_writer` is correct and
    no offset can reconcile two different atom orderings.)

    Only OpenMM has a bias path today. Any other engine gets a refusal naming
    the injection point rather than a silent engine swap.
    """
    if isinstance(adapter, OpenMMAdapter):

        def openmm_factory(plumed_input: str) -> MDAdapter:
            return OpenMMAdapter(
                work_dir=work_dir,
                seed=seed,
                spec=spec,
                plumed_input=plumed_input,
            )

        return openmm_factory

    engine = type(adapter).__name__

    def unsupported(plumed_input: str) -> MDAdapter:
        raise NotImplementedError(
            f"{engine} has no metadynamics path, and the CV's atom indices were "
            f"resolved against its topology, so they are not transferable to "
            f"another engine. Pass `biased_adapter_factory=` to run the biased "
            f"phase through a {engine}-compatible adapter."
        )

    return unsupported


# HILLS and COLVAR are to a biased round what the checkpoint is to a vanilla
# one: the state needed to continue. plumed.dat is the definition of the bias
# they were deposited under — and the live copy is rewritten in place by every
# pivot and CV revision, so without a per-round snapshot the rows for the
# rounds before a `switch_cv` point at a file describing a coordinate they
# never ran on. All of them are snapshotted together and restored together.
# Which HILLS files exist depends on how many coordinates are biased.


def _hills_files(active_cvs: list[MetadProposal]) -> list[str]:
    """The HILLS file(s) the current bias deposits into.

    One CV: `METAD` writes `HILLS`. Several: `PBMETAD` writes one per CV,
    `HILLS.<label>`. Must agree with what `_build_plumed_input` renders.
    """
    if len(active_cvs) <= 1:
        return [_HILLS_NAME]
    return [f"{_HILLS_NAME}.{p.label}" for p in active_cvs]


def _bias_state_files(active_cvs: list[MetadProposal]) -> list[str]:
    return _hills_files(active_cvs) + [_COLVAR_NAME, _PLUMED_DAT_NAME]


def _snapshot_bias_state(
    bias_dir: Path, rounds_dir: Path, round_index: int, names: list[str]
) -> None:
    """Copy the deposited bias alongside the round's checkpoint.

    Turning RESTART on makes the live HILLS load-bearing, which turns the
    existing mid-round crash window (D4: a crash between `save_checkpoint` and
    `append_round` leaves the round absent, so restart re-runs it) from
    "wasted time" into "wrong physics" — the re-run would deposit that round's
    hills a second time on top of the ones already on disk. Snapshotting at
    the same moment as the checkpoint means resume can restore the bias to
    exactly the point the positions correspond to.
    """
    for name in names:
        src = bias_dir / name
        if src.exists():
            shutil.copy2(src, _bias_snapshot_path(rounds_dir, round_index, name))


def _clear_bias_state(bias_dir: Path, names: list[str]) -> None:
    """Drop the live bias files so a replacement CV starts from zero bias.

    Hills deposited on the previous coordinate must not carry over: they
    describe a different CV and PLUMED would read them back as if they did
    not. Deleting rather than archiving is deliberate — the outgoing bias is
    already preserved at `rounds/round_NNN.hills*` by `_snapshot_bias_state`,
    and delete is idempotent, so a resume that re-enters this path cannot
    accumulate half-written archives of an interrupted round.
    """
    for name in names:
        (bias_dir / name).unlink(missing_ok=True)


def _restore_bias_state(
    bias_dir: Path, rounds_dir: Path, round_index: int, names: list[str]
) -> None:
    """Put the bias back to its state at the end of `round_index`.

    A missing snapshot is not an error: campaigns that pivoted before
    snapshotting existed have none, and leaving the live files untouched is
    the best available behaviour there.
    """
    bias_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        snapshot = _bias_snapshot_path(rounds_dir, round_index, name)
        if snapshot.exists():
            shutil.copy2(snapshot, bias_dir / name)


def _export_state(adapter: MDAdapter) -> str | None:
    """The walker's State, when the engine can hand it over (see `base.py`)."""
    export = getattr(adapter, "export_state_xml", None)
    return export() if callable(export) else None


def _write_state_snapshot(adapter: MDAdapter, rounds_dir: Path, round_index: int) -> None:
    """Beside the checkpoint, so a revision resumed after a crash warm-starts too."""
    xml = _export_state(adapter)
    if xml is not None:
        (rounds_dir / f"round_{round_index:03d}.{_STATE_NAME}").write_text(xml)


def _read_state_snapshot(rounds_dir: Path, round_index: int) -> str | None:
    path = rounds_dir / f"round_{round_index:03d}.{_STATE_NAME}"
    return path.read_text() if path.exists() else None


def _active_cvs(rows: list[store.RoundRow]) -> list[MetadProposal]:
    """The biased coordinate set, replayed from the persisted decisions.

    `switch_to_metad` and `switch_cv` each start a set of one; `add_cv`
    appends to it. Nothing else changes it, so the set is a pure function of
    the round history and never has to be stored separately.
    """
    cvs: list[MetadProposal] = []
    for r in rows:
        if r.metad_proposal is None:
            continue
        if r.decision in ("switch_to_metad", "switch_cv", "add_cv"):
            cvs = _revised_cvs(r.decision, cvs, MetadProposal.from_dict(r.metad_proposal))
    return cvs


def _revised_cvs(
    action: str, previous: list[MetadProposal], proposal: MetadProposal
) -> list[MetadProposal]:
    if action == "add_cv":
        return [*previous, proposal]
    return [proposal]


def _bias_snapshot_path(rounds_dir: Path, round_index: int, name: str) -> Path:
    return rounds_dir / f"round_{round_index:03d}.{name.lower()}"


def _round_report(
    trajectory_path: Path,
    topology_path: Path,
    *,
    plumed_dat_path: Path | None,
    temperature_k: float,
    fes_dir: Path,
    min_recrossings: int = 1,
    rounds_dir: Path | None = None,
    round_index: int | None = None,
    state_thresholds: tuple[float, float] | None = None,
    observable: ObservableSpec | None = None,
    cv_switches_used: int = 0,
    max_cv_switches: int = 0,
    active_cvs: list[MetadProposal] | None = None,
) -> dict[str, Any]:
    """Diagnostic bundle for one round, chosen by phase.

    With several coordinates biased in parallel the scientist still judges
    one *primary* surface: the campaign observable's own marginal when the
    observable is among them, otherwise the first coordinate's. The others
    are shown as the range each covered (`cv_ranges`), so a coordinate that
    is not moving is visible as such.

    A biased round also carries the free-energy surface *along the campaign
    observable*, reweighted from COLVAR — the surface the task's states are
    defined on, whatever CV is being biased, and the one a reference is
    compared against. PLUMED prints the observable on every COLVAR row beside
    the bias, so no trajectory frame has to be placed on its clock.

    A vanilla round gets the equilibrium convergence bundle. A biased round
    gets the free-energy bundle instead — *instead*, not alongside. The
    equilibrium statistics describe an equilibrium ensemble, and a biased
    trajectory is not one: the bias drives the observable, so a long
    autocorrelation means the bias is still filling and a bimodal marginal
    means the bias worked. Emitting them on a biased round would invite the
    scientist to read them as convergence evidence, which is the category
    error `diagnostics.free_energy` exists to remove.

    HILLS and COLVAR accumulate across the whole biased phase in a single
    file, so the surface integrated here is cumulative — which is what the
    well-tempered convergence test wants — not per-round.
    """
    if plumed_dat_path is None:
        report = make_report(trajectory_path, topology_path, observable)
        report["phase"] = "vanilla"
        return report

    # Only computed when there are task states to count against: it costs a
    # trajectory load, and without thresholds nothing consumes the result.
    series = observable_name = this_round = None
    if (
        state_thresholds is not None
        and rounds_dir is not None
        and round_index is not None
    ):
        series, observable_name, this_round = _accumulated_observable(
            rounds_dir, round_index, trajectory_path, topology_path, observable
        )

    bias_dir = plumed_dat_path.parent
    active = active_cvs or []
    report = metad_report(
        bias_dir / _primary_hills(active, observable),
        bias_dir / _COLVAR_NAME,
        fes_dir,
        temperature_k=temperature_k,
        min_recrossings=min_recrossings,
        observable=series,
        observable_name=observable_name,
        state_thresholds=state_thresholds,
    )
    if this_round is not None and this_round.size:
        # Where the walker was *this round*, not cumulatively. COLVAR appends
        # across the whole biased phase, so `cv_min`/`cv_max` keep reporting the
        # widest excursion the campaign ever made — which reads as "the full
        # range is being explored" long after the walker has stopped moving.
        # A campaign sat at 0.03-0.13 for two rounds while the scientist
        # correctly quoted a cumulative 0.39-10.01 and concluded the opposite.
        report["observable_min_this_round"] = float(this_round.min())
        report["observable_max_this_round"] = float(this_round.max())
        # Trapped-walker detection, computed rather than left to be inferred.
        # A campaign sat between 0.03 and 0.13 of its native contacts for two
        # rounds while the deposited bias grew past 115 kJ/mol and the
        # scientist, reading only cumulative ranges, concluded the full
        # coordinate was being explored.
        confined, rounds_confined = _confinement(
            rounds_dir, round_index, state_thresholds
        )
        report["confined_to_state"] = confined
        report["rounds_confined"] = rounds_confined
        # The other face of the trap. `rounds_confined` fires when a round sits
        # entirely inside one state; a walker that roams the disordered region
        # without ever re-entering the state it started from never trips it.
        # A real campaign sat at Q in [0.03, 0.55] for three rounds — below the
        # folded threshold of 0.7 every frame, straddling the unfolded one at
        # 0.3 — and reported `rounds_confined=0` throughout while the scientist
        # wrote "did not reach the folded state" three times and extended.
        since_low, since_high = _rounds_since_visited(
            rounds_dir, round_index, state_thresholds
        )
        report["rounds_since_low_visited"] = since_low
        report["rounds_since_high_visited"] = since_high
    colvar_path = bias_dir / _COLVAR_NAME
    if state_thresholds is not None and colvar_path.exists():
        profile = observable_surface_from_colvar(
            colvar_path, topology_path, observable, temperature_k
        )
        if profile is not None:
            report["observable_fes_path"] = str(
                write_fes(profile, fes_dir / "observable_fes.dat")
            )
            low, high = float(state_thresholds[0]), float(state_thresholds[1])
            report["delta_g_low_minus_high_kj_per_mol"] = delta_g_kj_per_mol(
                profile, low, high, temperature_k
            )
    report["biased_cvs"] = [p.label for p in active]
    if len(active) > 1 and colvar_path.exists():
        columns = load_colvar(colvar_path)
        report["cv_ranges"] = {
            p.label: [float(columns[p.label].min()), float(columns[p.label].max())]
            for p in active if p.label in columns
        }
    # The CV-revision allowance, so the scientist can weigh a switch against
    # what is left rather than only discovering the action is gone.
    report["cv_switches_used"] = cv_switches_used
    report["cv_switches_remaining"] = max(max_cv_switches - cv_switches_used, 0)
    report["phase"] = "metad"
    report["trajectory_path"] = str(trajectory_path)
    report["plumed_dat_path"] = str(plumed_dat_path)
    return report


def _primary_hills(active: list[MetadProposal], observable: ObservableSpec | None) -> str:
    """Which HILLS file the round's surface is integrated from."""
    if len(active) <= 1:
        return _HILLS_NAME
    spec = observable or ObservableSpec.ca_rmsd_angstrom()
    for p in active:
        if p.cv_type == spec.cv_type and tuple(p.selections) == tuple(spec.selections):
            return f"{_HILLS_NAME}.{p.label}"
    return f"{_HILLS_NAME}.{active[0].label}"


def _confinement(
    rounds_dir: Path | None,
    round_index: int | None,
    state_thresholds: tuple[float, float] | None,
) -> tuple[str | None, int]:
    """Which task state the walker has been stuck in, and for how many rounds.

    Walks backwards from this round and stops at the first one that was not
    confined, or was confined to the *other* state. Vanilla rounds write no
    observable file, so the count naturally stops at the pivot rather than
    running back into the unbiased phase.

    `None, 0` means the walker moved between the bands this round, which is the
    healthy case — a biased run is supposed to traverse them.
    """
    if rounds_dir is None or round_index is None or state_thresholds is None:
        return None, 0
    low, high = float(state_thresholds[0]), float(state_thresholds[1])
    state: str | None = None
    count = 0
    for index in range(round_index, 0, -1):
        path = rounds_dir / f"round_{index:03d}.obs.npy"
        if not path.exists():
            break
        series = np.load(path)
        if not series.size:
            break
        if series.max() <= low:
            here = "low"
        elif series.min() >= high:
            here = "high"
        else:
            break
        if state is None:
            state = here
        elif here != state:
            break
        count += 1
    return state, count


def observable_surface_from_colvar(
    colvar_path: Path,
    topology_path: Path,
    observable: ObservableSpec | None,
    temperature_k: float,
) -> Any:
    """The free-energy surface along the campaign observable, from COLVAR.

    Reads the observable column PLUMED printed (`COLVAR_OBSERVABLE_LABEL`),
    converts it to the observable's own units, and reweights it by the bias on
    the same row. None when COLVAR carries no such column — a campaign
    recorded before the observable was printed — or no bias column.

    Public because the benchmark scorer reads finished campaigns off disk
    with it.
    """
    spec = observable or ObservableSpec.ca_rmsd_angstrom()
    colvar = load_colvar(colvar_path)
    bias_column = next((k for k in colvar if k.endswith(".bias")), None)
    if COLVAR_OBSERVABLE_LABEL not in colvar or bias_column is None:
        return None
    reference = md.load(str(topology_path))
    with tempfile.TemporaryDirectory() as scratch:
        cv = design_cv(
            observable_cv_proposal(spec), reference.topology,
            reference=reference, output_dir=Path(scratch),
        )
    return reweighted_profile(
        colvar[COLVAR_OBSERVABLE_LABEL] * colvar_to_observable_factor(spec, cv),
        colvar[bias_column],
        temperature_k,
        label=spec.name,
    )


def _rounds_since_visited(
    rounds_dir: Path | None,
    round_index: int | None,
    state_thresholds: tuple[float, float] | None,
) -> tuple[int | None, int | None]:
    """Consecutive biased rounds, ending with this one, that never entered each state.

    Returned as `(low, high)`. 0 means the walker entered that state this
    round. Walks backwards over the per-round observable files exactly as
    `_confinement` does, so the count stops at the pivot.

    `None` when there is nothing to count against.
    """
    if rounds_dir is None or round_index is None or state_thresholds is None:
        return None, None
    low, high = float(state_thresholds[0]), float(state_thresholds[1])
    since = {"low": 0, "high": 0}
    open_ = {"low": True, "high": True}
    for index in range(round_index, 0, -1):
        path = rounds_dir / f"round_{index:03d}.obs.npy"
        if not path.exists():
            break
        series = np.load(path)
        if not series.size:
            break
        if open_["low"]:
            if series.min() <= low:
                open_["low"] = False
            else:
                since["low"] += 1
        if open_["high"]:
            if series.max() >= high:
                open_["high"] = False
            else:
                since["high"] += 1
        if not (open_["low"] or open_["high"]):
            break
    return since["low"], since["high"]


def _accumulated_observable(
    rounds_dir: Path,
    round_index: int,
    trajectory_path: Path,
    topology_path: Path,
    observable: ObservableSpec | None = None,
) -> tuple[np.ndarray, str, np.ndarray]:
    """The campaign observable over every biased round so far, in order.

    Cumulative, because the surface-derived count it replaces was cumulative:
    HILLS and COLVAR accumulate across the whole biased phase, so a per-round
    count would silently change what the number means.

    Each round's series is persisted next to its checkpoint rather than
    recomputed from the trajectories. The series is a few thousand floats
    (~16 KB) against a ~117 MB DCD, so concatenating the saved ones costs
    nothing while re-deriving them every round would mean re-reading the entire
    biased phase — over a gigabyte by the end of a 20 ns campaign.
    """
    traj = md.load(str(trajectory_path), top=str(topology_path))
    series, name = campaign_observable(traj, topology_path, observable)
    rounds_dir.mkdir(parents=True, exist_ok=True)
    np.save(rounds_dir / f"round_{round_index:03d}.obs.npy", series)

    chunks: list[np.ndarray] = []
    for index in range(1, round_index + 1):
        path = rounds_dir / f"round_{index:03d}.obs.npy"
        if path.exists():
            chunks.append(np.load(path))
    return (np.concatenate(chunks) if chunks else series), name, series


def _refuse_premature_stop(
    decision: Decision, report: dict[str, Any], remaining_steps: int | None
) -> tuple[Decision, str | None]:
    """Convert a biased-phase `stop` into an extend while the surface is unconverged.

    The system prompt already states the rule — `fes_converged=true` → stop,
    otherwise extend — but a rule the model can reason its way around is not a
    rule. On the first CLN025 campaign the scientist read `fes_converged=false`,
    wrote a paragraph rationalising the constituent numbers, and stopped with 16
    of 20 ns unspent and the done criterion one recrossing short.

    This does not take judgement away from the scientist: it still chooses the
    CV, sizes each extension, and decides when to stop once the diagnostic
    actually reports convergence. It removes only the ability to declare victory
    against the diagnostic. The refusal is written to the hypothesis ledger
    rather than swallowed, so the next round sees that it happened.
    """
    if decision.decision != "stop":
        return decision, None
    if report.get("fes_converged") is True:
        return decision, None
    if remaining_steps is not None and remaining_steps <= 0:
        return decision, None

    note = (
        f"stop refused: the scientist chose stop but fes_converged="
        f"{report.get('fes_converged')!r} "
        f"(drift={report.get('fes_drift_kj_per_mol')}, "
        f"recrossings={report.get('recrossings')}, "
        f"required>={report.get('min_recrossings')}). Budget remains, so the "
        f"round was converted to an extend. Reason given was: {decision.reason}"
    )
    return replace(decision, decision="extend", extra_ns=decision.extra_ns or 0.5), note


def _refuse_decision(refused: DecisionRefused) -> tuple[Decision, str]:
    """Convert a decision `decide()` could not get right into an extend, on record.

    `decide()` has already sent the problem back to the model
    `refused.attempts` times. Raising here would lose the round's MD (nothing
    is persisted yet) and, worse, restart would re-run it into the same
    answer. Extending keeps the campaign moving under its current Hamiltonian
    and writes the refusal to the ledger so the next round decides with the
    error in front of it — the same shape as `_refuse_premature_stop`.

    Two refusals arrive here. A proposal the topology cannot resolve: the
    proposal is dropped. Numbers cited that disagree with the report: the
    model's own ledger note is withheld too, since it was written from the
    misread values and the ledger is what every later round remembers.
    """
    d = refused.decision
    what = (
        "a CV the campaign topology cannot resolve"
        if isinstance(refused, UnresolvableProposal)
        else "numbers that disagree with the report it was given"
    )
    note = (
        f"decision refused: the scientist chose {d.decision} citing {what}, "
        f"{refused.attempts} time(s) running ({refused.error}). Converted to an "
        f"extend; its ledger note for this round was withheld. Reason given "
        f"was: {d.reason}"
    )
    return (
        replace(
            d, decision="extend", extra_ns=d.extra_ns or 0.5,
            metad_proposal=None, ledger_note=None,
        ),
        note,
    )


def _proposal_validator(reference: md.Trajectory) -> Callable[[MetadProposal], None]:
    """Resolve a proposal against the campaign topology, before it is committed.

    The same `design_cv` call the pivot makes, run on the proposal the moment
    it arrives. Until now the first resolution happened inside
    `_pivot_to_metad`, after `store.append_round` had recorded the switch: a
    selection string the topology could not resolve raised there, and every
    restart re-entered the pivot branch and raised again with `decide()` never
    called — the campaign was unrecoverable without editing state.db. Raised
    here, the error goes back to the model as a tool result and it proposes
    again.

    `rmsd` writes a reference PDB as a side effect of resolving; it goes to a
    scratch directory so a rejected proposal leaves nothing in the campaign.
    """

    def validate(proposal: MetadProposal) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            design_cv(
                CVProposal(
                    cv_type=proposal.cv_type,
                    selections=tuple(proposal.selections),
                    label=proposal.label,
                ),
                reference.topology,
                reference=reference,
                output_dir=Path(scratch),
            )

    return validate


def _record_notes(
    work_dir: Path,
    ledger_notes: list[store.LedgerNote],
    round_index: int,
    texts: list[str],
    *,
    skip_existing: bool = False,
) -> None:
    """Append notes to the ledger and to the copy the scientist is shown.

    `skip_existing` is for the resume paths: a pivot re-entered after a crash
    regenerates the wall notes the live run may already have written, and the
    ledger is append-only, so this is what keeps it from saying the same thing
    twice.
    """
    for text in texts:
        if skip_existing and any(
            n.round_index == round_index and n.text == text for n in ledger_notes
        ):
            continue
        store.append_ledger_note(work_dir, round_index=round_index, text=text)
        ledger_notes.append(store.LedgerNote(round_index=round_index, text=text))


def _extend_steps(
    extra_ns: float | None,
    max_extra_ns: float,
    steps_per_ns: int,
    report_interval_steps: int,
) -> int:
    """Steps for an extend round, clamped to the caller's `max_extra_ns`.

    What SQLite stores is the model's raw request, so the clamp has to be
    re-applied on every read. Applying it only in the live loop made a resumed
    campaign run a longer round than the uninterrupted one would have.

    Floored at one trajectory frame rather than one step. The reporter writes
    a frame every `report_interval_steps`, so a shorter round produces a DCD
    with no frames at all — a 0-byte file mdtraj cannot open — and the round
    report raises after the MD is spent; restart then re-derives the same
    length from the stored `extra_ns` and fails the same way. `extra_ns` has no
    lower bound in the tool schema, so the floor lives here.
    """
    return max(
        int(min(extra_ns or 0.5, max_extra_ns) * steps_per_ns), report_interval_steps
    )


def _revise_bias(
    action: str,
    *,
    previous: list[MetadProposal],
    proposal: MetadProposal,
    source_trajectory: Path,
    topology_path: Path,
    factory: BiasedAdapterFactory,
    plumed_dat_path: Path,
    temperature_k: float,
    cv_upper_wall_nm: float | None = None,
    bias_pace: int | None = None,
    bias_factor: float | None = None,
    observable: ObservableSpec | None = None,
    warm_state: str | None = None,
    notes: list[str] | None = None,
) -> MDAdapter:
    """Apply a coordinate revision to the bias on disk; return the started adapter.

    `switch_to_metad` and `switch_cv` start a fresh bias on one coordinate:
    every live bias file is dropped (the outgoing ones are already snapshotted
    beside their round). `add_cv` keeps the hills on the retained coordinates
    and biases the new one beside them in parallel — a single-CV `HILLS` is
    moved to the per-CV name `PBMETAD` reads, the new coordinate starts from
    nothing, COLVAR restarts (its columns change), and RESTART is turned on so
    the retained hills are read back rather than backed up. Every step is
    idempotent, so a resume that re-enters this path after a crash lands in
    the same place.

    `warm_state` places the walker where it was when the revision was
    decided; without it the adapter starts from the cached structure.
    """
    bias_dir = plumed_dat_path.parent
    bias_dir.mkdir(parents=True, exist_ok=True)
    revised = _revised_cvs(action, previous, proposal)
    if len(revised) > 3:
        raise ValueError(
            f"{action}: the biased set would have {len(revised)} coordinates; "
            f"the cap is three"
        )
    restart = False
    if action == "add_cv":
        if len(previous) == 1:
            single, per_cv = bias_dir / _HILLS_NAME, bias_dir / f"{_HILLS_NAME}.{previous[0].label}"
            if single.exists() and not per_cv.exists():
                single.rename(per_cv)
        (bias_dir / _COLVAR_NAME).unlink(missing_ok=True)
        restart = any((bias_dir / name).exists() for name in _hills_files(revised))
    else:
        _clear_bias_state(bias_dir, _bias_state_files(previous) + _bias_state_files(revised))

    plumed_input = _build_plumed_input(
        revised,
        source_trajectory,
        topology_path,
        bias_dir,
        temperature_k=temperature_k,
        cv_upper_wall_nm=cv_upper_wall_nm,
        bias_pace=bias_pace,
        bias_factor=bias_factor,
        observable=observable,
        notes=notes,
    )
    if restart:
        plumed_input = enable_restart(plumed_input)
    plumed_dat_path.write_text(plumed_input)
    biased = factory(plumed_input)
    biased.prepare()
    biased.start()
    load = getattr(biased, "load_state_xml", None)
    if warm_state is not None and callable(load):
        load(warm_state)
    return biased


# Grid bounds per coordinate type for the parallel bias. PBMETAD without a
# grid re-sums every hill every step and its cost grows with the run; with one
# the cost is constant, but a hill centre outside the grid is a fatal PLUMED
# error, so the bounds are generous. Torsions are periodic and their grid has
# to span exactly one period, which PLUMED spells `-pi`/`pi`.
_GRID_LENGTH_NM = (-0.2, 2.5)
_GRID_CONTACTS = (-0.3, 1.3)
_GRID_TORSION = ("-pi", "pi")


def _grid_bounds(cv: Any) -> tuple[Any, Any]:
    if isinstance(cv, TorsionCV):
        return _GRID_TORSION
    if isinstance(cv, ContactsCV):
        return _GRID_CONTACTS
    return _GRID_LENGTH_NM


def _build_plumed_input(
    proposals: list[MetadProposal],
    trajectory_path: Path,
    topology_path: Path,
    output_dir: Path,
    *,
    temperature_k: float,
    cv_upper_wall_nm: float | None = None,
    bias_pace: int | None = None,
    bias_factor: float | None = None,
    observable: ObservableSpec | None = None,
    notes: list[str] | None = None,
) -> str:
    """Proposals → resolved CVs → sized bias → rendered plumed.dat text.

    One proposal renders well-tempered `METAD` on it. Several render `PBMETAD`
    — one bias per coordinate, deposited in parallel, each converging to its
    own marginal — with a grid per coordinate. Every length-dimensioned
    coordinate gets the campaign's upper wall.

    `notes` collects anything the scientist needs to know about the bias that
    the diagnostics cannot show it — currently the walls. plumed.dat records
    the same thing as comments, but the scientist never reads plumed.dat.

    `observable`, when given, is resolved too and printed in COLVAR under
    `COLVAR_OBSERVABLE_LABEL`, unbiased. That column is what the surface along
    the observable is reweighted from.

    `output_dir` is where PLUMED writes HILLS and COLVAR. It has to be
    absolute and campaign-local: PLUMED resolves relative FILE= paths against
    the process working directory, so the deposited bias would otherwise land
    outside the campaign entirely.
    """
    if not proposals:
        raise ValueError("_build_plumed_input: at least one proposal")
    # Loaded with coordinates, not just connectivity: an `rmsd` CV measures
    # against a reference structure, and the campaign topology is the same
    # reference the vanilla observable uses, so both phases score against one
    # fixed structure rather than two different ones.
    reference = md.load(str(topology_path))
    cvs = [
        design_cv(
            CVProposal(cv_type=p.cv_type, selections=tuple(p.selections), label=p.label),
            reference.topology,
            reference=reference,
            output_dir=output_dir,
        )
        for p in proposals
    ]
    # None means "let bias_designer decide", so its defaults stay the single
    # definition of PACE and BIASFACTOR rather than being restated here.
    overrides = {
        k: v
        for k, v in (("pace", bias_pace), ("bias_factor", bias_factor))
        if v is not None
    }
    sized = [
        design_bias(cv, trajectory_path, topology_path, temperature_k=temperature_k, **overrides)
        for cv in cvs
    ]
    walls = []
    for cv in cvs:
        wall = design_upper_wall(
            cv, cv_upper_wall_nm, trajectory_path=trajectory_path, topology_path=topology_path,
        )
        if notes is not None:
            notes.extend(_wall_notes(cv, wall, cv_upper_wall_nm))
        if wall is not None:
            walls.append(wall)
    if len(cvs) == 1:
        bias: Any = sized[0]
    else:
        first = sized[0]
        bias = ParallelBias(
            cv_labels=tuple(cv.label for cv in cvs),
            sigma=tuple(b.sigma[0] for b in sized),
            height=first.height, pace=first.pace,
            bias_factor=first.bias_factor, temperature_k=first.temperature_k,
            grid=tuple(_grid_bounds(cv) for cv in cvs),
        )
    printed: tuple[Any, ...] = tuple(cvs)
    if observable is not None:
        printed += (
            design_cv(
                observable_cv_proposal(observable),
                reference.topology,
                reference=reference,
                output_dir=output_dir,
            ),
        )
    return PlumedInput(
        cvs=printed,
        bias=bias,
        walls=tuple(walls),
        output_dir=Path(output_dir).resolve(),
    ).render()


def _wall_notes(cv: Any, wall: Any, requested: float | None) -> list[str]:
    """What the scientist should know about the bound on this coordinate.

    Three cases matter. An unbounded coordinate with no bound at all is F6 —
    the bias drives the walker outward forever. A bound beyond what the box can
    hold is F11 — the solute reaches its own periodic image before the wall
    pushes back. And a bound derived from the box is worth saying out loud,
    because the campaign did not choose it.
    """
    from mdpilot.sampling.bias_designer import _LENGTH_DIMENSIONED

    if not isinstance(cv, _LENGTH_DIMENSIONED):
        return []
    if wall is None:
        return [
            f"WARNING: the biased CV '{cv.label}' is length-dimensioned and so "
            f"unbounded above, and no upper wall could be set — none was "
            f"configured and the box limit could not be measured from the "
            f"source trajectory. Well-tempered metadynamics will keep driving "
            f"the walker outward with nothing to turn it around (F6). Set "
            f"cv_upper_wall_nm for this campaign."
        ]
    if wall.derived_from_box:
        return [
            f"No wall position was configured for '{cv.label}', so one was "
            f"measured from the source trajectory: {wall.at:.2f} is where the "
            f"solute would reach its own periodic image. This bounds the "
            f"artifact, not the science — if the task's unfolded state lies "
            f"beyond it, the box is too small for the question being asked."
        ]
    if wall.exceeds_box_limit:
        return [
            f"WARNING: the wall on '{cv.label}' is at {wall.at:g}, but the box "
            f"can only hold {wall.box_limit_nm:.2f} before the solute reaches "
            f"its own periodic image (F11). Sampling past {wall.box_limit_nm:.2f} "
            f"is contaminated by self-interaction. Enlarge padding_nm or lower "
            f"the wall."
        ]
    return []


def _require_checkpoint(row: store.RoundRow) -> None:
    if row.checkpoint_path is None or not row.checkpoint_path.exists():
        raise FileNotFoundError(
            f"round {row.round_index} has no readable checkpoint; cannot resume"
        )


def _row_to_result(row: store.RoundRow) -> RoundResult:
    metad = (
        MetadProposal.from_dict(row.metad_proposal) if row.metad_proposal else None
    )
    return RoundResult(
        index=row.round_index,
        n_steps=row.n_steps,
        dcd_path=row.dcd_path,
        summary_path=row.dcd_path.with_suffix(".json"),
        report=row.report,
        decision=Decision(
            decision=cast(
                Literal["extend", "stop", "switch_to_metad", "switch_cv", "add_cv"],
                row.decision,
            ),
            reason=row.reason,
            extra_ns=row.extra_ns,
            metad_proposal=metad,
        ),
        plumed_dat_path=row.plumed_dat_path,
    )


def _compact_prior(r: RoundResult) -> dict[str, Any]:
    """Lean view of a prior round for the scientist's context — no raw report.

    Phase-keyed for the same reason the per-round report is: carrying `ess`
    and `plateau_reached` forward from a biased round would re-introduce the
    equilibrium statistics the biased report deliberately omits.
    """
    base = {
        "round_index": r.index,
        "n_steps": r.n_steps,
        "phase": r.report.get("phase", "vanilla"),
        "decision": r.decision.decision,
        "reason": r.decision.reason,
    }
    if r.plumed_dat_path is not None:
        base.update(
            # Which coordinate this round was judged on. Across a switch_cv the
            # history holds counts from two different CVs, and a bare list of
            # recrossings would invite exactly the comparison F7 was about.
            cv_label=r.report.get("cv_label"),
            biased_cvs=r.report.get("biased_cvs"),
            fes_drift_kj_per_mol=r.report.get("fes_drift_kj_per_mol"),
            recrossings=r.report.get("recrossings"),
            # Carried so the trend is visible: a range that shrinks round on
            # round is a walker settling into one state, whatever the
            # cumulative numbers say.
            observable_min_this_round=r.report.get("observable_min_this_round"),
            observable_max_this_round=r.report.get("observable_max_this_round"),
            rounds_since_low_visited=r.report.get("rounds_since_low_visited"),
            rounds_since_high_visited=r.report.get("rounds_since_high_visited"),
            # The boundaries move as the surface fills, so a count carried
            # forward without them is not comparable across rounds.
            recrossing_low=r.report.get("recrossing_low"),
            recrossing_high=r.report.get("recrossing_high"),
            fes_converged=r.report.get("fes_converged"),
        )
    else:
        base.update(
            trajectory_length_ns=r.report.get("trajectory_length_ns"),
            ess=r.report.get("ess"),
            plateau_reached=r.report.get("plateau_reached"),
        )
    return base


def _persist_round_json(
    path: Path,
    round_idx: int,
    n_steps: int,
    dcd: Path,
    report: dict[str, Any],
    decision: Decision,
    plumed_dat_path: Path | None,
) -> None:
    payload = {
        "round_index": round_idx,
        "n_steps": n_steps,
        "dcd_path": str(dcd),
        "plumed_dat_path": str(plumed_dat_path) if plumed_dat_path else None,
        "report": report,
        "decision": {
            "decision": decision.decision,
            "reason": decision.reason,
            "extra_ns": decision.extra_ns,
            "metad_proposal": (
                decision.metad_proposal.to_dict() if decision.metad_proposal else None
            ),
            # What the model said it read, kept with the round so a reason can
            # be audited against the numbers it claims to rest on.
            "cited": [{"field": f, "value": v} for f, v in decision.cited],
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON-serializable: {type(o)}")
