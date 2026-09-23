"""Compose the per-round diagnostic report bundle.

The report is the structured artifact handed to `scientist.decide()`. It must
stay compact (no raw trajectories, no per-level arrays) so that round summaries
fit in the LLM's context. Trajectory bytes stay on disk; the bundle holds
paths instead.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mdtraj as md
import numpy as np

from mdpilot.diagnostics.autocorrelation import autocorrelation
from mdpilot.diagnostics.block_averaging import block_average
from mdpilot.diagnostics.exploration import exploration
from mdpilot.observables import ObservableSpec, campaign_observable

_MIN_FRAMES_FOR_STATISTICS = 8

# Kept as a re-export: the campaign observable and its default now live in
# `mdpilot.observables`, because an observable is a collective variable and
# `sampling/` already knows how to compute five of them.
OBSERVABLE_NAME = ObservableSpec.ca_rmsd_angstrom().name

__all__ = ["OBSERVABLE_NAME", "campaign_observable", "make_report", "to_json"]


def make_report(
    dcd_path: Path,
    top_path: Path,
    observable: ObservableSpec | None = None,
) -> dict[str, Any]:
    """Load a trajectory, compute the campaign observable, summarize.

    `observable` defaults to CA-RMSD in Angstrom — the M1 observable — so a
    campaign that declares nothing behaves exactly as it did.

    The reference is the campaign's topology structure, *not* this round's
    first frame. A per-round reference makes every round a different
    observable: round 3 would measure displacement from wherever round 3
    happened to start, while the scientist is shown `ess` and
    `plateau_reached` across rounds as if they described one time series. A
    campaign drifting steadily away from its starting structure would then
    show a clean plateau in every round and stop. `top_path` is written once
    by the adapter's `start()` (post-equilibration) and is constant for the
    life of the campaign, so RMSD against it is comparable across rounds.
    """
    dcd_path = Path(dcd_path)
    top_path = Path(top_path)
    traj = md.load(str(dcd_path), top=str(top_path))
    series, observable_name = campaign_observable(traj, top_path, observable)

    if traj.n_frames > 1:
        frame_dt_ps = float(np.diff(traj.time).mean())
        length_ns = float(traj.time[-1] - traj.time[0]) / 1000.0
    else:
        frame_dt_ps = float("nan")
        length_ns = 0.0

    return _summarize(
        observable=series,
        observable_name=observable_name,
        dcd_path=dcd_path,
        top_path=top_path,
        frame_dt_ps=frame_dt_ps,
        length_ns=length_ns,
    )


def _summarize(
    *,
    observable: np.ndarray,
    observable_name: str,
    dcd_path: Path,
    top_path: Path,
    frame_dt_ps: float,
    length_ns: float,
) -> dict[str, Any]:
    n = int(observable.size)
    base: dict[str, Any] = {
        "trajectory_path": str(dcd_path),
        "topology_path": str(top_path),
        "n_frames": n,
        "frame_dt_ps": frame_dt_ps,
        "trajectory_length_ns": length_ns,
        "observable_name": observable_name,
    }
    if n < _MIN_FRAMES_FOR_STATISTICS:
        base.update(
            mean=float(observable.mean()) if n > 0 else None,
            plateau_reached=False,
            well_sampled=False,
            ess=float(n),
            statistical_inefficiency_block=None,
            statistical_inefficiency_autocorr=None,
            tau_int_frames=None,
            bimodality_coefficient=None,
            n_basins=None,
            minor_basin_occupancy=None,
            exploring=None,
            note=f"too few frames (n={n} < {_MIN_FRAMES_FOR_STATISTICS}) for statistics",
        )
        return base

    block = block_average(observable)
    autocorr = autocorrelation(observable)
    explore = exploration(observable)
    base.update(
        mean=block.mean,
        sem_blocked=block.sem,
        sem_naive=block.sem_naive,
        plateau_reached=block.plateau_reached,
        statistical_inefficiency_block=block.statistical_inefficiency,
        statistical_inefficiency_autocorr=autocorr.statistical_inefficiency,
        tau_int_frames=autocorr.tau_int,
        ess=autocorr.ess,
        well_sampled=autocorr.well_sampled,
        bimodality_coefficient=explore.bimodality_coefficient,
        n_basins=explore.n_basins,
        minor_basin_occupancy=explore.minor_basin_occupancy,
        exploring=explore.exploring,
    )
    return base


def to_json(report: dict[str, Any]) -> str:
    """Serialize a report to compact JSON (deterministic key order)."""
    return json.dumps(report, sort_keys=True, default=_json_default)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(f"not JSON-serializable: {type(o)}")


# --------------------------------------------------------------------------
# Three blocks: validity / precision / correctness
# --------------------------------------------------------------------------
#
# `make_report` and `metad_report` return flat bundles, as they always have.
# The report handed to the scientist is those numbers regrouped by what kind
# of claim each supports (docs/falsifiers.md §1):
#
#   validity     did the simulation do what it claimed, mechanically
#   precision    has the estimate stopped moving
#   correctness  the answer, and the falsifiers that tried to refute it
#
# Regrouping, not rewriting: every number keeps its name and value. A key
# that belongs to no block raises, so a new statistic has to be filed.

from mdpilot.diagnostics.falsifiers import summarize  # noqa: E402

_VALIDITY_KEYS = frozenset({
    "dcd_path", "top_path", "trajectory_path", "topology_path", "plumed_dat_path", "hills_path",
    "fes_path", "colvar_path", "observable_fes_path",
    "frame_dt_ps", "trajectory_length_ns", "n_frames", "note", "n_fes_estimates",
    "cv_start", "cv_min", "cv_max", "cv_ranges", "cv_label", "biased_cvs",
    "observable", "observable_name", "recrossing_observable",
})
_PRECISION_KEYS = frozenset({
    "mean", "sem_blocked", "sem_naive", "plateau_reached",
    "statistical_inefficiency_block", "statistical_inefficiency_autocorr",
    "tau_int_frames", "ess", "well_sampled",
    "bimodality_coefficient", "n_basins", "minor_basin_occupancy", "exploring",
    "fes_drift_kj_per_mol", "fes_converged",
    "recrossings", "barrier_crossed", "recrossing_low", "recrossing_high",
    "recrossing_basis", "min_recrossings",
    "n_basins_fes", "barrier_kj_per_mol", "fes_depth_kj_per_mol",
})
_ACTION_SPACE_KEYS = frozenset({"cv_switches_used", "cv_switches_remaining"})
_ANSWER_KEYS = frozenset({"delta_g_low_minus_high_kj_per_mol"})
_BLOCKS = ("validity", "precision", "action_space")


def precision_gate(phase: str, precision: dict[str, Any]) -> bool | None:
    """The precision block's one verdict: has the estimate stopped moving.

    Biased: `fes_converged` (drift < kT and enough recrossings). Unbiased:
    plateau reached and well sampled. None when the statistics could not be
    taken. It is a precision statement only; stop permission also needs the
    correctness block's `not_refuted`.
    """
    if phase == "metad":
        return precision.get("fes_converged")
    plateau = precision.get("plateau_reached")
    if plateau is None:
        return None
    return bool(plateau and precision.get("well_sampled"))


def group_report(
    flat: dict[str, Any],
    *,
    phase: str,
    falsifiers: dict[str, dict[str, Any]],
    answer: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Flat bundle -> the three-block report the scientist reads."""
    validity: dict[str, Any] = {}
    precision: dict[str, Any] = {}
    action: dict[str, Any] = {}
    answer = dict(answer or {})
    for key, value in flat.items():
        if key == "phase":
            continue
        if key in _VALIDITY_KEYS:
            validity[key] = value
        elif key in _PRECISION_KEYS:
            precision[key] = value
        elif key in _ACTION_SPACE_KEYS:
            action[key] = value
        elif key in _ANSWER_KEYS:
            answer[key] = value
        else:
            raise KeyError(
                f"group_report: {key!r} is filed under no block; add it to "
                f"_VALIDITY_KEYS, _PRECISION_KEYS, _ACTION_SPACE_KEYS or _ANSWER_KEYS"
            )
    precision["gate"] = precision_gate(phase, precision)
    return {
        "phase": phase,
        "validity": validity,
        "precision": precision,
        "correctness": {
            "answer": answer,
            "falsifiers": falsifiers,
            "summary": summarize(falsifiers),
        },
        "action_space": action,
    }


_ABSENT = object()


def _lookup(report: dict[str, Any], key: str) -> Any:
    """Find `key` in a grouped or a flat report; `_ABSENT` when nowhere.

    A dotted key is a path (`correctness.falsifiers.time_window_invariance.
    magnitude`). A bare key is searched at the top level, then in each block,
    then in the answer, then in every falsifier's inputs — so a citation the
    model writes as `recrossings` still resolves, and reports persisted before
    the blocks existed still read.
    """
    if "." in key:
        node: Any = report
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return _ABSENT
            node = node[part]
        return node
    if key in report:
        return report[key]
    for block in _BLOCKS:
        sub = report.get(block)
        if isinstance(sub, dict) and key in sub:
            return sub[key]
    correctness = report.get("correctness")
    if isinstance(correctness, dict):
        answer = correctness.get("answer")
        if isinstance(answer, dict) and key in answer:
            return answer[key]
        for f in (correctness.get("falsifiers") or {}).values():
            inputs = f.get("inputs") if isinstance(f, dict) else None
            if isinstance(inputs, dict) and key in inputs:
                return inputs[key]
    return _ABSENT


def has_report_field(report: dict[str, Any], key: str) -> bool:
    return _lookup(report, key) is not _ABSENT


def report_field(report: dict[str, Any], key: str, default: Any = None) -> Any:
    value = _lookup(report, key)
    return default if value is _ABSENT else value
