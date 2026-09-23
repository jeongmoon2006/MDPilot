"""Correctness falsifiers for a free-energy answer.

Design in `docs/falsifiers.md`. The short form: without an external reference
the quantity a campaign computes cannot be confirmed to be the quantity it
intended, only refuted. So every correctness check here is a *falsifier* —
a transformation under which the answer must be invariant, plus a tolerance —
and its states are

    refuted          the answer was not invariant beyond tolerance
    not_refuted      it was invariant; this is NOT "correct"
    not_evaluable    the transformation applies but the data cannot yet
                     support it (too few frames, a state never visited)
    not_applicable   the transformation has nothing to act on in this
                     campaign (no wall on the biased coordinates, the
                     observable is not biased, no second walker configured)

`not_evaluable` blocks acceptance; `not_applicable` does not, because an
invariance with nothing to be invariant under is not evidence either way. The
distinction is reported, never collapsed.

Universal by construction: nothing in this module names a CV type, a residue
or a system. Inputs are an observable series, a bias series, two thresholds,
a temperature and free-energy surfaces. Tolerances are data, passed in.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mdpilot.diagnostics.free_energy import (
    _KB_KJ_PER_MOL_K,
    FreeEnergySurface,
    delta_g_kj_per_mol,
    reweighted_profile,
    surface_rms_kj_per_mol,
)

REFUTED = "refuted"
NOT_REFUTED = "not_refuted"
NOT_EVALUABLE = "not_evaluable"
NOT_APPLICABLE = "not_applicable"

# The fixed set for the well-tempered metadynamics family, in report order.
# The LLM never selects from it; the loop evaluates every member every round.
FALSIFIER_NAMES: tuple[str, ...] = (
    "seed_invariance",
    "estimator_invariance",
    "time_window_invariance",
    "occupancy_invariance",
    "state_definition_invariance",
    "bias_accounting_invariance",
)

# One convention, stated once: the precision gate uses kT (drift < kT) and the
# reference match uses 1 kT, so no falsifier is laxer than either. Provisional
# until the calibration in docs/falsifiers.md §8 says otherwise.
TOLERANCE_KT = 1.0
TOLERANCE_SOURCE_KT = "provisional: 1 kT by convention (docs/falsifiers.md §8)"

# Fewer frames than this in a window and a reweighted ΔG is noise, not an
# estimate. Same floor `diagnostics.report` uses for the vanilla statistics.
_MIN_FRAMES_PER_WINDOW = 8
# PLUMED's wall bias is exactly zero below `AT`; anything above this is a
# frame the wall was acting on.
_WALL_ACTIVE_KJ_PER_MOL = 1e-9


def kt_kj_per_mol(temperature_k: float) -> float:
    return _KB_KJ_PER_MOL_K * temperature_k


def record(
    name: str,
    *,
    invariance: str,
    transformation: str,
    statistic: str,
    tolerance: float | None,
    tolerance_source: str,
    magnitude: float | None,
    state: str,
    note: str | None = None,
    inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One falsifier's record, the shape every entry in the report carries."""
    if state not in (REFUTED, NOT_REFUTED, NOT_EVALUABLE, NOT_APPLICABLE):
        raise ValueError(f"falsifier {name!r}: unknown state {state!r}")
    return {
        "name": name,
        "invariance": invariance,
        "transformation": transformation,
        "statistic": statistic,
        "magnitude": None if magnitude is None else float(magnitude),
        "tolerance": None if tolerance is None else float(tolerance),
        "tolerance_source": tolerance_source,
        "state": state,
        "fired": state == REFUTED,
        "note": note,
        "inputs": dict(inputs or {}),
    }


def _judge(magnitude: float, tolerance: float) -> str:
    return REFUTED if magnitude > tolerance else NOT_REFUTED


def not_applicable_set(reason: str) -> dict[str, dict[str, Any]]:
    """Every falsifier `not_applicable` for one stated reason — a campaign that
    declares no states has no ΔG to falsify."""
    return {
        name: record(
            name,
            invariance="(see docs/falsifiers.md)",
            transformation="(none)",
            statistic="(none)",
            tolerance=None,
            tolerance_source="",
            magnitude=None,
            state=NOT_APPLICABLE,
            note=reason,
        )
        for name in FALSIFIER_NAMES
    }


# --------------------------------------------------------------------------
# (a) initial condition / seed
# --------------------------------------------------------------------------

def seed_invariance(
    delta_g_a: float | None,
    delta_g_b: float | None,
    temperature_k: float,
    *,
    n_walkers: int,
) -> dict[str, Any]:
    """ΔG does not depend on which task state the walker started in.

    A single-walker campaign is `not_applicable`, not `not_evaluable`: the
    campaign was configured without a replica, so hysteresis is *untested*
    rather than unresolved, and the verdict layer reports it in those words.
    docs/falsifiers.md §10 decision 1 chose the stricter reading; the
    consequence — no single-walker campaign can ever stop on the scientist's
    say-so, and the ablation's false-accept rate is zero by construction — is
    why this constant exists in one place. Flip it to NOT_EVALUABLE to adopt
    the strict reading.
    """
    common = dict(
        invariance="ΔG(low−high) is independent of the walker's starting state and seed",
        transformation=(
            "second walker started in the other task state, same bias definition, "
            "independent hills"
        ),
        statistic="|ΔG_A − ΔG_B|",
        tolerance=TOLERANCE_KT * kt_kj_per_mol(temperature_k),
        tolerance_source=TOLERANCE_SOURCE_KT,
        inputs={"n_walkers": int(n_walkers), "delta_g_a": delta_g_a, "delta_g_b": delta_g_b},
    )
    if n_walkers < 2:
        return record(
            "seed_invariance", magnitude=None, state=_SINGLE_WALKER_STATE,
            note="single-walker campaign: hysteresis untested", **common,
        )
    if delta_g_a is None or delta_g_b is None:
        return record(
            "seed_invariance", magnitude=None, state=NOT_EVALUABLE,
            note="a walker has not populated both states yet", **common,
        )
    magnitude = abs(delta_g_a - delta_g_b)
    return record(
        "seed_invariance", magnitude=magnitude,
        state=_judge(magnitude, common["tolerance"]), **common,
    )


_SINGLE_WALKER_STATE = NOT_APPLICABLE


# --------------------------------------------------------------------------
# (b-i) two estimators of one surface
# --------------------------------------------------------------------------

def estimator_invariance(
    hills_surface: FreeEnergySurface | None,
    reweighted_surface: FreeEnergySurface | None,
    low: float,
    high: float,
    temperature_k: float,
) -> dict[str, Any]:
    """`sum_hills` on the observable's own hills and reweighting from COLVAR are
    two estimators of the same F(observable). Applicable only while the
    observable is among the biased coordinates."""
    kt = kt_kj_per_mol(temperature_k)
    common = dict(
        invariance="F(observable) from sum_hills and from reweighting agree",
        transformation="swap the estimator: hills marginal vs exp(V/kT) reweighting",
        statistic="max(|ΔG_hills − ΔG_rw|, rms(F_hills − F_rw) over the shared range)",
        tolerance=TOLERANCE_KT * kt,
        tolerance_source=TOLERANCE_SOURCE_KT,
    )
    if hills_surface is None:
        return record(
            "estimator_invariance", magnitude=None, state=NOT_APPLICABLE,
            note="the campaign observable is not among the biased coordinates", **common,
        )
    if reweighted_surface is None:
        return record(
            "estimator_invariance", magnitude=None, state=NOT_EVALUABLE,
            note="no reweighted surface yet", **common,
        )
    dg_h = delta_g_kj_per_mol(hills_surface, low, high, temperature_k)
    dg_r = delta_g_kj_per_mol(reweighted_surface, low, high, temperature_k)
    # The rms is taken where the task lives — the states and half a band
    # beyond each — not over the whole sampled range. The far edge of a
    # bounded coordinate is one bin that deepens for as long as the walker
    # sits in it, and the two estimators disagree there by construction while
    # the bias is still filling; that is a precision symptom (drift), not a
    # disagreement about the answer. Replayed on a real campaign the whole-
    # range rms was 4-5 kJ/mol every round while the ΔG gap was 0.3-2.6.
    half_band = 0.5 * (high - low)
    lo, hi = low - half_band, high + half_band
    rms = surface_rms_kj_per_mol(
        hills_surface.restricted_to(lo, hi), reweighted_surface.restricted_to(lo, hi)
    )
    inputs = {
        "delta_g_hills": dg_h, "delta_g_reweighted": dg_r, "rms_kj_per_mol": rms,
        "rms_range": [lo, hi],
    }
    if dg_h is None or dg_r is None or rms is None:
        return record(
            "estimator_invariance", magnitude=None, state=NOT_EVALUABLE,
            note="a state is unpopulated on one of the two surfaces", inputs=inputs, **common,
        )
    magnitude = max(abs(dg_h - dg_r), rms)
    return record(
        "estimator_invariance", magnitude=magnitude,
        state=_judge(magnitude, common["tolerance"]), inputs=inputs, **common,
    )


# --------------------------------------------------------------------------
# (c) time window
# --------------------------------------------------------------------------

def time_window_invariance(
    observable: np.ndarray,
    bias_kj_per_mol: np.ndarray,
    low: float,
    high: float,
    temperature_k: float,
) -> dict[str, Any]:
    """ΔG from the last half of the biased phase agrees with ΔG from all of it."""
    kt = kt_kj_per_mol(temperature_k)
    common = dict(
        invariance="ΔG(low−high) does not depend on which part of the biased phase it is computed from",
        transformation="reweight the last half of COLVAR only, vs the whole",
        statistic="|ΔG_last_half − ΔG_whole|",
        tolerance=TOLERANCE_KT * kt,
        tolerance_source=TOLERANCE_SOURCE_KT,
    )
    observable = np.asarray(observable, dtype=float).ravel()
    bias_kj_per_mol = np.asarray(bias_kj_per_mol, dtype=float).ravel()
    n = observable.size
    if n < 2 * _MIN_FRAMES_PER_WINDOW:
        return record(
            "time_window_invariance", magnitude=None, state=NOT_EVALUABLE,
            note=f"{n} frames; need {2 * _MIN_FRAMES_PER_WINDOW}", inputs={"n_frames": int(n)},
            **common,
        )
    half = n // 2
    whole = reweighted_profile(observable, bias_kj_per_mol, temperature_k)
    late = reweighted_profile(observable[half:], bias_kj_per_mol[half:], temperature_k)
    dg_whole = delta_g_kj_per_mol(whole, low, high, temperature_k)
    dg_late = delta_g_kj_per_mol(late, low, high, temperature_k)
    inputs = {"n_frames": int(n), "delta_g_whole": dg_whole, "delta_g_last_half": dg_late}
    if dg_whole is None or dg_late is None:
        return record(
            "time_window_invariance", magnitude=None, state=NOT_EVALUABLE,
            note="a state is unpopulated in one of the windows", inputs=inputs, **common,
        )
    magnitude = abs(dg_late - dg_whole)
    return record(
        "time_window_invariance", magnitude=magnitude,
        state=_judge(magnitude, common["tolerance"]), inputs=inputs, **common,
    )


# --------------------------------------------------------------------------
# (c′) occupancy — the trap signals, as one falsifier
# --------------------------------------------------------------------------

def occupancy_invariance(
    *,
    start_state: str | None,
    ns_since_low: float | None,
    ns_since_high: float | None,
    ns_confined: float | None,
    confined_to_state: str | None,
    tolerance_ns: float,
    inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Under a bias that has flattened the surface the walker keeps visiting
    both states. Measured as the biased time it has been away from the state
    it started in, and the biased time it has sat entirely inside one state.
    The tolerance is in time and comes from the task file."""
    common = dict(
        invariance="state occupancy is invariant across time windows: the walker keeps visiting both states",
        transformation="compare the most recent window's occupancy against the whole biased phase",
        statistic="max(ns absent from the starting state, ns confined to one state)",
        tolerance=float(tolerance_ns),
        tolerance_source="task file: done_criterion.absence_tolerance_ns",
        inputs=dict(inputs or {}),
    )
    if ns_since_low is None and ns_since_high is None:
        return record(
            "occupancy_invariance", magnitude=None, state=NOT_EVALUABLE,
            note="no per-round observable yet", **common,
        )
    if start_state == "low":
        absent = ns_since_low
    elif start_state == "high":
        absent = ns_since_high
    else:
        absent = max(x for x in (ns_since_low, ns_since_high) if x is not None)
    candidates = [x for x in (absent, ns_confined) if x is not None]
    magnitude = max(candidates) if candidates else 0.0
    state = _judge(magnitude, common["tolerance"])
    note = None
    if state == REFUTED:
        parts = []
        if absent is not None and absent > common["tolerance"]:
            parts.append(f"absent from the {start_state or 'starting'} state for {absent:g} ns")
        if ns_confined is not None and ns_confined > common["tolerance"]:
            parts.append(f"confined to the {confined_to_state} state for {ns_confined:g} ns")
        note = "; ".join(parts)
    return record("occupancy_invariance", magnitude=magnitude, state=state, note=note, **common)


# --------------------------------------------------------------------------
# (d) state definition
# --------------------------------------------------------------------------

def state_definition_invariance(
    surface: FreeEnergySurface | None,
    low: float,
    high: float,
    temperature_k: float,
) -> dict[str, Any]:
    """If the two states are separated by a barrier on the observable, ΔG does
    not depend on exactly where the thresholds are drawn."""
    kt = kt_kj_per_mol(temperature_k)
    band = high - low
    common = dict(
        invariance="ΔG(low−high) is stable when each threshold is moved by ±half a band",
        transformation="recompute ΔG with low±band/2 and with high±band/2",
        statistic="max |ΔG_perturbed − ΔG|",
        tolerance=TOLERANCE_KT * kt,
        tolerance_source=TOLERANCE_SOURCE_KT,
    )
    if surface is None:
        return record(
            "state_definition_invariance", magnitude=None, state=NOT_EVALUABLE,
            note="no surface on the observable yet", **common,
        )
    base = delta_g_kj_per_mol(surface, low, high, temperature_k)
    # The inward moves test the boundary between the states, which is what
    # the invariance is about; they are always required. The outward moves
    # ask about the far side of each state, and on a bounded coordinate that
    # region may not exist — a threshold moved past the sampled range has no
    # population to enclose, which is not evidence that the states are
    # unseparated. Those are skipped, and said so, rather than blocking.
    cv_lo, cv_hi = float(surface.cv.min()), float(surface.cv.max())
    perturbations = {
        "low_plus_half_band": ((low + 0.5 * band, high), True),
        "high_minus_half_band": ((low, high - 0.5 * band), True),
        "low_minus_half_band": ((low - 0.5 * band, high), low - 0.5 * band >= cv_lo),
        "high_plus_half_band": ((low, high + 0.5 * band), high + 0.5 * band <= cv_hi),
    }
    perturbed: dict[str, float | None] = {}
    skipped: list[str] = []
    for name, ((lo, hi), applicable) in perturbations.items():
        if not applicable:
            skipped.append(name)
            continue
        perturbed[name] = delta_g_kj_per_mol(surface, lo, hi, temperature_k)
    inputs: dict[str, Any] = {"delta_g": base, **perturbed, "skipped_outside_sampled_range": skipped}
    if base is None or any(v is None for v in perturbed.values()):
        return record(
            "state_definition_invariance", magnitude=None, state=NOT_EVALUABLE,
            note="a state is unpopulated under one of the perturbations", inputs=inputs, **common,
        )
    magnitude = max(abs(v - base) for v in perturbed.values())  # type: ignore[operator]
    return record(
        "state_definition_invariance", magnitude=magnitude,
        state=_judge(magnitude, common["tolerance"]), inputs=inputs, **common,
    )


# --------------------------------------------------------------------------
# (e) bias accounting — the wall
# --------------------------------------------------------------------------

def bias_accounting_invariance(
    observable: np.ndarray,
    total_bias_kj_per_mol: np.ndarray,
    wall_bias_kj_per_mol: np.ndarray | None,
    low: float,
    high: float,
    temperature_k: float,
) -> dict[str, Any]:
    """With the wall's bias in the reweighting, ΔG does not depend on whether
    frames the wall was acting on are included. Applicable only when a wall
    was printed to COLVAR."""
    kt = kt_kj_per_mol(temperature_k)
    common = dict(
        invariance="ΔG(low−high) does not depend on whether frames beyond the wall are included",
        transformation="reweight all frames vs only frames where the wall's bias is zero",
        statistic="|ΔG_all − ΔG_inside_wall|",
        tolerance=TOLERANCE_KT * kt,
        tolerance_source=TOLERANCE_SOURCE_KT,
    )
    if wall_bias_kj_per_mol is None:
        return record(
            "bias_accounting_invariance", magnitude=None, state=NOT_APPLICABLE,
            note="no wall on the biased coordinates", **common,
        )
    observable = np.asarray(observable, dtype=float).ravel()
    total = np.asarray(total_bias_kj_per_mol, dtype=float).ravel()
    wall = np.asarray(wall_bias_kj_per_mol, dtype=float).ravel()
    inside = wall <= _WALL_ACTIVE_KJ_PER_MOL
    fraction_past = float(1.0 - inside.mean()) if wall.size else 0.0
    inputs: dict[str, Any] = {"fraction_of_frames_past_wall": fraction_past}
    if inside.sum() < _MIN_FRAMES_PER_WINDOW or observable.size < _MIN_FRAMES_PER_WINDOW:
        return record(
            "bias_accounting_invariance", magnitude=None, state=NOT_EVALUABLE,
            note="too few frames inside the wall", inputs=inputs, **common,
        )
    dg_all = delta_g_kj_per_mol(
        reweighted_profile(observable, total, temperature_k), low, high, temperature_k
    )
    dg_inside = delta_g_kj_per_mol(
        reweighted_profile(observable[inside], total[inside], temperature_k),
        low, high, temperature_k,
    )
    inputs.update({"delta_g_all": dg_all, "delta_g_inside_wall": dg_inside})
    if dg_all is None or dg_inside is None:
        return record(
            "bias_accounting_invariance", magnitude=None, state=NOT_EVALUABLE,
            note="a state is unpopulated inside the wall", inputs=inputs, **common,
        )
    magnitude = abs(dg_all - dg_inside)
    return record(
        "bias_accounting_invariance", magnitude=magnitude,
        state=_judge(magnitude, common["tolerance"]), inputs=inputs, **common,
    )


# --------------------------------------------------------------------------
# the block's summary
# --------------------------------------------------------------------------

def summarize(falsifiers: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Counts and the one boolean the stop rule reads.

    `not_refuted` is true only when nothing is refuted AND nothing is
    `not_evaluable`. `not_applicable` entries do not block: an invariance with
    nothing to act on is not evidence. The word "verified" does not appear
    here and must not — that is the reference's word, at evaluation time.
    """
    by_state: dict[str, list[str]] = {
        REFUTED: [], NOT_REFUTED: [], NOT_EVALUABLE: [], NOT_APPLICABLE: [],
    }
    for name, f in falsifiers.items():
        by_state[f["state"]].append(name)
    none_refuted = not by_state[REFUTED]
    all_evaluable = not by_state[NOT_EVALUABLE]
    return {
        "n_evaluated": len(falsifiers),
        "refuted": by_state[REFUTED],
        "not_evaluable": by_state[NOT_EVALUABLE],
        "not_applicable": by_state[NOT_APPLICABLE],
        "none_refuted": none_refuted,
        "all_evaluable": all_evaluable,
        "not_refuted": none_refuted and all_evaluable,
    }
