"""The correctness falsifiers, the three-block report, and the stop rule.

Design in docs/falsifiers.md. Every falsifier is a transformation the answer
must be invariant under plus a tolerance; these tests drive each one into
each of its states with synthetic data, then check the grouping, the
nested-aware accessor the citation check and the loop read through, and the
stop rule that now needs the precision gate AND an unrefuted correctness
block.
"""

from __future__ import annotations

import numpy as np
import pytest

from mdpilot.diagnostics import falsifiers as fz
from mdpilot.diagnostics.free_energy import FreeEnergySurface, _KB_KJ_PER_MOL_K
from mdpilot.diagnostics.report import (
    group_report,
    has_report_field,
    precision_gate,
    report_field,
)
from mdpilot.orchestrator.loop import _refuse_premature_stop
from mdpilot.orchestrator.scientist import Decision, check_citations

T = 300.0
KT = _KB_KJ_PER_MOL_K * T
LOW, HIGH = 0.3, 0.7


def _surface(f_of_x, n: int = 101) -> FreeEnergySurface:
    x = np.linspace(0.0, 1.0, n)
    f = np.array([f_of_x(v) for v in x], dtype=float)
    return FreeEnergySurface(cv_label="q", cv=x, free_energy=f - f.min(), periodic=False)


def _double_well(x: float) -> float:
    # Two basins at 0.15 and 0.85, a 6 kT barrier between them.
    return 6.0 * KT * np.exp(-((x - 0.5) ** 2) / 0.02)


# ---------- the record ----------

def test_a_record_carries_the_fields_the_report_contract_names() -> None:
    r = fz.record(
        "x", invariance="i", transformation="t", statistic="s", tolerance=1.0,
        tolerance_source="src", magnitude=0.5, state=fz.NOT_REFUTED,
    )
    assert set(r) >= {"statistic", "tolerance", "fired", "magnitude", "state"}
    assert r["fired"] is False
    with pytest.raises(ValueError, match="unknown state"):
        fz.record("x", invariance="i", transformation="t", statistic="s", tolerance=1.0,
                  tolerance_source="src", magnitude=None, state="correct")


def test_the_not_applicable_set_names_every_falsifier() -> None:
    s = fz.not_applicable_set("no states")
    assert tuple(s) == fz.FALSIFIER_NAMES
    assert all(v["state"] == fz.NOT_APPLICABLE for v in s.values())


# ---------- (a) seed ----------

def test_seed_invariance_is_not_applicable_on_one_walker_and_judged_on_two() -> None:
    assert fz.seed_invariance(-5.0, None, T, n_walkers=1)["state"] == fz.NOT_APPLICABLE
    assert fz.seed_invariance(-5.0, None, T, n_walkers=2)["state"] == fz.NOT_EVALUABLE
    assert fz.seed_invariance(-5.0, -5.5, T, n_walkers=2)["state"] == fz.NOT_REFUTED
    r = fz.seed_invariance(-5.0, 2.0, T, n_walkers=2)
    assert r["state"] == fz.REFUTED and r["fired"] and r["magnitude"] == pytest.approx(7.0)


# ---------- (b-i) estimator ----------

def test_estimator_invariance_states() -> None:
    well = _surface(_double_well)
    assert fz.estimator_invariance(None, well, LOW, HIGH, T)["state"] == fz.NOT_APPLICABLE
    assert fz.estimator_invariance(well, None, LOW, HIGH, T)["state"] == fz.NOT_EVALUABLE
    same = fz.estimator_invariance(well, well, LOW, HIGH, T)
    assert same["state"] == fz.NOT_REFUTED and same["magnitude"] == pytest.approx(0.0)
    # The same shape with the high basin lifted by 3 kT: ΔG disagrees by 3 kT.
    lifted = _surface(lambda x: _double_well(x) + (3.0 * KT if x > 0.5 else 0.0))
    r = fz.estimator_invariance(well, lifted, LOW, HIGH, T)
    assert r["state"] == fz.REFUTED and r["magnitude"] > KT


# ---------- (c) time window ----------

def _frames(n_low: int, n_high: int, rng: np.random.Generator) -> np.ndarray:
    return np.concatenate([rng.uniform(0.0, LOW, n_low), rng.uniform(HIGH, 1.0, n_high)])


def test_time_window_invariance_is_silent_on_a_stationary_series_and_fires_on_a_walking_one() -> None:
    rng = np.random.default_rng(0)
    first = _frames(500, 500, rng)
    rng.shuffle(first)
    second = _frames(500, 500, rng)
    rng.shuffle(second)
    stationary = np.concatenate([first, second])
    bias = np.zeros_like(stationary)
    r = fz.time_window_invariance(stationary, bias, LOW, HIGH, T)
    assert r["state"] == fz.NOT_REFUTED and r["magnitude"] < KT

    # First half 90 % low; second half 90 % high: the answer walks by ~4 kT.
    walking = np.concatenate([_frames(900, 100, rng), _frames(100, 900, rng)])
    r = fz.time_window_invariance(walking, np.zeros_like(walking), LOW, HIGH, T)
    assert r["state"] == fz.REFUTED and r["magnitude"] > KT

    assert fz.time_window_invariance(np.ones(5), np.ones(5), LOW, HIGH, T)["state"] == fz.NOT_EVALUABLE
    # Second half never in the high state: the window cannot be judged.
    one_way = np.concatenate([_frames(50, 50, rng), rng.uniform(0.0, LOW, 100)])
    assert fz.time_window_invariance(one_way, np.zeros(200), LOW, HIGH, T)["state"] == fz.NOT_EVALUABLE


# ---------- (c') occupancy ----------

def test_occupancy_invariance_reads_the_starting_state_and_the_task_tolerance() -> None:
    r = fz.occupancy_invariance(
        start_state="high", ns_since_low=20.0, ns_since_high=2.0, ns_confined=0.0,
        confined_to_state=None, tolerance_ns=8.0,
    )
    assert r["state"] == fz.NOT_REFUTED and r["magnitude"] == 2.0     # away from *its* state
    r = fz.occupancy_invariance(
        start_state="high", ns_since_low=0.0, ns_since_high=10.0, ns_confined=0.0,
        confined_to_state=None, tolerance_ns=8.0,
    )
    assert r["state"] == fz.REFUTED and "absent from the high state for 10 ns" in r["note"]
    r = fz.occupancy_invariance(
        start_state=None, ns_since_low=0.0, ns_since_high=0.0, ns_confined=9.0,
        confined_to_state="low", tolerance_ns=8.0,
    )
    assert r["state"] == fz.REFUTED and "confined to the low state" in r["note"]
    assert r["tolerance"] == 8.0 and "task file" in r["tolerance_source"]
    assert fz.occupancy_invariance(
        start_state=None, ns_since_low=None, ns_since_high=None, ns_confined=None,
        confined_to_state=None, tolerance_ns=8.0,
    )["state"] == fz.NOT_EVALUABLE


# ---------- (d) state definition ----------

def test_state_definition_invariance_needs_a_barrier_between_the_states() -> None:
    r = fz.state_definition_invariance(_surface(_double_well), LOW, HIGH, T)
    assert r["state"] == fz.NOT_REFUTED and r["magnitude"] < KT
    # A flat surface: moving a threshold by half a band changes the population
    # it encloses by a factor of three, ~1.1 kT.
    r = fz.state_definition_invariance(_surface(lambda x: 0.0), LOW, HIGH, T)
    assert r["state"] == fz.REFUTED
    assert fz.state_definition_invariance(None, LOW, HIGH, T)["state"] == fz.NOT_EVALUABLE


# ---------- (e) bias accounting ----------

def test_bias_accounting_invariance_states() -> None:
    rng = np.random.default_rng(3)
    obs = _frames(500, 500, rng)
    total = np.zeros_like(obs)
    assert fz.bias_accounting_invariance(obs, total, None, LOW, HIGH, T)["state"] == fz.NOT_APPLICABLE
    quiet = fz.bias_accounting_invariance(obs, total, np.zeros_like(obs), LOW, HIGH, T)
    assert quiet["state"] == fz.NOT_REFUTED and quiet["magnitude"] == pytest.approx(0.0)
    assert quiet["inputs"]["fraction_of_frames_past_wall"] == 0.0
    # The wall was acting on 90 % of the high-state frames: excluding them
    # moves ΔG by kT ln 10.
    wall = np.zeros_like(obs)
    wall[500:950] = 5.0
    r = fz.bias_accounting_invariance(obs, total + wall, wall, LOW, HIGH, T)
    assert r["state"] == fz.REFUTED and r["magnitude"] > KT
    assert r["inputs"]["fraction_of_frames_past_wall"] == pytest.approx(0.45)


# ---------- summary ----------

def test_summary_blocks_on_not_evaluable_but_not_on_not_applicable() -> None:
    def rec(state):
        return fz.record("f", invariance="", transformation="", statistic="",
                         tolerance=1.0, tolerance_source="", magnitude=None, state=state)
    s = fz.summarize({"a": rec(fz.NOT_REFUTED), "b": rec(fz.NOT_APPLICABLE)})
    assert s["not_refuted"] is True and s["not_applicable"] == ["b"]
    s = fz.summarize({"a": rec(fz.NOT_REFUTED), "b": rec(fz.NOT_EVALUABLE)})
    assert s["not_refuted"] is False and s["none_refuted"] is True
    s = fz.summarize({"a": rec(fz.REFUTED)})
    assert s["not_refuted"] is False and s["refuted"] == ["a"]
    assert "verified" not in s


# ---------- the three-block report ----------

_FLAT_METAD = {
    "hills_path": "/h", "fes_path": "/f", "cv_label": "q", "cv_min": 0.0, "cv_max": 1.0,
    "n_fes_estimates": 4, "n_basins_fes": 2, "barrier_kj_per_mol": 10.0,
    "fes_depth_kj_per_mol": 20.0, "fes_drift_kj_per_mol": 1.0, "recrossings": 3,
    "barrier_crossed": True, "fes_converged": True, "min_recrossings": 2,
    "cv_switches_used": 0, "cv_switches_remaining": 2, "biased_cvs": ["q"],
    "delta_g_low_minus_high_kj_per_mol": -4.0,
}


def test_group_report_files_every_number_and_refuses_an_unfiled_one() -> None:
    report = group_report(_FLAT_METAD, phase="metad", falsifiers=fz.not_applicable_set("x"))
    assert set(report) == {"phase", "validity", "precision", "correctness", "action_space"}
    assert report["precision"]["recrossings"] == 3 and report["precision"]["gate"] is True
    assert report["validity"]["cv_label"] == "q"
    assert report["action_space"]["cv_switches_remaining"] == 2
    assert report["correctness"]["answer"]["delta_g_low_minus_high_kj_per_mol"] == -4.0
    assert report["correctness"]["summary"]["not_refuted"] is True
    with pytest.raises(KeyError, match="filed under no block"):
        group_report({"brand_new_statistic": 1}, phase="metad", falsifiers={})


def test_the_precision_gate_per_phase() -> None:
    assert precision_gate("metad", {"fes_converged": False}) is False
    assert precision_gate("vanilla", {"plateau_reached": True, "well_sampled": True}) is True
    assert precision_gate("vanilla", {"plateau_reached": True, "well_sampled": False}) is False
    assert precision_gate("vanilla", {"plateau_reached": None}) is None


def test_report_field_reads_nested_flat_dotted_and_falsifier_inputs() -> None:
    occ = fz.occupancy_invariance(
        start_state="high", ns_since_low=0.0, ns_since_high=10.0, ns_confined=0.0,
        confined_to_state=None, tolerance_ns=8.0, inputs={"rounds_confined": 0},
    )
    nested = group_report(_FLAT_METAD, phase="metad", falsifiers={"occupancy_invariance": occ})
    assert report_field(nested, "recrossings") == 3
    assert report_field(nested, "gate") is True
    assert report_field(nested, "delta_g_low_minus_high_kj_per_mol") == -4.0
    assert report_field(nested, "rounds_confined") == 0                      # inside inputs
    assert report_field(nested, "correctness.falsifiers.occupancy_invariance.magnitude") == 10.0
    assert report_field(nested, "correctness.falsifiers.occupancy_invariance.state") == "refuted"
    assert report_field(nested, "no_such_thing", "dflt") == "dflt"
    assert not has_report_field(nested, "ess")
    assert report_field({"recrossings": 1}, "recrossings") == 1              # legacy flat


def test_citations_resolve_dotted_paths_and_input_leaves() -> None:
    occ = fz.occupancy_invariance(
        start_state="high", ns_since_low=0.0, ns_since_high=10.0, ns_confined=0.0,
        confined_to_state=None, tolerance_ns=8.0, inputs={"rounds_confined": 0},
    )
    nested = group_report(_FLAT_METAD, phase="metad", falsifiers={"occupancy_invariance": occ})
    ok = (
        ("recrossings", 3),
        ("correctness.falsifiers.occupancy_invariance.magnitude", 10.0),
        ("correctness.falsifiers.occupancy_invariance.state", "refuted"),
        ("rounds_confined", 0),
        ("gate", True),
    )
    assert check_citations(ok, nested) is None
    bad = check_citations((("correctness.falsifiers.occupancy_invariance.magnitude", 2.0),), nested)
    assert bad is not None and "report says 10.0" in bad


# ---------- stop permission ----------

def _stop() -> Decision:
    return Decision(decision="stop", reason="done", extra_ns=None)


def _with(falsifiers: dict) -> dict:
    return group_report(_FLAT_METAD, phase="metad", falsifiers=falsifiers)


def _rec(state: str, name: str = "f") -> dict:
    return fz.record(name, invariance="", transformation="", statistic="",
                     tolerance=1.0, tolerance_source="", magnitude=None, state=state)


def test_stop_needs_the_precision_gate_and_an_unrefuted_correctness_block() -> None:
    allowed, note = _refuse_premature_stop(_stop(), _with({"f": _rec(fz.NOT_REFUTED)}), 100)
    assert allowed.decision == "stop" and note is None

    allowed, note = _refuse_premature_stop(_stop(), _with({"f": _rec(fz.NOT_APPLICABLE)}), 100)
    assert allowed.decision == "stop" and note is None

    refused, note = _refuse_premature_stop(_stop(), _with({"f": _rec(fz.REFUTED)}), 100)
    assert refused.decision == "extend" and "refuted=['f']" in note

    refused, note = _refuse_premature_stop(_stop(), _with({"f": _rec(fz.NOT_EVALUABLE)}), 100)
    assert refused.decision == "extend" and "not_evaluable=['f']" in note

    unconverged = group_report({**_FLAT_METAD, "fes_converged": False}, phase="metad",
                               falsifiers={"f": _rec(fz.NOT_REFUTED)})
    refused, note = _refuse_premature_stop(_stop(), unconverged, 100)
    assert refused.decision == "extend" and "precision.gate=False" in note

    # Budget spent: the stop stands whatever the blocks say.
    allowed, _ = _refuse_premature_stop(_stop(), _with({"f": _rec(fz.REFUTED)}), 0)
    assert allowed.decision == "stop"


def test_a_report_persisted_before_the_blocks_existed_is_judged_on_precision_alone() -> None:
    allowed, note = _refuse_premature_stop(_stop(), {"fes_converged": True}, 100)
    assert allowed.decision == "stop" and note is None
    refused, _ = _refuse_premature_stop(_stop(), {"fes_converged": False}, 100)
    assert refused.decision == "extend"


def test_no_module_in_the_verifier_names_a_system_or_a_cv_type() -> None:
    """The falsifiers are universal by construction (docs/falsifiers.md §11)."""
    import inspect

    source = inspect.getsource(fz)
    for word in ("hairpin", "CLN025", "chignolin", "contacts", "rmsd", "gyration", "torsion", "ligand"):
        assert word not in source, word
    for word in ("verified", "is_correct", "correct_"):
        assert word.lower() not in source.lower() or word == "verified", word


def test_state_definition_skips_an_outward_move_past_the_sampled_range() -> None:
    """On a bounded coordinate the far side of a state may not exist. A
    threshold moved past the sampled range is skipped and named, not a block:
    a real campaign whose contact fraction never exceeded 0.89 reported this
    falsifier `not_evaluable` in every round because high+band/2 = 0.9."""
    x = np.linspace(0.0, 0.89, 90)                       # never reaches 0.9
    f = np.array([_double_well(v) for v in x])
    surface = FreeEnergySurface(cv_label="q", cv=x, free_energy=f - f.min(), periodic=False)

    r = fz.state_definition_invariance(surface, LOW, HIGH, T)

    assert r["state"] == fz.NOT_REFUTED
    assert r["inputs"]["skipped_outside_sampled_range"] == ["high_plus_half_band"]
    assert "high_plus_half_band" not in r["inputs"]
    assert {"low_plus_half_band", "high_minus_half_band", "low_minus_half_band"} <= set(r["inputs"])


def test_estimator_rms_is_taken_over_the_task_range_only() -> None:
    """A disagreement confined to the far edge of the coordinate — the bin a
    still-filling bias deepens — is drift, not a disagreement about ΔG."""
    # Two harmonic wells at 0.15 and 0.85, 6 kT up at q = 0.05: the far edge
    # holds e^-6 of the low basin's population.
    def wells(x: float) -> float:
        return 6.0 * KT * min(((x - 0.15) / 0.1) ** 2, ((x - 0.85) / 0.1) ** 2)
    well = _surface(wells)
    # Same surface, lifted by 8 kT below q = 0.05: outside states±band/2, and
    # a whole-range rms of ~4 kJ/mol that would have refuted on its own.
    edge = _surface(lambda x: wells(x) + (8.0 * KT if x < 0.05 else 0.0))
    r = fz.estimator_invariance(well, edge, LOW, HIGH, T)
    assert r["state"] == fz.NOT_REFUTED
    assert r["inputs"]["rms_range"] == pytest.approx([LOW - 0.2, HIGH + 0.2])
