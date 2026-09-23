"""The fault registry, the scripted policy, and the score (`mdpilot.faults`).

No MD and no API: the registry is checked for integrity, the policy for what
it emits, and the scorer on a synthetic fault directory laid out the way
`benchmarks/run_faults.py` and `mdpilot.ablation` write it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mdpilot import faults
from mdpilot.diagnostics import falsifiers as fz
from mdpilot.orchestrator.scientist import MetadProposal


# ---------- registry ----------

def test_every_fault_names_known_falsifiers_and_at_least_one_expected_to_fire() -> None:
    assert len(faults.FAULTS) == 6
    for name, f in faults.FAULTS.items():
        assert f.name == name
        assert f.expected_to_fire, name
        assert set(f.expected_to_fire) <= set(fz.FALSIFIER_NAMES), name
        assert set(f.expected_silent) <= set(fz.FALSIFIER_NAMES), name
        assert f.ground_truth and f.description
        d = f.to_dict()
        assert d["cv"] == f.cv.to_dict() and d["overrides"] == f.overrides


def test_a_fault_with_an_unknown_or_empty_expectation_is_refused() -> None:
    cv = MetadProposal(cv_type="rmsd", selections=("name CA",), label="r")
    with pytest.raises(ValueError, match="unknown falsifier"):
        faults.Fault(name="x", description="d", ground_truth="g", cv=cv,
                     expected_to_fire=("gravity_invariance",))
    with pytest.raises(ValueError, match="registers no falsifier"):
        faults.Fault(name="x", description="d", ground_truth="g", cv=cv)


def test_the_six_faults_cover_the_registered_scenarios() -> None:
    by = faults.FAULTS
    assert by["rmsd_no_configured_wall"].overrides == {"cv_upper_wall_nm": None}
    assert by["wall_inside_transition_region"].overrides["cv_upper_wall_nm"] == 0.3
    assert by["thresholds_inside_one_basin"].overrides["state_thresholds"] == (0.75, 0.85)
    assert by["truncated_budget"].overrides["max_biased_ns"] == 2.0
    assert by["walker_pinned_weak_bias"].overrides["bias_pace"] == 50_000
    assert by["gamma_too_small"].overrides["bias_factor"] == 1.5
    assert by["rmsd_no_configured_wall"].cv.cv_type == "rmsd"
    assert by["gamma_too_small"].cv.cv_type == "contacts"


# ---------- the scripted policy ----------

def test_the_scripted_policy_pivots_once_and_then_only_extends() -> None:
    fault = faults.FAULTS["gamma_too_small"]
    drive = faults.scripted_policy(fault, extend_ns=2.0)

    pivot = drive({"phase": "vanilla"}, phase="vanilla", task_expectation="t",
                  validate_proposal=lambda p: None, max_extra_ns=2.0)
    assert pivot.decision == "switch_to_metad" and pivot.metad_proposal == fault.cv

    later = drive({"phase": "metad"}, phase="metad", allow_cv_switch=True)
    assert later.decision == "extend" and later.extra_ns == 2.0
    # Whatever the report says — a refuted falsifier, a converged gate — the
    # policy is the cause of the fault, not a verifier of it.
    still = drive({"phase": "metad", "precision": {"gate": True}}, phase="metad")
    assert still.decision == "extend"


# ---------- the score ----------

def _rec(state: str) -> dict:
    return fz.record("f", invariance="", transformation="", statistic="", tolerance=1.0,
                     tolerance_source="", magnitude=None, state=state)


def _fault_dir(
    tmp_path: Path, fault_name: str, *, states_by_round: list[dict[str, str]],
    arms: dict[str, dict], biased_ns: list[float],
) -> Path:
    d = tmp_path / fault_name
    ab = d / faults.ABLATION_DIR
    (ab / "reports").mkdir(parents=True)
    faults.write_fault_record(d, faults.FAULTS[fault_name])
    for i, (states, _ns) in enumerate(zip(states_by_round, biased_ns, strict=True), start=1):
        report = {"phase": "metad", "correctness": {"falsifiers": {
            n: {**_rec(s), "name": n} for n, s in states.items()
        }}}
        (ab / "reports" / f"round_{i:03d}.json").write_text(json.dumps(report))
    summary = {"biased_ns": biased_ns[-1], "arms": arms}
    (ab / "summary.json").write_text(json.dumps(summary))
    for arm in arms:
        (ab / arm).mkdir()
        rows = [{"round": i, "biased_ns": ns} for i, ns in enumerate(biased_ns, start=1)]
        (ab / arm / "verdicts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return d


def test_false_accept_rate_and_latency_per_arm_and_detection_per_fault(tmp_path: Path) -> None:
    accepted = {"accepted": True, "accepted_at": {"biased_ns": 4.0},
                "first_non_continue": {"biased_ns": 4.0, "verdict": "accept"},
                "final_delta_g_low_minus_high_kj_per_mol": 3.0}
    refused = {"accepted": False, "accepted_at": None,
               "first_non_continue": {"biased_ns": 2.0, "verdict": "revise"},
               "final_delta_g_low_minus_high_kj_per_mol": -1.0}
    never = {"accepted": False, "accepted_at": None, "first_non_continue": None,
             "final_delta_g_low_minus_high_kj_per_mol": None}

    # gamma_too_small: occupancy not evaluable at 2 ns, refuted from 4 ns.
    a = _fault_dir(
        tmp_path, "gamma_too_small",
        states_by_round=[
            {"occupancy_invariance": fz.NOT_EVALUABLE, "time_window_invariance": fz.NOT_REFUTED},
            {"occupancy_invariance": fz.REFUTED, "time_window_invariance": fz.REFUTED},
            {"occupancy_invariance": fz.REFUTED, "time_window_invariance": fz.NOT_REFUTED},
        ],
        arms={"gates_only": refused, "lm_only": accepted, "lm_plus_gates": never},
        biased_ns=[2.0, 4.0, 6.0],
    )
    # truncated_budget: nothing ever fires — a gap the score has to show.
    b = _fault_dir(
        tmp_path, "truncated_budget",
        states_by_round=[{"time_window_invariance": fz.NOT_REFUTED}],
        arms={"gates_only": accepted, "lm_only": accepted, "lm_plus_gates": accepted},
        biased_ns=[2.0],
    )

    report = faults.score([a, b])

    arms = report["arms"]
    assert arms["gates_only"]["false_accept_rate"] == 0.5
    assert arms["gates_only"]["accepted"] == ["truncated_budget"]
    assert arms["lm_only"]["false_accept_rate"] == 1.0
    assert arms["lm_plus_gates"]["false_accept_rate"] == 0.5
    assert arms["gates_only"]["latency_ns"] == {"gamma_too_small": 2.0, "truncated_budget": 4.0}
    assert arms["lm_plus_gates"]["latency_ns"]["gamma_too_small"] is None

    g = report["faults"]["gamma_too_small"]
    assert g["detected"] is True
    assert g["fired_at_ns"] == {"occupancy_invariance": 4.0}
    assert g["blocked_at_ns"] == {"occupancy_invariance": 2.0}      # not_evaluable counts as a block
    assert g["unexpected_fires"] == ["time_window_invariance"]
    assert g["arms"]["lm_only"]["accepted"] and g["arms"]["lm_only"]["accepted_at_ns"] == 4.0

    t = report["faults"]["truncated_budget"]
    assert t["detected"] is False and t["fired_at_ns"] == {"time_window_invariance": None}
