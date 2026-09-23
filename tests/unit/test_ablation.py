"""The verifier ablation harness (`mdpilot.ablation`).

Three arms judge one and the same replayed campaign. These tests drive the
harness with a fake `decide` and a fake report replay, so no MD and no API;
what they pin is the contract: the arms differ only in what the verifier
sees, the comparability assertion fails loudly, the raw view really carries
no verdicts or tolerances (and neither does the raw prompt), and the
per-round verdicts plus the final ΔG are logged per arm.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mdpilot import ablation
from mdpilot.adapters.system_spec import SystemSpec
from mdpilot.diagnostics import falsifiers as fz
from mdpilot.diagnostics.report import group_report
from mdpilot.memory import store
from mdpilot.orchestrator.scientist import Decision, build_system_prompt, knowledge_keys

_FLAT = {
    "hills_path": "/h", "fes_path": "/f", "cv_label": "q", "cv_min": 0.0, "cv_max": 1.0,
    "n_fes_estimates": 4, "n_basins_fes": 2, "barrier_kj_per_mol": 10.0,
    "fes_depth_kj_per_mol": 20.0, "fes_drift_kj_per_mol": 1.0, "recrossings": 3,
    "barrier_crossed": True, "fes_converged": True, "min_recrossings": 2,
    "cv_switches_used": 0, "cv_switches_remaining": 2, "biased_cvs": ["q"],
    "delta_g_low_minus_high_kj_per_mol": -4.0,
}


def _rec(state: str, magnitude: float | None = 0.3, name: str = "f") -> dict:
    return fz.record(name, invariance="i", transformation="t", statistic="s",
                     tolerance=2.49, tolerance_source="src", magnitude=magnitude,
                     state=state, note="why", inputs={"x": 1})


def _metad(**over: Any) -> dict:
    flat = {**_FLAT, **{k: v for k, v in over.items() if k in _FLAT}}
    falsifiers = over.get("falsifiers") or {"f": _rec(fz.NOT_REFUTED)}
    return group_report(flat, phase="metad", falsifiers=falsifiers)


def _vanilla(**over: Any) -> dict:
    flat = {"ess": 60.0, "plateau_reached": True, "well_sampled": True, "exploring": False,
            "n_basins": 1, "trajectory_length_ns": 1.0, "mean": 0.8, **over}
    return group_report(flat, phase="vanilla", falsifiers={})


# ---------- the raw view ----------

def test_the_raw_view_keeps_every_magnitude_and_drops_every_verdict_and_tolerance() -> None:
    full = _metad(falsifiers={"f": _rec(fz.REFUTED, 5.0)})
    raw = ablation.raw_view(full)

    assert raw["precision"]["fes_drift_kj_per_mol"] == 1.0
    assert raw["precision"]["recrossings"] == 3
    for verdict in ("gate", "fes_converged", "barrier_crossed", "min_recrossings"):
        assert verdict not in raw["precision"], verdict
    f = raw["correctness"]["falsifiers"]["f"]
    assert f == {"name": "f", "statistic": "s", "magnitude": 5.0, "inputs": {"x": 1}}
    assert "summary" not in raw["correctness"]
    assert raw["correctness"]["answer"]["delta_g_low_minus_high_kj_per_mol"] == -4.0
    # the full report is untouched
    assert full["precision"]["gate"] is True and full["correctness"]["summary"]["refuted"] == ["f"]

    v = ablation.raw_view(_vanilla())
    for verdict in ("gate", "plateau_reached", "well_sampled", "exploring"):
        assert verdict not in v["precision"], verdict
    assert v["precision"]["ess"] == 60.0


def test_the_raw_prompt_names_no_threshold_flag_or_state() -> None:
    """A tolerance smuggled in as prose is a gate by another route."""
    for phase in ("vanilla", "metad"):
        keys = knowledge_keys(phase, can_propose_cv=True, allow_cv_switch=phase == "metad", view="raw")
        assert keys[0] == "role_raw" and keys[1] == f"phase_{phase}_raw"
        text = build_system_prompt(phase, can_propose_cv=True, allow_cv_switch=phase == "metad", view="raw")
        for banned in ("refuted", "not_evaluable", "fired", "tolerance", "gate", "kT",
                       "kj/mol at", "ess>=", ">= 50", "8 ns", "precision.gate"):
            assert banned not in text, banned
    gated = build_system_prompt("metad", can_propose_cv=True, allow_cv_switch=True)
    assert "refuted" in gated and "precision.gate" in gated       # and the campaign's still does


# ---------- the deterministic arm ----------

def test_gates_only_verdicts() -> None:
    accept = _metad()
    assert ablation.gates_only_verdict(accept, task_expectation="x")[0] == "accept"
    refuted = _metad(falsifiers={"f": _rec(fz.REFUTED, 9.0)})
    assert ablation.gates_only_verdict(refuted, task_expectation="x")[0] == "revise"
    pending = _metad(falsifiers={"f": _rec(fz.NOT_EVALUABLE, None)})
    assert ablation.gates_only_verdict(pending, task_expectation="x")[0] == "continue"
    unconverged = _metad(fes_converged=False)
    assert ablation.gates_only_verdict(unconverged, task_expectation="x")[0] == "continue"

    assert ablation.gates_only_verdict(_vanilla(), task_expectation=None)[0] == "accept"
    assert ablation.gates_only_verdict(_vanilla(), task_expectation="needs a transition")[0] == "revise"
    assert ablation.gates_only_verdict(_vanilla(exploring=True), task_expectation="t")[0] == "continue"


# ---------- comparability ----------

def _campaign(tmp_path: Path, name: str, *, seed: int = 42, n_rounds: int = 3) -> Path:
    work = tmp_path / name
    work.mkdir()
    store.init_campaign(work, {
        "seed": seed, "initial_steps": 100, "report_interval_steps": 50,
        "equilibration_steps": 0, "system_spec": SystemSpec.trpcage().to_dict(),
        "engine": "_FakeAdapter", "task_expectation": "fold it", "cv_upper_wall_nm": None,
        "state_thresholds": [0.3, 0.7], "min_recrossings": 2, "absence_tolerance_ns": 8.0,
    })
    rounds = work / "rounds"
    rounds.mkdir()
    for i in range(1, n_rounds + 1):
        biased = i > 1
        (rounds / f"round_{i:03d}.dcd").write_text("DCD")
        store.append_round(
            work, round_index=i, n_steps=1000, dcd_path=rounds / f"round_{i:03d}.dcd",
            checkpoint_path=None, report={}, decision="switch_to_metad" if i == 1 else "extend",
            reason="", extra_ns=None,
            metad_proposal={"cv_type": "gyration", "selections": ["backbone"], "label": "rg"} if i == 1 else None,
            plumed_dat_path=(work / "plumed.dat") if biased else None,
        )
    return work


def test_arms_over_the_same_campaign_are_comparable_and_a_different_seed_is_refused(tmp_path: Path) -> None:
    a = _campaign(tmp_path, "a")
    ablation.Comparison((ablation.ArmConfig("gates_only", a), ablation.ArmConfig("lm_only", a)))

    copy = _campaign(tmp_path, "copy")                       # same config, same rounds
    ablation.Comparison((ablation.ArmConfig("gates_only", a), ablation.ArmConfig("lm_only", copy)))

    other_seed = _campaign(tmp_path, "seed", seed=7)
    with pytest.raises(ablation.ArmsNotComparable, match="seed: 42 != 7"):
        ablation.Comparison((ablation.ArmConfig("gates_only", a), ablation.ArmConfig("lm_only", other_seed)))

    shorter = _campaign(tmp_path, "short", n_rounds=2)
    with pytest.raises(ablation.ArmsNotComparable, match="different round table"):
        ablation.Comparison((ablation.ArmConfig("gates_only", a), ablation.ArmConfig("lm_only", shorter)))

    with pytest.raises(ablation.ArmsNotComparable, match="different models"):
        ablation.Comparison((ablation.ArmConfig("lm_only", a, model="m1"),
                             ablation.ArmConfig("lm_plus_gates", a, model="m2")))
    with pytest.raises(ValueError, match="unknown arm"):
        ablation.ArmConfig("llm_alone", a)  # type: ignore[arg-type]


# ---------- the replay ----------

def test_three_arms_judge_the_same_rounds_and_differ_only_in_what_they_saw(tmp_path: Path) -> None:
    """Round 1 vanilla; round 2 biased with a refuted falsifier and a stable
    surface; round 3 biased, everything clean. A fake LLM that always says
    `stop` shows the arms apart: lm_only accepts at round 2 (it saw no flag
    and no gate), lm_plus_gates is refused at round 2 and accepts at 3,
    gates_only revises at 2 and accepts at 3."""
    work = _campaign(tmp_path, "c")
    reports = {
        1: _vanilla(),
        2: _metad(falsifiers={"f": _rec(fz.REFUTED, 9.0)}, delta_g_low_minus_high_kj_per_mol=+3.0),
        3: _metad(delta_g_low_minus_high_kj_per_mol=-4.0),
    }
    seen: list[tuple[str, int, dict]] = []

    def fake_report(work_dir, row, rows, config, scratch):  # noqa: ANN001
        return reports[row.round_index]

    def fake_decide(report, **kw):  # noqa: ANN001
        seen.append((kw["view"], len(kw["prior_round_summaries"]) + 1, report))
        if kw["phase"] == "vanilla":
            return Decision(decision="extend", reason="more", extra_ns=1.0)
        return Decision(decision="stop", reason="looks done", extra_ns=None, ledger_note="note")

    arms = tuple(ablation.ArmConfig(a, work) for a in ablation.ARMS)
    result = ablation.run_comparison(
        ablation.Comparison(arms), out_dir=tmp_path / "out",
        decide_fn=fake_decide, report_fn=fake_report,
    )

    verdicts = {a: result["arms"][a]["verdicts"] for a in ablation.ARMS}
    assert verdicts["gates_only"] == ["revise", "revise", "accept"]
    assert verdicts["lm_only"] == ["continue", "accept", "accept"]
    assert verdicts["lm_plus_gates"] == ["continue", "continue", "accept"]

    # The final ΔG is the answer at the round each arm accepted — the early
    # acceptance shows up as the wrong-signed number, not as a missing one.
    assert result["arms"]["lm_only"]["final_delta_g_low_minus_high_kj_per_mol"] == 3.0
    assert result["arms"]["lm_plus_gates"]["final_delta_g_low_minus_high_kj_per_mol"] == -4.0
    assert result["arms"]["gates_only"]["accepted_at"]["round"] == 3
    # Latency counts biased rounds only: the vanilla pivot is the campaign's
    # design, not a detection. gates_only revised at round 2 (first biased).
    assert result["arms"]["gates_only"]["first_non_continue"]["round"] == 2
    assert result["arms"]["lm_only"]["first_non_continue"]["round"] == 2
    # 1000 steps at 2 fs per biased round
    assert result["gates_first_fire"] == {"round": 2, "biased_ns": 0.002, "refuted": ["f"]}

    # What each LLM arm was shown, round 2: raw for lm_only, full for the other.
    raw_r2 = next(r for v, i, r in seen if v == "raw" and i == 2)
    full_r2 = next(r for v, i, r in seen if v == "gated" and i == 2)
    assert "gate" not in raw_r2["precision"] and "summary" not in raw_r2["correctness"]
    assert full_r2["precision"]["gate"] is True and full_r2["correctness"]["summary"]["refuted"] == ["f"]
    assert raw_r2["precision"]["recrossings"] == full_r2["precision"]["recrossings"] == 3

    # The override is on the record for the arm it applies to, and the LLM's
    # own verdict is kept beside the final one.
    rows = [json.loads(ln) for ln in (tmp_path / "out" / "lm_plus_gates" / "verdicts.jsonl").read_text().splitlines()]
    assert rows[1]["verdict_llm"] == "accept" and rows[1]["verdict"] == "continue"
    assert "stop refused" in rows[1]["override"]
    assert (tmp_path / "out" / "reports" / "round_002.json").exists()
    assert json.loads((tmp_path / "out" / "summary.json").read_text())["n_rounds"] == 3


def test_each_llm_arm_keeps_its_own_ledger_and_priors(tmp_path: Path) -> None:
    work = _campaign(tmp_path, "c")
    ledgers: dict[str, list] = {}

    def fake_report(work_dir, row, rows, config, scratch):  # noqa: ANN001
        return _vanilla() if row.round_index == 1 else _metad()

    def fake_decide(report, **kw):  # noqa: ANN001
        ledgers[kw["view"]] = list(kw["hypothesis_ledger"])
        return Decision(decision="extend", reason="r", extra_ns=1.0,
                        ledger_note=f"{kw['view']} note")

    arms = (ablation.ArmConfig("lm_only", work), ablation.ArmConfig("lm_plus_gates", work))
    ablation.run_comparison(ablation.Comparison(arms), out_dir=tmp_path / "out",
                            decide_fn=fake_decide, report_fn=fake_report)

    assert ledgers["raw"] == ["R1: raw note", "R2: raw note"]
    assert ledgers["gated"] == ["R1: gated note", "R2: gated note"]
