"""The end-to-end scorer reads a campaign off disk and judges it.

Synthetic campaigns only: `campaigns/` and `benchmarks/data/` are gitignored,
so a test that read them would pass locally and fail in CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks import run_cln025_e2e as e2e
from mdpilot.diagnostics.free_energy import (
    _KB_KJ_PER_MOL_K,
    FreeEnergySurface,
    write_fes,
)
from mdpilot.memory import store
from mdpilot.task_file import load_task_file

_KT = _KB_KJ_PER_MOL_K * 300.0


def _hairpin_surface(dg_low_minus_high: float, label: str = "q") -> FreeEnergySurface:
    """Two wells on [0, 1]: unfolded near 0.15, folded near 0.85, folded lower."""
    q = np.linspace(0.02, 0.98, 60)
    f = 12.0 * np.exp(-((q - 0.5) / 0.12) ** 2)                 # a barrier at the middle
    f = f + np.where(q < 0.5, dg_low_minus_high, 0.0)            # lift the unfolded well
    f = f + 6.0 * ((q - np.where(q < 0.5, 0.15, 0.85)) / 0.15) ** 2 * 0.3
    return FreeEnergySurface(label, q, f - f.min(), False)


def _campaign(tmp_path: Path, *, surface_path: Path | None, switch: bool = False) -> Path:
    work = tmp_path / "campaign"
    store.init_campaign(work, {
        "seed": 42, "initial_steps": 100, "report_interval_steps": 50,
        "equilibration_steps": 0, "system_spec": {"pdb_id": "5AWL"},
        "engine": "Fake", "state_thresholds": [0.3, 0.7],
    })
    store.append_round(
        work, round_index=1, n_steps=100, dcd_path=work / "r1.dcd", checkpoint_path=None,
        report={"exploring": False, "n_basins": 1, "bimodality_coefficient": 0.37, "ess": 177.0},
        decision="switch_to_metad", reason="pinned", extra_ns=None,
        metad_proposal={"cv_type": "contacts", "selections": ["name CA"], "label": "nc"},
    )
    report = {
        "cv_label": "nc", "fes_drift_kj_per_mol": 20.0, "recrossings": 1,
        "rounds_since_high_visited": 3, "fes_depth_kj_per_mol": 30.0,
    }
    if surface_path is not None:
        report["observable_fes_path"] = str(surface_path)
    store.append_round(
        work, round_index=2, n_steps=200, dcd_path=work / "r2.dcd", checkpoint_path=None,
        report=report, decision="switch_cv" if switch else "extend", reason="not returning",
        extra_ns=None if switch else 1.0, plumed_dat_path=work / "plumed.dat",
        metad_proposal=(
            {"cv_type": "rmsd", "selections": ["name CA"], "label": "r"} if switch else None
        ),
    )
    store.append_ledger_note(work, round_index=2, text="stop refused: drift too high")
    return work


def test_identical_surfaces_match_and_a_shifted_well_does_not(tmp_path: Path) -> None:
    task = load_task_file(Path("benchmarks/tasks/cln025_contacts.yaml"))
    reference = _hairpin_surface(4.0)
    ref_path = write_fes(reference, tmp_path / "reference_fes.dat")
    (tmp_path / "reference.json").write_text(json.dumps({"biased_ns": 50, "seed": 7}))

    same = write_fes(_hairpin_surface(4.0), tmp_path / "same.dat")
    work = _campaign(tmp_path, surface_path=same, switch=True)
    v = e2e.verdict(work, task, reference_path=ref_path, reference_meta=tmp_path / "reference.json")

    assert v["pivoted"] and v["pivot"]["exploring"] is False
    assert v["self_corrected"] and v["cv_switches"][0]["rounds_since_high_visited"] == 3
    assert v["loop_refusals"][0]["note"].startswith("stop refused")
    assert v["literature"]["consistent"]                      # 0 < ΔG < 10 kJ/mol
    assert v["reference"]["matches"] and v["reference"]["rms_kj_per_mol"] < 1e-6
    assert v["passed"]

    # The unfolded well 3 kT too shallow: same shape, wrong thermodynamics.
    off = write_fes(_hairpin_surface(4.0 + 3 * _KT), tmp_path / "off.dat")
    v = e2e.verdict(
        _campaign(tmp_path / "b", surface_path=off), task,
        reference_path=ref_path, reference_meta=tmp_path / "reference.json",
    )
    assert not v["self_corrected"]
    assert v["reference"]["delta_g_gap_kj_per_mol"] == pytest.approx(3 * _KT, abs=0.3)
    assert not v["reference"]["matches"]
    assert not v["passed"]


def test_the_literature_is_reported_and_only_the_reference_gates(tmp_path: Path) -> None:
    """The F13 campaigns put 100+ kJ/mol between the states: outside anything
    experiment supports, and said so. But ff14SB/TIP3P is documented to fold
    CLN025 poorly, so the literature cannot be the pass line — the reference is,
    and without one the verdict is incomplete rather than a pass on the pivot."""
    task = load_task_file(Path("benchmarks/tasks/cln025_contacts.yaml"))
    runaway = write_fes(_hairpin_surface(116.0), tmp_path / "runaway.dat")

    v = e2e.verdict(
        _campaign(tmp_path, surface_path=runaway), task,
        reference_path=tmp_path / "absent.dat",
    )

    assert v["reference"] is None
    assert not v["literature"]["consistent"] and "not gated" in v["literature"]["note"]
    assert not v["passed"]
    assert v["verdict"].startswith("INCOMPLETE")

    # A surface the literature would reject can still pass against a reference
    # that agrees with it: the agent found what the method converges to.
    ref = write_fes(_hairpin_surface(-20.0), tmp_path / "ref.dat")      # unfolded favoured
    same = write_fes(_hairpin_surface(-20.0), tmp_path / "same.dat")
    (tmp_path / "ref.json").write_text(json.dumps({"biased_ns": 50, "converged": True}))
    v = e2e.verdict(
        _campaign(tmp_path / "b", surface_path=same), task,
        reference_path=ref, reference_meta=tmp_path / "ref.json",   # never the real data dir
    )
    assert not v["literature"]["consistent"]
    assert v["reference"]["matches"] and v["passed"] and v["verdict"] == "PASS"


def test_the_surface_is_rebuilt_from_colvar_for_older_campaigns(tmp_path: Path) -> None:
    """A campaign recorded before the loop printed the observable in COLVAR
    can still be scored when the CV it biased *is* the observable: the column
    is there under the biased CV's own label."""
    task = load_task_file(Path("benchmarks/tasks/cln025_contacts.yaml"))
    work = _campaign(tmp_path, surface_path=None)
    (work / "COLVAR").write_text(
        "#! FIELDS time native_contacts_fraction metad.bias\n"
        + "".join(f"{200 + i:.3f} {q:.3f} 3.0\n" for i, q in enumerate((0.9, 0.85, 0.2, 0.15, 0.5)))
    )
    # the row's cv_label is the observable's name -> the column is the observable
    rows = store.list_rounds(work)
    assert rows[-1].report["cv_label"] == "nc"                      # as seeded: not the observable
    store_path = work / "state.db"
    import sqlite3
    with sqlite3.connect(store_path) as conn:
        conn.execute(
            "UPDATE rounds SET report_json = replace(report_json, '\"cv_label\": \"nc\"', "
            "'\"cv_label\": \"native_contacts_fraction\"') WHERE round_index = 2"
        )

    v = e2e.verdict(work, task, reference_path=tmp_path / "absent.dat")

    assert v["observable_surface"] is not None
    lo, hi = v["observable_surface"]["sampled_range"]
    assert 0.15 <= lo < hi <= 0.9


def test_surface_deviation_ignores_where_zero_was_put() -> None:
    a = _hairpin_surface(4.0)
    b = FreeEnergySurface(a.cv_label, a.cv, a.free_energy + 7.5, False)   # same, offset

    dev = e2e.surface_deviation(a, b)

    assert dev["rms_kj_per_mol"] < 1e-9
    assert dev["max_kj_per_mol"] < 1e-9


def test_case_study_renders_every_round(tmp_path: Path) -> None:
    task = load_task_file(Path("benchmarks/tasks/cln025_contacts.yaml"))
    work = _campaign(tmp_path, surface_path=None, switch=True)
    v = e2e.verdict(work, task, reference_path=tmp_path / "absent.dat")

    text = e2e.case_study(work, task, v)

    assert "| 1 | vanilla |" in text and "| 2 | metad |" in text
    assert "switch_cv" in text and "since_high=3" in text
    assert "stop refused" in text


def test_an_unconverged_reference_is_compared_but_cannot_be_matched(tmp_path: Path) -> None:
    """Three reference schemes and 175 ns did not converge F(Q) for this force
    field. A surface with a `reference.json` saying so is still compared —
    the numbers are informative — but the verdict stays incomplete."""
    task = load_task_file(Path("benchmarks/tasks/cln025_contacts.yaml"))
    ref = write_fes(_hairpin_surface(4.0), tmp_path / "unconverged.dat")
    (tmp_path / "reference.json").write_text(json.dumps({"biased_ns": 75, "converged": False}))
    same = write_fes(_hairpin_surface(4.0), tmp_path / "same.dat")

    v = e2e.verdict(
        _campaign(tmp_path, surface_path=same), task,
        reference_path=ref, reference_meta=tmp_path / "reference.json",
    )

    assert v["reference"]["converged"] is False
    assert v["reference"]["rms_kj_per_mol"] < 1e-6          # compared all the same
    assert not v["reference"]["matches"] and not v["passed"]
    assert v["verdict"] == "INCOMPLETE (reference unconverged)"


def test_the_observables_own_marginal_is_preferred_when_it_is_biased(tmp_path: Path) -> None:
    """After an `add_cv` the COLVAR restarts, so the reweighted surface sees
    only the rounds since; the `sum_hills` marginal on the observable's own
    hills is cumulative across the addition and is the surface to score."""
    task = load_task_file(Path("benchmarks/tasks/cln025_contacts.yaml"))
    work = _campaign(tmp_path, surface_path=write_fes(_hairpin_surface(-30.0), tmp_path / "rw.dat"))
    marginal = write_fes(_hairpin_surface(4.0), tmp_path / "fes.dat")
    import sqlite3
    with sqlite3.connect(work / "state.db") as conn:      # the biased CV is the observable
        conn.execute(
            "UPDATE rounds SET report_json = replace(report_json, '\"cv_label\": \"nc\"', "
            f"'\"cv_label\": \"{task.observable_name}\", \"fes_path\": \"{marginal}\"') "
            "WHERE round_index = 2"
        )
    (work / "rounds").mkdir(exist_ok=True)
    np.save(work / "rounds" / "round_002.obs.npy", np.array([0.05, 0.5, 0.95]))

    v = e2e.verdict(work, task, reference_path=tmp_path / "absent.dat")

    dg = v["observable_surface"]["delta_g_low_minus_high_kj_per_mol"]
    assert dg is not None and dg > 0          # the marginal (+4), not the reweighted (-30)
