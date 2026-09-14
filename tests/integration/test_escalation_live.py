"""Escalation from one biased coordinate to several, live under PLUMED.

The unit tests prove the loop's bookkeeping with fakes. This proves the parts
only PLUMED and OpenMM can: that `PBMETAD` reads the single-CV `HILLS` back
after it is moved to the per-CV name, that a missing hills file for the
added coordinate is tolerated on RESTART, that a `State` exported from one
adapter loads into another whose System carries a different `PlumedForce`,
and that the warm start actually happens — COLVAR's clock continues from
where the vanilla round ended instead of restarting at the cached time.

Decisions are scripted; no API key needed. Needs the conda environment
(`openmm-plumed`, `plumed` on PATH) and a few minutes of GPU or CPU.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from mdpilot.diagnostics.free_energy import load_colvar, plumed_available
from mdpilot.orchestrator import loop as loop_mod
from mdpilot.orchestrator.scientist import Decision, MetadProposal
from mdpilot.task_file import load_task_file

pytestmark = pytest.mark.skipif(
    not plumed_available() or shutil.which("plumed") is None,
    reason="needs a PLUMED runtime",
)

_TASK = Path("benchmarks/tasks/cln025_contacts.yaml")


def _scripted(decisions: list[Decision]):
    queue = list(decisions)

    def decide(report, **kwargs):  # noqa: ANN001
        return queue.pop(0)

    return decide


def test_add_cv_escalates_to_parallel_bias_and_warm_starts(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("openmmplumed")
    task = load_task_file(_TASK)
    contacts = MetadProposal(cv_type="contacts", selections=("name CA",),
                             label="native_contacts_fraction")
    rmsd = MetadProposal(cv_type="rmsd", selections=("name CA",), label="rmsd_ca")
    monkeypatch.setattr(loop_mod, "decide", _scripted([
        Decision("switch_to_metad", "pinned", None, metad_proposal=contacts),
        Decision("add_cv", "crosses but cannot return", None, metad_proposal=rmsd),
        Decision("extend", "still filling", 0.01),
        Decision("extend", "still filling", 0.01),   # a `stop` here would be refused: not converged
    ]))

    work = tmp_path / "campaign"
    adapter = task.build_adapter(work)
    steps_per_ns = loop_mod.steps_per_ns_for(adapter)
    kwargs = task.run_kwargs(
        initial_steps=int(0.01 * steps_per_ns),          # 10 ps rounds
        report_interval_steps=int(0.001 * steps_per_ns),  # 1 ps frames
        max_rounds=4, max_extra_ns=0.01, max_cv_switches=2,
    )
    result = loop_mod.run_campaign(work_dir=work, adapter=adapter, **kwargs)

    assert [r.decision.decision for r in result.rounds] == [
        "switch_to_metad", "add_cv", "extend", "extend",
    ]
    assert result.stop_reason == "max_rounds_reached"

    # --- the bias on disk after the escalation ---
    plumed = (work / "plumed.dat").read_text()
    assert plumed.startswith("RESTART")
    assert "PBMETAD ARG=native_contacts_fraction,rmsd_ca" in plumed
    assert "GRID_MIN=" in plumed and "UPPER_WALLS ARG=rmsd_ca" in plumed
    kept = work / "HILLS.native_contacts_fraction"
    assert kept.exists() and not (work / "HILLS").exists()
    n_kept = sum(1 for ln in kept.read_text().splitlines() if ln and not ln.startswith("#"))
    assert n_kept >= 20, "hills from the single-CV round and the parallel rounds"
    assert (work / "HILLS.rmsd_ca").exists(), "PLUMED created the added coordinate's file"

    # --- the reports the scientist saw ---
    r2, r3 = result.rounds[1].report, result.rounds[2].report
    assert r2["biased_cvs"] == ["native_contacts_fraction"]
    assert r3["biased_cvs"] == ["native_contacts_fraction", "rmsd_ca"]
    assert set(r3["cv_ranges"]) == {"native_contacts_fraction", "rmsd_ca"}
    assert r3["cv_label"] == "native_contacts_fraction"      # the observable's own marginal
    assert r3["cv_switches_used"] == 1 and r3["cv_switches_remaining"] == 1

    # --- the warm start: COLVAR's clock continues from where the walker was ---
    colvar = load_colvar(work / "COLVAR")                    # restarted at the add_cv
    t_first = float(colvar["time"][0])
    equil_ps = (task.spec.equilibration.nvt_ps + task.spec.equilibration.npt_ps)
    # vanilla 10 ps + first biased round 10 ps after the 200 ps equilibration
    assert t_first == pytest.approx(equil_ps + 20.0, abs=1.0), (
        f"COLVAR starts at {t_first} ps; a cold restart would start at {equil_ps}"
    )
    # and the per-round State snapshots exist for a crash to resume from
    assert (work / "rounds" / "round_001.state.xml").exists()
    assert (work / "rounds" / "round_002.state.xml").exists()

    # the observable is continuous across the escalation, not reset to folded
    obs_before = np.load(work / "rounds" / "round_002.obs.npy")[-1]
    obs_after = np.load(work / "rounds" / "round_003.obs.npy")[0]
    assert abs(obs_after - obs_before) < 0.3
