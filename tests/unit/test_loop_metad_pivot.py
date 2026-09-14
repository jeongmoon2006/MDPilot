"""In-place metaD pivot orchestration (M4 step 4), with fake adapters.

The pivot's *physics* (bias sizing, PLUMED rendering, biased MD) is covered by
bias_designer / plumed_writer / cv_designer unit tests and, end-to-end, by the
PLUMED-enabled live test. Here we pin the *orchestration*: on switch_to_metad
the loop builds a biased adapter, writes plumed.dat, marks subsequent rounds
with plumed_dat_path, and resumes correctly into the metaD phase — none of which
needs OpenMM or a PLUMED runtime. Collaborators that touch disk/LLM
(`make_report`, `decide`, `_build_plumed_input`) are stubbed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mdpilot.adapters.system_spec import SystemSpec
from mdpilot.memory import store
from mdpilot.orchestrator import loop as loop_mod
from mdpilot.orchestrator.loop import run_campaign
from mdpilot.orchestrator.scientist import Decision, MetadProposal


# A real (if tiny) topology, not a marker file. `run_campaign`'s pre-flight
# loads this and computes the campaign observable on it before any dynamics,
# so a fake that writes "PDB" would exercise a path no real adapter takes.
# Four alanines in a line: enough CA atoms for an rmsd observable to resolve.
_MINIMAL_PDB = "".join(
    f"ATOM  {i * 2 + 1:>5d}  N   ALA A{i + 1:>4d}    "
    f"{i * 3.8:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           N\n"
    f"ATOM  {i * 2 + 2:>5d}  CA  ALA A{i + 1:>4d}    "
    f"{i * 3.8 + 1.0:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C\n"
    for i in range(4)
) + "END\n"


class _FakeAdapter:
    """Minimal MDAdapter that writes marker files and records calls."""

    def __init__(
        self,
        work_dir: Path,
        *,
        spec: SystemSpec,
        plumed_input: str | None = None,
        timestep_fs: float = 2.0,
        temperature_k: float = 300.0,
    ):
        self._work_dir = Path(work_dir)
        self._spec = spec
        self.plumed_input = plumed_input
        self._timestep_fs = timestep_fs
        self._temperature_k = temperature_k
        self.run_calls: list[int] = []
        self.loaded: list[Path] = []
        self.started = False
        self.state_loaded: str | None = None
        self._topology_path = self._work_dir / "topology.pdb"

    @property
    def spec(self) -> SystemSpec:
        return self._spec

    @property
    def timestep_fs(self) -> float:
        return self._timestep_fs

    @property
    def temperature_k(self) -> float:
        return self._temperature_k

    @property
    def trajectory_extension(self) -> str:
        return ".dcd"

    @property
    def topology_path(self) -> Path:
        return self._topology_path

    def prepare(self) -> None:
        pass

    def start(self) -> None:
        self.started = True
        self._topology_path.parent.mkdir(parents=True, exist_ok=True)
        self._topology_path.write_text(_MINIMAL_PDB)

    def run_steps(self, n_steps, *, trajectory_path=None, report_interval_steps=500):
        self.run_calls.append(n_steps)
        if trajectory_path is not None:
            Path(trajectory_path).parent.mkdir(parents=True, exist_ok=True)
            Path(trajectory_path).write_text("DCD")
        return trajectory_path

    def save_checkpoint(self, path: Path) -> Path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("CHK")
        return Path(path)

    def load_checkpoint(self, path: Path) -> None:
        self.loaded.append(Path(path))

    def export_state_xml(self) -> str:
        return f"STATE after {sum(self.run_calls)} steps"

    def load_state_xml(self, xml: str) -> None:
        self.state_loaded = xml


_REPORT = {
    "trajectory_length_ns": 0.01,
    "ess": 5.0,
    "plateau_reached": True,
    "exploring": False,
    "n_basins": 1,
}

# What `diagnostics.free_energy.metad_report` returns for a biased round. Note
# what is *not* here: no ess, no plateau_reached, no exploring. That omission is
# the contract under test, not an abbreviation of the fixture.
_METAD_REPORT = {
    "fes_drift_kj_per_mol": 0.9,
    "recrossings": 3,
    "recrossing_low": -0.4,
    "recrossing_high": 0.6,
    "barrier_crossed": True,
    "fes_converged": True,
    "n_basins_fes": 2,
    "barrier_kj_per_mol": 21.4,
}


def _stub_collaborators(monkeypatch, decisions: list[Decision]) -> dict:
    """Patch decide / both report builders / _build_plumed_input.

    Returns a record of what the loop asked for: `plumed_texts` (one entry per
    rendered bias), `phases` (the phase passed to each decide call),
    `allow_cv_switch` (whether CV revision was on the table that round) and
    `cv_labels` (the label of each CV the loop asked to have rendered), so
    tests can assert the loop routed each round to the right contract.
    """
    queue = list(decisions)
    record: dict = {
        "plumed_texts": [], "phases": [], "allow_cv_switch": [], "cv_labels": [],
    }

    monkeypatch.setattr(loop_mod, "make_report", lambda *a, **k: dict(_REPORT))
    monkeypatch.setattr(loop_mod, "metad_report", lambda *a, **k: dict(_METAD_REPORT))

    def fake_decide(report, **kwargs):  # noqa: ANN001
        record["phases"].append(kwargs.get("phase"))
        record["allow_cv_switch"].append(kwargs.get("allow_cv_switch"))
        return queue.pop(0)

    monkeypatch.setattr(loop_mod, "decide", fake_decide)

    def fake_build(proposals, traj, top, output_dir, **kwargs):  # noqa: ANN001
        text = "PLUMED-TEXT\n"
        record["plumed_texts"].append(text)
        record["cv_labels"].append(proposals[-1].label)
        record.setdefault("cv_sets", []).append([p.label for p in proposals])
        return text

    monkeypatch.setattr(loop_mod, "_build_plumed_input", fake_build)
    return record


def _proposal() -> MetadProposal:
    return MetadProposal(
        cv_type="gyration", selections=("backbone",), label="rg_back"
    )


def _switch() -> Decision:
    return Decision(
        decision="switch_to_metad",
        reason="pinned single basin; task needs a transition the budget can't reach",
        extra_ns=None,
        metad_proposal=_proposal(),
    )


def _extend() -> Decision:
    return Decision(decision="extend", reason="surface still moving", extra_ns=0.5)


def _stop() -> Decision:
    return Decision(decision="stop", reason="biased run has sampled the transition", extra_ns=None)


def _full_config(tmp_path: Path) -> dict:
    return {
        "seed": 42,
        "initial_steps": 100,
        "report_interval_steps": 50,
        "equilibration_steps": 0,
        "system_spec": SystemSpec.trpcage().to_dict(),
        "engine": "_FakeAdapter",
        "task_expectation": None,
        "cv_upper_wall_nm": None,
        "state_thresholds": None,
        "min_recrossings": 1,
    }


def _run_kwargs() -> dict:
    return dict(initial_steps=100, seed=42, report_interval_steps=50, equilibration_steps=0)


def test_in_call_pivot_runs_biased_phase(tmp_path: Path, monkeypatch) -> None:
    _stub_collaborators(monkeypatch, [_switch(), _stop()])
    base = _FakeAdapter(tmp_path, spec=SystemSpec.trpcage())
    built_biased: list[_FakeAdapter] = []

    def factory(plumed_input: str) -> _FakeAdapter:
        a = _FakeAdapter(tmp_path, spec=SystemSpec.trpcage(), plumed_input=plumed_input)
        built_biased.append(a)
        return a

    result = run_campaign(
        work_dir=tmp_path,
        adapter=base,
        biased_adapter_factory=factory,
        max_rounds=5,
        **_run_kwargs(),
    )

    assert result.stop_reason == "scientist_said_stop"
    assert [r.decision.decision for r in result.rounds] == ["switch_to_metad", "stop"]
    # The switch round is vanilla; the metaD round is marked with the bias path.
    assert result.rounds[0].plumed_dat_path is None
    assert result.rounds[1].plumed_dat_path == tmp_path / "plumed.dat"

    # plumed.dat was written with exactly the rendered text, once.
    assert (tmp_path / "plumed.dat").read_text() == "PLUMED-TEXT\n"
    assert len(built_biased) == 1
    assert built_biased[0].plumed_input == "PLUMED-TEXT\n"
    assert built_biased[0].started
    # The biased adapter ran the metaD round (initial_steps), not the base one.
    assert built_biased[0].run_calls == [100]
    assert base.run_calls == [100]  # only the vanilla switch round

    # Persistence agrees: round 2 carries the plumed_dat_path.
    rows = store.list_rounds(tmp_path)
    assert rows[0].plumed_dat_path is None
    assert rows[1].plumed_dat_path == tmp_path / "plumed.dat"


def test_resume_after_switch_pivots_into_metad_phase(tmp_path: Path, monkeypatch) -> None:
    """A campaign whose last persisted round said switch_to_metad now resumes by
    building the biased adapter from the stored proposal + trajectory and running
    the metaD phase — it is no longer terminal."""
    store.init_campaign(tmp_path, _full_config(tmp_path))
    store.append_round(
        tmp_path,
        round_index=1,
        n_steps=100,
        dcd_path=tmp_path / "rounds/round_001.dcd",
        checkpoint_path=tmp_path / "rounds/round_001.chk",
        report=dict(_REPORT),
        decision="switch_to_metad",
        reason="pinned + task wants a transition",
        extra_ns=None,
        metad_proposal=_proposal().to_dict(),
    )

    _stub_collaborators(monkeypatch, [_stop()])
    base = _FakeAdapter(tmp_path, spec=SystemSpec.trpcage())
    built_biased: list[_FakeAdapter] = []

    def factory(plumed_input: str) -> _FakeAdapter:
        a = _FakeAdapter(tmp_path, spec=SystemSpec.trpcage(), plumed_input=plumed_input)
        built_biased.append(a)
        return a

    result = run_campaign(
        work_dir=tmp_path,
        adapter=base,
        biased_adapter_factory=factory,
        max_rounds=5,
        **_run_kwargs(),
    )

    assert result.stop_reason == "scientist_said_stop"
    # Round 1 (the stored switch) + round 2 (first metaD round).
    assert [r.index for r in result.rounds] == [1, 2]
    assert result.rounds[1].plumed_dat_path == tmp_path / "plumed.dat"
    assert len(built_biased) == 1
    # The metaD phase starts fresh from cache — no vanilla checkpoint reload.
    assert base.loaded == []
    assert built_biased[0].loaded == []
    assert built_biased[0].run_calls == [100]


def test_second_switch_in_metad_phase_terminates(tmp_path: Path, monkeypatch) -> None:
    """Once biased, a further switch_to_metad ends the campaign for human review
    rather than rebuilding the bias in a loop."""
    _stub_collaborators(monkeypatch, [_switch(), _switch()])
    base = _FakeAdapter(tmp_path, spec=SystemSpec.trpcage())

    result = run_campaign(
        work_dir=tmp_path,
        adapter=base,
        biased_adapter_factory=lambda p: _FakeAdapter(
            tmp_path, spec=SystemSpec.trpcage(), plumed_input=p
        ),
        max_rounds=5,
        **_run_kwargs(),
    )

    assert result.stop_reason == "switch_to_metad_requested"
    assert [r.decision.decision for r in result.rounds] == [
        "switch_to_metad",
        "switch_to_metad",
    ]
    # The second (metaD-phase) switch round is marked biased.
    assert result.rounds[1].plumed_dat_path == tmp_path / "plumed.dat"


def test_resume_after_second_switch_stays_terminal(tmp_path: Path, monkeypatch) -> None:
    """A switch_to_metad recorded on an already-biased round is terminal on
    resume too, not just in-process.

    The stored row carries decision='switch_to_metad' *and* a plumed_dat_path.
    Without the phase check, resume would take the pivot branch and rebuild the
    bias from that biased round's trajectory — sizing SIGMA off a spread the
    bias itself produced, which is exactly the second pivot the live loop
    declined to perform.
    """
    store.init_campaign(tmp_path, _full_config(tmp_path))
    plumed_dat = tmp_path / "plumed.dat"
    plumed_dat.write_text("PLUMED-TEXT\n")
    store.append_round(
        tmp_path,
        round_index=1,
        n_steps=100,
        dcd_path=tmp_path / "rounds/round_001.dcd",
        checkpoint_path=tmp_path / "rounds/round_001.chk",
        report=dict(_REPORT),
        decision="switch_to_metad",
        reason="second switch, already biased",
        extra_ns=None,
        metad_proposal=_proposal().to_dict(),
        plumed_dat_path=plumed_dat,
    )

    base = _FakeAdapter(tmp_path, spec=SystemSpec.trpcage())
    result = run_campaign(
        work_dir=tmp_path, adapter=base, max_rounds=5, **_run_kwargs()
    )

    assert result.stop_reason == "switch_to_metad_requested"
    assert [r.index for r in result.rounds] == [1]
    # Nothing ran: no engine work, no rebuilt bias.
    assert base.run_calls == []
    assert base.started is False


def test_default_biased_factory_refuses_a_non_openmm_engine(
    tmp_path: Path, monkeypatch
) -> None:
    """Without an injected factory, a pivot from a non-OpenMM adapter raises.

    The CV's atom indices were resolved against the vanilla engine's topology.
    Silently building an OpenMM biased adapter would bias whichever atoms sat
    at those indices in a differently-solvated, differently-ordered system.
    """
    _stub_collaborators(monkeypatch, [_switch()])
    base = _FakeAdapter(tmp_path, spec=SystemSpec.trpcage())

    with pytest.raises(NotImplementedError, match="_FakeAdapter"):
        run_campaign(work_dir=tmp_path, adapter=base, max_rounds=5, **_run_kwargs())


def test_resumed_extend_round_is_clamped_to_max_extra_ns(
    tmp_path: Path, monkeypatch
) -> None:
    """SQLite stores the model's raw extra_ns, so the clamp has to be re-applied
    on read. Applying it only in the live loop let a resumed campaign run a
    longer round than the uninterrupted one would have.
    """
    store.init_campaign(tmp_path, _full_config(tmp_path))
    store.append_round(
        tmp_path,
        round_index=1,
        n_steps=100,
        dcd_path=tmp_path / "rounds/round_001.dcd",
        checkpoint_path=tmp_path / "rounds/round_001.chk",
        report=dict(_REPORT),
        decision="extend",
        reason="far from converged",
        extra_ns=20.0,  # well past any sane max_extra_ns
    )
    (tmp_path / "rounds").mkdir(parents=True, exist_ok=True)
    (tmp_path / "rounds/round_001.chk").write_text("CHK")

    _stub_collaborators(monkeypatch, [_stop()])
    base = _FakeAdapter(tmp_path, spec=SystemSpec.trpcage())
    run_campaign(
        work_dir=tmp_path,
        adapter=base,
        max_rounds=2,
        max_extra_ns=2.0,
        **_run_kwargs(),
    )

    # 2.0 ns at 2 fs = 1_000_000 steps, not the 10_000_000 the row asked for.
    assert base.run_calls == [1_000_000]


def test_biased_round_gets_the_free_energy_report_not_the_equilibrium_one(
    tmp_path: Path, monkeypatch
) -> None:
    """The pivot swaps the diagnostic contract, not just the adapter.

    A biased trajectory is not an equilibrium ensemble, so the vanilla
    convergence fields must be *absent* from a biased round's report rather
    than present-and-ignorable. This pins the omission, the phase label, and
    the action space handed to the scientist in each phase.
    """
    record = _stub_collaborators(monkeypatch, [_switch(), _stop()])
    base = _FakeAdapter(tmp_path, spec=SystemSpec.trpcage())

    result = run_campaign(
        work_dir=tmp_path,
        adapter=base,
        biased_adapter_factory=lambda p: _FakeAdapter(
            tmp_path, spec=SystemSpec.trpcage(), plumed_input=p
        ),
        max_rounds=5,
        **_run_kwargs(),
    )

    vanilla, biased = result.rounds[0].report, result.rounds[1].report
    assert vanilla["phase"] == "vanilla"
    assert vanilla["ess"] == 5.0

    assert biased["phase"] == "metad"
    assert biased["fes_converged"] is True
    assert biased["recrossings"] == 3
    for equilibrium_field in ("ess", "plateau_reached", "exploring", "n_basins"):
        assert equilibrium_field not in biased, equilibrium_field

    # The scientist was offered the matching action space each round.
    assert record["phases"] == ["vanilla", "metad"]


def test_prior_round_summaries_do_not_leak_equilibrium_fields_from_biased_rounds(
    tmp_path: Path, monkeypatch
) -> None:
    """The compact prior-round view is phase-keyed too — otherwise the fields
    the biased report omits would reappear via the campaign history."""
    _stub_collaborators(monkeypatch, [_switch(), _extend(), _stop()])
    base = _FakeAdapter(tmp_path, spec=SystemSpec.trpcage())

    result = run_campaign(
        work_dir=tmp_path,
        adapter=base,
        biased_adapter_factory=lambda p: _FakeAdapter(
            tmp_path, spec=SystemSpec.trpcage(), plumed_input=p
        ),
        max_rounds=5,
        **_run_kwargs(),
    )

    summaries = [loop_mod._compact_prior(r) for r in result.rounds]
    vanilla_summary, biased_summary = summaries[0], summaries[1]

    assert vanilla_summary["phase"] == "vanilla"
    assert "ess" in vanilla_summary

    assert biased_summary["phase"] == "metad"
    assert biased_summary["fes_converged"] is True
    assert "ess" not in biased_summary
    assert "plateau_reached" not in biased_summary


def _seed_metad_phase_round(tmp_path: Path, *, round_index: int = 1) -> Path:
    """A campaign already inside the metaD phase: one completed biased round,
    its checkpoint, its plumed.dat, and the bias snapshot paired with it."""
    plumed_dat = tmp_path / "plumed.dat"
    plumed_dat.write_text("PLUMED-TEXT\n")
    rounds = tmp_path / "rounds"
    rounds.mkdir(parents=True, exist_ok=True)
    (rounds / f"round_{round_index:03d}.chk").write_text("CHK")
    (rounds / f"round_{round_index:03d}.hills").write_text("SNAPSHOT-HILLS\n")
    (rounds / f"round_{round_index:03d}.colvar").write_text("SNAPSHOT-COLVAR\n")

    store.init_campaign(tmp_path, _full_config(tmp_path))
    store.append_round(
        tmp_path,
        round_index=round_index,
        n_steps=100,
        dcd_path=rounds / f"round_{round_index:03d}.dcd",
        checkpoint_path=rounds / f"round_{round_index:03d}.chk",
        report=dict(_METAD_REPORT),
        decision="extend",
        reason="surface still moving",
        extra_ns=0.5,
        plumed_dat_path=plumed_dat,
    )
    return plumed_dat


def test_mid_metad_resume_enables_restart_and_restores_the_bias(
    tmp_path: Path, monkeypatch
) -> None:
    """Resuming inside the biased phase must continue the deposited bias.

    Without RESTART, PLUMED backs HILLS up to bck.0.HILLS and refills from
    zero while the coordinates carry on from a biased configuration — so the
    surface the campaign integrates is the sum of two disjoint fillings. The
    HILLS/COLVAR snapshot is the other half: it puts the bias back to the
    point the restored checkpoint corresponds to.
    """
    _seed_metad_phase_round(tmp_path)
    # A live HILLS left over from a round that crashed before it was recorded.
    (tmp_path / "HILLS").write_text("SNAPSHOT-HILLS\nSTALE-EXTRA-HILL\n")

    _stub_collaborators(monkeypatch, [_stop()])
    base = _FakeAdapter(tmp_path, spec=SystemSpec.trpcage())
    handed: list[str] = []

    def factory(plumed_input: str) -> _FakeAdapter:
        handed.append(plumed_input)
        return _FakeAdapter(tmp_path, spec=SystemSpec.trpcage(), plumed_input=plumed_input)

    run_campaign(
        work_dir=tmp_path,
        adapter=base,
        biased_adapter_factory=factory,
        max_rounds=2,
        **_run_kwargs(),
    )

    assert len(handed) == 1
    assert handed[0].splitlines()[0].startswith("RESTART")
    assert "PLUMED-TEXT" in handed[0]
    # The stale hill from the unrecorded round is gone; the bias matches the
    # checkpoint it is paired with.
    assert (tmp_path / "HILLS").read_text() == "SNAPSHOT-HILLS\n"
    assert (tmp_path / "COLVAR").read_text() == "SNAPSHOT-COLVAR\n"


def test_fresh_pivot_does_not_enable_restart(tmp_path: Path, monkeypatch) -> None:
    """A pivot has no prior bias for this campaign. RESTART there would read
    back whatever HILLS happened to be lying in the campaign directory."""
    _stub_collaborators(monkeypatch, [_switch(), _stop()])
    base = _FakeAdapter(tmp_path, spec=SystemSpec.trpcage())
    handed: list[str] = []

    def factory(plumed_input: str) -> _FakeAdapter:
        handed.append(plumed_input)
        return _FakeAdapter(tmp_path, spec=SystemSpec.trpcage(), plumed_input=plumed_input)

    run_campaign(
        work_dir=tmp_path,
        adapter=base,
        biased_adapter_factory=factory,
        max_rounds=5,
        **_run_kwargs(),
    )

    assert handed == ["PLUMED-TEXT\n"]


def test_biased_round_snapshots_the_bias_beside_its_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    """Every biased round leaves a bias snapshot next to its checkpoint, so a
    later resume has a consistent pair to restore."""
    _stub_collaborators(monkeypatch, [_switch(), _stop()])
    base = _FakeAdapter(tmp_path, spec=SystemSpec.trpcage())

    def factory(plumed_input: str) -> _FakeAdapter:
        # Stand in for PLUMED depositing hills during the biased round.
        (tmp_path / "HILLS").write_text("HILL-1\nHILL-2\n")
        (tmp_path / "COLVAR").write_text("ROW-1\n")
        return _FakeAdapter(tmp_path, spec=SystemSpec.trpcage(), plumed_input=plumed_input)

    run_campaign(
        work_dir=tmp_path,
        adapter=base,
        biased_adapter_factory=factory,
        max_rounds=5,
        **_run_kwargs(),
    )

    rounds = tmp_path / "rounds"
    # Round 1 was vanilla — no bias existed yet, so nothing was snapshotted.
    assert not (rounds / "round_001.hills").exists()
    # Round 2 was biased.
    assert (rounds / "round_002.hills").read_text() == "HILL-1\nHILL-2\n"
    assert (rounds / "round_002.colvar").read_text() == "ROW-1\n"


def test_biased_budget_clamps_the_last_round_and_ends_the_campaign(
    tmp_path: Path, monkeypatch
) -> None:
    """`max_biased_ns` is a hard cap on cumulative biased simulation time.

    A budget stated only in `task_expectation` is advisory — the model reads
    it and can still ask for more — which is exactly the wrong property for an
    unattended multi-hour run. The round that would overshoot is shortened to
    land on the budget rather than skipped, so its hills still count.
    """
    _stub_collaborators(monkeypatch, [_switch(), _extend(), _extend()])
    base = _FakeAdapter(tmp_path, spec=SystemSpec.trpcage())
    biased: list[_FakeAdapter] = []

    def factory(plumed_input: str) -> _FakeAdapter:
        a = _FakeAdapter(tmp_path, spec=SystemSpec.trpcage(), plumed_input=plumed_input)
        biased.append(a)
        return a

    result = run_campaign(
        work_dir=tmp_path,
        adapter=base,
        biased_adapter_factory=factory,
        max_rounds=10,
        max_biased_ns=0.0003,          # 150 steps at 2 fs
        **_run_kwargs(),               # initial_steps=100
    )

    assert result.stop_reason == "biased_budget_exhausted"
    # 100 steps, then the second biased round clamped from 250_000 to 50.
    assert biased[0].run_calls == [100, 50]
    assert sum(biased[0].run_calls) == 150
    # The vanilla switch round is not charged to the biased budget.
    assert base.run_calls == [100]


def test_biased_budget_survives_a_resume(tmp_path: Path, monkeypatch) -> None:
    """The meter is recomputed from the persisted biased rounds, so restarting
    a campaign cannot silently buy it another full budget."""
    _seed_metad_phase_round(tmp_path)   # one completed biased round, 100 steps

    _stub_collaborators(monkeypatch, [_extend(), _extend()])
    base = _FakeAdapter(tmp_path, spec=SystemSpec.trpcage())
    biased: list[_FakeAdapter] = []

    def factory(plumed_input: str) -> _FakeAdapter:
        a = _FakeAdapter(tmp_path, spec=SystemSpec.trpcage(), plumed_input=plumed_input)
        biased.append(a)
        return a

    result = run_campaign(
        work_dir=tmp_path,
        adapter=base,
        biased_adapter_factory=factory,
        max_rounds=10,
        max_biased_ns=0.0003,          # 150 steps; 100 already spent before the kill
        **_run_kwargs(),
    )

    assert result.stop_reason == "biased_budget_exhausted"
    assert biased[0].run_calls == [50]   # only the 50 steps still owed


def test_prior_summaries_carry_the_boundaries_a_recrossing_count_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The two basins are re-derived from the current surface each round, so
    the boundaries move. A count carried into the campaign history without them
    is not comparable across rounds — the same defect that phase-keying fixed
    for `ess`, arriving through the history channel instead."""
    _stub_collaborators(
        monkeypatch,
        [
            Decision("switch_to_metad", "pivot", None, metad_proposal=_proposal()),
            Decision("stop", "converged", None),
        ],
    )
    base = _FakeAdapter(tmp_path, spec=SystemSpec.trpcage())
    result = run_campaign(
        work_dir=tmp_path / "campaign",
        adapter=base,
        biased_adapter_factory=lambda p: _FakeAdapter(
            tmp_path, spec=SystemSpec.trpcage(), plumed_input=p
        ),
        max_rounds=5,
        **_run_kwargs(),
    )

    biased = [s for s in map(loop_mod._compact_prior, result.rounds)
              if s["phase"] == "metad"]
    assert biased
    for summary in biased:
        assert summary["recrossings"] == 3
        assert summary["recrossing_low"] == -0.4
        assert summary["recrossing_high"] == 0.6


# ---------- switch_cv: CV revision inside the biased phase ----------

def _replacement() -> MetadProposal:
    return MetadProposal(
        cv_type="contacts", selections=("name CA",), label="q_native"
    )


def _switch_cv() -> Decision:
    return Decision(
        decision="switch_cv",
        reason="recrossings counted between boundaries both above the task's "
               "extended threshold; the walker never returned to cv_start",
        extra_ns=None,
        metad_proposal=_replacement(),
    )


def _hills(work_dir: Path) -> Path:
    return work_dir / "HILLS"


def test_switch_cv_rebuilds_the_bias_on_the_replacement_cv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The core of CV revision: a second biased adapter is built against the
    new proposal, and the campaign continues in the same call rather than
    terminating for a human to restart."""
    _stub_collaborators(monkeypatch, [_switch(), _switch_cv(), _stop()])
    base = _FakeAdapter(tmp_path, spec=SystemSpec.trpcage())
    built: list[_FakeAdapter] = []

    def factory(plumed_input: str) -> _FakeAdapter:
        a = _FakeAdapter(tmp_path, spec=SystemSpec.trpcage(), plumed_input=plumed_input)
        built.append(a)
        return a

    result = run_campaign(
        work_dir=tmp_path / "campaign",
        adapter=base,
        biased_adapter_factory=factory,
        max_rounds=5,
        **_run_kwargs(),
    )

    # Two biased adapters: the pivot's, and the replacement's.
    assert len(built) == 2
    assert result.stop_reason == "scientist_said_stop"
    assert [r.decision.decision for r in result.rounds] == [
        "switch_to_metad", "switch_cv", "stop",
    ]
    # The round after the switch is still biased.
    assert result.rounds[2].plumed_dat_path is not None


def test_switch_cv_clears_the_outgoing_hills(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hills on the old coordinate must not carry into the new bias — PLUMED
    would read them back as if they described the new CV. The round snapshot
    is what preserves them."""
    _stub_collaborators(monkeypatch, [_switch(), _switch_cv(), _stop()])
    base = _FakeAdapter(tmp_path, spec=SystemSpec.trpcage())
    work_dir = tmp_path / "campaign"

    def factory(plumed_input: str) -> _FakeAdapter:
        # Simulate PLUMED depositing as soon as a biased adapter starts.
        _hills(work_dir).parent.mkdir(parents=True, exist_ok=True)
        _hills(work_dir).write_text("old-cv hills")
        return _FakeAdapter(tmp_path, spec=SystemSpec.trpcage(), plumed_input=plumed_input)

    run_campaign(
        work_dir=work_dir,
        adapter=base,
        biased_adapter_factory=factory,
        max_rounds=5,
        **_run_kwargs(),
    )

    # The replacement factory rewrote HILLS after the clear, so what matters is
    # that the *snapshot* of the pre-switch round survives as the record.
    snapshot = work_dir / "rounds" / "round_002.hills"
    assert snapshot.exists()
    assert snapshot.read_text() == "old-cv hills"


def test_switch_cv_is_withdrawn_once_the_allowance_is_spent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Past the cap the action is dropped from the tool schema rather than
    refused after the fact, so the model never emits a decision the loop will
    not honour — the same reason a second switch_to_metad is unrepresentable."""
    record = _stub_collaborators(
        monkeypatch, [_switch(), _switch_cv(), _extend(), _stop()]
    )
    base = _FakeAdapter(tmp_path, spec=SystemSpec.trpcage())

    run_campaign(
        work_dir=tmp_path / "campaign",
        adapter=base,
        biased_adapter_factory=lambda p: _FakeAdapter(
            tmp_path, spec=SystemSpec.trpcage(), plumed_input=p
        ),
        max_rounds=6,
        max_cv_switches=1,
        **_run_kwargs(),
    )

    # Round 1 vanilla: not offered. Round 2 biased with one switch left: offered.
    # Rounds 3+ biased with the allowance spent: withdrawn.
    assert record["allow_cv_switch"] == [False, True, False, False]


def test_switch_cv_allowance_is_recomputed_from_disk_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A restart must not hand the scientist a second allowance, for the same
    reason it must not hand it a second biased budget."""
    work_dir = tmp_path / "campaign"
    factory = lambda p: _FakeAdapter(  # noqa: E731
        tmp_path, spec=SystemSpec.trpcage(), plumed_input=p
    )

    _stub_collaborators(monkeypatch, [_switch(), _switch_cv()])
    run_campaign(
        work_dir=work_dir,
        adapter=_FakeAdapter(tmp_path, spec=SystemSpec.trpcage()),
        biased_adapter_factory=factory,
        max_rounds=2,          # stop right after the switch round
        max_cv_switches=1,
        **_run_kwargs(),
    )

    record = _stub_collaborators(monkeypatch, [_stop()])
    run_campaign(
        work_dir=work_dir,
        adapter=_FakeAdapter(tmp_path, spec=SystemSpec.trpcage()),
        biased_adapter_factory=factory,
        max_rounds=4,
        max_cv_switches=1,
        **_run_kwargs(),
    )

    assert record["allow_cv_switch"] == [False]


def test_resumed_switch_cv_round_builds_the_replacement_not_the_rejected_cv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Branch-ordering guard. A switch_cv round is itself biased, so the
    generic `plumed_dat_path is not None` resume branch matches it too — and
    would restart the campaign on the coordinate the scientist just rejected,
    with RESTART reading its hills back."""
    work_dir = tmp_path / "campaign"
    factory = lambda p: _FakeAdapter(  # noqa: E731
        tmp_path, spec=SystemSpec.trpcage(), plumed_input=p
    )

    _stub_collaborators(monkeypatch, [_switch(), _switch_cv()])
    run_campaign(
        work_dir=work_dir,
        adapter=_FakeAdapter(tmp_path, spec=SystemSpec.trpcage()),
        biased_adapter_factory=factory,
        max_rounds=2,
        **_run_kwargs(),
    )

    record = _stub_collaborators(monkeypatch, [_stop()])
    run_campaign(
        work_dir=work_dir,
        adapter=_FakeAdapter(tmp_path, spec=SystemSpec.trpcage()),
        biased_adapter_factory=factory,
        max_rounds=4,
        **_run_kwargs(),
    )

    # The bias rebuilt on resume is the replacement CV, not the original.
    assert record["cv_labels"] == [_replacement().label]


# ---------- strictness: a biased phase needs the task's states ----------

def test_a_campaign_that_can_pivot_must_supply_state_thresholds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without them the biased phase would fall back to counting between the
    two deepest basins of the current surface, which F9 showed is not
    comparable round to round. `task_expectation` is the sole input gating
    `switch_to_metad`, so it is the predicate for "can reach a biased phase"."""
    _stub_collaborators(monkeypatch, [_stop()])

    with pytest.raises(ValueError, match="state_thresholds is None"):
        run_campaign(
            work_dir=tmp_path / "campaign",
            adapter=_FakeAdapter(tmp_path, spec=SystemSpec.trpcage()),
            task_expectation="cross the barrier",
            **_run_kwargs(),
        )


def test_the_guard_fires_before_any_simulation_is_paid_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Raising at the pivot instead would throw away the whole vanilla phase —
    hours of GPU time on a real campaign."""
    _stub_collaborators(monkeypatch, [_stop()])
    adapter = _FakeAdapter(tmp_path, spec=SystemSpec.trpcage())

    with pytest.raises(ValueError):
        run_campaign(
            work_dir=tmp_path / "campaign",
            adapter=adapter,
            task_expectation="cross the barrier",
            **_run_kwargs(),
        )

    assert adapter.run_calls == []
    assert not (tmp_path / "campaign" / "rounds").exists()


def test_a_pure_convergence_campaign_needs_no_state_thresholds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Strict, not indiscriminate. With no task_expectation the scientist
    cannot propose a pivot, so no biased phase can occur and there is nothing
    for the thresholds to anchor.

    What makes that premise true is
    `test_a_campaign_with_no_expectation_cannot_express_a_pivot` in
    test_scientist.py: `switch_to_metad` leaves the tool enum entirely. Before
    that it was only asserted here and discouraged in the prompt, so the
    action stayed emittable and a pivot could still reach a biased phase with
    no band to count recrossings against.
    """
    _stub_collaborators(monkeypatch, [_stop()])

    result = run_campaign(
        work_dir=tmp_path / "campaign",
        adapter=_FakeAdapter(tmp_path, spec=SystemSpec.trpcage()),
        **_run_kwargs(),
    )

    assert result.stop_reason == "scientist_said_stop"


def test_inverted_state_thresholds_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`count_recrossings` returns 0 for an inverted band rather than raising,
    so a swapped pair would read as a campaign that never crossed."""
    _stub_collaborators(monkeypatch, [_stop()])

    with pytest.raises(ValueError, match="high > low"):
        run_campaign(
            work_dir=tmp_path / "campaign",
            adapter=_FakeAdapter(tmp_path, spec=SystemSpec.trpcage()),
            task_expectation="cross the barrier",
            state_thresholds=(4.0, 1.5),
            **_run_kwargs(),
        )


def test_state_thresholds_are_locked_against_a_changed_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resuming with a different band would splice two definitions of "a
    transition" into one campaign's history."""
    work_dir = tmp_path / "campaign"
    kwargs = dict(
        adapter=_FakeAdapter(tmp_path, spec=SystemSpec.trpcage()),
        task_expectation="cross the barrier",
        **_run_kwargs(),
    )
    _stub_collaborators(monkeypatch, [_stop()])
    run_campaign(work_dir=work_dir, state_thresholds=(1.5, 4.0), **kwargs)

    _stub_collaborators(monkeypatch, [_stop()])
    with pytest.raises(ValueError, match="different config"):
        run_campaign(work_dir=work_dir, state_thresholds=(2.0, 5.0), **kwargs)


def test_min_recrossings_is_locked_against_a_changed_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other half of the same definition. `state_thresholds` says where the
    states are; `min_recrossings` says how many transitions between them count
    as done — it is the threshold `fes_converged` compares against, and
    `_refuse_premature_stop` reads that verdict to decide whether the scientist
    may stop. Changing it mid-campaign re-judges rounds already decided under
    the old value.
    """
    work_dir = tmp_path / "campaign"
    kwargs = dict(
        adapter=_FakeAdapter(tmp_path, spec=SystemSpec.trpcage()),
        task_expectation="cross the barrier",
        state_thresholds=(1.5, 4.0),
        **_run_kwargs(),
    )
    _stub_collaborators(monkeypatch, [_stop()])
    run_campaign(work_dir=work_dir, min_recrossings=2, **kwargs)

    _stub_collaborators(monkeypatch, [_stop()])
    with pytest.raises(ValueError, match="different config"):
        run_campaign(work_dir=work_dir, min_recrossings=1, **kwargs)


# ---------- the loop reads its physics constants off the adapter ----------

def test_round_length_follows_the_adapter_timestep(
    tmp_path: Path, monkeypatch
) -> None:
    """`extra_ns` is nanoseconds, and the step count has to be derived from the
    engine's own dt. Against a hardcoded 2 fs, an engine at 4 fs would have run
    every round at twice the requested length with `extra_ns` silently no
    longer meaning nanoseconds anywhere in the campaign record.
    """
    _stub_collaborators(monkeypatch, [_extend(), _stop()])
    base = _FakeAdapter(tmp_path, spec=SystemSpec.trpcage(), timestep_fs=4.0)

    run_campaign(
        work_dir=tmp_path,
        adapter=base,
        max_rounds=2,
        max_extra_ns=2.0,
        **_run_kwargs(),
    )

    # `_extend()` asks for 0.5 ns. At 4 fs that is 125_000 steps, not the
    # 250_000 a 2 fs assumption would have produced.
    assert base.run_calls == [100, 125_000]


def test_biased_phase_uses_the_adapter_thermostat_temperature(
    tmp_path: Path, monkeypatch
) -> None:
    """PLUMED's `METAD ... TEMP=` must match the thermostat or the
    well-tempered scaling factor is computed against the wrong temperature and
    the bias converges to something other than -(1 - 1/gamma)F(s) — with no
    error anywhere. The same temperature sets the kT the free-energy
    convergence threshold is taken against.
    """
    record = _stub_collaborators(monkeypatch, [_switch(), _stop()])
    seen: dict = {}

    def spy_build(proposals, traj, top, output_dir, **kwargs):  # noqa: ANN001
        proposal = proposals[-1]
        seen["build"] = kwargs["temperature_k"]
        record["cv_labels"].append(proposal.label)
        return "PLUMED-TEXT\n"

    def spy_metad_report(*_args, **kwargs):
        seen["report"] = kwargs["temperature_k"]
        return dict(_METAD_REPORT)

    monkeypatch.setattr(loop_mod, "_build_plumed_input", spy_build)
    monkeypatch.setattr(loop_mod, "metad_report", spy_metad_report)

    base = _FakeAdapter(tmp_path, spec=SystemSpec.trpcage(), temperature_k=277.0)
    run_campaign(
        work_dir=tmp_path,
        adapter=base,
        biased_adapter_factory=lambda text: _FakeAdapter(
            tmp_path, spec=SystemSpec.trpcage(), plumed_input=text, temperature_k=277.0
        ),
        max_rounds=3,
        **_run_kwargs(),
    )

    assert seen == {"build": 277.0, "report": 277.0}


# ---------- bias shape overrides ----------
#
# PACE and BIASFACTOR were already keyword arguments on `design_bias`; the loop
# simply never passed them, so no campaign could change the shape of its own
# bias. These pin the threading and the resume lock that has to come with it.

def _two_atom_traj(tmp_path: Path) -> tuple[Path, Path]:
    import mdtraj as md
    import numpy as np

    top = md.Topology()
    res = top.add_residue("ALA", top.add_chain(), resSeq=1)
    top.add_atom("A", md.element.carbon, res)
    top.add_atom("B", md.element.carbon, res)
    xyz = np.zeros((20, 2, 3))
    xyz[:, 1, 0] = np.linspace(0.9, 1.4, 20)   # a real, non-degenerate spread
    traj = md.Trajectory(xyz=xyz.astype(np.float32), topology=top)
    pdb, dcd = tmp_path / "top.pdb", tmp_path / "traj.dcd"
    traj[0].save_pdb(str(pdb))
    traj.save_dcd(str(dcd))
    return dcd, pdb


def _render(tmp_path: Path, **overrides) -> str:
    from mdpilot.orchestrator.loop import _build_plumed_input

    dcd, pdb = _two_atom_traj(tmp_path)
    return _build_plumed_input(
        [MetadProposal(cv_type="distance", selections=("name A", "name B"), label="d")],
        dcd,
        pdb,
        tmp_path.resolve(),
        temperature_k=300.0,
        **overrides,
    )


def test_bias_overrides_reach_the_rendered_plumed_dat(tmp_path: Path) -> None:
    rendered = _render(tmp_path, bias_pace=200, bias_factor=15.0)

    assert "PACE=200" in rendered
    assert "BIASFACTOR=15" in rendered


def test_unset_bias_overrides_leave_the_designer_defaults(tmp_path: Path) -> None:
    """None means "let bias_designer decide" — the loop must not restate its
    defaults, or the two drift apart silently."""
    from mdpilot.sampling.bias_designer import _DEFAULT_BIAS_FACTOR, _DEFAULT_PACE

    rendered = _render(tmp_path)

    assert f"PACE={_DEFAULT_PACE}" in rendered
    assert f"BIASFACTOR={_DEFAULT_BIAS_FACTOR:g}" in rendered


def test_bias_shape_locks_into_the_campaign_config_only_when_set(
    tmp_path: Path, monkeypatch
) -> None:
    """Biased-phase physics, so it has to lock — but adding the keys
    unconditionally would break resume for every campaign predating them."""
    from mdpilot.memory import store

    for kwargs, expected in (
        ({}, False),
        ({"bias_factor": 15.0}, True),
    ):
        work = tmp_path / f"c{int(expected)}"
        _stub_collaborators(monkeypatch, [_stop()])
        adapter = _FakeAdapter(work, spec=SystemSpec.trpcage())
        run_campaign(
            work_dir=work, adapter=adapter, max_rounds=1, **_run_kwargs(), **kwargs
        )
        config = store.get_campaign_config(work)
        assert ("bias_factor" in config) is expected, kwargs


def test_every_config_key_is_covered_by_the_compatibility_table(
    tmp_path: Path, monkeypatch
) -> None:
    """Forget-proofing for the resume guard.

    Adding a key to the campaign config without a compatibility entry silently
    strands every campaign already on disk — the bug `_LEGACY_CONFIG_DEFAULTS`
    exists to prevent, and one that had already cost four real campaigns before
    it existed. If this fails, add the new key to
    `store._LEGACY_CONFIG_DEFAULTS` with the behaviour that was in force
    *before* the key existed, which is not necessarily its current default.
    """
    from mdpilot.memory import store
    from mdpilot.memory.store import _LEGACY_CONFIG_DEFAULTS, _ORIGINAL_CONFIG_KEYS
    from mdpilot.observables import ObservableSpec

    _stub_collaborators(monkeypatch, [_stop()])
    run_campaign(
        work_dir=tmp_path,
        adapter=_FakeAdapter(tmp_path, spec=SystemSpec.trpcage()),
        max_rounds=1,
        # Every optional parameter set, so the recorded config is the widest
        # one run_campaign can produce.
        task_expectation="fold it",
        state_thresholds=(1.0, 2.0),
        min_recrossings=2,
        cv_upper_wall_nm=0.8,
        bias_pace=200,
        bias_factor=15.0,
        observable=ObservableSpec(
            cv_type="gyration", selections=("name CA",), name="rg_nm"
        ),
        **_run_kwargs(),
    )

    recorded = set(store.get_campaign_config(tmp_path))
    uncovered = recorded - _ORIGINAL_CONFIG_KEYS - set(_LEGACY_CONFIG_DEFAULTS)
    assert not uncovered, (
        f"config key(s) {sorted(uncovered)} have no compatibility entry in "
        f"store._LEGACY_CONFIG_DEFAULTS; campaigns already on disk would be "
        f"stranded by them"
    )


def test_the_real_task_file_drives_the_loop_and_reopens_cleanly(
    tmp_path: Path, monkeypatch
) -> None:
    """The seam between `task_file` and `run_campaign`, which nothing else
    covers: the two were built separately and only met in a benchmark script.

    Also the round trip that matters operationally — the same task file must
    reopen its own campaign, which is only true if the rendered
    `task_expectation` is deterministic.
    """
    from mdpilot.memory import store
    from mdpilot.task_file import load_task_file

    task = load_task_file(Path("benchmarks/tasks/cln025_folding.yaml"))
    kwargs = task.run_kwargs(
        max_extra_ns=2.0, seed=42, initial_steps=100,
        report_interval_steps=50, equilibration_steps=0, max_rounds=1,
    )

    _stub_collaborators(monkeypatch, [_stop()])
    result = run_campaign(
        work_dir=tmp_path,
        adapter=_FakeAdapter(tmp_path, spec=task.spec),
        **kwargs,
    )
    assert result.stop_reason == "scientist_said_stop"

    config = store.get_campaign_config(tmp_path)
    assert config["state_thresholds"] == [1.5, 4.0]
    assert config["min_recrossings"] == 2
    assert config["task_expectation"] == kwargs["task_expectation"]

    _stub_collaborators(monkeypatch, [_stop()])
    run_campaign(
        work_dir=tmp_path,
        adapter=_FakeAdapter(tmp_path, spec=task.spec),
        **kwargs,
    )


def test_the_biased_report_says_where_the_walker_was_this_round(
    tmp_path: Path, monkeypatch
) -> None:
    """COLVAR appends across the whole biased phase, so `cv_min`/`cv_max` keep
    reporting the widest excursion the campaign ever made. A real campaign sat
    between 0.03 and 0.13 for two rounds while the scientist correctly quoted a
    cumulative 0.39-10.01 and concluded the full range was being explored."""
    import numpy as np

    from mdpilot.orchestrator import loop as loop_mod

    captured: dict = {}

    def fake_accumulated(rounds_dir, round_index, traj, top, observable=None):
        this_round = np.linspace(0.03, 0.13, 50)          # stuck, unfolded
        cumulative = np.concatenate([np.linspace(0.03, 0.78, 50), this_round])
        return cumulative, "native_contacts_fraction", this_round

    monkeypatch.setattr(loop_mod, "_accumulated_observable", fake_accumulated)
    monkeypatch.setattr(
        loop_mod, "metad_report",
        lambda *a, **k: (captured.update(k) or dict(_METAD_REPORT)),
    )

    report = loop_mod._round_report(
        tmp_path / "r.dcd", tmp_path / "top.pdb",
        plumed_dat_path=tmp_path / "plumed.dat", temperature_k=300.0,
        fes_dir=tmp_path / "fes", rounds_dir=tmp_path, round_index=4,
        state_thresholds=(0.3, 0.7),
    )

    assert report["observable_min_this_round"] == pytest.approx(0.03)
    assert report["observable_max_this_round"] == pytest.approx(0.13)
    # Entirely inside one state — the signal the cumulative range cannot give.
    assert report["observable_max_this_round"] < 0.3
    # The recrossing count is still taken against the cumulative series.
    assert captured["observable"].size == 100


# ---------- the CV-switch escape hatch out of a trap ----------
#
# Replays the observable from `campaigns/ui_campaign_chignolin`, which left the
# folded state in round 2 and then sat between 0.03 and 0.13 of its native
# contacts while the deposited bias grew past 115 kJ/mol. A contact count maps
# every disordered conformation onto roughly the same value, so the bias fills
# one degenerate bin and cannot lead the chain back. `switch_cv` is the way out.

_TRAP_SERIES = [
    np.linspace(0.78, 0.11, 40),   # round 2: crosses both bands
    np.linspace(0.44, 0.04, 40),   # round 3: still crosses 0.3
    np.linspace(0.13, 0.03, 40),   # round 4: confined below 0.3
    np.linspace(0.12, 0.03, 40),   # round 5: still confined -> rounds_confined=2
]


def _stub_trap(monkeypatch, decisions, depths) -> dict:
    """Like `_stub_collaborators`, but the observable really is trapped and the
    surface really keeps deepening, so `_round_report` computes confinement
    from data instead of being handed a verdict."""
    record = _stub_collaborators(monkeypatch, decisions)
    record["reports"] = []
    biased_round = {"n": 0}

    def fake_accumulated(rounds_dir, round_index, traj, top, observable=None):
        i = biased_round["n"]
        biased_round["n"] += 1
        series = _TRAP_SERIES[min(i, len(_TRAP_SERIES) - 1)]
        # Persist it so `_confinement` reads the same files the real loop would.
        Path(rounds_dir).mkdir(parents=True, exist_ok=True)
        np.save(Path(rounds_dir) / f"round_{round_index:03d}.obs.npy", series)
        cumulative = np.concatenate(_TRAP_SERIES[: i + 1])
        return cumulative, "native_contacts_fraction", series

    monkeypatch.setattr(loop_mod, "_accumulated_observable", fake_accumulated)
    depth_iter = iter(depths)

    def fake_metad_report(*a, **k):
        report = dict(_METAD_REPORT)
        report.update(
            fes_depth_kj_per_mol=next(depth_iter), recrossings=1,
            fes_converged=False, recrossing_basis="task_states",
        )
        return report

    monkeypatch.setattr(loop_mod, "metad_report", fake_metad_report)

    inner = loop_mod.decide

    def recording_decide(report, **kwargs):
        record["reports"].append(report)
        return inner(report, **kwargs)

    monkeypatch.setattr(loop_mod, "decide", recording_decide)
    return record


def test_a_contact_space_trap_is_detected_and_escaped(tmp_path, monkeypatch) -> None:
    record = _stub_trap(
        monkeypatch,
        [_switch(), _extend(), _extend(), _extend(), _switch_cv(), _stop()],
        depths=[51.0, 92.0, 116.0, 131.0, 20.0],
    )
    base = _FakeAdapter(tmp_path, spec=SystemSpec.trpcage())
    result = run_campaign(
        work_dir=tmp_path, adapter=base,
        biased_adapter_factory=lambda text: _FakeAdapter(
            tmp_path, spec=SystemSpec.trpcage(), plumed_input=text
        ),
        max_rounds=6, max_cv_switches=1,
        state_thresholds=(0.3, 0.7), task_expectation="fold it",
        **_run_kwargs(),
    )

    biased = [r for r in record["reports"] if r.get("phase") == "metad"]
    confined = [(r.get("confined_to_state"), r.get("rounds_confined")) for r in biased]

    # Rounds 2 and 3 traverse the bands; 4 and 5 are stuck below the low one.
    assert confined[:2] == [(None, 0), (None, 0)]
    assert confined[2] == ("low", 1)
    assert confined[3] == ("low", 2)          # the trap is now unambiguous
    # And the surface kept deepening while it sat there.
    assert [r["fes_depth_kj_per_mol"] for r in biased][:4] == [51.0, 92.0, 116.0, 131.0]

    # The escape actually happened: a fresh bias on the replacement coordinate.
    assert "switch_cv" in [r.decision.decision for r in result.rounds]
    assert record["cv_labels"][-1] != record["cv_labels"][0]


def test_the_switch_is_offered_only_while_the_allowance_lasts(
    tmp_path, monkeypatch
) -> None:
    """`max_cv_switches` is the budget; once spent the action leaves the tool
    rather than being emitted and refused."""
    record = _stub_trap(
        monkeypatch,
        [_switch(), _extend(), _extend(), _extend(), _switch_cv(), _stop()],
        depths=[51.0, 92.0, 116.0, 131.0, 20.0],
    )
    run_campaign(
        work_dir=tmp_path, adapter=_FakeAdapter(tmp_path, spec=SystemSpec.trpcage()),
        biased_adapter_factory=lambda text: _FakeAdapter(
            tmp_path, spec=SystemSpec.trpcage(), plumed_input=text
        ),
        max_rounds=6, max_cv_switches=1,
        state_thresholds=(0.3, 0.7), task_expectation="fold it",
        **_run_kwargs(),
    )

    offered = record["allow_cv_switch"]
    assert offered[0] is False                      # vanilla round: not biased yet
    assert all(offered[1:5])                        # biased, one switch in hand
    assert offered[5] is False                      # spent — withdrawn from the tool


def test_the_report_tells_the_scientist_what_allowance_is_left(
    tmp_path, monkeypatch
) -> None:
    record = _stub_trap(
        monkeypatch, [_switch(), _extend(), _switch_cv(), _stop()],
        depths=[51.0, 92.0, 20.0],
    )
    run_campaign(
        work_dir=tmp_path, adapter=_FakeAdapter(tmp_path, spec=SystemSpec.trpcage()),
        biased_adapter_factory=lambda text: _FakeAdapter(
            tmp_path, spec=SystemSpec.trpcage(), plumed_input=text
        ),
        max_rounds=4, max_cv_switches=1,
        state_thresholds=(0.3, 0.7), task_expectation="fold it",
        **_run_kwargs(),
    )

    biased = [r for r in record["reports"] if r.get("phase") == "metad"]
    assert biased[0]["cv_switches_remaining"] == 1
    assert biased[-1]["cv_switches_used"] == 1 and biased[-1]["cv_switches_remaining"] == 0


def test_wall_warnings_reach_the_scientist_through_the_ledger(
    tmp_path: Path, monkeypatch
) -> None:
    """plumed.dat records the same thing as comments, but the scientist never
    reads plumed.dat. An unbounded CV with no wall is F6 and it has to be said
    somewhere the next round will see."""
    from mdpilot.adapters.plumed_writer import RmsdCV
    from mdpilot.orchestrator.loop import _wall_notes

    cv = RmsdCV(label="rmsd_ca", atoms=(0, 1, 2), reference_path=Path("/tmp/r.pdb"))

    # No wall at all: the F6 warning.
    (note,) = _wall_notes(cv, None, None)
    assert "unbounded above" in note and "F6" in note

    # A wall the box cannot honour: the F11 warning.
    from mdpilot.adapters.plumed_writer import UpperWall

    (note,) = _wall_notes(
        cv, UpperWall(cv_label="rmsd_ca", at=0.8, kappa=1000.0, box_limit_nm=0.55), 0.8
    )
    assert "0.55" in note and "periodic image" in note

    # A wall the campaign did not choose: say so, without alarm.
    (note,) = _wall_notes(
        cv,
        UpperWall(cv_label="rmsd_ca", at=0.55, kappa=1000.0,
                  box_limit_nm=0.55, derived_from_box=True),
        None,
    )
    assert "measured from the source trajectory" in note
    assert "too small for the question" in note

    # A bounded coordinate has nothing to warn about.
    from mdpilot.adapters.plumed_writer import ContactsCV

    assert _wall_notes(ContactsCV(label="q", pairs=((0, 1),), r0_nm=0.75), None, None) == []


# ---------- a proposal the topology cannot resolve ----------

def test_an_unresolvable_proposal_becomes_an_extend_on_record(
    tmp_path: Path, monkeypatch
) -> None:
    """Before this, the first resolution of a proposal happened inside the
    pivot, after the round was persisted as `switch_to_metad`: it raised, and
    every restart re-entered the pivot branch and raised again with `decide()`
    never called. The campaign was unrecoverable without editing state.db.
    Now the refusal is an extend with the error on the ledger, and the next
    round proposes again."""
    from mdpilot.orchestrator.scientist import UnresolvableProposal

    _stub_collaborators(monkeypatch, [_extend(), _stop()])
    scripted = loop_mod.decide
    refused = UnresolvableProposal(
        _switch(), "cv_designer: selection 'name CB' resolved to 0 atoms", 3
    )
    calls = {"n": 0}

    def refuse_first(report, **kwargs):  # noqa: ANN001
        calls["n"] += 1
        if calls["n"] == 1:
            raise refused
        return scripted(report, **kwargs)

    monkeypatch.setattr(loop_mod, "decide", refuse_first)
    events: list[tuple[str, dict]] = []

    result = run_campaign(
        work_dir=tmp_path,
        adapter=_FakeAdapter(tmp_path, spec=SystemSpec.trpcage()),
        max_rounds=5,
        on_event=lambda name, payload: events.append((name, payload)),
        **_run_kwargs(),
    )

    rows = store.list_rounds(tmp_path)
    assert [r.decision for r in rows] == ["extend", "extend", "stop"]
    assert rows[0].metad_proposal is None            # nothing for a resume to rebuild
    assert result.stop_reason == "scientist_said_stop"
    notes = [n.text for n in store.list_ledger_notes(tmp_path) if n.round_index == 1]
    assert any("decision refused" in n and "resolved to 0 atoms" in n for n in notes)
    overrides = [p["note"] for name, p in events if name == "override"]
    assert len(overrides) == 1 and "decision refused" in overrides[0]


def test_the_proposal_validator_resolves_against_the_campaign_topology(
    tmp_path: Path,
) -> None:
    import mdtraj as md

    adapter = _FakeAdapter(tmp_path, spec=SystemSpec.trpcage())
    adapter.start()
    validate = loop_mod._proposal_validator(md.load(str(adapter.topology_path)))

    validate(MetadProposal(cv_type="gyration", selections=("name CA",), label="rg"))
    with pytest.raises(ValueError, match="resolved to 0 atoms"):
        validate(MetadProposal(cv_type="gyration", selections=("name CB",), label="rg"))
    with pytest.raises(ValueError, match="distance requires 2"):
        validate(MetadProposal(cv_type="distance", selections=("name CA",), label="d"))
    # `rmsd` writes a reference PDB while resolving; a rejected or merely
    # validated proposal must leave nothing in the campaign directory.
    validate(MetadProposal(cv_type="rmsd", selections=("name CA",), label="r"))
    assert not list(tmp_path.rglob("r_reference.pdb"))


def test_the_extension_ceiling_and_validator_are_passed_to_the_scientist(
    tmp_path: Path, monkeypatch
) -> None:
    _stub_collaborators(monkeypatch, [_stop()])
    scripted = loop_mod.decide
    seen: dict = {}

    def spy(report, **kwargs):  # noqa: ANN001
        seen.update(kwargs)
        return scripted(report, **kwargs)

    monkeypatch.setattr(loop_mod, "decide", spy)
    run_campaign(
        work_dir=tmp_path,
        adapter=_FakeAdapter(tmp_path, spec=SystemSpec.trpcage()),
        max_extra_ns=0.75,
        **_run_kwargs(),
    )

    assert seen["max_extra_ns"] == 0.75
    assert callable(seen["validate_proposal"])


# ---------- rounds shorter than one trajectory frame ----------

def test_an_extend_shorter_than_one_frame_is_floored_to_a_frame() -> None:
    """`extra_ns` has no lower bound in the tool schema. A round shorter than
    the reporter interval writes a 0-byte DCD that mdtraj cannot open, the
    report raises after the MD is spent, and restart re-derives the same
    length from the stored `extra_ns`."""
    assert loop_mod._extend_steps(0.0005, 2.0, 500_000, 500) == 500
    assert loop_mod._extend_steps(-1.0, 2.0, 500_000, 500) == 500
    assert loop_mod._extend_steps(1.0, 2.0, 500_000, 500) == 500_000


def test_an_opening_round_shorter_than_a_frame_is_refused_before_any_md(
    tmp_path: Path,
) -> None:
    adapter = _FakeAdapter(tmp_path, spec=SystemSpec.trpcage())

    with pytest.raises(ValueError, match="initial_steps=10 is shorter"):
        run_campaign(
            work_dir=tmp_path, adapter=adapter,
            initial_steps=10, report_interval_steps=50, seed=42, equilibration_steps=0,
        )

    assert adapter.run_calls == []
    assert not adapter.started


def test_a_budget_remainder_shorter_than_a_frame_ends_the_campaign(
    tmp_path: Path, monkeypatch
) -> None:
    _stub_collaborators(monkeypatch, [_switch(), _extend(), _extend()])
    biased: list[_FakeAdapter] = []

    def factory(plumed_input: str) -> _FakeAdapter:
        a = _FakeAdapter(tmp_path, spec=SystemSpec.trpcage(), plumed_input=plumed_input)
        biased.append(a)
        return a

    result = run_campaign(
        work_dir=tmp_path,
        adapter=_FakeAdapter(tmp_path, spec=SystemSpec.trpcage()),
        biased_adapter_factory=factory,
        max_rounds=10,
        # ~135 steps: 100 for the opening biased round, then under one
        # 50-step frame left — not worth a round that cannot be diagnosed.
        max_biased_ns=0.00027,
        **_run_kwargs(),
    )

    assert result.stop_reason == "biased_budget_exhausted"
    assert biased[0].run_calls == [100]


# ---------- provenance across a CV switch ----------

def test_each_biased_round_keeps_the_plumed_dat_it_ran_under(
    tmp_path: Path, monkeypatch
) -> None:
    """Every pivot rewrites the one live plumed.dat, so after a `switch_cv`
    the rows for the earlier biased rounds pointed at a file describing a CV
    they never ran on. The snapshot beside the checkpoint is the record."""
    _stub_collaborators(monkeypatch, [_switch(), _switch_cv(), _extend(), _stop()])

    def labelled_build(proposals, traj, top, output_dir, **kwargs):  # noqa: ANN001
        return f"# bias on {proposals[-1].label}\n"

    monkeypatch.setattr(loop_mod, "_build_plumed_input", labelled_build)

    def factory(plumed_input: str) -> _FakeAdapter:
        return _FakeAdapter(tmp_path, spec=SystemSpec.trpcage(), plumed_input=plumed_input)

    run_campaign(
        work_dir=tmp_path,
        adapter=_FakeAdapter(tmp_path, spec=SystemSpec.trpcage()),
        biased_adapter_factory=factory,
        max_rounds=6,
        **_run_kwargs(),
    )

    rounds = tmp_path / "rounds"
    first, replacement = _proposal().label, _replacement().label
    assert not (rounds / "round_001.plumed.dat").exists()          # vanilla
    assert (rounds / "round_002.plumed.dat").read_text() == f"# bias on {first}\n"
    assert (rounds / "round_003.plumed.dat").read_text() == f"# bias on {replacement}\n"
    # The live file now describes the replacement — which is why the
    # snapshot has to exist for round 2 to be readable afterwards.
    assert (tmp_path / "plumed.dat").read_text() == f"# bias on {replacement}\n"


# ---------- wall notes across a crash at the pivot ----------

def test_wall_notes_lost_to_a_crash_are_written_on_the_pivot_resume(
    tmp_path: Path, monkeypatch
) -> None:
    """The live pivot writes its wall notes after `append_round`; a crash
    between the two left the F6/F11 warnings off the ledger for good, because
    the resume branches rebuilt the bias without collecting them. They are
    re-derived on resume, and the ones the live run did get written are not
    written twice."""
    store.init_campaign(tmp_path, _full_config(tmp_path))
    store.append_round(
        tmp_path, round_index=1, n_steps=100,
        dcd_path=tmp_path / "rounds/round_001.dcd",
        checkpoint_path=tmp_path / "rounds/round_001.chk",
        report=dict(_REPORT), decision="switch_to_metad", reason="pinned",
        extra_ns=None, metad_proposal=_proposal().to_dict(),
    )
    store.append_ledger_note(tmp_path, round_index=1, text="WARNING: already on record")

    _stub_collaborators(monkeypatch, [_stop()])

    def noting_build(proposals, traj, top, output_dir, *, notes=None, **kwargs):  # noqa: ANN001
        if notes is not None:
            notes.extend(["WARNING: already on record", "WARNING: wall beyond the box"])
        return "PLUMED-TEXT\n"

    monkeypatch.setattr(loop_mod, "_build_plumed_input", noting_build)

    def factory(plumed_input: str) -> _FakeAdapter:
        return _FakeAdapter(tmp_path, spec=SystemSpec.trpcage(), plumed_input=plumed_input)

    run_campaign(
        work_dir=tmp_path,
        adapter=_FakeAdapter(tmp_path, spec=SystemSpec.trpcage()),
        biased_adapter_factory=factory,
        max_rounds=5,
        **_run_kwargs(),
    )

    texts = [n.text for n in store.list_ledger_notes(tmp_path) if n.round_index == 1]
    assert texts == ["WARNING: already on record", "WARNING: wall beyond the box"]


# ---------- the trap seen from the other side ----------

def _write_obs(rounds_dir: Path, index: int, lo: float, hi: float) -> None:
    rounds_dir.mkdir(parents=True, exist_ok=True)
    np.save(rounds_dir / f"round_{index:03d}.obs.npy", np.linspace(lo, hi, 50))


def test_rounds_since_visited_counts_the_walker_that_never_comes_back(tmp_path: Path) -> None:
    """The real campaign: unfolded in round 2, then three rounds at Q in
    [0.03, 0.55] — below the folded threshold every frame, straddling the
    unfolded one — with `rounds_confined=0` throughout."""
    rounds = tmp_path / "rounds"
    _write_obs(rounds, 2, 0.157, 0.811)   # visited both states
    _write_obs(rounds, 3, 0.026, 0.550)
    _write_obs(rounds, 4, 0.030, 0.547)
    _write_obs(rounds, 5, 0.164, 0.529)   # dips into unfolded, never above 0.7

    assert loop_mod._confinement(rounds, 5, (0.3, 0.7)) == (None, 0)          # the old signal
    assert loop_mod._rounds_since_visited(rounds, 5, (0.3, 0.7)) == (0, 3)     # the new one
    assert loop_mod._rounds_since_visited(rounds, 2, (0.3, 0.7)) == (0, 0)
    assert loop_mod._rounds_since_visited(rounds, 5, None) == (None, None)


def test_rounds_since_visited_stops_at_the_pivot(tmp_path: Path) -> None:
    rounds = tmp_path / "rounds"
    _write_obs(rounds, 2, 0.4, 0.6)       # round 1 was vanilla: no file
    _write_obs(rounds, 3, 0.4, 0.6)

    assert loop_mod._rounds_since_visited(rounds, 3, (0.3, 0.7)) == (2, 2)


def test_a_misquoted_report_becomes_an_extend_with_the_note_withheld(
    tmp_path: Path, monkeypatch
) -> None:
    from mdpilot.orchestrator.scientist import MisquotedReport

    _stub_collaborators(monkeypatch, [_stop()])
    scripted = loop_mod.decide
    misread = Decision(
        decision="stop", reason="recrossings rose to 2", extra_ns=None,
        ledger_note="crossing requirement satisfied",
        cited=(("recrossings", 2),),
    )
    calls = {"n": 0}

    def refuse_first(report, **kwargs):  # noqa: ANN001
        calls["n"] += 1
        if calls["n"] == 1:
            raise MisquotedReport(misread, "`recrossings`: cited 2, report says 1", 3)
        return scripted(report, **kwargs)

    monkeypatch.setattr(loop_mod, "decide", refuse_first)

    run_campaign(
        work_dir=tmp_path,
        adapter=_FakeAdapter(tmp_path, spec=SystemSpec.trpcage()),
        max_rounds=5,
        **_run_kwargs(),
    )

    rows = store.list_rounds(tmp_path)
    assert [r.decision for r in rows] == ["extend", "stop"]
    notes = [n.text for n in store.list_ledger_notes(tmp_path) if n.round_index == 1]
    assert notes == [notes[0]]                                   # exactly one
    assert "decision refused" in notes[0] and "cited 2, report says 1" in notes[0]
    assert "crossing requirement satisfied" not in notes[0]      # the misread note never landed


def test_the_round_json_keeps_what_the_model_said_it_read(tmp_path: Path, monkeypatch) -> None:
    import json

    _stub_collaborators(monkeypatch, [
        Decision(decision="stop", reason="ess=5", extra_ns=None, cited=(("ess", 5.0),)),
    ])
    run_campaign(
        work_dir=tmp_path, adapter=_FakeAdapter(tmp_path, spec=SystemSpec.trpcage()),
        **_run_kwargs(),
    )

    payload = json.loads((tmp_path / "rounds" / "round_001.json").read_text())
    assert payload["decision"]["cited"] == [{"field": "ess", "value": 5.0}]


# ---------- the surface on the campaign observable ----------

def test_the_observable_is_printed_in_colvar_beside_the_biased_cv(tmp_path: Path) -> None:
    """Whatever CV the scientist biases, PLUMED also evaluates the campaign
    observable every step, unbiased, so the surface along it can be reweighted
    from the same COLVAR row as the bias."""
    from mdpilot.observables import COLVAR_OBSERVABLE_LABEL, ObservableSpec

    traj, top = _two_atom_traj(tmp_path)
    biased = MetadProposal(cv_type="distance", selections=("index 0", "index 1"), label="d")
    text = loop_mod._build_plumed_input(
        [biased], traj, top, tmp_path, temperature_k=300.0,
        observable=ObservableSpec(cv_type="distance", selections=("index 0", "index 1"),
                                  name="d_nm"),
    )

    assert f"{COLVAR_OBSERVABLE_LABEL}: DISTANCE" in text
    assert f"PRINT ARG=d,{COLVAR_OBSERVABLE_LABEL},metad.bias" in text
    assert "METAD ARG=d " in text                            # still biased on the proposal alone


def test_a_biased_round_reports_the_surface_on_the_observable(tmp_path: Path, monkeypatch) -> None:
    """From COLVAR's observable and bias columns the biased report carries a
    reweighted surface *on the observable* and the free-energy difference
    between the task's states on it — the numbers a reference is compared
    against, whatever CV is being biased."""
    from mdpilot.diagnostics.free_energy import load_fes
    from mdpilot.observables import ObservableSpec

    _stub_collaborators(monkeypatch, [_switch(), _stop()])
    rng = np.random.default_rng(1)

    def fake_accumulated(rounds_dir, round_index, trajectory_path, topology_path, observable=None):  # noqa: ANN001
        series = rng.uniform(0.0, 1.0, 2)
        return series, "q", series

    monkeypatch.setattr(loop_mod, "_accumulated_observable", fake_accumulated)
    written: dict = {}

    def factory(plumed_input: str) -> _FakeAdapter:
        # A COLVAR PLUMED would have written: the biased CV, the observable
        # (a CA-CA distance here, printed in nm) and the bias, 1 ps rows.
        rows = ["#! FIELDS time rg observable metad.bias"] + [
            f"{200 + i:.6f} 0.5 {0.3 + 0.1 * i:.6f} {2.0 * i:.6f}" for i in range(0, 5)
        ]
        (tmp_path / "COLVAR").write_text("\n".join(rows) + "\n")
        return _FakeAdapter(tmp_path, spec=SystemSpec.trpcage(), plumed_input=plumed_input)

    result = run_campaign(
        work_dir=tmp_path,
        adapter=_FakeAdapter(tmp_path, spec=SystemSpec.trpcage()),
        biased_adapter_factory=factory,
        task_expectation="fold it", state_thresholds=(3.0, 7.0),
        # the observable is that distance in Angstrom: PLUMED's nm column x 10
        observable=ObservableSpec(cv_type="distance", selections=("index 0", "index 1"),
                                  name="d_angstrom", scale=10.0),
        max_rounds=3,
        **_run_kwargs(),
    )

    report = result.rounds[1].report
    assert report["phase"] == "metad"
    surface = load_fes(Path(report["observable_fes_path"]))
    assert surface.cv_label == "d_angstrom"
    assert 3.0 <= surface.cv.min() and surface.cv.max() <= 7.0     # 0.3-0.7 nm, in Angstrom
    assert surface.free_energy.min() == 0.0
    assert "delta_g_low_minus_high_kj_per_mol" in report
    written.update(report)


# ---------- escalation: one coordinate, then several in parallel ----------

def _added() -> MetadProposal:
    return MetadProposal(cv_type="gyration", selections=("backbone",), label="rg_added")


def _add_cv() -> Decision:
    return Decision(
        decision="add_cv", reason="crosses but cannot return", extra_ns=None,
        metad_proposal=_added(),
    )


def test_active_cvs_replays_the_persisted_decisions() -> None:
    def row(decision, proposal=None):
        return store.RoundRow(
            round_index=0, n_steps=1, dcd_path=Path("x"), checkpoint_path=None, report={},
            decision=decision, reason="", extra_ns=None,
            metad_proposal=proposal.to_dict() if proposal else None,
        )
    a, b, c, d = (MetadProposal("rmsd", ("name CA",), lbl) for lbl in ("a", "b", "c", "d"))
    rows = [row("switch_to_metad", a), row("extend"), row("add_cv", b), row("extend"),
            row("switch_cv", c), row("add_cv", d)]

    assert [p.label for p in loop_mod._active_cvs(rows[:4])] == ["a", "b"]
    assert [p.label for p in loop_mod._active_cvs(rows)] == ["c", "d"]
    assert loop_mod._active_cvs([row("extend")]) == []


def test_hills_files_follow_the_size_of_the_set() -> None:
    one, two = _proposal(), _replacement()
    assert loop_mod._hills_files([one]) == ["HILLS"]
    assert loop_mod._hills_files([one, two]) == [f"HILLS.{one.label}", f"HILLS.{two.label}"]


def test_the_primary_surface_is_the_observables_own_marginal_when_biased() -> None:
    from mdpilot.observables import ObservableSpec

    q = ObservableSpec.native_contact_fraction()
    rmsd = MetadProposal("rmsd", ("name CA",), "rmsd_ca")
    contacts = MetadProposal("contacts", ("name CA",), "q_biased")

    assert loop_mod._primary_hills([rmsd], q) == "HILLS"
    assert loop_mod._primary_hills([rmsd, contacts], q) == "HILLS.q_biased"
    assert loop_mod._primary_hills([rmsd, _added()], q) == "HILLS.rmsd_ca"


def test_add_cv_keeps_the_deposited_hills_and_biases_in_parallel(
    tmp_path: Path, monkeypatch
) -> None:
    """The first coordinate was insufficient, not wrong: its hills stay, moved
    to the per-CV file PBMETAD reads, the new coordinate starts from nothing,
    and RESTART is on so the retained hills are read back."""
    record = _stub_collaborators(monkeypatch, [_switch(), _add_cv(), _extend(), _stop()])
    events: list[tuple[str, dict]] = []

    def factory(plumed_input: str) -> _FakeAdapter:
        # what PLUMED would leave behind while the single-CV bias ran
        (tmp_path / "HILLS").write_text("#! FIELDS time rg\n0 1\n")
        return _FakeAdapter(tmp_path, spec=SystemSpec.trpcage(), plumed_input=plumed_input)

    result = run_campaign(
        work_dir=tmp_path, adapter=_FakeAdapter(tmp_path, spec=SystemSpec.trpcage()),
        biased_adapter_factory=factory, max_rounds=6,
        on_event=lambda n, p: events.append((n, p)), **_run_kwargs(),
    )

    assert [r.decision.decision for r in result.rounds] == ["switch_to_metad", "add_cv", "extend", "stop"]
    assert record["cv_sets"] == [[_proposal().label], [_proposal().label, _added().label]]
    assert (tmp_path / f"HILLS.{_proposal().label}").exists()          # kept, renamed
    assert (tmp_path / "plumed.dat").read_text().startswith("RESTART")  # read back
    pivot = next(p for n, p in events if n == "pivot" and p["kind"] == "add_cv")
    assert pivot["biased_cvs"] == [_proposal().label, _added().label]
    # both revisions draw on one allowance
    assert result.rounds[2].report["cv_switches_used"] == 1
    assert result.rounds[1].report["cv_switches_remaining"] == 2 - 0


def test_a_switch_after_an_add_drops_every_hills_file(tmp_path: Path, monkeypatch) -> None:
    _stub_collaborators(monkeypatch, [_switch(), _add_cv(), _switch_cv(), _stop()])

    built = 0

    def factory(plumed_input: str) -> _FakeAdapter:
        # What PLUMED leaves behind under each bias: `HILLS` under the first,
        # single-CV bias; the added coordinate's file under the parallel one.
        nonlocal built
        built += 1
        if built == 1:
            (tmp_path / "HILLS").write_text("h\n")
        elif built == 2:
            (tmp_path / f"HILLS.{_added().label}").write_text("h\n")
        return _FakeAdapter(tmp_path, spec=SystemSpec.trpcage(), plumed_input=plumed_input)

    run_campaign(
        work_dir=tmp_path, adapter=_FakeAdapter(tmp_path, spec=SystemSpec.trpcage()),
        biased_adapter_factory=factory, max_rounds=6, max_cv_switches=2, **_run_kwargs(),
    )

    # the factory re-creates `HILLS` at the final switch; nothing per-CV survives
    assert not list(tmp_path.glob("HILLS.*"))
    assert not (tmp_path / "plumed.dat").read_text().startswith("RESTART")


def test_the_biased_set_is_capped_at_three(tmp_path: Path) -> None:
    three = [MetadProposal("rmsd", ("name CA",), f"cv{i}") for i in range(3)]
    with pytest.raises(ValueError, match="cap is three"):
        loop_mod._revise_bias(
            "add_cv", previous=three, proposal=_added(), source_trajectory=tmp_path / "x",
            topology_path=tmp_path / "t.pdb", factory=lambda t: None,
            plumed_dat_path=tmp_path / "plumed.dat", temperature_k=300.0,
        )


def test_a_revision_warm_starts_from_the_walkers_state(tmp_path: Path, monkeypatch) -> None:
    """The new adapter is placed where the walker was when the revision was
    decided, not at the cached start — that cost a campaign eight of eleven
    nanoseconds re-unfolding a hairpin it had already unfolded."""
    _stub_collaborators(monkeypatch, [_switch(), _stop()])
    biased: list[_FakeAdapter] = []

    def factory(plumed_input: str) -> _FakeAdapter:
        a = _FakeAdapter(tmp_path, spec=SystemSpec.trpcage(), plumed_input=plumed_input)
        biased.append(a)
        return a

    run_campaign(
        work_dir=tmp_path, adapter=_FakeAdapter(tmp_path, spec=SystemSpec.trpcage()),
        biased_adapter_factory=factory, max_rounds=3, **_run_kwargs(),
    )

    assert biased[0].state_loaded == "STATE after 100 steps"     # the vanilla walker
    assert (tmp_path / "rounds" / "round_001.state.xml").read_text() == "STATE after 100 steps"


def test_a_revision_resumed_after_a_crash_warm_starts_from_the_snapshot(
    tmp_path: Path, monkeypatch
) -> None:
    store.init_campaign(tmp_path, _full_config(tmp_path))
    store.append_round(
        tmp_path, round_index=1, n_steps=100, dcd_path=tmp_path / "rounds/round_001.dcd",
        checkpoint_path=tmp_path / "rounds/round_001.chk", report=dict(_REPORT),
        decision="switch_to_metad", reason="pinned", extra_ns=None,
        metad_proposal=_proposal().to_dict(),
    )
    (tmp_path / "rounds").mkdir(exist_ok=True)
    (tmp_path / "rounds" / "round_001.state.xml").write_text("STATE from the snapshot")
    _stub_collaborators(monkeypatch, [_stop()])
    biased: list[_FakeAdapter] = []

    def factory(plumed_input: str) -> _FakeAdapter:
        a = _FakeAdapter(tmp_path, spec=SystemSpec.trpcage(), plumed_input=plumed_input)
        biased.append(a)
        return a

    run_campaign(
        work_dir=tmp_path, adapter=_FakeAdapter(tmp_path, spec=SystemSpec.trpcage()),
        biased_adapter_factory=factory, max_rounds=3, **_run_kwargs(),
    )

    assert biased[0].state_loaded == "STATE from the snapshot"


def test_two_proposals_render_a_parallel_bias_with_grids(tmp_path: Path) -> None:
    traj, top = _two_atom_traj(tmp_path)
    d1 = MetadProposal(cv_type="distance", selections=("index 0", "index 1"), label="d1")
    d2 = MetadProposal(cv_type="distance", selections=("index 1", "index 0"), label="d2")

    text = loop_mod._build_plumed_input([d1, d2], traj, top, tmp_path, temperature_k=300.0)

    line = next(ln for ln in text.splitlines() if "PBMETAD" in ln)
    assert "ARG=d1,d2" in line
    assert "GRID_MIN=-0.2,-0.2 GRID_MAX=2.5,2.5" in line
    assert f"FILE={tmp_path.resolve() / 'HILLS.d1'},{tmp_path.resolve() / 'HILLS.d2'}" in line
    assert "PRINT ARG=d1,d2,pb.bias" in text


def test_a_torsion_in_the_set_gets_a_periodic_grid() -> None:
    from mdpilot.adapters.plumed_writer import ContactsCV, RmsdCV, TorsionCV

    assert loop_mod._grid_bounds(TorsionCV("t", (0, 1, 2, 3))) == ("-pi", "pi")
    assert loop_mod._grid_bounds(ContactsCV("q", ((0, 1),), 0.75)) == (-0.3, 1.3)
    assert loop_mod._grid_bounds(RmsdCV("r", (0, 1, 2), Path("/abs/r.pdb"))) == (-0.2, 2.5)
