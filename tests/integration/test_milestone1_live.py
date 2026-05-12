"""Milestone 1 done-criterion tests.

Three assertions, covering both directions of the extend/stop decision:

1. EXTEND on a planted under-converged trajectory (50 ps real Trp-cage MD)
   — directly satisfies ROADMAP's done criterion.
2. EXTEND on a real 5 ns Trp-cage trajectory whose RMSD-from-first-frame has
   tau_int ≈ 500 ps (so only ~9 autocorrelation times sampled) — guards
   against false-positive convergence on real-but-undersampled MD data.
3. STOP on a synthetic iid time series wrapped as a diagnostic report —
   covers the stop path on unambiguous data without needing a multi-µs MD run.

The end-to-end loop wiring is exercised separately by test_loop_live.py.

Skipped without ANTHROPIC_API_KEY. The two MD-dependent halves are also
skipped if their planted trajectory file is missing — generate them with:

    python -m benchmarks.generate_trpcage_planted --traj under
    python -m benchmarks.generate_trpcage_planted --traj converged   # ~9 hours
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from mdpilot.diagnostics.report import _summarize, make_report
from mdpilot.orchestrator.scientist import decide

pytestmark = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set in env",
)

_DATA = Path("benchmarks/data/trpcage")
_TOPOLOGY = _DATA / "topology.pdb"
_UNDER = _DATA / "under_converged.dcd"
_CONVERGED = _DATA / "converged_ref.dcd"


@pytest.mark.skipif(
    not _UNDER.exists(),
    reason=f"{_UNDER} missing — run `python -m benchmarks.generate_trpcage_planted --traj under`",
)
def test_scientist_extends_on_planted_under_converged() -> None:
    report = make_report(_UNDER, _TOPOLOGY)
    # The diagnostic must actually flag this as not-converged before the test is meaningful
    assert report["plateau_reached"] is False or report["ess"] < 50, report
    decision = decide(report)
    assert decision.decision == "extend", decision
    assert decision.extra_ns is not None and decision.extra_ns > 0


@pytest.mark.skipif(
    not _CONVERGED.exists(),
    reason=f"{_CONVERGED} missing — run `python -m benchmarks.generate_trpcage_planted --traj converged` (~9 hours CPU)",
)
def test_scientist_extends_on_5ns_trpcage_due_to_long_autocorrelation() -> None:
    """No-false-positive guard.

    5 ns of vanilla MD on Trp-cage with RMSD-from-first-frame as the observable
    has tau_int ≈ 500 ps — only ~9 autocorrelation times within the trajectory.
    The diagnostic correctly reports plateau_reached=False and ess<50; the
    scientist must therefore say `extend`, not `stop`. This is exactly the
    failure mode MDPilot exists to prevent: silently declaring vanilla MD
    adequate when it isn't.
    """
    report = make_report(_CONVERGED, _TOPOLOGY)
    assert report["plateau_reached"] is False, report
    assert report["ess"] < 50, report
    decision = decide(report)
    assert decision.decision == "extend", decision


def test_scientist_stops_on_synthetic_well_sampled_report() -> None:
    """Stop-path coverage on a synthetic iid time series.

    Real Trp-cage on the timescales we can afford does not yield a converged
    RMSD-from-first time series; rather than run multi-microsecond MD, this
    test feeds the scientist a synthetic well-sampled report so that the
    `stop` direction of the binary decision has explicit coverage.
    """
    rng = np.random.default_rng(42)
    obs = rng.normal(loc=1.5, scale=0.1, size=10_000)
    report = _summarize(
        observable=obs,
        observable_name="rmsd_synthetic_iid",
        dcd_path=Path("/tmp/synth.dcd"),
        top_path=Path("/tmp/synth.pdb"),
        frame_dt_ps=1.0,
        length_ns=10.0,
    )
    assert report["plateau_reached"] is True, report
    assert report["ess"] >= 50, report
    decision = decide(report)
    assert decision.decision == "stop", decision
    assert decision.extra_ns is None
