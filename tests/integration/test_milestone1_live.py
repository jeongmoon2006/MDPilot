"""Milestone 1 done-criterion test (per ROADMAP.md).

Two assertions:
1. Scientist says EXTEND on a planted under-converged trajectory (50 ps).
2. Scientist says STOP  on a planted converged trajectory (5 ns).

Both load pre-generated DCDs and call make_report + decide directly. The
end-to-end loop wiring is exercised separately by test_loop_live.py.

Skipped without ANTHROPIC_API_KEY. Each half also skipped if its planted
trajectory file is missing — generate them with:

    python -m benchmarks.generate_trpcage_planted --traj under
    python -m benchmarks.generate_trpcage_planted --traj converged   # ~9 hours
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from mdpilot.diagnostics.report import make_report
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
def test_scientist_stops_on_planted_converged() -> None:
    report = make_report(_CONVERGED, _TOPOLOGY)
    # The diagnostic must actually flag this as converged before the test is meaningful
    assert report["plateau_reached"] is True, report
    assert report["ess"] >= 50, report
    decision = decide(report)
    assert decision.decision == "stop", decision
    assert decision.extra_ns is None
