"""End-to-end integration: build → simulate → diagnose → persist → decide.

Runs a real OpenMM Trp-cage simulation and a real Claude call. Skipped when
ANTHROPIC_API_KEY is unset. Uses a tiny step count so a single round
completes in ~3 minutes on CPU.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from mdpilot.orchestrator.loop import run_campaign

pytestmark = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set in env",
)


def test_single_round_under_converged_yields_extend(tmp_path: Path) -> None:
    """A 20 ps Trp-cage trajectory is far from converged — scientist must say extend."""
    result = run_campaign(
        work_dir=tmp_path,
        initial_steps=10_000,   # 20 ps at 2 fs
        max_rounds=1,
        seed=42,
    )
    assert len(result.rounds) == 1
    round_0 = result.rounds[0]

    # Loop produced the expected on-disk artifacts
    assert round_0.dcd_path.exists()
    assert round_0.summary_path.exists()
    persisted = json.loads(round_0.summary_path.read_text())
    assert persisted["round_index"] == 1
    assert persisted["decision"]["decision"] == round_0.decision.decision

    # The diagnostic on a 20 ps run should not yet show plateau / well-sampled
    assert round_0.report["plateau_reached"] is False or round_0.report["ess"] < 50

    # Therefore the scientist should extend
    assert round_0.decision.decision == "extend", round_0.decision
    assert round_0.decision.extra_ns is not None and round_0.decision.extra_ns > 0
