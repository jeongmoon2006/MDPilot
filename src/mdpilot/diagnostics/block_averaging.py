"""Flyvbjerg-Petersen block-averaging for 1D MD observables.

Reference: Flyvbjerg & Petersen, "Error estimates on averages of correlated
data", J. Chem. Phys. 91:461 (1989).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BlockLevel:
    n_blocks: int
    sem: float
    sem_err: float


@dataclass(frozen=True)
class BlockAverageResult:
    mean: float
    sem: float
    sem_naive: float
    statistical_inefficiency: float
    plateau_reached: bool
    n_levels: int
    levels: tuple[BlockLevel, ...]


def block_average(
    x: np.ndarray,
    *,
    min_blocks: int = 8,
    plateau_window: int = 3,
) -> BlockAverageResult:
    """Block-averaging error estimate for a 1D time series.

    Iteratively averages adjacent samples and tracks the block standard
    error (SEM). A plateau across `plateau_window` consecutive levels —
    where the SEM spread is within 2x the largest FP error bar — is taken
    as the converged SEM. If no plateau is found the highest available
    level's SEM is reported and `plateau_reached` is False.

    The statistical inefficiency g = (sem / sem_naive)^2 is roughly the
    number of correlated samples per independent one (for AR(1) with
    correlation phi, g = (1+phi)/(1-phi)).
    """
    arr = np.asarray(x, dtype=float).ravel()
    n0 = arr.size
    if n0 < min_blocks:
        raise ValueError(f"need at least {min_blocks} samples, got {n0}")

    mean = float(np.mean(arr))
    sem_naive = float(np.std(arr, ddof=1) / np.sqrt(n0))

    levels: list[BlockLevel] = []
    cur = arr.copy()
    while cur.size >= min_blocks:
        n = cur.size
        sem = float(np.std(cur, ddof=1) / np.sqrt(n))
        sem_err = sem / np.sqrt(2.0 * (n - 1))
        levels.append(BlockLevel(n_blocks=n, sem=sem, sem_err=sem_err))
        if n % 2 == 1:
            cur = cur[:-1]
        cur = 0.5 * (cur[::2] + cur[1::2])

    plateau_idx = _detect_plateau(levels, window=plateau_window)
    plateau_reached = plateau_idx is not None
    sem_at_plateau = (
        levels[plateau_idx].sem if plateau_reached else levels[-1].sem
    )
    g = (sem_at_plateau / sem_naive) ** 2 if sem_naive > 0 else float("nan")

    return BlockAverageResult(
        mean=mean,
        sem=sem_at_plateau,
        sem_naive=sem_naive,
        statistical_inefficiency=float(g),
        plateau_reached=plateau_reached,
        n_levels=len(levels),
        levels=tuple(levels),
    )


def _detect_plateau(levels: list[BlockLevel], *, window: int) -> int | None:
    if len(levels) < window:
        return None
    for k in range(len(levels) - window + 1):
        sems = [lv.sem for lv in levels[k : k + window]]
        max_err = max(lv.sem_err for lv in levels[k : k + window])
        if max(sems) - min(sems) < 2.0 * max_err:
            return k
    return None
