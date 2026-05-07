"""Geyer initial-monotone-positive-sequence (IMPS) estimator of integrated
autocorrelation time and effective sample size.

Reference: Geyer, "Practical Markov Chain Monte Carlo", Stat. Sci. 7:473 (1992).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AutocorrResult:
    tau_int: float
    statistical_inefficiency: float  # g = 2*tau_int (matches block-averaging plateau)
    ess: float
    well_sampled: bool
    n_samples: int


def autocorrelation(
    x: np.ndarray,
    *,
    target_ess: float = 50.0,
) -> AutocorrResult:
    """Integrated autocorrelation time via Geyer IMPS, and effective sample size.

    For AR(1) with correlation phi: tau_int = (1+phi) / (2*(1-phi)) and
    ESS = n / (2*tau_int). The statistical inefficiency g = 2*tau_int is
    in the same units as block-averaging's plateau ratio, so the two
    diagnostics agree on the same input.

    `well_sampled` is True when ESS >= target_ess. The threshold is a
    blunt heuristic — adjust per-observable in the report bundle.
    """
    arr = np.asarray(x, dtype=float).ravel()
    n = arr.size
    if n < 4:
        raise ValueError(f"need at least 4 samples, got {n}")

    centered = arr - arr.mean()
    var = float(np.dot(centered, centered) / n)
    if var <= 0.0:
        return AutocorrResult(
            tau_int=0.5,
            statistical_inefficiency=1.0,
            ess=float(n),
            well_sampled=n >= target_ess,
            n_samples=n,
        )

    # Unbiased autocovariance via FFT
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(centered, n=nfft)
    acov = np.fft.irfft(f * np.conj(f), n=nfft)[:n]
    acov = acov / (n - np.arange(n))
    rho = acov / acov[0]

    # Geyer IMPS: pair adjacent ρ(2m), ρ(2m+1); truncate when pair sum ≤ 0;
    # enforce non-increasing monotone clamp on the pair sums.
    sum_pairs = 0.0
    prev = np.inf
    m_max = n // 2
    for m in range(m_max):
        if 2 * m + 1 >= n:
            break
        gamma = float(rho[2 * m] + rho[2 * m + 1])
        if gamma <= 0.0:
            break
        if gamma > prev:
            gamma = prev
        sum_pairs += gamma
        prev = gamma

    tau_int = -0.5 + sum_pairs
    if tau_int < 0.5:
        tau_int = 0.5  # floor at the iid limit
    g = 2.0 * tau_int
    ess = n / g

    return AutocorrResult(
        tau_int=float(tau_int),
        statistical_inefficiency=float(g),
        ess=float(ess),
        well_sampled=bool(ess >= target_ess),
        n_samples=n,
    )
