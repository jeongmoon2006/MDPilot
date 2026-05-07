"""Block-averaging diagnostic recovers known statistical inefficiency on AR(1)."""

from __future__ import annotations

import numpy as np
import pytest

from mdpilot.diagnostics.block_averaging import block_average


def _ar1(n: int, phi: float, *, seed: int) -> np.ndarray:
    """Generate an AR(1) sample: x_t = phi x_{t-1} + eps_t, eps ~ N(0, 1-phi^2).

    Stationary variance is 1.0; population statistical inefficiency g = (1+phi)/(1-phi).
    """
    rng = np.random.default_rng(seed)
    sigma_eps = np.sqrt(1.0 - phi * phi)
    x = np.empty(n)
    x[0] = rng.normal()
    for t in range(1, n):
        x[t] = phi * x[t - 1] + sigma_eps * rng.normal()
    return x


@pytest.mark.parametrize("phi,n,tol_frac", [(0.9, 200_000, 0.20), (0.5, 100_000, 0.20)])
def test_block_average_recovers_ar1_inefficiency(phi: float, n: int, tol_frac: float) -> None:
    expected_g = (1.0 + phi) / (1.0 - phi)
    x = _ar1(n, phi, seed=12345)
    result = block_average(x)
    assert result.plateau_reached, "expected a plateau on a long stationary AR(1)"
    rel_err = abs(result.statistical_inefficiency - expected_g) / expected_g
    assert rel_err < tol_frac, (
        f"phi={phi}: g_est={result.statistical_inefficiency:.2f} vs expected={expected_g:.2f} "
        f"(rel_err={rel_err:.2%})"
    )


def test_block_average_iid_gives_g_near_1() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(50_000)
    result = block_average(x)
    assert result.plateau_reached
    assert 0.7 < result.statistical_inefficiency < 1.3, result.statistical_inefficiency


def test_block_average_rejects_too_few_samples() -> None:
    with pytest.raises(ValueError, match="at least"):
        block_average(np.zeros(4))
