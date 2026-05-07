"""Autocorrelation diagnostic recovers known tau_int on AR(1)."""

from __future__ import annotations

import numpy as np
import pytest

from mdpilot.diagnostics.autocorrelation import autocorrelation


def _ar1(n: int, phi: float, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sigma_eps = np.sqrt(1.0 - phi * phi)
    x = np.empty(n)
    x[0] = rng.normal()
    for t in range(1, n):
        x[t] = phi * x[t - 1] + sigma_eps * rng.normal()
    return x


@pytest.mark.parametrize("phi,n,tol_frac", [(0.9, 200_000, 0.25), (0.5, 100_000, 0.25)])
def test_autocorrelation_recovers_ar1_tau(phi: float, n: int, tol_frac: float) -> None:
    expected_tau = (1.0 + phi) / (2.0 * (1.0 - phi))
    x = _ar1(n, phi, seed=98765)
    result = autocorrelation(x)
    rel_err = abs(result.tau_int - expected_tau) / expected_tau
    assert rel_err < tol_frac, (
        f"phi={phi}: tau_est={result.tau_int:.3f} vs expected={expected_tau:.3f} "
        f"(rel_err={rel_err:.2%})"
    )


def test_autocorrelation_iid_tau_near_half() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(50_000)
    result = autocorrelation(x)
    assert 0.45 <= result.tau_int <= 0.8, result.tau_int
    assert result.ess > 30_000
    assert result.well_sampled


def test_autocorrelation_well_sampled_threshold() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(200)
    r1 = autocorrelation(x, target_ess=50.0)
    r2 = autocorrelation(x, target_ess=10_000.0)
    assert r1.well_sampled is True
    assert r2.well_sampled is False


def test_autocorrelation_rejects_tiny_input() -> None:
    with pytest.raises(ValueError, match="at least"):
        autocorrelation(np.zeros(3))
