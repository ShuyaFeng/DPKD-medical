# -*- coding: utf-8 -*-
"""
Unit tests for sigma helper functions.

Run with:  python test_sigma.py
All tests pass silently; any failure prints the assertion error.
"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from drive_pate_poc import correct_uniform_sigma
from drive_pate_canal_combined import correct_waterfilling_sigma, add_dp_noise_to_importance


def test_uniform_sigma_known_value():
    """Delta=1, rho=0.5 → sigma = sqrt(1/(2*0.5)) = 1.0 exactly."""
    deltas = torch.tensor([1.0])
    sigma = correct_uniform_sigma(deltas, rho=0.5)
    assert sigma.shape == (1,), f"Expected shape (1,), got {sigma.shape}"
    assert abs(sigma[0].item() - 1.0) < 1e-6, (
        f"FAIL: expected sigma=1.0, got {sigma[0].item():.8f}. "
        f"Check that correct_uniform_sigma divides by 2*rho, not just rho."
    )


def test_uniform_sigma_multi_channel():
    """Two channels, each Delta=1, rho=1 → sigma = sqrt(2/(2*1)) = 1.0."""
    deltas = torch.tensor([1.0, 1.0])
    sigma = correct_uniform_sigma(deltas, rho=1.0)
    expected = math.sqrt(2.0 / (2.0 * 1.0))  # = 1.0
    assert abs(sigma[0].item() - expected) < 1e-6, (
        f"FAIL: expected sigma={expected:.6f}, got {sigma[0].item():.8f}"
    )
    assert abs(sigma[0].item() - sigma[1].item()) < 1e-9, "sigma must be equal across channels"


def test_uniform_sigma_zcdp_budget_satisfied():
    """Verify the returned sigma actually satisfies the zCDP constraint: sum(Delta^2 / (2*sigma^2)) == rho."""
    deltas = torch.tensor([1.0, 2.0, 0.5])
    rho = 2.0
    sigma = correct_uniform_sigma(deltas, rho=rho)
    cost = (deltas.pow(2) / (2.0 * sigma.pow(2))).sum().item()
    assert abs(cost - rho) < 1e-5, (
        f"FAIL: zCDP cost={cost:.8f} does not equal rho={rho}. sigma is wrong."
    )


def test_waterfilling_sigma_equals_uniform_when_importance_uniform():
    """When all importance scores are equal, CANAL must equal uniform allocation."""
    C = 4
    deltas = torch.tensor([1.0] * C)
    importance = torch.tensor([1.0] * C)
    rho = 1.0

    sigma_wf = correct_waterfilling_sigma(deltas, importance, rho)
    sigma_uni = correct_uniform_sigma(deltas, rho)

    for c in range(C):
        assert abs(sigma_wf[c].item() - sigma_uni[c].item()) < 1e-5, (
            f"FAIL at channel {c}: WF={sigma_wf[c].item():.8f}, "
            f"uniform={sigma_uni[c].item():.8f}. "
            f"With uniform importance, CANAL must equal uniform allocation."
        )


def test_waterfilling_sigma_zcdp_budget_satisfied():
    """Verify CANAL sigma satisfies the zCDP budget: sum(Delta^2 / (2*sigma^2)) == rho."""
    deltas = torch.tensor([1.0, 2.0, 0.5, 1.5])
    importance = torch.tensor([0.8, 0.2, 0.5, 1.0])
    rho = 1.5

    sigma = correct_waterfilling_sigma(deltas, importance, rho)
    cost = (deltas.pow(2) / (2.0 * sigma.pow(2))).sum().item()
    assert abs(cost - rho) < 1e-5, (
        f"FAIL: WF zCDP cost={cost:.8f} does not equal rho={rho}. "
        f"Check the factor of 2 in kappa."
    )


def test_importance_noise_sigma_has_factor_of_two():
    """add_dp_noise_to_importance: sigma_imp = sensitivity / sqrt(2 * rho_imp)."""
    # With sensitivity=1 and rho_imp=0.5: sigma_imp = 1/sqrt(1) = 1.0
    # Noise std should be 1.0; we verify via many samples (mean~0, std~1).
    torch.manual_seed(0)
    sensitivity = 1.0
    rho_imp = 0.5  # → sigma_imp = 1/sqrt(2*0.5) = 1.0
    expected_sigma = sensitivity / math.sqrt(2.0 * rho_imp)  # = 1.0

    importance = torch.zeros(10000)
    noisy = add_dp_noise_to_importance(importance, sensitivity, rho_imp, seed=42)
    # clamp(min=1e-6) shifts mean slightly above 0 for negative draws; use std
    # of the raw noise by subtracting the clean importance
    noise = noisy - importance.clamp(min=1e-6)
    # approximate: use absolute values distribution mean ≈ sigma * sqrt(2/pi)
    approx_sigma = noise.abs().mean().item() / math.sqrt(2.0 / math.pi)
    assert abs(approx_sigma - expected_sigma) < 0.05, (
        f"FAIL: estimated sigma_imp={approx_sigma:.4f}, expected {expected_sigma:.4f}. "
        f"Check that add_dp_noise_to_importance divides by sqrt(2*rho_imp)."
    )


if __name__ == "__main__":
    tests = [
        test_uniform_sigma_known_value,
        test_uniform_sigma_multi_channel,
        test_uniform_sigma_zcdp_budget_satisfied,
        test_waterfilling_sigma_equals_uniform_when_importance_uniform,
        test_waterfilling_sigma_zcdp_budget_satisfied,
        test_importance_noise_sigma_has_factor_of_two,
    ]

    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1

    print()
    if failed == 0:
        print(f"All {len(tests)} tests passed.")
    else:
        print(f"{failed}/{len(tests)} tests FAILED.")
        sys.exit(1)
