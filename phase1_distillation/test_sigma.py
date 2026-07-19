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
    """Delta=1, rho=0.5 -> sigma = sqrt(1/(2*0.5)) = 1.0 exactly."""
    deltas = torch.tensor([1.0])
    sigma = correct_uniform_sigma(deltas, rho=0.5)
    assert sigma.shape == (1,), "Expected shape (1,), got {}".format(sigma.shape)
    assert abs(sigma[0].item() - 1.0) < 1e-6, (
        "FAIL: expected sigma=1.0, got {:.8f}. "
        "Check that correct_uniform_sigma divides by 2*rho, not just rho.".format(sigma[0].item())
    )


def test_uniform_sigma_multi_channel():
    """Two channels, each Delta=1, rho=1 -> sigma = sqrt(2/(2*1)) = 1.0."""
    deltas = torch.tensor([1.0, 1.0])
    sigma = correct_uniform_sigma(deltas, rho=1.0)
    expected = math.sqrt(2.0 / (2.0 * 1.0))
    assert abs(sigma[0].item() - expected) < 1e-6, (
        "FAIL: expected sigma={:.6f}, got {:.8f}".format(expected, sigma[0].item())
    )
    assert abs(sigma[0].item() - sigma[1].item()) < 1e-9, "sigma must be equal across channels"


def test_uniform_sigma_zcdp_budget_satisfied():
    """Verify the returned sigma satisfies the zCDP constraint: sum(Delta^2 / (2*sigma^2)) == rho."""
    deltas = torch.tensor([1.0, 2.0, 0.5])
    rho = 2.0
    sigma = correct_uniform_sigma(deltas, rho=rho)
    cost = (deltas.pow(2) / (2.0 * sigma.pow(2))).sum().item()
    assert abs(cost - rho) < 1e-5, (
        "FAIL: zCDP cost={:.8f} does not equal rho={}. sigma is wrong.".format(cost, rho)
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
            "FAIL at channel {}: WF={:.8f}, uniform={:.8f}. "
            "With uniform importance, CANAL must equal uniform allocation.".format(
                c, sigma_wf[c].item(), sigma_uni[c].item()
            )
        )


def test_waterfilling_sigma_zcdp_budget_satisfied():
    """Verify CANAL sigma satisfies the zCDP budget: sum(Delta^2 / (2*sigma^2)) == rho."""
    deltas = torch.tensor([1.0, 2.0, 0.5, 1.5])
    importance = torch.tensor([0.8, 0.2, 0.5, 1.0])
    rho = 1.5

    sigma = correct_waterfilling_sigma(deltas, importance, rho)
    cost = (deltas.pow(2) / (2.0 * sigma.pow(2))).sum().item()
    assert abs(cost - rho) < 1e-5, (
        "FAIL: WF zCDP cost={:.8f} does not equal rho={}. "
        "Check the factor of 2 in kappa.".format(cost, rho)
    )


def test_importance_noise_sigma_has_factor_of_two():
    """add_dp_noise_to_importance: sigma_imp = sensitivity / sqrt(2 * rho_imp)."""
    sensitivity = 1.0
    rho_imp = 0.5
    expected_sigma = sensitivity / math.sqrt(2.0 * rho_imp)  # = 1.0

    importance = torch.zeros(10000)
    noisy = add_dp_noise_to_importance(importance, sensitivity, rho_imp, seed=42)
    noise = noisy - importance.clamp(min=1e-6)
    approx_sigma = noise.abs().mean().item() / math.sqrt(2.0 / math.pi)
    assert abs(approx_sigma - expected_sigma) < 0.05, (
        "FAIL: estimated sigma_imp={:.4f}, expected {:.4f}. "
        "Check that add_dp_noise_to_importance divides by sqrt(2*rho_imp).".format(
            approx_sigma, expected_sigma
        )
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
            print("  PASS  {}".format(t.__name__))
        except AssertionError as e:
            print("  FAIL  {}: {}".format(t.__name__, e))
            failed += 1

    print("")
    if failed == 0:
        print("All {} tests passed.".format(len(tests)))
    else:
        print("{}/{} tests FAILED.".format(failed, len(tests)))
        sys.exit(1)
