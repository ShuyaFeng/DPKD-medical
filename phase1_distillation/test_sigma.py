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
from drive_pate_canal_combined import (
    correct_waterfilling_sigma, add_dp_noise_to_importance,
    importance_sensitivity,
)
from synthetic_demo import uniform_sigma, waterfilling_sigma


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
    """add_dp_noise_to_importance: sigma_imp = sensitivity / sqrt(2 * rho_imp).

    Use large importance values so clamp(min=1e-6) never activates,
    then measure std of the noise directly.
    """
    sensitivity = 1.0
    rho_imp = 0.5
    expected_sigma = sensitivity / math.sqrt(2.0 * rho_imp)  # = 1.0

    # Large base so clamp(min=1e-6) never fires; noise is unmodified
    importance = torch.full((10000,), 1e6)
    noisy = add_dp_noise_to_importance(importance, sensitivity, rho_imp, seed=42)
    noise = noisy - importance
    sigma_est = noise.std().item()
    assert abs(sigma_est - expected_sigma) < 0.05, (
        "FAIL: estimated sigma_imp={:.4f}, expected {:.4f}. "
        "Check that add_dp_noise_to_importance divides by sqrt(2*rho_imp).".format(
            sigma_est, expected_sigma
        )
    )


def test_legacy_sigma_functions_equal_correct_versions():
    """synthetic_demo's uniform_sigma / waterfilling_sigma must be IDENTICAL
    to the correct_ versions (the missing-/2 bug regression test).
    Several local helper copies call the synthetic_demo versions, so any
    drift here silently un-fixes the factor-of-2 bug."""
    torch.manual_seed(7)
    for _ in range(20):
        deltas = torch.rand(12) + 0.05
        imp = torch.rand(12) * 4 + 1e-6
        rho = float(torch.rand(1)) * 3 + 0.01
        assert torch.allclose(
            uniform_sigma(deltas, rho), correct_uniform_sigma(deltas, rho),
            rtol=1e-6,
        ), "FAIL: synthetic_demo.uniform_sigma drifted from correct_uniform_sigma"
        assert torch.allclose(
            waterfilling_sigma(deltas, imp, rho),
            correct_waterfilling_sigma(deltas, imp, rho), rtol=1e-6,
        ), "FAIL: synthetic_demo.waterfilling_sigma drifted from correct_waterfilling_sigma"


def test_masked_helper_copies_spend_exact_budget():
    """Every masked/thresholded helper copy must spend exactly rho on the
    active set and release nothing (sigma=0) off it."""
    helpers = []
    from drive_pate_pruning_joint import thresholded_uniform_sigma as h1
    helpers.append(("drive_pate_pruning_joint", h1))
    try:
        from drive_pruning_ablation import masked_uniform_sigma as h2
        helpers.append(("drive_pruning_ablation", h2))
    except ImportError as e:
        print("    [skip] drive_pruning_ablation: {}".format(e))
    try:
        from drive_per_teacher_importance import thresholded_uniform_sigma as h3
        helpers.append(("drive_per_teacher_importance", h3))
    except ImportError as e:
        print("    [skip] drive_per_teacher_importance: {}".format(e))
    deltas = torch.full((16,), 2.0 / 3.0)
    mask = torch.zeros(16, dtype=torch.bool)
    mask[[1, 5, 6, 12]] = True
    rho = 0.7
    for name, h in helpers:
        sigma = h(deltas, rho, mask)
        cost = (deltas[mask].pow(2) / (2.0 * sigma[mask].pow(2))).sum().item()
        assert abs(cost - rho) < 1e-4, (
            "FAIL {}: zCDP cost={:.6f} != rho={} (missing /2?)".format(name, cost, rho))
        assert (sigma[~mask] == 0).all(), name


def test_importance_sensitivity_teacher_level():
    """Delta_imp = (2 clip / K) * (1 + (K-1)/N): teacher-level bound.
    K=1 -> 2*clip (whole average can flip); N->inf -> 2*clip/K."""
    clip = 100.0
    assert abs(importance_sensitivity(clip, 1, 900) - 2 * clip) < 1e-9
    assert abs(importance_sensitivity(clip, 10, 10**12) - 2 * clip / 10) < 1e-6
    K, N = 10, 900
    expect = (2 * clip / K) * (1 + (K - 1) / N)
    assert abs(importance_sensitivity(clip, K, N) - expect) < 1e-9
    # must never fall below either the old (invalid) 2*clip/N or 2*clip/K
    assert importance_sensitivity(clip, K, N) >= 2 * clip / N
    assert importance_sensitivity(clip, K, N) >= 2 * clip / K


if __name__ == "__main__":
    tests = [
        test_uniform_sigma_known_value,
        test_uniform_sigma_multi_channel,
        test_uniform_sigma_zcdp_budget_satisfied,
        test_waterfilling_sigma_equals_uniform_when_importance_uniform,
        test_waterfilling_sigma_zcdp_budget_satisfied,
        test_importance_noise_sigma_has_factor_of_two,
        test_legacy_sigma_functions_equal_correct_versions,
        test_masked_helper_copies_spend_exact_budget,
        test_importance_sensitivity_teacher_level,
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
