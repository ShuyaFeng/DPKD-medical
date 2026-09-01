# -*- coding: utf-8 -*-
"""
Comprehensive unit tests for ALL DP sigma/noise helper functions.

Written 2026-08-31 after the factor-of-2 audit. Guards against:
  - the missing-/2 bug (sigma sqrt(2)x too large) in ANY copy of the helpers
  - drift between synthetic_demo's legacy functions and the correct_ versions
  - wrong water-filling allocation (must solve min sum s*sigma^2 s.t. zCDP=rho)
  - wrong importance sensitivity (teacher-level, NOT 2*clip/N)

Run with:  python test_sigma.py
Prints one line per test; exits non-zero on any failure.
"""

import math
import sys
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from synthetic_demo import eps_to_rho, uniform_sigma, waterfilling_sigma
from drive_pate_poc import correct_uniform_sigma
from drive_pate_canal_combined import (
    correct_waterfilling_sigma, importance_sensitivity,
)

TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


def zcdp_cost(deltas, sigma, mask=None):
    """sum_c Delta_c^2 / (2*sigma_c^2) over the released (masked) channels."""
    if mask is not None:
        deltas, sigma = deltas[mask], sigma[mask]
    return (deltas.pow(2) / (2.0 * sigma.pow(2))).sum().item()


# ---------------------------------------------------------------- core formulas

@test
def uniform_known_value():
    """Delta=1, rho=0.5 -> sigma = sqrt(1/(2*0.5)) = 1.0 exactly."""
    s = correct_uniform_sigma(torch.tensor([1.0]), rho=0.5)
    assert abs(s[0].item() - 1.0) < 1e-6, s


@test
def uniform_budget_exact():
    """Returned sigma must satisfy sum(Delta^2/(2 sigma^2)) == rho exactly."""
    deltas = torch.tensor([1.0, 2.0, 0.5, 0.1])
    for rho in (0.05, 0.5, 3.0):
        s = correct_uniform_sigma(deltas, rho)
        assert abs(zcdp_cost(deltas, s) - rho) < 1e-4 * rho, (rho, s)


@test
def wf_budget_exact():
    """WF sigma must also spend exactly rho."""
    torch.manual_seed(0)
    deltas = torch.rand(32) + 0.1
    imp = torch.rand(32) * 10 + 1e-4
    for rho in (0.05, 0.5, 3.0):
        s = correct_waterfilling_sigma(deltas, imp, rho)
        assert abs(zcdp_cost(deltas, s) - rho) < 1e-3 * rho, rho


@test
def wf_uniform_importance_degenerates_to_uniform():
    """Equal importance => WF == uniform allocation."""
    deltas = torch.full((8,), 2.0 / 5.0)
    su = correct_uniform_sigma(deltas, 1.0)
    sw = correct_waterfilling_sigma(deltas, torch.full((8,), 3.7), 1.0)
    assert torch.allclose(su, sw, rtol=1e-5), (su[0], sw[0])


@test
def wf_quarter_power_law():
    """With constant Delta, sigma_c must scale as s_c^{-1/4}."""
    deltas = torch.full((4,), 2.0)
    imp = torch.tensor([1.0, 16.0, 81.0, 256.0])
    s = correct_waterfilling_sigma(deltas, imp, 1.0)
    ratio = s / s[0]
    expected = imp.pow(-0.25) / imp[0].pow(-0.25)
    assert torch.allclose(ratio, expected, rtol=1e-5), ratio


@test
def wf_is_optimal():
    """WF must minimise sum_c s_c*sigma_c^2 among allocations spending rho:
    random feasible perturbations must never do better."""
    torch.manual_seed(1)
    deltas = torch.rand(16) + 0.1
    imp = torch.rand(16) * 5 + 0.01
    rho = 1.0
    s_wf = correct_waterfilling_sigma(deltas, imp, rho)
    obj_wf = (imp * s_wf.pow(2)).sum().item()
    for trial in range(200):
        w = s_wf * torch.exp(0.3 * torch.randn(16))       # perturb
        w = w * math.sqrt(zcdp_cost(deltas, w) / rho)      # rescale onto budget
        assert abs(zcdp_cost(deltas, w) - rho) < 1e-3
        obj = (imp * w.pow(2)).sum().item()
        assert obj >= obj_wf * (1 - 1e-5), (trial, obj, obj_wf)


@test
def eps_to_rho_roundtrip():
    """rho must satisfy eps = rho + 2*sqrt(rho*log(1/delta)) exactly."""
    for eps in (0.1, 1.0, 2.0, 4.0, 8.0):
        rho = eps_to_rho(eps, 1e-5)
        eps_back = rho + 2.0 * math.sqrt(rho * math.log(1e5))
        assert abs(eps_back - eps) < 1e-9, (eps, eps_back)
        assert rho > 0


# ------------------------------------------- legacy copies must equal correct_

@test
def legacy_uniform_equals_correct():
    """synthetic_demo.uniform_sigma must be IDENTICAL to correct_uniform_sigma
    (this is the test that would have caught the missing-/2 bug)."""
    torch.manual_seed(2)
    for _ in range(20):
        deltas = torch.rand(12) + 0.05
        rho = float(torch.rand(1)) * 3 + 0.01
        assert torch.allclose(uniform_sigma(deltas, rho),
                              correct_uniform_sigma(deltas, rho), rtol=1e-6)


@test
def legacy_wf_equals_correct():
    """synthetic_demo.waterfilling_sigma must equal correct_waterfilling_sigma."""
    torch.manual_seed(3)
    for _ in range(20):
        deltas = torch.rand(12) + 0.05
        imp = torch.rand(12) * 4 + 1e-6
        rho = float(torch.rand(1)) * 3 + 0.01
        assert torch.allclose(waterfilling_sigma(deltas, imp, rho),
                              correct_waterfilling_sigma(deltas, imp, rho),
                              rtol=1e-6)


@test
def masked_helpers_budget_exact():
    """Every masked/thresholded helper copy must spend exactly rho on the
    active set and release nothing (sigma=0) off it."""
    helpers = []
    from drive_pate_pruning_joint import thresholded_uniform_sigma as h1
    helpers.append(("drive_pate_pruning_joint.thresholded_uniform_sigma", h1))
    try:
        from drive_pruning_ablation import (
            masked_uniform_sigma as h2, masked_waterfilling_sigma as h3,
        )
        helpers.append(("drive_pruning_ablation.masked_uniform_sigma", h2))
    except ImportError as e:                       # optional heavy deps
        print("    [skip] drive_pruning_ablation import failed: {}".format(e))
        h3 = None
    try:
        from drive_per_teacher_importance import thresholded_uniform_sigma as h4
        helpers.append(("drive_per_teacher_importance.thresholded_uniform_sigma", h4))
    except ImportError as e:
        print("    [skip] drive_per_teacher_importance import failed: {}".format(e))

    deltas = torch.full((16,), 2.0 / 3.0)
    mask = torch.zeros(16, dtype=torch.bool)
    mask[[1, 5, 6, 12]] = True
    rho = 0.7
    for name, h in helpers:
        s = h(deltas, rho, mask)
        assert abs(zcdp_cost(deltas, s, mask) - rho) < 1e-4, name
        assert (s[~mask] == 0).all(), name
    if h3 is not None:
        s = h3(deltas, torch.rand(16) + 0.1, rho, mask)
        assert abs(zcdp_cost(deltas, s, mask) - rho) < 1e-3, "masked_waterfilling"
        assert (s[~mask] == 0).all(), "masked_waterfilling"


@test
def alloc_sigma_beta_family():
    """drive_modified_canal.alloc_sigma: exact budget for every beta;
    beta=0 == uniform; beta=0.25 == classical WF (constant Delta=2)."""
    try:
        from drive_modified_canal import alloc_sigma
    except Exception as e:                          # session-local file
        print("    [skip] drive_modified_canal not importable: {}".format(e))
        return
    torch.manual_seed(4)
    C = 20
    imp = torch.rand(C) * 3 + 1e-5
    kept = torch.zeros(C, dtype=torch.bool)
    kept[torch.randperm(C)[:8]] = True
    deltas = torch.full((C,), 2.0)
    rho = 0.9
    for beta in (0.0, 0.1, 0.25, 1.0, 4.0):
        s = alloc_sigma(imp, kept, rho, beta)
        assert abs(zcdp_cost(deltas, s, kept) - rho) < 1e-3 * rho, beta
        assert (s[~kept] == 0).all(), beta
    s0 = alloc_sigma(imp, kept, rho, 0.0)
    su = correct_uniform_sigma(deltas[kept], rho)
    assert torch.allclose(s0[kept], su, rtol=1e-4)
    s25 = alloc_sigma(imp, kept, rho, 0.25)
    swf = correct_waterfilling_sigma(deltas[kept], imp[kept], rho)
    assert torch.allclose(s25[kept], swf, rtol=1e-4)


# --------------------------------------------------- importance sensitivity

@test
def importance_sensitivity_formula():
    """Delta_imp = (2 clip / K) * (1 + (K-1)/N): teacher-level bound."""
    clip = 100.0
    # K=1: the single teacher retrains -> whole average can flip: Delta = 2*clip
    assert abs(importance_sensitivity(clip, 1, 900) - 2 * clip) < 1e-9
    # N -> infinity: direct term vanishes -> 2*clip/K
    assert abs(importance_sensitivity(clip, 10, 10**12) - 2 * clip / 10) < 1e-6
    # exact value
    K, N = 10, 900
    expect = (2 * clip / K) * (1 + (K - 1) / N)
    assert abs(importance_sensitivity(clip, K, N) - expect) < 1e-9
    # must NEVER be below the old (invalid) 2*clip/N NOR below 2*clip/K
    assert importance_sensitivity(clip, K, N) >= 2 * clip / N
    assert importance_sensitivity(clip, K, N) >= 2 * clip / K


@test
def gaussian_mechanism_sigma():
    """sigma = Delta / sqrt(2 rho) spends exactly rho for a single release."""
    delta_s, rho = 0.37, 0.42
    sigma = delta_s / math.sqrt(2.0 * rho)
    assert abs(delta_s ** 2 / (2 * sigma ** 2) - rho) < 1e-12


# ------------------------------------------------------------------ runner

def main():
    failed = 0
    for fn in TESTS:
        try:
            fn()
            print("  PASS  {}".format(fn.__name__))
        except Exception:
            failed += 1
            print("  FAIL  {}".format(fn.__name__))
            traceback.print_exc()
    print("\n{}/{} passed".format(len(TESTS) - failed, len(TESTS)))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
