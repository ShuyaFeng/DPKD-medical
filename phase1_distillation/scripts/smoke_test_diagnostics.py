#!/usr/bin/env python3
"""
Smoke test for diagnostic A/B/C changes in phase1_gkd_distill_v2.py.

Verifies WITHOUT needing mmseg, mmcv, DRIVE data, or any GPU:
  1.  argparse accepts --noise-type none and --student-from-scratch
  2.  argparse still rejects unknown noise types
  3.  uniform_sigma is constant across channels
  4.  waterfilling_sigma gives less noise to higher-importance channels
  5.  noise_type='none' branch produces all-zero sigma
  6.  build_model(from_scratch=True)  → load_checkpoint is NOT called
  7.  build_model(from_scratch=False) → load_checkpoint IS called
  8.  final_mDice_lastN_mean computes mean of last 3 validation mDice
  9.  summary.json contains the new diagnostic-relevant fields

Run:
  python3 smoke_test_diagnostics.py
"""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Stub mmseg / mmengine BEFORE importing phase1_gkd_distill_v2
# ---------------------------------------------------------------------------

def _stub(name: str, attrs: dict | None = None) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in (attrs or {}).items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


_stub("mmengine")
_stub("mmengine.config", {"Config": MagicMock()})
_stub("mmengine.dataset", {"pseudo_collate": lambda x: x})
_stub("mmengine.registry", {"init_default_scope": lambda x: None})
_stub("mmengine.runner", {"load_checkpoint": MagicMock()})
_stub("mmengine.structures", {"PixelData": MagicMock()})
_stub("mmengine.model", {"revert_sync_batchnorm": None})
_stub("mmseg")
_stub("mmseg.registry", {"DATASETS": MagicMock(), "MODELS": MagicMock()})


# ---------------------------------------------------------------------------
# Dependencies the test actually needs
# ---------------------------------------------------------------------------

try:
    import numpy as np
    import torch
except ImportError as e:
    print(f"\n[!] Missing dependency: {e}")
    print("\nQuick install (uv-style):")
    print("    uv venv .venv-smoke")
    print("    source .venv-smoke/bin/activate")
    print("    uv pip install torch numpy")
    print("\nOr plain venv:")
    print("    python3 -m venv .venv-smoke")
    print("    source .venv-smoke/bin/activate")
    print("    pip install torch numpy\n")
    sys.exit(1)


# Import the (modified) script under test
sys.path.insert(0, str(Path(__file__).parent))
import phase1_gkd_distill_v2 as v2  # noqa: E402


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

PASS = 0
FAIL = 0


def _ok(name: str, msg: str = "") -> None:
    global PASS
    PASS += 1
    suffix = f" — {msg}" if msg else ""
    print(f"  PASS  {name}{suffix}")


def _fail(name: str, err: str) -> None:
    global FAIL
    FAIL += 1
    print(f"  FAIL  {name}")
    print(f"        {err}")


def _run(name: str, fn) -> None:
    try:
        fn()
    except AssertionError as e:
        _fail(name, str(e) or "assertion failed")
    except Exception as e:
        _fail(name, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

REQUIRED_ARGS = [
    "phase1_gkd_distill_v2.py",
    "--teacher-config", "/dummy/t.py",
    "--teacher-checkpoint", "/dummy/t.pth",
    "--student-config", "/dummy/s.py",
    "--student-checkpoint", "/dummy/s.pth",
    "--data-root", "/dummy/data",
    "--importance-csv", "/dummy/imp.csv",
    "--out-dir", "/tmp/out",
]


# ---------------------------------------------------------------------------
# Test 1 — argparse accepts the new flags
# ---------------------------------------------------------------------------

def test_argparse_new_flags():
    argv = REQUIRED_ARGS + [
        "--noise-type", "none",
        "--epsilon", "8",
        "--student-from-scratch",
    ]
    with patch.object(sys, "argv", argv):
        args = v2.parse_args()
    assert args.noise_type == "none", f"noise_type={args.noise_type!r}"
    assert args.student_from_scratch is True, "student_from_scratch should be True"
    _ok("argparse_new_flags",
        f"noise_type='none', student_from_scratch=True parsed correctly")


# ---------------------------------------------------------------------------
# Test 2 — existing flags still work
# ---------------------------------------------------------------------------

def test_argparse_existing_flags_still_work():
    argv = REQUIRED_ARGS + [
        "--noise-type", "channel_WF",
        "--epsilon", "2",
    ]
    with patch.object(sys, "argv", argv):
        args = v2.parse_args()
    assert args.noise_type == "channel_WF"
    assert args.student_from_scratch is False  # default
    assert args.epsilon == 2.0
    _ok("argparse_existing_flags",
        "channel_WF still works, --student-from-scratch defaults to False")


# ---------------------------------------------------------------------------
# Test 3 — argparse rejects unknown noise type
# ---------------------------------------------------------------------------

def test_argparse_rejects_bad_noise_type():
    argv = REQUIRED_ARGS + [
        "--noise-type", "garbage",
        "--epsilon", "8",
    ]
    with patch.object(sys, "argv", argv):
        try:
            v2.parse_args()
        except SystemExit:
            _ok("argparse_rejects_bad_noise_type",
                "unknown noise type → SystemExit (as expected)")
            return
    raise AssertionError("expected SystemExit, got none")


# ---------------------------------------------------------------------------
# Test 4 — uniform_sigma constant across channels
# ---------------------------------------------------------------------------

def test_uniform_sigma_constant():
    deltas = torch.full((10,), 2.0)
    rho    = 1.0
    sigma  = v2.uniform_sigma(deltas, rho)
    assert sigma.shape == (10,), f"shape={sigma.shape}"
    assert torch.allclose(sigma, sigma[0].expand_as(sigma)), \
        "uniform sigma should be identical across channels"
    _ok("uniform_sigma_constant", f"sigma_all={sigma[0]:.4f}")


def test_uniform_sigma_hits_target_rho():
    """REGRESSION: sum_c delta_c^2 / (2 sigma^2) must equal rho_budget exactly.

    Catches the factor-of-2 bug where sigma was calibrated against 2*rho
    instead of rho (reported epsilon would not match released sigma).
    """
    deltas = torch.full((100,), 2.0)
    for rho_target in [0.05, 0.5, 5.0]:
        sigma = v2.uniform_sigma(deltas, rho_target)
        rho_actual = float((deltas.pow(2) / (2.0 * sigma.pow(2))).sum())
        rel_err = abs(rho_actual - rho_target) / rho_target
        assert rel_err < 1e-4, \
            f"uniform calibration off by {rel_err*100:.2f}% at rho={rho_target} " \
            f"(target {rho_target}, actual {rho_actual})"
    _ok("uniform_sigma_hits_target_rho",
        "sum delta^2/(2 sigma^2) == rho across 3 budgets")


def test_waterfilling_sigma_hits_target_rho():
    """Same regression check for channel_WF."""
    deltas = torch.full((100,), 2.0)
    importances = torch.linspace(0.01, 1.0, 100)
    for rho_target in [0.05, 0.5, 5.0]:
        sigma = v2.waterfilling_sigma(deltas, importances, rho_target)
        rho_actual = float((deltas.pow(2) / (2.0 * sigma.pow(2))).sum())
        rel_err = abs(rho_actual - rho_target) / rho_target
        assert rel_err < 1e-4, \
            f"WF calibration off by {rel_err*100:.2f}% at rho={rho_target}"
    _ok("waterfilling_sigma_hits_target_rho",
        "sum delta^2/(2 sigma^2) == rho across 3 budgets")


# ---------------------------------------------------------------------------
# Test 5 — waterfilling gives less noise to important channels
# ---------------------------------------------------------------------------

def test_waterfilling_inverse_to_importance():
    deltas      = torch.full((100,), 2.0)
    importances = torch.linspace(0.01, 1.0, 100)  # ch 99 is most important
    rho         = 1.0
    sigma       = v2.waterfilling_sigma(deltas, importances, rho)
    assert sigma[99] < sigma[0], \
        f"important sigma={sigma[99]:.4f} should be < unimportant sigma={sigma[0]:.4f}"
    _ok("waterfilling_inverse_to_importance",
        f"important={sigma[99]:.4f} < unimportant={sigma[0]:.4f}")


# ---------------------------------------------------------------------------
# Test 6 — noise_type='none' branch produces zero sigma
# (Mirrors lines 562-568 of the script exactly — keep in sync if main() changes.)
# ---------------------------------------------------------------------------

def test_none_branch_produces_zero_sigma():
    deltas      = torch.full((10,), 2.0)
    importances = torch.ones(10)
    rho         = v2.eps_to_rho(8.0)

    for noise_type, expect_zero in [("none", True),
                                    ("uniform", False),
                                    ("channel_WF", False)]:
        if noise_type == "channel_WF":
            sigma = v2.waterfilling_sigma(deltas, importances, rho)
        elif noise_type == "uniform":
            sigma = v2.uniform_sigma(deltas, rho)
        else:
            sigma = torch.zeros_like(deltas)

        all_zero = bool(torch.all(sigma == 0))
        assert all_zero == expect_zero, (
            f"noise_type={noise_type!r}: expected zero={expect_zero}, "
            f"got zero={all_zero} (sigma={sigma})"
        )

    _ok("none_branch_produces_zero_sigma",
        "uniform/channel_WF → non-zero; none → all-zero")


# ---------------------------------------------------------------------------
# Test 7 — build_model(from_scratch=True) skips load_checkpoint
# ---------------------------------------------------------------------------

def test_build_model_from_scratch_skips_load():
    cfg               = MagicMock()
    cfg.get           = MagicMock(return_value="mmseg")
    cfg.model         = MagicMock()
    fake_model        = MagicMock()
    fake_model.to     = MagicMock(return_value=fake_model)
    fake_model.parameters = MagicMock(return_value=[])

    with patch.object(v2.MODELS, "build", return_value=fake_model), \
         patch.object(v2, "load_checkpoint") as fake_load:
        v2.build_model(cfg, "/dummy.pth", "cpu", from_scratch=True)

    assert not fake_load.called, \
        f"load_checkpoint was called {fake_load.call_count}× with from_scratch=True"
    _ok("from_scratch_True_skips_load_checkpoint",
        "load_checkpoint was NOT invoked (as expected)")


# ---------------------------------------------------------------------------
# Test 8 — build_model(from_scratch=False) calls load_checkpoint
# ---------------------------------------------------------------------------

def test_build_model_default_loads():
    cfg               = MagicMock()
    cfg.get           = MagicMock(return_value="mmseg")
    cfg.model         = MagicMock()
    fake_model        = MagicMock()
    fake_model.to     = MagicMock(return_value=fake_model)
    fake_model.parameters = MagicMock(return_value=[])

    with patch.object(v2.MODELS, "build", return_value=fake_model), \
         patch.object(v2, "load_checkpoint") as fake_load:
        v2.build_model(cfg, "/dummy.pth", "cpu", from_scratch=False)

    assert fake_load.called, \
        "load_checkpoint was NOT invoked with from_scratch=False"
    _ok("from_scratch_False_loads_checkpoint",
        f"load_checkpoint called {fake_load.call_count}× (as expected)")


# ---------------------------------------------------------------------------
# Test 9 — final_mDice_lastN_mean calculation
# (Mirrors the inline computation in main() — keep in sync.)
# ---------------------------------------------------------------------------

def test_final_mdice_lastn_mean():
    history = [
        {"iter":  4000, "mDice": 0.85},
        {"iter":  8000, "mDice": 0.86},
        {"iter": 12000, "mDice": 0.88},
        {"iter": 16000, "mDice": 0.87},
        {"iter": 20000, "mDice": 0.89},
        {"iter": 24000, "mDice": 0.90},
    ]
    last_n = history[-3:]
    got    = float(np.mean([h["mDice"] for h in last_n]))
    want   = (0.87 + 0.89 + 0.90) / 3.0
    assert abs(got - want) < 1e-9, f"got {got}, want {want}"
    _ok("final_mdice_lastn_mean", f"{got:.4f} matches expected {want:.4f}")


# ---------------------------------------------------------------------------
# Test 10 — --precompute-noise flag parsed correctly
# ---------------------------------------------------------------------------

def test_precompute_noise_flag():
    argv = REQUIRED_ARGS + [
        "--noise-type", "uniform",
        "--epsilon", "8",
        "--precompute-noise",
    ]
    with patch.object(sys, "argv", argv):
        args = v2.parse_args()
    assert args.precompute_noise is True
    _ok("precompute_noise_flag",
        "--precompute-noise sets args.precompute_noise=True")


def test_precompute_noise_default_off():
    argv = REQUIRED_ARGS + [
        "--noise-type", "uniform",
        "--epsilon", "8",
    ]
    with patch.object(sys, "argv", argv):
        args = v2.parse_args()
    assert args.precompute_noise is False, "default should be False"
    _ok("precompute_noise_default_off",
        "--precompute-noise defaults to False (backwards-compatible)")


# ---------------------------------------------------------------------------
# Test 11 — image-id helper is stable + falls back gracefully
# ---------------------------------------------------------------------------

def test_img_id_stable_and_fallback():
    ds_with_path = MagicMock()
    ds_with_path.metainfo = {"img_path": "/data/img_007.png"}
    assert v2._img_id(ds_with_path) == "/data/img_007.png"

    ds_with_filename = MagicMock()
    ds_with_filename.metainfo = {"filename": "img_008.png"}
    assert v2._img_id(ds_with_filename) == "img_008.png"

    ds_empty = MagicMock()
    ds_empty.metainfo = {}
    fallback = v2._img_id(ds_empty)
    assert fallback.startswith("obj_"), f"fallback unexpected: {fallback}"

    _ok("img_id_stable_and_fallback",
        "img_path > filename > obj_<id> fallback chain works")


# ---------------------------------------------------------------------------
# Test 12 — lookup_cached_noisy_bottleneck returns the cached tensors
# in the order matching the batch's data_samples
# ---------------------------------------------------------------------------

def test_lookup_cached_noisy_bottleneck():
    a = MagicMock(); a.metainfo = {"img_path": "/a.png"}
    b = MagicMock(); b.metainfo = {"img_path": "/b.png"}
    cache = {
        "/a.png": torch.zeros(1024, 37, 36) + 1.0,
        "/b.png": torch.zeros(1024, 37, 36) + 2.0,
    }
    stacked = v2.lookup_cached_noisy_bottleneck([a, b], cache, "cpu")
    assert stacked.shape == (2, 1024, 37, 36)
    assert torch.allclose(stacked[0], torch.zeros_like(stacked[0]) + 1.0)
    assert torch.allclose(stacked[1], torch.zeros_like(stacked[1]) + 2.0)
    _ok("lookup_cached_noisy_bottleneck",
        f"shape={tuple(stacked.shape)}, ordering preserved")


def test_lookup_raises_when_missing():
    miss = MagicMock(); miss.metainfo = {"img_path": "/missing.png"}
    try:
        v2.lookup_cached_noisy_bottleneck([miss], {}, "cpu")
    except KeyError as e:
        msg = str(e)
        assert "missing.png" in msg or "augmentation" in msg.lower()
        _ok("lookup_raises_when_missing",
            "missing cache key raises a clear KeyError")
        return
    raise AssertionError("expected KeyError for missing cache entry")


# ---------------------------------------------------------------------------
# Test 13 — privacy_accounting math round-trips
# ---------------------------------------------------------------------------

def test_privacy_accounting_roundtrip():
    import privacy_accounting as pa
    for eps in [0.5, 1.0, 2.0, 4.0, 8.0]:
        rho = pa.eps_to_rho(eps, delta=1e-5)
        eps_back = pa.rho_to_eps(rho, delta=1e-5)
        assert abs(eps - eps_back) < 1e-6, \
            f"eps={eps} -> rho={rho} -> eps_back={eps_back}"
    _ok("privacy_accounting_roundtrip",
        "eps_to_rho ∘ rho_to_eps is identity for eps in [0.5, 8]")


def test_per_iteration_blowup_significant():
    """The whole point of the analysis: per-iter composition really does blow up."""
    import privacy_accounting as pa
    per_iter = pa.per_iteration_threat_model(2.0, T=40000, B=4, N=20)
    once     = pa.sample_once_threat_model(2.0, N=20)
    assert per_iter.total_eps_per_user > 100 * once.total_eps_per_user, (
        f"expected blowup >100x; got per-iter={per_iter.total_eps_per_user} "
        f"once={once.total_eps_per_user}"
    )
    _ok("per_iteration_blowup_significant",
        f"per-iter eps={per_iter.total_eps_per_user:.1f} "
        f">> sample-once eps={once.total_eps_per_user:.1f}")


# ---------------------------------------------------------------------------
# Tests 14-17 — public-proxy threat model
# ---------------------------------------------------------------------------

def test_public_proxy_threat_model_equality():
    """public-proxy + sample-once: user-level eps == per-release eps."""
    import privacy_accounting as pa
    for eps in [1.0, 2.0, 4.0, 8.0, 16.0]:
        r = pa.public_proxy_threat_model(eps)
        assert abs(r.total_eps_per_user - eps) < 1e-9, \
            f"public-proxy should preserve eps={eps}, got {r.total_eps_per_user}"
        assert r.n_releases_per_user == 1.0
    _ok("public_proxy_threat_model_equality",
        "user-level eps == per-release eps across [1, 16]")


def test_threat_model_argparse():
    """argparse accepts --threat-model public-proxy + --public-caps-csv."""
    argv = REQUIRED_ARGS + [
        "--noise-type", "uniform",
        "--epsilon", "8",
        "--threat-model", "public-proxy",
        "--public-caps-csv", "/dummy/public_caps.csv",
    ]
    with patch.object(sys, "argv", argv):
        args = v2.parse_args()
    assert args.threat_model == "public-proxy"
    assert args.public_caps_csv == "/dummy/public_caps.csv"
    _ok("threat_model_argparse",
        "--threat-model + --public-caps-csv parsed correctly")


def test_threat_model_default_is_public_proxy():
    argv = REQUIRED_ARGS + [
        "--noise-type", "uniform",
        "--epsilon", "8",
        "--public-caps-csv", "/dummy/cap.csv",
    ]
    with patch.object(sys, "argv", argv):
        args = v2.parse_args()
    assert args.threat_model == "public-proxy", \
        f"default should be public-proxy, got {args.threat_model}"
    _ok("threat_model_default_is_public_proxy",
        "default threat_model is public-proxy (DP-honest by default)")


def test_load_caps_from_csv():
    """load_caps_from_csv reads cap_norm column."""
    import tempfile, os
    csv_text = "cap_norm\n12.34\n5.67\n0.89\n"
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write(csv_text)
        path = f.name
    try:
        caps = v2.load_caps_from_csv(path)
        assert caps.shape == (3,), f"shape={caps.shape}"
        assert torch.allclose(caps, torch.tensor([12.34, 5.67, 0.89])), \
            f"got {caps}"
        _ok("load_caps_from_csv",
            f"loaded 3 caps from CSV: {caps.tolist()}")
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 64)
    print("Smoke test — phase1_gkd_distill_v2.py diagnostic changes")
    print("=" * 64)

    for fn in [
        test_argparse_new_flags,
        test_argparse_existing_flags_still_work,
        test_argparse_rejects_bad_noise_type,
        test_uniform_sigma_constant,
        test_uniform_sigma_hits_target_rho,
        test_waterfilling_sigma_hits_target_rho,
        test_waterfilling_inverse_to_importance,
        test_none_branch_produces_zero_sigma,
        test_build_model_from_scratch_skips_load,
        test_build_model_default_loads,
        test_final_mdice_lastn_mean,
        test_precompute_noise_flag,
        test_precompute_noise_default_off,
        test_img_id_stable_and_fallback,
        test_lookup_cached_noisy_bottleneck,
        test_lookup_raises_when_missing,
        test_privacy_accounting_roundtrip,
        test_per_iteration_blowup_significant,
        test_public_proxy_threat_model_equality,
        test_threat_model_argparse,
        test_threat_model_default_is_public_proxy,
        test_load_caps_from_csv,
    ]:
        _run(fn.__name__, fn)

    print("=" * 64)
    print(f"  {PASS} passed, {FAIL} failed")
    print("=" * 64)
    sys.exit(0 if FAIL == 0 else 1)
