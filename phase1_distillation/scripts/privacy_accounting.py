#!/usr/bin/env python3
"""
Privacy accounting for DP feature distillation.

Provides:
  1. Core conversions: (eps, delta)-DP <-> rho-zCDP (Bun & Steinke 2016).
  2. Gaussian-mechanism sigma <-> rho conversion.
  3. Composition under two threat models:
        per-iteration release  — current code's implicit model
        sample-once-per-image  — the threat model we adopt for WACV
  4. analyze_experiments(): reads existing summary.json files and prints
     a comparison table of *reported* eps vs *true* user-level eps under
     each threat model. This table is the empirical motivation for §3.

Run:
    python3 privacy_accounting.py                       # default analysis
    python3 privacy_accounting.py --results-dir <path>  # custom results dir

Definitions used throughout
---------------------------
  rho             zCDP parameter (Bun & Steinke 2016)
  delta_DP        the delta in (eps, delta)-DP            (default 1e-5)
  sensitivity     L2 sensitivity of the released vector  (= 2/K after
                  per-channel L2 normalisation in our pipeline)
  T               number of training iterations           (default 40000)
  B               batch size                              (default 4)
  N               number of unique training images        (default 20 for DRIVE)
  Q_i             expected #releases involving image i    = T*B/N
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def eps_to_rho(eps: float, delta: float = 1e-5) -> float:
    """
    (eps, delta)-DP --> rho-zCDP.  Bun & Steinke 2016 Prop 1.13:
        eps = rho + 2*sqrt(rho * log(1/delta))
    Solve as a quadratic in sqrt(rho).
    """
    log_inv_d = math.log(1.0 / delta)
    b = 2.0 * math.sqrt(log_inv_d)
    sqrt_rho = (-b + math.sqrt(b * b + 4.0 * eps)) / 2.0
    return sqrt_rho * sqrt_rho


def rho_to_eps(rho: float, delta: float = 1e-5) -> float:
    """rho-zCDP --> (eps, delta)-DP via Bun & Steinke 2016 Prop 1.3."""
    if rho <= 0.0:
        return 0.0
    return rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))


def gaussian_sigma_for_rho(sensitivity: float, rho: float) -> float:
    """Gaussian mechanism: rho = sensitivity^2 / (2 * sigma^2)  =>  sigma."""
    if rho <= 0.0:
        return float("inf")
    return sensitivity / math.sqrt(2.0 * rho)


def rho_for_gaussian_sigma(sensitivity: float, sigma: float) -> float:
    """Inverse of gaussian_sigma_for_rho."""
    if sigma <= 0.0:
        return float("inf")
    return (sensitivity ** 2) / (2.0 * sigma ** 2)


# ---------------------------------------------------------------------------
# Channel-aware sigma (multi-coordinate Gaussian mechanism)
# ---------------------------------------------------------------------------

def total_rho_per_release_from_sigmas(
    sensitivities: Iterable[float],
    sigmas: Iterable[float],
) -> float:
    """
    For a vector-valued Gaussian mechanism with per-channel sensitivities
    delta_c and per-channel noise stddev sigma_c, the per-release zCDP cost
    is the sum over channels:
            rho_release = sum_c   delta_c^2 / (2 * sigma_c^2)
    """
    return sum(
        (d ** 2) / (2.0 * s ** 2)
        for d, s in zip(sensitivities, sigmas)
        if s > 0.0
    )


# ---------------------------------------------------------------------------
# Composition under two threat models
# ---------------------------------------------------------------------------

@dataclass
class CompositionResult:
    """One row in the analysis table."""
    threat_model: str
    per_release_eps: float
    per_release_rho: float
    n_releases_per_user: float
    total_rho_per_user: float
    total_eps_per_user: float

    def __str__(self) -> str:
        return (
            f"{self.threat_model:24s} | "
            f"per-release eps={self.per_release_eps:6.3f} | "
            f"#releases/user={self.n_releases_per_user:>10.1f} | "
            f"TOTAL eps/user={self.total_eps_per_user:>10.2f}"
        )


def compose_naive_zcdp(rho_per_release: float, n_releases: int) -> float:
    """Basic zCDP composition: T releases of rho-zCDP -> (T*rho)-zCDP."""
    return n_releases * rho_per_release


def per_iteration_threat_model(
    per_release_eps: float,
    T: int,
    B: int,
    N: int,
    delta_DP: float = 1e-5,
) -> CompositionResult:
    """
    Current code's implicit threat model.

    Every training iteration draws fresh noise on a fresh batch and
    releases the noisy teacher bottleneck. Worst-case (most-visited)
    user appears in Q_i = T * B / N batches; each batch release is
    a fresh Gaussian mechanism invocation involving that user.

    Conservative naive composition: total rho for that user
        rho_user = Q_i * rho_per_release
    """
    rho_per_release = eps_to_rho(per_release_eps, delta_DP)
    q_per_user = T * B / N
    rho_per_user = compose_naive_zcdp(rho_per_release, int(q_per_user))
    eps_per_user = rho_to_eps(rho_per_user, delta_DP)
    return CompositionResult(
        threat_model="per-iteration release",
        per_release_eps=per_release_eps,
        per_release_rho=rho_per_release,
        n_releases_per_user=q_per_user,
        total_rho_per_user=rho_per_user,
        total_eps_per_user=eps_per_user,
    )


def sample_once_threat_model(
    per_release_eps: float,
    N: int,
    delta_DP: float = 1e-5,
) -> CompositionResult:
    """
    Sample-once-per-image release model.

    Each training image's noisy teacher bottleneck is computed ONCE,
    stored, and reused across all training iterations. Each user
    contributes exactly 1 release.

    By PARALLEL composition of disjoint per-image mechanisms,
    user-level rho is bounded by the single-release rho (NOT N * rho).
        rho_user = rho_per_release
    """
    rho_per_release = eps_to_rho(per_release_eps, delta_DP)
    return CompositionResult(
        threat_model="sample-once (release only)",
        per_release_eps=per_release_eps,
        per_release_rho=rho_per_release,
        n_releases_per_user=1.0,
        total_rho_per_user=rho_per_release,
        total_eps_per_user=per_release_eps,
    )


def public_proxy_threat_model(
    per_release_eps: float,
    delta_DP: float = 1e-5,
) -> CompositionResult:
    """
    Public-proxy + sample-once threat model — the one we adopt in the paper.

    Caps and importance are estimated on a PUBLIC retinal dataset
    (HRF / STARE / CHASE-DB1) disjoint from the private training set,
    so they incur ZERO privacy cost on training data. The per-image
    feature release is parallel-composed over disjoint per-image data,
    yielding clean per-user privacy equal to the reported epsilon.

        rho_user = rho_caps + rho_imp + rho_rel
                 =    0     +    0    + rho_per_release
                 = rho_per_release
    """
    rho_per_release = eps_to_rho(per_release_eps, delta_DP)
    return CompositionResult(
        threat_model="public-proxy + sample-once",
        per_release_eps=per_release_eps,
        per_release_rho=rho_per_release,
        n_releases_per_user=1.0,
        total_rho_per_user=rho_per_release,
        total_eps_per_user=per_release_eps,
    )


def calibrate_per_release_eps_for_target_user_eps(
    target_user_eps: float,
    threat_model: str,
    T: int,
    B: int,
    N: int,
    delta_DP: float = 1e-5,
) -> float:
    """
    Inverse direction: given a target *user-level* eps, return the
    per-release eps that should be used in code.

    Useful for designing experiments that report a meaningful user-level
    privacy guarantee.
    """
    target_user_rho = eps_to_rho(target_user_eps, delta_DP)
    if threat_model == "per-iteration":
        n_releases = T * B / N
    elif threat_model == "sample-once":
        n_releases = 1
    else:
        raise ValueError(f"Unknown threat model: {threat_model}")
    per_release_rho = target_user_rho / n_releases
    return rho_to_eps(per_release_rho, delta_DP)


# ---------------------------------------------------------------------------
# Read existing experiment summaries and analyse
# ---------------------------------------------------------------------------

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DEFAULT_T   = 40000
DEFAULT_B   = 4
DEFAULT_N   = 20
DEFAULT_DELTA = 1e-5


def collect_unique_per_release_eps(results_dir: Path) -> List[float]:
    """Read all summary.json files and return sorted unique epsilons."""
    seen = set()
    for p in sorted(results_dir.glob("GKD_V*_summary.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        eps = d.get("epsilon")
        if isinstance(eps, (int, float)):
            seen.add(float(eps))
    return sorted(seen)


def analyse(
    epsilons: Optional[List[float]] = None,
    T: int = DEFAULT_T,
    B: int = DEFAULT_B,
    N: int = DEFAULT_N,
    delta_DP: float = DEFAULT_DELTA,
) -> None:
    if epsilons is None:
        epsilons = collect_unique_per_release_eps(DEFAULT_RESULTS_DIR)
        if not epsilons:
            epsilons = [2.0, 4.0, 8.0, 16.0]

    print("=" * 88)
    print(" Privacy accounting — reported eps vs TRUE user-level eps")
    print("=" * 88)
    print(f"  Composition:      naive (no subsampling amplification)")
    print(f"  Conversion:       Bun & Steinke 2016  rho-zCDP <-> (eps, delta)-DP")
    print(f"  Training iters T  = {T}")
    print(f"  Batch size B      = {B}")
    print(f"  Train images N    = {N}")
    print(f"  Per-user releases Q = T*B/N = {T*B/N:.0f}  (per-iter model)")
    print(f"  delta             = {delta_DP}")
    print("=" * 88)
    print()
    print(f"  {'reported eps':<13s}  "
          f"{'threat model':<24s}  "
          f"{'#releases/user':>16s}  "
          f"{'TRUE eps/user':>16s}")
    print("  " + "-" * 84)

    for eps in epsilons:
        per_iter = per_iteration_threat_model(eps, T, B, N, delta_DP)
        once     = sample_once_threat_model(eps, N, delta_DP)
        public   = public_proxy_threat_model(eps, delta_DP)
        print(f"  {eps:<13.2f}  {per_iter.threat_model:<27s}  "
              f"{per_iter.n_releases_per_user:>13.0f}  "
              f"{per_iter.total_eps_per_user:>16.2f}")
        print(f"  {'':13s}  {once.threat_model:<27s}  "
              f"{once.n_releases_per_user:>13.0f}  "
              f"{once.total_eps_per_user:>16.2f}")
        print(f"  {'':13s}  {public.threat_model:<27s}  "
              f"{public.n_releases_per_user:>13.0f}  "
              f"{public.total_eps_per_user:>16.2f}  <-- adopted in paper")
        ratio = per_iter.total_eps_per_user / max(public.total_eps_per_user, 1e-9)
        print(f"  {'':13s}  {'   --> per-iter blowup':<27s}  "
              f"{'':>13s}  {ratio:>15.1f}x  vs public-proxy")
        print()

    print("=" * 88)
    print(" Inverse view: per-release eps required to deliver a target user-level eps")
    print("=" * 88)
    print(f"  {'target user eps':<16s}  "
          f"{'threat model':<24s}  "
          f"{'required per-release eps':>26s}")
    print("  " + "-" * 84)
    for target in [1.0, 2.0, 4.0, 8.0]:
        per_iter_eps = calibrate_per_release_eps_for_target_user_eps(
            target, "per-iteration", T, B, N, delta_DP)
        once_eps = calibrate_per_release_eps_for_target_user_eps(
            target, "sample-once", T, B, N, delta_DP)
        print(f"  {target:<16.2f}  {'per-iteration release':<24s}  "
              f"{per_iter_eps:>26.6f}")
        print(f"  {'':16s}  {'sample-once per image':<24s}  "
              f"{once_eps:>26.4f}")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Privacy accounting analyser.")
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR,
                   help="Directory containing GKD_V*_summary.json files.")
    p.add_argument("--T", type=int, default=DEFAULT_T)
    p.add_argument("--B", type=int, default=DEFAULT_B)
    p.add_argument("--N", type=int, default=DEFAULT_N)
    p.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    args = p.parse_args()

    eps_list = collect_unique_per_release_eps(args.results_dir) or [2.0, 4.0, 8.0, 16.0]
    analyse(eps_list, T=args.T, B=args.B, N=args.N, delta_DP=args.delta)


if __name__ == "__main__":
    main()
