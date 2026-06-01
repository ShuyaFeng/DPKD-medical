#!/usr/bin/env python3
"""
Budget split analysis for DP feature distillation (paper §3.4).

The total privacy budget rho_total is split across THREE data-dependent
quantities used by the pipeline:

    rho_total  =  rho_caps  +  rho_imp  +  rho_rel
        ^             ^           ^          ^
        |             |           |          \\
        |             |           |           per-release noise on the
        |             |           |           teacher bottleneck
        |             |           |
        |             |           noisy importance score estimation
        |             |
        |             noisy per-channel L2 cap estimation
        |
        full budget

This module:

  1. Provides a closed-form analytical model of the budget trade-off:
        - more rho_caps  -> tighter clipping caps  -> less wasted budget
                            on per-release noise (smaller effective sensitivity)
        - more rho_imp   -> cleaner importance scores -> WF works better
        - more rho_rel   -> less per-release noise (cleaner teacher targets)

  2. Loads the teacher's gradient_abs_mean importance CSV (when available)
     and SIMULATES the effect of adding Gaussian noise to the importance
     scores under various rho_imp values. Computes the resulting
     channel-WF allocation and reports:
        - top-K channel ranking overlap with clean importance
        - implied per-release sigma distribution
        - WF objective value vs. uniform baseline

  3. Sweeps several budget splits and emits a CSV usable directly as a
     figure / table in the paper.

Run (no real data needed; falls back to a synthetic Zipf importance vector):

    python3 budget_split_analysis.py
    python3 budget_split_analysis.py --importance-csv /path/to/channel_*.csv
    python3 budget_split_analysis.py --total-eps 4 --output out.csv

Outputs:
    - prints a markdown-style table to stdout
    - writes <output>.csv with all sweep rows
"""

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

# privacy_accounting.py is alongside this file
sys.path.insert(0, str(Path(__file__).resolve().parent))
import privacy_accounting as pa


# ---------------------------------------------------------------------------
# Noisy importance simulation
# ---------------------------------------------------------------------------

def load_importance_csv(csv_path: Path) -> List[float]:
    """Load gradient_abs_mean column from the bottleneck-channel CSV."""
    vals = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            vals.append(float(row["gradient_abs_mean"]))
    return vals


def synthetic_importance(C: int = 1024, alpha: float = 1.0) -> List[float]:
    """Fallback Zipf-like importance vector when no CSV is available."""
    # Sorted descending; rough match to observed bottleneck-importance shape.
    return [1.0 / ((i + 1) ** alpha) for i in range(C)]


def add_gaussian_noise(
    values: Sequence[float],
    sigma: float,
    rng_seed: int = 0,
) -> List[float]:
    """Deterministic Gaussian noise via Python random (avoids torch dep)."""
    import random
    rnd = random.Random(rng_seed)
    return [v + rnd.gauss(0.0, sigma) for v in values]


def importance_sensitivity(values: Sequence[float], topk_clip: int = 100) -> float:
    """
    Replace-one L2 sensitivity of the importance vector after clipping each
    entry to the top-k-th largest value. Used only to convert a desired
    rho_imp into an importance-noise sigma.

    NOTE: This is a rough conservative bound (treats each coordinate as
    independently clippable). In a tight DP analysis one would clip the
    whole vector's L2 norm; we use this for the analytical sweep only.
    """
    sorted_v = sorted(values, reverse=True)
    cap = sorted_v[min(topk_clip, len(values) - 1)]
    return 2.0 * cap  # replace-one |v_c - v_c'| <= 2 * cap


def waterfilling_objective(
    importances: Sequence[float],
    sensitivities: Sequence[float],
    rho_rel: float,
) -> float:
    """
    Minimised value of  sum_c s_c * sigma_c^2  under the per-release budget
    sum_c delta_c^2 / (2 sigma_c^2) <= rho_rel.

    Plugging in the WF optimum sigma_c = kappa * sqrt(delta_c) / s_c^{1/4}
    and the kappa from Theorem 1 gives:

        WF_obj = (1 / (2 * rho_rel)) * (sum_c delta_c * sqrt(s_c))^2
    """
    if rho_rel <= 0.0:
        return float("inf")
    s = [max(v, 1e-12) for v in importances]
    inner = sum(d * math.sqrt(si) for d, si in zip(sensitivities, s))
    return (inner ** 2) / (2.0 * rho_rel)


def uniform_objective(
    importances: Sequence[float],
    sensitivities: Sequence[float],
    rho_rel: float,
) -> float:
    """Same objective under uniform sigma (all channels same sigma)."""
    if rho_rel <= 0.0:
        return float("inf")
    sigma_sq = sum(d * d for d in sensitivities) / (2.0 * rho_rel)
    return sigma_sq * sum(importances)


def topk_overlap(
    a: Sequence[float],
    b: Sequence[float],
    k: int = 100,
) -> float:
    """Jaccard overlap of top-k argmax indices between two score vectors."""
    idx_a = set(sorted(range(len(a)), key=lambda i: a[i], reverse=True)[:k])
    idx_b = set(sorted(range(len(b)), key=lambda i: b[i], reverse=True)[:k])
    return len(idx_a & idx_b) / k


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def sweep_budget_split(
    importances: List[float],
    total_eps: float,
    delta_DP: float,
    splits: Sequence[Tuple[float, float, float]],   # (f_caps, f_imp, f_rel)
    K: int,
    seed: int,
) -> List[dict]:
    """
    For each split fraction (f_caps, f_imp, f_rel) summing to 1.0:
      - allocate rho's
      - simulate noisy importance with sigma_imp implied by rho_imp
      - compute WF objective and uniform objective under rho_rel using the
        NOISY importance
      - record top-k overlap with clean importance
    """
    rho_tot = pa.eps_to_rho(total_eps, delta_DP)
    C = len(importances)

    # Per-channel sensitivity of the released bottleneck after L2 normalisation
    # (paper §3.2 — replace-one, K = number of simulated teachers)
    deltas = [2.0 / K] * C

    # Importance sensitivity (rough bound — see importance_sensitivity docstring)
    imp_sensitivity = importance_sensitivity(importances)

    rows = []
    for (f_caps, f_imp, f_rel) in splits:
        assert abs((f_caps + f_imp + f_rel) - 1.0) < 1e-9, "fractions must sum to 1"

        rho_caps = f_caps * rho_tot
        rho_imp  = f_imp  * rho_tot
        rho_rel  = f_rel  * rho_tot

        # Sigma for noisy importance release (Gaussian mechanism on imp vector)
        if rho_imp > 0:
            sigma_imp = pa.gaussian_sigma_for_rho(imp_sensitivity, rho_imp)
            noisy_imp = add_gaussian_noise(importances, sigma_imp, rng_seed=seed)
            # Clip negatives — importance scores are non-negative
            noisy_imp = [max(v, 1e-12) for v in noisy_imp]
        else:
            # f_imp = 0 -> "dishonest baseline": current code uses CLEAN
            # importance for free. We model this row as such; it does NOT
            # satisfy DP, but it represents what prior work effectively does.
            sigma_imp = float("inf")
            noisy_imp = [max(v, 1e-12) for v in importances]

        # WF + uniform under NOISY importance and the rho_rel budget
        wf_obj   = waterfilling_objective(noisy_imp, deltas, rho_rel)
        uni_obj  = uniform_objective(importances, deltas, rho_rel)
        # uniform doesn't use importance — use clean to be fair

        # Also: oracle WF using CLEAN importance for comparison
        wf_obj_clean = waterfilling_objective(importances, deltas, rho_rel)

        rows.append({
            "f_caps":           round(f_caps, 4),
            "f_imp":            round(f_imp,  4),
            "f_rel":            round(f_rel,  4),
            "rho_caps":         rho_caps,
            "rho_imp":          rho_imp,
            "rho_rel":          rho_rel,
            "sigma_imp":        sigma_imp if math.isfinite(sigma_imp) else None,
            "topk_overlap_100": round(topk_overlap(importances, noisy_imp, k=100), 4),
            "wf_obj_noisy":     wf_obj,
            "wf_obj_clean":     wf_obj_clean,
            "uniform_obj":      uni_obj,
            "wf_advantage_noisy": round((uni_obj - wf_obj) / max(uni_obj, 1e-12), 4),
            "wf_advantage_clean": round((uni_obj - wf_obj_clean) / max(uni_obj, 1e-12), 4),
        })

    return rows


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------

def print_sweep_table(rows: List[dict], total_eps: float) -> None:
    print(f"\n=== Budget split sweep — total user-level eps = {total_eps} ===\n")
    print(f"  {'f_caps':>7s} {'f_imp':>7s} {'f_rel':>7s}  "
          f"{'top100 overlap':>15s}  "
          f"{'WF adv (noisy)':>15s}  {'WF adv (clean)':>15s}  "
          f"{'note':<20s}")
    print("  " + "-" * 96)
    for r in rows:
        adv_n = 100.0 * r["wf_advantage_noisy"]
        adv_c = 100.0 * r["wf_advantage_clean"]
        note  = "DISHONEST (no DP)" if r["f_imp"] == 0.0 else ""
        print(f"  {r['f_caps']:>7.2f} {r['f_imp']:>7.2f} {r['f_rel']:>7.2f}  "
              f"{r['topk_overlap_100']:>15.3f}  "
              f"{adv_n:>14.2f}%  {adv_c:>14.2f}%  {note:<20s}")
    print()
    print("  Reading the table:")
    print("    - top100 overlap     = fraction of top-100 importance ranks")
    print("                            preserved after noise (1.0 = perfect)")
    print("    - WF adv (noisy)     = (uniform_obj - WF_obj) / uniform_obj")
    print("                            with NOISY importance — what an honest")
    print("                            DP pipeline actually achieves.")
    print("    - WF adv (clean)     = same ratio but with the ORACLE (clean)")
    print("                            importance — upper bound on WF gain.")
    print("    - f_imp = 0 row      = current code's implicit behaviour:")
    print("                            uses clean importance for free, but")
    print("                            this is NOT a valid DP guarantee.")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_SPLITS = [
    # (caps, imp, rel)  — must sum to 1.0
    (0.00, 0.00, 1.00),   # no privacy budget for caps/imp (current code, dishonest)
    (0.01, 0.01, 0.98),   # tiny slice each (the minimal honest version)
    (0.05, 0.05, 0.90),
    (0.05, 0.10, 0.85),
    (0.05, 0.20, 0.75),
    (0.10, 0.30, 0.60),
    (0.10, 0.45, 0.45),   # extreme: half budget on importance
]


def main() -> None:
    p = argparse.ArgumentParser(description="Budget split analysis (paper §3.4).")
    p.add_argument("--importance-csv", type=Path, default=None,
                   help="Optional path to channel_importance_*_gradient.csv.")
    p.add_argument("--total-eps", type=float, default=2.0)
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--K", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=Path("budget_split_sweep.csv"))
    args = p.parse_args()

    if args.importance_csv and args.importance_csv.exists():
        importances = load_importance_csv(args.importance_csv)
        src = f"loaded from {args.importance_csv.name}"
    else:
        importances = synthetic_importance(C=1024, alpha=1.0)
        src = "synthetic Zipf(1024, alpha=1.0)"

    print(f"Importance source : {src}")
    print(f"C                 : {len(importances)}")
    print(f"min/max/mean      : {min(importances):.3e}/{max(importances):.3e}/"
          f"{sum(importances)/len(importances):.3e}")

    rows = sweep_budget_split(
        importances=importances,
        total_eps=args.total_eps,
        delta_DP=args.delta,
        splits=DEFAULT_SPLITS,
        K=args.K,
        seed=args.seed,
    )

    print_sweep_table(rows, args.total_eps)

    # Write CSV
    fieldnames = list(rows[0].keys())
    with args.output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  Wrote sweep CSV -> {args.output}\n")


if __name__ == "__main__":
    main()
