"""
Importance scaling and reverse CANAL.

Strategy 1 (importance scaling): apply a power transform to the importance
distribution before water-filling, making it MORE skewed so WF has more
variance to exploit.

  imp_scaled = (imp / imp.mean()) ** alpha    (normalize → power → use in WF)

  alpha=1.0  — standard CANAL (no scaling)
  alpha=2.0  — moderate skew boost
  alpha=4.0  — strong skew boost
  alpha=8.0  — extreme skew boost

Strategy 8 (reverse CANAL / sanity check): flip the importance so HIGHER
importance gets MORE noise (inverse water-filling).

  imp_reversed = 1.0 / imp.clamp(min=1e-12)   (then renormalized)

If reverse hurts vs uniform  → confirms CANAL theory (important channels need less noise).
If reverse helps vs standard CANAL → suggests a different mechanism is operating.

Fixed config: top 10% channels, K=3, all 3 datasets.
Budget split: rho_imp=10%, rho_rel=90%.

Usage:
  python canal_importance_scaling.py --dataset isic --seeds 3 --te 60 --se 40
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from drive_pate_poc import train_K_teachers, correct_uniform_sigma
from drive_pate_pruning_joint import shared_importance, precompute_joint_cache
from drive_pate_canal_combined import add_dp_noise_to_importance, correct_waterfilling_sigma
from drive_student_distill import train_student_distill
from synthetic_demo import eps_to_rho

HERE = Path(__file__).parent
(HERE / "results").mkdir(exist_ok=True)

ALPHA_IMP = 0.1
CLIP_IMP = 1e-8
K = 3
KEEP_FRAC = 0.1


def get_dataset(name, split, size):
    if name == "isic":
        from isic_dataset import ISICDataset
        return ISICDataset(split, size), 3
    if name == "kvasir":
        from kvasir_dataset import KvasirDataset
        return KvasirDataset(split, size), 3
    if name == "busi":
        from busi_dataset import BUSIDataset
        return BUSIDataset(split, size), 1
    raise ValueError(name)


def scale_importance(imp, alpha):
    """Normalize then apply power transform to boost skew."""
    imp_norm = imp / imp.mean().clamp(min=1e-12)
    return imp_norm.pow(alpha)


def reverse_importance(imp):
    """Invert importance: high-importance channels get high 'reversed' score
    so water-filling assigns them MORE noise (sanity check experiment)."""
    inv = 1.0 / imp.clamp(min=1e-12)
    return inv / inv.mean()  # normalize so scale matches original


def build_wf_sigma(Cb, top_indices, deltas, imp_effective, rho_rel, device):
    """WF sigma over the top-n_keep active channel set."""
    active_mask = torch.zeros(Cb, dtype=torch.bool, device=device)
    active_mask[top_indices] = True
    sigma = torch.zeros(Cb, device=device)
    sigma[active_mask] = correct_waterfilling_sigma(
        deltas[active_mask], imp_effective[active_mask], rho_rel
    )
    return active_mask, sigma


def run_students(train_ds, val_loader, teachers, caps_list, active_mask, sigma,
                 device, seeds, n_epochs_student, in_ch, cache_seed):
    cache = precompute_joint_cache(
        teachers, caps_list, train_ds, sigma, active_mask, device, seed=cache_seed
    )
    dices = []
    for s in seeds:
        best, _ = train_student_distill(
            train_ds, val_loader, cache, device,
            student_base=16, teacher_base=32,
            n_epochs=n_epochs_student, lr=1e-3, lambda_feat=0.4,
            seed=s, in_ch=in_ch,
        )
        dices.append(best)
    return dices


def cell_stats(dices):
    m = float(np.mean(dices))
    s = float(np.std(dices))
    sem = s / math.sqrt(len(dices))
    return {"dices": dices, "mean": m, "std": s, "sem": sem}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="isic", choices=["isic", "kvasir", "busi"])
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--te", type=int, default=60)
    ap.add_argument("--se", type=int, default=40)
    ap.add_argument("--epsilons", default="0.5,1,2,4,6")
    ap.add_argument("--alphas", default="1.0,2.0,4.0,8.0",
                    help="power-transform exponents for importance scaling")
    args = ap.parse_args()

    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else "cpu")
    EPS = [float(e) for e in args.epsilons.split(",")]
    ALPHAS = [float(a) for a in args.alphas.split(",")]
    SEEDS = list(range(100, 100 + args.seeds * 100, 100))

    print(f"[{args.dataset}] device={dev}  K={K}  eps={EPS}")
    print(f"  alphas={ALPHAS}  seeds={SEEDS}")

    train_ds, in_ch = get_dataset(args.dataset, "train", args.size)
    val_ds, _ = get_dataset(args.dataset, "val", args.size)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    N = len(train_ds)
    print(f"  train={N}  val={len(val_ds)}  in_ch={in_ch}")

    teachers, caps_list = train_K_teachers(train_ds, K, dev, n_epochs=args.te, in_ch=in_ch)
    Cb = teachers[0].base * 4
    n_keep = max(1, int(round(KEEP_FRAC * Cb)))
    print(f"  bottleneck C={Cb}  n_keep={n_keep} ({KEEP_FRAC*100:.0f}%)")

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=False)
    importance = shared_importance(teachers, train_loader, dev).to(dev)

    # Print raw importance distribution for diagnostics
    imp_cpu = importance.cpu()
    print(f"  importance: min={imp_cpu.min():.4f}  max={imp_cpu.max():.4f}  "
          f"mean={imp_cpu.mean():.4f}  ratio={imp_cpu.max()/imp_cpu.min():.1f}x  "
          f"max/mean={imp_cpu.max()/imp_cpu.mean():.1f}x")

    sens_imp = 2.0 * CLIP_IMP / float(N)
    deltas = torch.full((Cb,), 2.0 / K, device=dev)

    results = {
        "dataset": args.dataset, "K": K, "keep_frac": KEEP_FRAC, "n_keep": n_keep,
        "epsilons": EPS, "alphas": ALPHAS, "seeds": SEEDS, "ALPHA_IMP": ALPHA_IMP,
        "C": Cb,
        "importance_stats": {
            "min": float(imp_cpu.min()), "max": float(imp_cpu.max()),
            "mean": float(imp_cpu.mean()), "std": float(imp_cpu.std()),
            "ratio_max_min": float(imp_cpu.max() / imp_cpu.min()),
            "ratio_max_mean": float(imp_cpu.max() / imp_cpu.mean()),
        },
        "sweep": {},  # [eps] → {top_uniform, canal_alpha{a}, reverse_canal}
    }

    for eps in EPS:
        rho = eps_to_rho(eps)
        rho_imp = ALPHA_IMP * rho
        rho_rel = (1.0 - ALPHA_IMP) * rho
        noise_seed = int(eps * 100) + 7
        imp_noisy = add_dp_noise_to_importance(importance, sens_imp, rho_imp, seed=noise_seed)
        rank_desc = torch.argsort(imp_noisy, descending=True)
        top_indices = rank_desc[:n_keep]

        print(f"\n{'='*64}\neps={eps}  rho_rel={rho_rel:.4f}")
        ep_res = {}

        # --- Baseline: top-n_keep + uniform ---
        top_mask = torch.zeros(Cb, dtype=torch.bool, device=dev)
        top_mask[top_indices] = True
        sigma_uni = torch.zeros(Cb, device=dev)
        sigma_uni[top_mask] = correct_uniform_sigma(deltas[top_mask], rho_rel)

        print("  [baseline] top_uniform ...", end="", flush=True)
        t0 = time.time()
        dices = run_students(train_ds, val_loader, teachers, caps_list,
                             top_mask, sigma_uni, dev, SEEDS, args.se, in_ch,
                             cache_seed=int(eps * 1000) + 1)
        ep_res["top_uniform"] = cell_stats(dices)
        print(f"  Dice={ep_res['top_uniform']['mean']:.4f}  ({time.time()-t0:.1f}s)")

        # --- CANAL with power-scaled importance ---
        for alpha in ALPHAS:
            imp_scaled = scale_importance(imp_noisy, alpha)
            # Diagnostics for this alpha
            imp_sc_active = imp_scaled[top_indices]
            ratio = float(imp_sc_active.max() / imp_sc_active.min().clamp(min=1e-12))
            print(f"  [alpha={alpha:.1f}] imp_ratio_active={ratio:.1f}x ...", end="", flush=True)

            cond_key = f"canal_alpha{alpha:.1f}"
            t0 = time.time()
            am, sig = build_wf_sigma(Cb, top_indices, deltas, imp_scaled, rho_rel, dev)
            dices = run_students(train_ds, val_loader, teachers, caps_list,
                                 am, sig, dev, SEEDS, args.se, in_ch,
                                 cache_seed=int(eps * 1000) + int(alpha * 10) + 50)
            ep_res[cond_key] = cell_stats(dices)
            ep_res[cond_key]["alpha"] = alpha
            ep_res[cond_key]["imp_ratio_active"] = ratio
            print(f"  Dice={ep_res[cond_key]['mean']:.4f}  diff_vs_uni={ep_res[cond_key]['mean']-ep_res['top_uniform']['mean']:+.4f}  ({time.time()-t0:.1f}s)")

        # --- Reverse CANAL (sanity check) ---
        imp_rev = reverse_importance(imp_noisy)
        imp_rev_active = imp_rev[top_indices]
        ratio_rev = float(imp_rev_active.max() / imp_rev_active.min().clamp(min=1e-12))
        print(f"  [reverse] imp_ratio_active={ratio_rev:.1f}x ...", end="", flush=True)
        t0 = time.time()
        am_rev, sig_rev = build_wf_sigma(Cb, top_indices, deltas, imp_rev, rho_rel, dev)
        dices = run_students(train_ds, val_loader, teachers, caps_list,
                             am_rev, sig_rev, dev, SEEDS, args.se, in_ch,
                             cache_seed=int(eps * 1000) + 99)
        ep_res["reverse_canal"] = cell_stats(dices)
        ep_res["reverse_canal"]["imp_ratio_active"] = ratio_rev
        print(f"  Dice={ep_res['reverse_canal']['mean']:.4f}  diff_vs_uni={ep_res['reverse_canal']['mean']-ep_res['top_uniform']['mean']:+.4f}  ({time.time()-t0:.1f}s)")

        results["sweep"][str(eps)] = ep_res

    # Summary
    for eps in EPS:
        print(f"\n{'='*72}")
        print(f"IMPORTANCE SCALING  ({args.dataset.upper()}, eps={eps})")
        r = results["sweep"][str(eps)]
        print(f"  top_uniform:          {r['top_uniform']['mean']:.4f}  (reference)")
        for alpha in ALPHAS:
            k = f"canal_alpha{alpha:.1f}"
            diff = r[k]["mean"] - r["top_uniform"]["mean"]
            print(f"  canal_alpha={alpha:.1f}:     {r[k]['mean']:.4f}  diff={diff:+.4f}  "
                  f"imp_ratio={r[k]['imp_ratio_active']:.1f}x")
        diff_rev = r["reverse_canal"]["mean"] - r["top_uniform"]["mean"]
        print(f"  reverse_canal:        {r['reverse_canal']['mean']:.4f}  diff={diff_rev:+.4f}  "
              f"imp_ratio={r['reverse_canal']['imp_ratio_active']:.1f}x")

    out = HERE / "results" / f"{args.dataset}_importance_scaling_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
