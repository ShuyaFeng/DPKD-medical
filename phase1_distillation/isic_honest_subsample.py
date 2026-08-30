# -*- coding: utf-8 -*-
"""
ISIC honest CANAL + channel subsampling.

Pay rho_imp for noisy importance (Gaussian mechanism with clipped per-sample
contribution), then KEEP the top keep_frac channels by NOISY importance and
drop the rest. Allocate uniform sigma over the kept channels using rho_rel.

Compared at each (K=10, eps):
  PATE+uniform_all       : all C channels, uniform sigma, rho_total for noise
  PATE+subsample_honest  : top-keep_frac by NOISY importance, uniform on kept;
                           rho split as rho_imp + rho_rel (alpha default 0.1)
  PATE+subsample_random  : random keep_frac, uniform on kept;
                           no importance used, so full rho_total for noise
                           (acts as a control: does importance ranking help vs
                           random pruning?)

Total zCDP cost = rho_total for all three (apples-to-apples).

Usage:
  python isic_honest_subsample.py
  python isic_honest_subsample.py --keep_fracs 0.5,0.25,0.1,0.05 --seeds 5
"""

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from isic_dataset import ISICDataset
from drive_local_demo import TinyUNet, evaluate_vessel_dice
from drive_pate_poc import train_K_teachers
from drive_pate_pruning_joint import (
    thresholded_uniform_sigma, precompute_joint_cache,
)
from drive_pate_canal_combined import (
    correct_waterfilling_sigma, correct_uniform_sigma,
)
from drive_student_distill import train_student_distill
from synthetic_demo import eps_to_rho


HERE = Path(__file__).parent
(HERE / "results").mkdir(exist_ok=True)


def cleanup(device):
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


@torch.no_grad()
def shared_actnorm_importance_clipped(teachers, loader, device, clip):
    """Per-teacher act_norm with per-sample L2-clipping, averaged across K.
    Returns (shared_imp [C], N_per_teacher, diagnostics)."""
    Cb = teachers[0].base * 4
    imps = []
    N_per_teacher = 0
    per_sample_l2_max = 0.0
    per_sample_l2_sum = 0.0
    n_seen = 0
    for t in teachers:
        t.eval()
        sum_clipped = torch.zeros(Cb, device=device)
        N = 0
        for x, _ in loader:
            x = x.to(device)
            _, _, e3 = t.encode(x)
            per_sample = e3.flatten(2).norm(dim=2)  # [B, C]
            l2 = per_sample.norm(dim=1, keepdim=True)
            per_sample_l2_max = max(per_sample_l2_max, float(l2.max()))
            per_sample_l2_sum += float(l2.sum())
            n_seen += l2.shape[0]
            scale = (clip / l2).clamp(max=1.0)
            clipped = per_sample * scale
            sum_clipped += clipped.sum(dim=0)
            N += clipped.shape[0]
        imps.append(sum_clipped / N)
        N_per_teacher = N
    shared = torch.stack(imps, dim=0).mean(dim=0).cpu()
    diag = {
        "per_sample_l2_max": per_sample_l2_max,
        "per_sample_l2_mean": per_sample_l2_sum / max(n_seen, 1),
    }
    return shared, N_per_teacher, diag


def add_dp_noise_to_importance(imp, sensitivity, rho_imp, device, seed):
    sigma_imp = sensitivity / math.sqrt(2.0 * rho_imp)
    g = torch.Generator(); g.manual_seed(int(seed))
    noise = torch.randn(imp.shape, generator=g) * sigma_imp
    noisy = (imp + noise).clamp(min=1e-6)
    return noisy.to(device), sigma_imp


def topk_mask(importance, keep_frac, device):
    """Return a boolean mask keeping the top keep_frac fraction of channels."""
    C = importance.numel()
    K_kept = max(1, int(round(keep_frac * C)))
    order = torch.argsort(importance, descending=True)
    mask = torch.zeros(C, dtype=torch.bool, device=device)
    mask[order[:K_kept]] = True
    return mask, K_kept


def random_mask(C, keep_frac, device, seed):
    K_kept = max(1, int(round(keep_frac * C)))
    g = torch.Generator(); g.manual_seed(int(seed))
    idx = torch.randperm(C, generator=g)[:K_kept].to(device)
    mask = torch.zeros(C, dtype=torch.bool, device=device)
    mask[idx] = True
    return mask, K_kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epsilons", type=str, default="0.5,1,2")
    ap.add_argument("--keep_fracs", type=str, default="0.5,0.25,0.1,0.05")
    ap.add_argument("--alpha_imp", type=float, default=0.10)
    ap.add_argument("--clip_imp", type=float, default=100.0)
    ap.add_argument("--te", type=int, default=50)
    ap.add_argument("--se", type=int, default=40)
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--teacher_base", type=int, default=32)
    ap.add_argument("--student_base", type=int, default=0)
    ap.add_argument("--out_tag", type=str, default="")
    ap.add_argument("--skip_random", action="store_true",
                    help="skip the random_subsample control to save compute")
    args = ap.parse_args()
    if args.student_base == 0:
        args.student_base = max(8, args.teacher_base // 2)

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    EPS = [float(e) for e in args.epsilons.split(",")]
    KFS = [float(k) for k in args.keep_fracs.split(",")]
    SEEDS = list(range(100, 100 + args.seeds * 100, 100))

    train_ds = ISICDataset(split="train", size=args.size)
    val_ds = ISICDataset(split="val", size=args.size)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    print(f"[ISIC honest subsample] K={args.K}  eps={EPS}  keep_fracs={KFS}  "
          f"seeds={SEEDS}  alpha={args.alpha_imp}  clip={args.clip_imp}")

    # ---- train K teachers ----
    print(f"\n[teachers] training K={args.K}")
    t0 = time.time()
    teachers, caps_list = train_K_teachers(
        train_ds, K=args.K, device=device, n_epochs=args.te, in_ch=3,
        base=args.teacher_base,
    )
    print(f"  K={args.K} done in {time.time()-t0:.0f}s")

    # ---- clipped importance + sensitivity ----
    imp_loader = DataLoader(train_ds, batch_size=8, shuffle=False)
    imp_clipped, N_per_teacher, diag = shared_actnorm_importance_clipped(
        teachers, imp_loader, device, clip=args.clip_imp,
    )
    R_clipped = (imp_clipped.max() / imp_clipped.min()).item()
    sensitivity = 2.0 * args.clip_imp / N_per_teacher
    print(f"  R_clipped={R_clipped:.2f}  sensitivity={sensitivity:.6f}  "
          f"per_sample_L2 mean={diag['per_sample_l2_mean']:.1f} max={diag['per_sample_l2_max']:.1f}")
    cleanup(device)

    Cb = teachers[0].base * 4
    deltas = torch.full((Cb,), 2.0 / args.K, device=device)

    tag_suffix = f"_{args.out_tag}" if args.out_tag else ""
    save_path = HERE / "results" / f"isic_honest_subsample{tag_suffix}_results.json"
    results = {
        "dataset": "isic", "K": args.K, "epsilons": EPS, "keep_fracs": KFS,
        "seeds": SEEDS, "alpha_imp": args.alpha_imp, "clip_imp": args.clip_imp,
        "R_clipped": R_clipped, "sensitivity_imp": sensitivity,
        "N_per_teacher": N_per_teacher, "C": Cb,
        "teacher_base": args.teacher_base, "student_base": args.student_base,
        "series": {},
    }

    def run_cell(sigma_vec, mask, label, eps, seed_offset=0):
        cache = precompute_joint_cache(
            teachers, caps_list, train_ds, sigma_vec, mask, device,
            seed=42 + int(eps * 10) + seed_offset,
        )
        dices = []
        for s in SEEDS:
            t0 = time.time()
            best, _ = train_student_distill(
                train_ds, val_loader, cache, device,
                student_base=args.student_base, teacher_base=args.teacher_base,
                n_epochs=args.se, lr=1e-3, lambda_feat=0.4,
                seed=s, in_ch=3,
            )
            dices.append(best)
            print(f"      {label} eps={eps} seed={s}: {best:.4f}  ({time.time()-t0:.0f}s)")
            cleanup(device)
        del cache
        cleanup(device)
        return {
            "dices": dices, "mean": float(np.mean(dices)),
            "std": float(np.std(dices)),
            "sem": float(np.std(dices) / np.sqrt(len(dices))),
        }

    # Series keys
    results["series"]["PATE+uniform_all"] = {}
    for kf in KFS:
        results["series"][f"PATE+subsample_honest_keep{int(kf*100)}%"] = {}
        if not args.skip_random:
            results["series"][f"PATE+subsample_random_keep{int(kf*100)}%"] = {}

    # ---- main sweep ----
    cells_per_eps = 1 + (2 if not args.skip_random else 1) * len(KFS)
    total = len(EPS) * cells_per_eps
    done = 0
    job_t0 = time.time()
    mask_all = torch.ones(Cb, dtype=torch.bool, device=device)

    for eps in EPS:
        rho_total = eps_to_rho(eps)
        rho_imp = args.alpha_imp * rho_total
        rho_rel = (1.0 - args.alpha_imp) * rho_total
        sigma_imp_val = sensitivity / math.sqrt(2.0 * rho_imp)
        print(f"\n========== eps={eps}  rho={rho_total:.5f}  "
              f"(rho_imp={rho_imp:.5f}, rho_rel={rho_rel:.5f}, sigma_imp={sigma_imp_val:.2f}) ==========")

        # 1) PATE+uniform_all baseline (full rho on noise, no importance)
        sigma_unif = correct_uniform_sigma(deltas, rho_total)
        print(f"   [uniform_all]      sigma={sigma_unif[0].item():.4f}  (K_kept={Cb})")
        results["series"]["PATE+uniform_all"][str(eps)] = run_cell(
            sigma_unif, mask_all, "uniform_all", eps,
        )
        done += 1
        elapsed = time.time() - job_t0
        eta = elapsed / done * (total - done)
        print(f"   [progress] {done}/{total}  elapsed={elapsed/60:.1f}min  ETA={eta/60:.1f}min")
        save_path.write_text(json.dumps(results, indent=2))

        # Noise importance ONCE per eps (same noisy_imp shared across keep_fracs)
        imp_noisy, _ = add_dp_noise_to_importance(
            imp_clipped, sensitivity, rho_imp, device,
            seed=12345 + int(eps * 10),
        )

        for kf in KFS:
            # 2) subsample_honest: top-keep% by noisy importance + uniform on kept
            mask_h, K_kept = topk_mask(imp_noisy, kf, device)
            sigma_h = thresholded_uniform_sigma(deltas, rho_rel, mask_h)
            sigma_kept_val = float(sigma_h[mask_h].mean())
            print(f"   [honest keep{int(kf*100)}%]   K_kept={K_kept}/{Cb}  "
                  f"sigma_kept={sigma_kept_val:.4f}")
            results["series"][f"PATE+subsample_honest_keep{int(kf*100)}%"][str(eps)] = run_cell(
                sigma_h, mask_h, f"honest_keep{int(kf*100)}%", eps, seed_offset=1,
            )
            done += 1
            elapsed = time.time() - job_t0
            eta = elapsed / done * (total - done)
            print(f"   [progress] {done}/{total}  elapsed={elapsed/60:.1f}min  ETA={eta/60:.1f}min")
            save_path.write_text(json.dumps(results, indent=2))

            # 3) subsample_random control (skip if --skip_random)
            if not args.skip_random:
                mask_r, _ = random_mask(Cb, kf, device, seed=12345 + int(eps * 10))
                sigma_r = thresholded_uniform_sigma(deltas, rho_total, mask_r)
                sigma_kept_r = float(sigma_r[mask_r].mean())
                print(f"   [random keep{int(kf*100)}%]   K_kept={K_kept}/{Cb}  "
                      f"sigma_kept={sigma_kept_r:.4f}")
                results["series"][f"PATE+subsample_random_keep{int(kf*100)}%"][str(eps)] = run_cell(
                    sigma_r, mask_r, f"random_keep{int(kf*100)}%", eps, seed_offset=2,
                )
                done += 1
                elapsed = time.time() - job_t0
                eta = elapsed / done * (total - done)
                print(f"   [progress] {done}/{total}  elapsed={elapsed/60:.1f}min  ETA={eta/60:.1f}min")
                save_path.write_text(json.dumps(results, indent=2))

    # ---- summary ----
    print("\n" + "=" * 92)
    print(f"Honest-subsample summary  K={args.K}  alpha={args.alpha_imp}  clip={args.clip_imp}")
    print("=" * 92)
    hdr = f"{'series':<40s} " + "  ".join(f"{e:>7.2f}" for e in EPS)
    print(hdr); print("-" * len(hdr))
    for key, data in results["series"].items():
        row = "  ".join(f"{data[str(e)]['mean']:>7.4f}" if str(e) in data
                        else f"{'--':>7s}" for e in EPS)
        print(f"{key:<40s} {row}")
    print("=" * 92)

    save_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved JSON: {save_path}")

    # ---- plot ----
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 6.5))
        keys = list(results["series"].keys())
        colors = plt.cm.tab10(np.linspace(0, 1, len(keys)))
        for k, c in zip(keys, colors):
            data = results["series"][k]
            xs = sorted([float(e) for e in data.keys()])
            ys = [data[str(e)]["mean"] for e in xs]
            es = [data[str(e)]["std"]  for e in xs]
            ls = "-" if "honest" in k or k == "PATE+uniform_all" else "--"
            ax.errorbar(xs, ys, yerr=es, fmt=f"o{ls}", color=c, lw=2, ms=8,
                        capsize=4, label=k)
        ax.set_xscale("log")
        ax.set_xlabel("privacy budget  eps", fontsize=12)
        ax.set_ylabel("Dice (ISIC val)", fontsize=12)
        ax.set_title(f"Honest subsample (K={args.K}, alpha={args.alpha_imp}, "
                     f"clip={args.clip_imp}, {args.seeds} seeds)",
                     fontsize=12, fontweight="bold")
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9, ncol=2)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        plot_path = HERE / f"fig_isic_honest_subsample{tag_suffix}.png"
        fig.savefig(plot_path, dpi=150)
        print(f"Saved plot: {plot_path}")
    except Exception as e:
        print(f"(plot skipped: {e})")

    cleanup(device)


if __name__ == "__main__":
    main()
