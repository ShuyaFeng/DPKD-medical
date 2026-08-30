# -*- coding: utf-8 -*-
"""
ISIC honest-CANAL comparison: charge rho_imp for the importance computation.

For a fixed K (default 10), at 7 epsilons, compares 3 methods:

  PATE+uniform        : sigma_unif from rho_total, no importance used
  PATE+CANAL (raw)    : unclipped importance, FREE (current behavior, NOT honest)
  PATE+CANAL (honest) : per-sample L2-clipped importance + Gaussian noise paid for,
                        rho split as rho_imp + rho_rel = rho_total

Honest accounting:
  rho_total = rho_imp + rho_rel    (alpha = rho_imp / rho_total, default 0.1)
  per-sample L2-clipped to <= clip (default 100)
  per-record sensitivity Delta_imp = 2 * clip / N_per_teacher
  sigma_imp = Delta_imp / sqrt(2 * rho_imp)
  shared_imp_noisy = clipped_shared_imp + Gaussian(sigma_imp)   [clamped > 0]
  sigma_canal = waterfilling(deltas, shared_imp_noisy, rho_rel)

Total zCDP cost matches uniform (both consume rho_total). Apples-to-apples.

Usage:
  python isic_honest_canal_compare.py
  python isic_honest_canal_compare.py --alpha_imp 0.1 --clip_imp 100.0 --seeds 5
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
from drive_local_demo import TinyUNet, evaluate_vessel_dice, compute_importance_actnorm
from drive_pate_poc import train_K_teachers, correct_uniform_sigma, precompute_pate_cache
from drive_pate_canal_combined import correct_waterfilling_sigma
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
    """Per-teacher act_norm with per-sample L2-clipping, then averaged.

    Per-sample per-channel act-norm vector is L2-clipped to <= clip.
    Returns (shared_imp [C], N_per_teacher).
    """
    K = len(teachers)
    Cb = teachers[0].base * 4
    imps = []
    N_per_teacher = 0
    per_sample_l2_max = 0.0
    per_sample_l2_mean_acc = 0.0
    n_seen = 0
    for t in teachers:
        t.eval()
        sum_clipped = torch.zeros(Cb, device=device)
        N = 0
        for x, _ in loader:
            x = x.to(device)
            _, _, e3 = t.encode(x)
            per_sample = e3.flatten(2).norm(dim=2)  # [B, C]
            l2 = per_sample.norm(dim=1, keepdim=True)  # [B, 1]
            per_sample_l2_max = max(per_sample_l2_max, float(l2.max()))
            per_sample_l2_mean_acc += float(l2.sum())
            n_seen += l2.shape[0]
            scale = (clip / l2).clamp(max=1.0)
            clipped = per_sample * scale
            sum_clipped += clipped.sum(dim=0)
            N += clipped.shape[0]
        imps.append(sum_clipped / N)
        N_per_teacher = N
    shared = torch.stack(imps, dim=0).mean(dim=0)
    diag = {
        "per_sample_l2_max": per_sample_l2_max,
        "per_sample_l2_mean": per_sample_l2_mean_acc / max(n_seen, 1),
        "clip_used": clip,
    }
    return shared.cpu(), N_per_teacher, diag


def add_dp_noise_to_importance(imp, sensitivity, rho_imp, device, seed):
    """Gaussian mechanism on importance vector. Returns (noisy_imp [C], sigma_imp)."""
    sigma_imp = sensitivity / math.sqrt(2.0 * rho_imp)
    gen = torch.Generator()
    gen.manual_seed(seed)
    noise = torch.randn(imp.shape, generator=gen) * sigma_imp
    noisy = (imp + noise).clamp(min=1e-6)
    return noisy.to(device), sigma_imp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epsilons", type=str, default="0.1,0.5,1,2,4,6,8")
    ap.add_argument("--alpha_imp", type=float, default=0.1,
                    help="fraction of rho_total spent on releasing importance")
    ap.add_argument("--clip_imp", type=float, default=100.0,
                    help="per-sample L2 clip on the per-channel act-norm vector")
    ap.add_argument("--te", type=int, default=50)
    ap.add_argument("--se", type=int, default=40)
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--teacher_base", type=int, default=32,
                    help="teacher TinyUNet base width (bottleneck C = 4 * teacher_base)")
    ap.add_argument("--student_base", type=int, default=0,
                    help="student TinyUNet base width (0 = teacher_base // 2)")
    ap.add_argument("--out_tag", type=str, default="")
    args = ap.parse_args()
    if args.student_base == 0:
        args.student_base = max(8, args.teacher_base // 2)

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    EPS = [float(e) for e in args.epsilons.split(",")]
    SEEDS = list(range(100, 100 + args.seeds * 100, 100))

    train_ds = ISICDataset(split="train", size=args.size)
    val_ds = ISICDataset(split="val", size=args.size)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    print(f"[ISIC honest CANAL] K={args.K} eps={EPS} seeds={SEEDS}")
    print(f"  alpha_imp={args.alpha_imp}  clip_imp={args.clip_imp}")

    # ---- train K teachers ----
    print(f"\n[teachers] training K={args.K}")
    t0 = time.time()
    teachers, caps_list = train_K_teachers(
        train_ds, K=args.K, device=device, n_epochs=args.te, in_ch=3,
        base=args.teacher_base,
    )
    print(f"  K={args.K} done in {time.time()-t0:.0f}s "
          f"(teacher_base={args.teacher_base}, bottleneck C={args.teacher_base*4})")

    # ---- raw importance (unclipped, used by raw CANAL = current behavior) ----
    imp_loader = DataLoader(train_ds, batch_size=8, shuffle=False)
    raw_imps = [compute_importance_actnorm(t, imp_loader, device) for t in teachers]
    imp_raw = torch.stack(raw_imps, dim=0).mean(dim=0).to(device)
    R_raw = (imp_raw.max() / imp_raw.min()).item()
    print(f"  R_raw(unclipped) = {R_raw:.2f}")

    # ---- clipped importance + sensitivity (for honest CANAL) ----
    imp_clipped, N_per_teacher, diag = shared_actnorm_importance_clipped(
        teachers, imp_loader, device, clip=args.clip_imp,
    )
    R_clipped = (imp_clipped.max() / imp_clipped.min()).item()
    sensitivity = 2.0 * args.clip_imp / N_per_teacher
    print(f"  R_clipped = {R_clipped:.2f}  (clip={args.clip_imp}, "
          f"obs per-sample L2 max={diag['per_sample_l2_max']:.2f} "
          f"mean={diag['per_sample_l2_mean']:.2f})")
    print(f"  Delta_imp (replace-one sensitivity) = {sensitivity:.6f}")

    Cb = teachers[0].base * 4
    deltas = torch.full((Cb,), 2.0 / args.K, device=device)

    tag_suffix = f"_{args.out_tag}" if args.out_tag else ""
    save_path = HERE / "results" / f"isic_honest_canal{tag_suffix}_results.json"
    results = {
        "dataset": "isic", "K": args.K,
        "epsilons": EPS, "seeds": SEEDS,
        "alpha_imp": args.alpha_imp, "clip_imp": args.clip_imp,
        "R_raw_unclipped": R_raw, "R_clipped": R_clipped,
        "sensitivity_imp": sensitivity, "N_per_teacher": N_per_teacher,
        "obs_per_sample_l2_max": diag["per_sample_l2_max"],
        "obs_per_sample_l2_mean": diag["per_sample_l2_mean"],
        "C": Cb,
        "series": {
            "PATE+uniform":        {},
            "PATE+CANAL (raw)":    {},
            "PATE+CANAL (honest)": {},
        },
        "sigma_imp_by_eps": {},
    }

    def run_cell(sigma, label, eps):
        cache = precompute_pate_cache(
            teachers, caps_list, train_ds, sigma, device,
            seed=42 + int(eps * 10),
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

    total = 3 * len(EPS)
    done = 0
    job_t0 = time.time()
    for eps in EPS:
        rho = eps_to_rho(eps)
        rho_imp = args.alpha_imp * rho
        rho_rel = (1.0 - args.alpha_imp) * rho
        sigma_imp_at_eps = sensitivity / math.sqrt(2.0 * rho_imp)
        results["sigma_imp_by_eps"][str(eps)] = sigma_imp_at_eps
        print(f"\n========== eps={eps}  rho={rho:.5f}  "
              f"(rho_imp={rho_imp:.5f}, rho_rel={rho_rel:.5f}, "
              f"sigma_imp={sigma_imp_at_eps:.3f}) ==========")

        # 1) uniform: full rho budget
        sigma_unif = correct_uniform_sigma(deltas, rho)
        print(f"   [uniform]      sigma={sigma_unif[0].item():.4f}")
        results["series"]["PATE+uniform"][str(eps)] = run_cell(
            sigma_unif, "uniform", eps,
        )
        done += 1
        save_path.write_text(json.dumps(results, indent=2))

        # 2) CANAL raw: unclipped imp, free, full rho budget
        sigma_canal_raw = correct_waterfilling_sigma(deltas, imp_raw, rho)
        top10 = torch.argsort(imp_raw, descending=True)[:10]
        bot10 = torch.argsort(imp_raw, descending=False)[:10]
        ratio = (sigma_canal_raw[bot10].mean() / sigma_canal_raw[top10].mean()).item()
        print(f"   [CANAL raw]    sigma top10={sigma_canal_raw[top10].mean().item():.4f} "
              f"bot10={sigma_canal_raw[bot10].mean().item():.4f} ratio={ratio:.3f}x")
        results["series"]["PATE+CANAL (raw)"][str(eps)] = run_cell(
            sigma_canal_raw, "CANAL_raw", eps,
        )
        done += 1
        save_path.write_text(json.dumps(results, indent=2))

        # 3) CANAL honest: pay rho_imp for noisy imp, rho_rel for noise
        imp_noisy, _ = add_dp_noise_to_importance(
            imp_clipped, sensitivity, rho_imp, device,
            seed=12345 + int(eps * 10),
        )
        sigma_canal_honest = correct_waterfilling_sigma(
            deltas, imp_noisy, rho_rel,
        )
        top10_h = torch.argsort(imp_noisy, descending=True)[:10]
        bot10_h = torch.argsort(imp_noisy, descending=False)[:10]
        ratio_h = (
            sigma_canal_honest[bot10_h].mean()
            / sigma_canal_honest[top10_h].mean()
        ).item()
        print(f"   [CANAL honest] sigma top10={sigma_canal_honest[top10_h].mean().item():.4f} "
              f"bot10={sigma_canal_honest[bot10_h].mean().item():.4f} ratio={ratio_h:.3f}x")
        print(f"      imp_noisy range [{imp_noisy.min():.3f}, {imp_noisy.max():.3f}] "
              f"vs imp_clipped [{imp_clipped.min():.3f}, {imp_clipped.max():.3f}]")
        results["series"]["PATE+CANAL (honest)"][str(eps)] = run_cell(
            sigma_canal_honest, "CANAL_honest", eps,
        )
        done += 1
        save_path.write_text(json.dumps(results, indent=2))

        elapsed = time.time() - job_t0
        eta = elapsed / done * (total - done)
        print(f"   [progress] {done}/{total}  "
              f"elapsed={elapsed/60:.1f}min  ETA={eta/60:.1f}min")

    # ---- summary ----
    print("\n" + "=" * 92)
    print(f"Honest-CANAL summary  K={args.K} alpha={args.alpha_imp} clip={args.clip_imp}")
    print("=" * 92)
    hdr = f"{'series':<25s} " + " ".join(f"{e:>7.2f}" for e in EPS)
    print(hdr)
    print("-" * len(hdr))
    for key, data in results["series"].items():
        row = " ".join(f"{data[str(e)]['mean']:>7.4f}" for e in EPS)
        print(f"{key:<25s} {row}")
    print("=" * 92)

    save_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {save_path}")

    # ---- plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 6.5))
        colors = {
            "PATE+uniform":        "#2ca02c",
            "PATE+CANAL (raw)":    "#9467bd",
            "PATE+CANAL (honest)": "#d62728",
        }
        markers = {
            "PATE+uniform":        "o",
            "PATE+CANAL (raw)":    "^",
            "PATE+CANAL (honest)": "s",
        }
        for name, data in results["series"].items():
            means = [data[str(e)]["mean"] for e in EPS]
            stds = [data[str(e)]["std"] for e in EPS]
            ax.errorbar(
                EPS, means, yerr=stds, fmt=f"{markers[name]}-",
                color=colors[name], lw=2.2, ms=10, capsize=6, label=name,
            )
        ax.set_xscale("log")
        ax.set_xticks(EPS)
        ax.set_xticklabels([str(e) for e in EPS])
        ax.set_xlabel("eps")
        ax.set_ylabel("Dice (ISIC val)")
        ax.set_title(
            f"Honest CANAL  (K={args.K}, alpha_imp={args.alpha_imp}, "
            f"clip={args.clip_imp}, {args.seeds} seeds)"
        )
        ax.legend(loc="best", fontsize=10)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        plot_path = HERE / f"fig_isic_honest_canal{tag_suffix}.png"
        fig.savefig(plot_path, dpi=150)
        print(f"Saved: {plot_path}")
    except Exception as e:
        print(f"(plot skipped: {e})")

    cleanup(device)


if __name__ == "__main__":
    main()
