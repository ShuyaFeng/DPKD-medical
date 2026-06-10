"""
Q1 experiment for paper Table 3: per-iteration vs sample-once release at
MATCHED user-level ε.

Per-iteration model (prior work's implicit model):
  - student trains T=40000 iters, batch B=4, dataset N=20
  - each image is touched Q = T·B/N = 8000 times during student training
  - to achieve TARGET user-level ε, per-release ε must be tiny:
      ρ_per_release = ρ_user_target / Q
  - σ_per_iter = √(C · Δ² / (2 ρ_per_release))   — astronomical
  - student sees fresh noise on a fresh teacher query every iter

Sample-once model (ours):
  - teacher is forwarded once per image, noise added once, cached
  - per-release ε = user-level ε  (parallel composition over k=1 images
    per patient)
  - σ_sample_once = √(C · Δ² / (2 ρ_user_target))
  - student reads the same cached noisy bottleneck every iter

At matched user-level ε, σ_per_iter ≫ σ_sample_once (factor √Q ≈ 90×),
so the per-iter student is trained against essentially pure noise.
The Dice collapse is the empirical signal Tab. 3 reports.

Local PoC: T=40 epochs × 5 batches/epoch = 200 iters per training run,
so Q_local = 200·4/20 = 40 releases per patient if per-iter mode is used.
For paper Tab. 3 we report the formula at T=40000 (paper config) so the
EFFECTIVE per-iter σ used here is calibrated to the PAPER's Q=8000, not
to the local Q=40. This matches the privacy claim of Tab. 1 and tests
the utility consequence of that claim.

Conditions tested (5 seeds × 4 user-level ε × 3 modes):
  per-iter           : large σ, fresh per batch
  sample-once (ours) : small σ, cached
  non-private        : σ=0, full clean features
"""

import json
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from drive_local_demo import (
    DriveDataset, TinyUNet, train_teacher, evaluate_vessel_dice,
    compute_importance, collect_caps, vessel_dice,
)
from synthetic_demo import (
    eps_to_rho, clip_and_normalise, denormalise, task_loss,
)
from drive_student_distill import Adapter


# Paper config (matches Tab. 1)
PAPER_T = 40_000
PAPER_B = 4
PAPER_N = 20


@torch.no_grad()
def precompute_clean_cache(teacher, train_ds, caps, device):
    """Cache CLEAN (clip+normalised) teacher bottlenecks. No noise added.
    Per-iter mode adds noise on top of this; sample-once mode shifts the
    addition to a separate precompute path."""
    teacher.eval()
    cache_clean = []
    loader = DataLoader(train_ds, batch_size=4, shuffle=False)
    for x, _ in loader:
        x = x.to(device)
        _, _, e3 = teacher.encode(x)
        bn_norm = clip_and_normalise(e3, caps)
        for b in range(bn_norm.shape[0]):
            cache_clean.append(bn_norm[b].detach().cpu().clone())
    return cache_clean


@torch.no_grad()
def precompute_sample_once_cache(teacher, train_ds, caps, sigma, device, seed):
    """Sample-once: clean+normalize, ADD noise once, denormalize, cache."""
    teacher.eval()
    torch.manual_seed(seed)
    cache = []
    loader = DataLoader(train_ds, batch_size=4, shuffle=False)
    for x, _ in loader:
        x = x.to(device)
        _, _, e3 = teacher.encode(x)
        bn_norm = clip_and_normalise(e3, caps)
        B, C, H, W = bn_norm.shape
        noise = torch.randn(B, C, H, W, device=device) * sigma.view(1, C, 1, 1)
        bn_noisy = denormalise(bn_norm + noise, caps)
        for b in range(B):
            cache.append(bn_noisy[b].detach().cpu().clone())
    return cache


def train_student_inline_noise(train_ds, val_loader, clean_cache, caps, sigma,
                               device, mode: str,
                               student_base=16, teacher_base=32,
                               n_epochs=40, lr=1e-3, lambda_feat=0.4, seed=2):
    """
    Train student.  mode ∈ {"per_iter", "sample_once", "non_private"}.

    per_iter      : at each batch, freshly noise the CLEAN cached bn
                    using `sigma`. Different noise every iter.
    sample_once   : `clean_cache` is already the noisy cache (caller passes
                    the sample-once cache here as `clean_cache`).  No
                    additional noise. Same target every iter.
    non_private   : no noise; clean_cache is the clean (denormalized) bn.
    """
    torch.manual_seed(seed)
    student = TinyUNet(in_ch=3, num_classes=2, base=student_base).to(device)
    adapter = Adapter(student_base * 4, teacher_base * 4).to(device)
    opt = torch.optim.Adam(
        list(student.parameters()) + list(adapter.parameters()), lr=lr,
    )

    N = len(train_ds)
    batch_size = 4
    best_dice = 0.0

    # For per_iter mode, clean_cache holds NORMALISED clean bn (so we can
    # add noise then denormalise per batch).  For sample_once / non_private
    # it holds the actual target the student should match.
    caps_view = caps.view(1, -1, 1, 1)

    for ep in range(n_epochs):
        student.train(); adapter.train()
        perm = torch.randperm(N)
        for i in range(0, N, batch_size):
            idxs = perm[i:i + batch_size].tolist()
            xs = torch.stack([train_ds[idx][0] for idx in idxs]).to(device)
            ys = torch.stack([train_ds[idx][1] for idx in idxs]).to(device)
            cached = torch.stack([clean_cache[idx] for idx in idxs]).to(device)

            if mode == "per_iter":
                # Freshly noise the normalised clean bn.
                B, C, H, W = cached.shape
                noise = torch.randn(B, C, H, W, device=device) * sigma.view(1, C, 1, 1)
                target = (cached + noise) * caps_view
            elif mode == "sample_once" or mode == "non_private":
                # cached IS the target (already denormalised in precompute).
                target = cached
            else:
                raise ValueError(mode)

            opt.zero_grad()
            e1, e2, e3 = student.encode(xs)
            s_proj  = adapter(e3)
            f_loss  = F.mse_loss(s_proj, target)
            logits  = student.decode(e1, e2, e3)
            t_loss  = task_loss(logits, ys)
            loss    = t_loss + lambda_feat * f_loss
            loss.backward()
            opt.step()

        val_dice = evaluate_vessel_dice(student, val_loader, device)
        if val_dice > best_dice:
            best_dice = val_dice
    return best_dice


def correct_uniform_sigma(deltas: torch.Tensor, rho: float) -> torch.Tensor:
    sigma_val = (deltas.pow(2).sum() / (2.0 * rho)).sqrt()
    return sigma_val.expand(deltas.shape[0]).clone()


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)
    print(f"Device: {device}")

    train_ds = DriveDataset("train", size=96)
    val_ds   = DriveDataset("val",   size=96)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    print("\n[1/3] Training K=1 teacher...")
    torch.manual_seed(1000)
    teacher = TinyUNet(in_ch=3, num_classes=2, base=32).to(device)
    t_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    train_teacher(teacher, t_loader, n_epochs=60, lr=1e-3, device=device)
    clean_dice = evaluate_vessel_dice(teacher, val_loader, device)
    caps = collect_caps(teacher, t_loader, device).to(device)
    Cb_T = caps.shape[0]
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    print(f"  teacher clean Dice = {clean_dice:.4f}")

    # Caches (clean normalised + clean denormalised, both reusable)
    print("\n[2/3] Building per-image caches (clean normalised + clean denormalised)...")
    clean_norm_cache = precompute_clean_cache(teacher, train_ds, caps, device)
    clean_denorm_cache = [cn * caps.cpu().view(-1, 1, 1) for cn in clean_norm_cache]

    # Paper composition factor at T=40000, B=4, N=20:
    Q = PAPER_T * PAPER_B // PAPER_N    # 8000 releases per patient under per-iter
    print(f"  Per-iter releases per patient under PAPER config: Q = {Q}")

    K = 1
    Delta = 2.0 / K
    deltas = torch.full((Cb_T,), Delta, device=device)

    eps_targets = [2.0, 4.0, 8.0, 16.0]
    student_seeds = [100, 200, 300, 400, 500]

    results = {
        "clean_teacher_dice": clean_dice,
        "K": K, "Delta": Delta, "Cb_T": Cb_T,
        "Q_paper": Q, "T": PAPER_T, "B": PAPER_B, "N": PAPER_N,
        "epsilons": eps_targets, "student_seeds": student_seeds,
        "sweep": {},
    }

    print("\n[3/3] Running per-iter vs sample-once vs non-private...")
    for eps in eps_targets:
        rho_user = eps_to_rho(eps)
        rho_per_release_per_iter = rho_user / Q
        rho_per_release_once = rho_user

        sigma_per_iter = correct_uniform_sigma(deltas, rho_per_release_per_iter)
        sigma_once     = correct_uniform_sigma(deltas, rho_per_release_once)

        print(f"\n=== user-level ε = {eps}  (ρ_user = {rho_user:.4f}) ===")
        print(f"  per-iter: ρ_per_release = {rho_per_release_per_iter:.2e},  σ = {sigma_per_iter[0].item():.2f}")
        print(f"  once:     ρ_per_release = {rho_per_release_once:.4f},      σ = {sigma_once[0].item():.4f}")

        results["sweep"][eps] = {
            "rho_user": rho_user,
            "sigma_per_iter": float(sigma_per_iter[0]),
            "sigma_sample_once": float(sigma_once[0]),
        }

        # --- per-iter ---
        print(f"\n  --- per-iter mode (σ={sigma_per_iter[0].item():.2f}) ---")
        per_iter_dices = []
        for s in student_seeds:
            t0 = time.time()
            best = train_student_inline_noise(
                train_ds, val_loader, clean_norm_cache, caps, sigma_per_iter,
                device, mode="per_iter",
                n_epochs=40, lr=1e-3, lambda_feat=0.4, seed=s,
            )
            per_iter_dices.append(best)
            print(f"    seed={s}: best={best:.4f}  ({time.time()-t0:.1f}s)")
        results["sweep"][eps]["per_iter"] = {
            "dices": per_iter_dices,
            "mean": float(np.mean(per_iter_dices)),
            "std":  float(np.std(per_iter_dices)),
        }
        print(f"    per-iter mean = {results['sweep'][eps]['per_iter']['mean']:.4f}")

        # --- sample-once ---
        print(f"\n  --- sample-once mode (σ={sigma_once[0].item():.4f}) ---")
        once_cache = precompute_sample_once_cache(teacher, train_ds, caps,
                                                  sigma_once, device,
                                                  seed=42 + int(eps * 10))
        once_dices = []
        for s in student_seeds:
            t0 = time.time()
            best = train_student_inline_noise(
                train_ds, val_loader, once_cache, caps, sigma_once,
                device, mode="sample_once",
                n_epochs=40, lr=1e-3, lambda_feat=0.4, seed=s,
            )
            once_dices.append(best)
            print(f"    seed={s}: best={best:.4f}  ({time.time()-t0:.1f}s)")
        results["sweep"][eps]["sample_once"] = {
            "dices": once_dices,
            "mean": float(np.mean(once_dices)),
            "std":  float(np.std(once_dices)),
        }
        print(f"    sample-once mean = {results['sweep'][eps]['sample_once']['mean']:.4f}")

    # --- non-private reference (run once, eps-independent) ---
    print(f"\n=== non-private (ε = ∞) ===")
    np_dices = []
    for s in student_seeds:
        t0 = time.time()
        best = train_student_inline_noise(
            train_ds, val_loader, clean_denorm_cache, caps, torch.zeros_like(deltas),
            device, mode="non_private",
            n_epochs=40, lr=1e-3, lambda_feat=0.4, seed=s,
        )
        np_dices.append(best)
        print(f"  seed={s}: best={best:.4f}  ({time.time()-t0:.1f}s)")
    results["non_private"] = {
        "dices": np_dices,
        "mean": float(np.mean(np_dices)),
        "std":  float(np.std(np_dices)),
    }
    print(f"  non-private mean = {results['non_private']['mean']:.4f}")

    # --- summary ---
    print("\n" + "=" * 92)
    print(f"Tab. 3 (paper Table 3): per-iter vs sample-once at MATCHED user-level ε")
    print("-" * 92)
    print(f"{'user-level ε':>13}  {'σ_per_iter':>10}  {'per-iter Dice':>15}  "
          f"{'σ_once':>9}  {'sample-once Dice':>16}  {'gap':>8}")
    print("-" * 92)
    for eps in eps_targets:
        s = results["sweep"][eps]
        gap = s["sample_once"]["mean"] - s["per_iter"]["mean"]
        print(f"{eps:>13.1f}  {s['sigma_per_iter']:>10.2f}  "
              f"{s['per_iter']['mean']:.4f} ± {s['per_iter']['std']:.4f}  "
              f"{s['sigma_sample_once']:>9.4f}  "
              f"{s['sample_once']['mean']:.4f} ± {s['sample_once']['std']:.4f}  "
              f"{gap:>+8.4f}")
    print(f"{'∞ (non-private)':>13}  {'—':>10}  {'—':>15}  "
          f"{'0':>9}  {results['non_private']['mean']:.4f} ± {results['non_private']['std']:.4f}  {'—':>8}")
    print("=" * 92)

    # save
    out_path = Path(__file__).parent / "drive_q1_per_iter_vs_sample_once_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved JSON: {out_path}")

    # plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.axhline(results["non_private"]["mean"], color="#555", ls=":", lw=2.5,
               label=f"non-private (ε=∞) = {results['non_private']['mean']:.3f}")

    pi_means = [results["sweep"][e]["per_iter"]["mean"]   for e in eps_targets]
    pi_stds  = [results["sweep"][e]["per_iter"]["std"]    for e in eps_targets]
    so_means = [results["sweep"][e]["sample_once"]["mean"] for e in eps_targets]
    so_stds  = [results["sweep"][e]["sample_once"]["std"]  for e in eps_targets]

    ax.errorbar(eps_targets, pi_means, yerr=pi_stds, fmt="s--", color="#d62728",
                lw=2.5, ms=11, capsize=6,
                label="per-iter model at MATCHED user-level ε")
    ax.errorbar(eps_targets, so_means, yerr=so_stds, fmt="o-", color="#2ca02c",
                lw=2.5, ms=11, capsize=6,
                label="sample-once model at user-level ε (ours)")

    ax.set_xscale("log", base=2)
    ax.set_xticks(eps_targets)
    ax.set_xticklabels([str(int(e)) for e in eps_targets])
    ax.set_xlabel("user-level ε  (sample-once: per-release ε = user-level ε)")
    ax.set_ylabel("vessel Dice")
    ax.set_title("Q1 — same user-level privacy, very different utility (5 seeds)\n"
                 "per-iter must inject √Q ≈ 90× more noise per release to absorb composition")
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plot_path = Path(__file__).parent / "drive_q1_per_iter_vs_sample_once.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
