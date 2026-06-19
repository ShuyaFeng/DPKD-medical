"""
Joint (K × keep) ablation: PATE teacher-count × channel-pruning.

Two ORTHOGONAL denoising levers, swept together:
  - PATE K:    sensitivity Δ = 2/K       → σ ∝ 1/K   (linear, strong)
  - pruning f: keep top-f·C channels      → σ ∝ √f    (sqrt)
Combined:      σ ∝ (1/K) · √f

This single sweep subsumes:
  - the K=1 column      = single-teacher channel-pruning
  - the keep=1.0 row    = PATE K-saturation
  - the interior        = the joint composition (the paper heatmap)

CRITICAL DESIGN DECISION (channel mask)
---------------------------------------
The channel mask is a SINGLE shared set chosen from PUBLIC importance,
used by ALL K teachers. NOT per-teacher top-channels (those would be
different channel sets and could not be aggregated). The shared mask
also partially mitigates the channel-alignment problem: teachers only
average over the same public-important channel indices.

(Local stand-in for public importance: we average each teacher's
train-data importance into a shared ranking. In the paper this comes
from HRF, costing zero privacy on DRIVE.)

GRID
----
3 K × 4 keep × 3 ε × 5 seeds = 180 student trainings (~2-3 GPU-h)
plus training 1+5+10 = 16 teachers.

OUTPUT
------
  drive_pate_pruning_joint_results.json   (nested K → keep → ε → dices)
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from drive_local_demo import DriveDataset, compute_importance
from synthetic_demo import (
    eps_to_rho, uniform_sigma, clip_and_normalise, denormalise,
)
from drive_pate_poc import partition_dataset, train_K_teachers
from drive_student_distill import train_student_distill


def shared_importance(teachers, train_loader, device):
    """Average each teacher's importance into ONE shared channel ranking.

    Stand-in for public-data importance. The point is that all teachers
    use the SAME channel mask so their bottlenecks can be aggregated.
    """
    imps = [compute_importance(t, train_loader, device) for t in teachers]
    return torch.stack(imps, dim=0).mean(dim=0)


def thresholded_uniform_sigma(deltas, rho, active_mask):
    sigma = torch.zeros_like(deltas)
    if active_mask.any():
        sigma[active_mask] = uniform_sigma(deltas[active_mask], rho)
    return sigma


@torch.no_grad()
def precompute_joint_cache(teachers, caps_list, train_ds, sigma,
                           active_mask, device, seed):
    """
    Aggregate K teachers (mean of normalised bottlenecks), keep only the
    shared active channels, add Gaussian noise on the active set.
    Returns a list keyed by dataset index.
    """
    for t in teachers:
        t.eval()
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)  # CPU generator (mps-safe)
    K = len(teachers)
    C = active_mask.shape[0]
    mean_caps = torch.stack([c.to(device) for c in caps_list]).mean(dim=0)
    cache = []
    loader = DataLoader(train_ds, batch_size=4, shuffle=False)
    for x, _ in loader:
        x = x.to(device)
        agg = None
        for t, caps in zip(teachers, caps_list):
            _, _, z = t.encode(x)
            z_norm = clip_and_normalise(z, caps.to(device))
            agg = z_norm if agg is None else agg + z_norm
        agg = agg / K                                   # mean normalised bottleneck
        B, Cc, H, W = agg.shape
        agg = agg * active_mask.view(1, Cc, 1, 1).float()   # drop inactive channels
        noise = torch.randn(B, Cc, H, W, generator=g).to(device) * sigma.view(1, Cc, 1, 1)
        z_noisy = denormalise(agg + noise, mean_caps)
        for b in range(B):
            cache.append(z_noisy[b].detach().cpu())
    return cache


def main():
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(0); np.random.seed(0)
    print(f"Device: {device}")

    train_ds = DriveDataset("train", size=96)
    val_ds   = DriveDataset("val",   size=96)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=False)
    val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False)
    N_train = len(train_ds)

    K_values      = [1, 5, 10]
    keep_fracs    = [1.0, 0.3, 0.2, 0.1, 0.05, 0.02]
    epsilons      = [1.0, 2.0, 3.0, 4.0, 5.0]
    student_seeds = [100, 200, 300, 400, 500]
    base_T        = 32
    Cb_T          = base_T * 4

    results = {
        "K_values": K_values, "keep_fractions": keep_fracs,
        "epsilons": epsilons, "student_seeds": student_seeds,
        "C": Cb_T, "sweep": {},
    }

    for K in K_values:
        print(f"\n{'#'*72}\n#  K = {K}\n{'#'*72}")
        partitions = partition_dataset(N_train, K)
        print(f"  partition sizes: {[len(p) for p in partitions]}")
        teachers, caps_list = train_K_teachers(train_ds, K, device, n_epochs=60)

        # ONE shared channel ranking for all K teachers
        importance = shared_importance(teachers, train_loader, device).to(device)
        rank_desc = torch.argsort(importance, descending=True)
        deltas = torch.full((Cb_T,), 2.0 / K, device=device)

        results["sweep"][str(K)] = {}
        for eps in epsilons:
            rho = eps_to_rho(eps)
            results["sweep"][str(K)][str(eps)] = {}
            print(f"\n  {'='*64}\n  K={K}  ε={eps}  ρ={rho:.4f}  Δ=2/K={2/K:.3f}\n  {'='*64}")
            for f in keep_fracs:
                n_active = max(1, int(round(f * Cb_T)))
                active_mask = torch.zeros(Cb_T, dtype=torch.bool, device=device)
                active_mask[rank_desc[:n_active]] = True
                sigma = thresholded_uniform_sigma(deltas, rho, active_mask)
                sig_a = sigma[active_mask].mean().item()
                print(f"\n    keep={f*100:4.0f}%  n_active={n_active:3d}  σ_active={sig_a:.3f}")

                cache_seed = 42 + int(eps * 10) + int(f * 1000) + K * 7
                cache = precompute_joint_cache(teachers, caps_list, train_ds,
                                               sigma, active_mask, device,
                                               seed=cache_seed)
                dices = []
                for s in student_seeds:
                    t0 = time.time()
                    best, _ = train_student_distill(
                        train_ds, val_loader, cache, device,
                        student_base=16, teacher_base=base_T,
                        n_epochs=40, lr=1e-3, lambda_feat=0.4, seed=s,
                    )
                    dices.append(best)
                    print(f"      seed={s}: Dice={best:.4f}  ({time.time()-t0:.1f}s)")
                results["sweep"][str(K)][str(eps)][f"{f:.2f}"] = {
                    "n_active": n_active,
                    "sigma_active": sig_a,
                    "dices": dices,
                    "mean": float(np.mean(dices)),
                    "std":  float(np.std(dices)),
                    "sem":  float(np.std(dices) / np.sqrt(len(dices))),
                }

    out_path = Path(__file__).parent / "drive_pate_pruning_joint_results.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote: {out_path}")

    # quick best-cell summary per ε
    print("\nBest (K, keep) per ε:")
    for eps in epsilons:
        best = None
        for K in K_values:
            for f in keep_fracs:
                cell = results["sweep"][str(K)][str(eps)][f"{f:.2f}"]
                if best is None or cell["mean"] > best[2]:
                    best = (K, f, cell["mean"])
        print(f"  ε={eps}: best Dice={best[2]:.4f} at K={best[0]}, keep={best[1]*100:.0f}%")


if __name__ == "__main__":
    main()
