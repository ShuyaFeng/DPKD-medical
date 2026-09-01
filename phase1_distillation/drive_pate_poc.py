"""
M1: Honest PATE multi-teacher feature aggregation — proof of concept.

Train K small teachers on disjoint patient cohorts of DRIVE train (20 imgs
total). For each image, all K teachers compute their bottleneck. We clip
each teacher's bottleneck to per-teacher caps so each channel has L2 norm
≤ 1, then aggregate by mean. Add Gaussian noise with Δ = 2/K (replace-one
a patient affects exactly 1 of K teachers; the mean changes by ≤ 2/K per
channel under unit-norm clipping).

K ∈ {1, 3, 5} swept. At fixed per-release ε, σ scales as 1/K (σ ∝ Δ = 2/K)
→ utility should climb as K grows. This is the "real DP-mechanism advance" path:
the lift comes from a TIGHTER sensitivity, not from anything CANAL does.

Compared at the same per-release ε via paired 3-seed student distillation.
"""

import json
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent))
from drive_local_demo import (
    DriveDataset, TinyUNet, train_teacher, evaluate_vessel_dice,
    compute_importance, collect_caps, vessel_dice,
)
from synthetic_demo import (
    eps_to_rho, clip_and_normalise, denormalise, task_loss,
)
from drive_student_distill import train_student_distill


def correct_uniform_sigma(deltas: torch.Tensor, rho: float) -> torch.Tensor:
    """
    Uniform σ with the CORRECT factor-of-2 in zCDP.
      Σ Δ_c²/(2σ²) = ρ  →  σ = √(Σ Δ_c² / (2ρ))
    (synthetic_demo's uniform_sigma had the /2 missing until 2026-08-31;
    it is now fixed and identical to this function. Kept here as the
    canonical version — test_sigma.py asserts the two stay in sync.)
    """
    sigma_val = (deltas.pow(2).sum() / (2.0 * rho)).sqrt()
    return sigma_val.expand(deltas.shape[0]).clone()


def partition_dataset(n: int, K: int) -> List[List[int]]:
    """Disjoint K partitions of {0, ..., n-1}."""
    sizes = [n // K + (1 if i < n % K else 0) for i in range(K)]
    out, start = [], 0
    for size in sizes:
        out.append(list(range(start, start + size)))
        start += size
    return out


def train_K_teachers(train_ds, K: int, device, n_epochs=60, lr=1e-3, base=32, in_ch=3):
    """Train K teachers, each on its disjoint patient cohort. in_ch=4 for BraTS."""
    partitions = partition_dataset(len(train_ds), K)
    teachers, caps_list = [], []
    for k, idxs in enumerate(partitions):
        torch.manual_seed(1000 + k)
        teacher = TinyUNet(in_ch=in_ch, num_classes=2, base=base).to(device)
        subset = Subset(train_ds, idxs)
        bs = min(4, len(idxs))
        sub_loader = DataLoader(subset, batch_size=bs, shuffle=True)
        print(f"  teacher {k+1}/{K} on patients {idxs}...")
        train_teacher(teacher, sub_loader, n_epochs=n_epochs, lr=lr, device=device)
        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher.eval()
        # caps computed on this teacher's OWN training subset
        cap_loader = DataLoader(subset, batch_size=bs, shuffle=False)
        caps = collect_caps(teacher, cap_loader, device).to(device)
        teachers.append(teacher)
        caps_list.append(caps)
    return teachers, caps_list


@torch.no_grad()
def precompute_pate_cache(teachers, caps_list, train_ds, sigma, device,
                          seed: int):
    """
    Sample-once PATE precompute. For each image x:
      1. Each teacher k: enc(x) → clip + normalize by caps_k → unit-norm bn_k
      2. Aggregate by mean: bn_agg = mean(bn_k for k in K)
      3. Add noise: bn_noisy_norm = bn_agg + N(0, σ²)
      4. Denormalize using mean of caps
      5. Cache by index
    """
    torch.manual_seed(seed)
    K = len(teachers)
    caps_mean = torch.stack(caps_list).mean(dim=0)  # (C,)
    cache = []
    loader = DataLoader(train_ds, batch_size=4, shuffle=False)
    for x, _ in loader:
        x = x.to(device)
        # Each teacher's clip+normalized bottleneck (norm ≤ 1 per channel)
        normalized_bns = []
        for teacher, caps in zip(teachers, caps_list):
            _, _, e3 = teacher.encode(x)
            bn_norm = clip_and_normalise(e3, caps)
            normalized_bns.append(bn_norm)
        agg = torch.stack(normalized_bns, dim=0).mean(dim=0)  # (B, C, H, W)
        B, C, H, W = agg.shape
        noise = torch.randn(B, C, H, W, device=device) * sigma.view(1, C, 1, 1)
        agg_noisy_norm = agg + noise
        # Denormalize with mean cap so student lives in roughly the same scale
        agg_noisy = denormalise(agg_noisy_norm, caps_mean)
        for b in range(B):
            cache.append(agg_noisy[b].detach().cpu())
    return cache


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)
    print(f"Device: {device}")

    train_ds = DriveDataset("train", size=96)
    val_ds   = DriveDataset("val",   size=96)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)
    N_train = len(train_ds)
    print(f"train={N_train}  val={len(val_ds)}")

    K_values      = [1, 3, 5]
    epsilons      = [2.0, 8.0, 16.0]
    student_seeds = [100, 200, 300]
    base_T        = 32
    Cb_T          = base_T * 4   # 128 channels at bottleneck

    results = {
        "K_values":      K_values,
        "epsilons":      epsilons,
        "student_seeds": student_seeds,
        "sweep":         {},
    }

    for K in K_values:
        partitions = partition_dataset(N_train, K)
        sizes = [len(p) for p in partitions]
        print(f"\n{'='*72}")
        print(f"K = {K}   partition sizes = {sizes}   Δ = 2/K = {2/K:.3f}")
        print(f"{'='*72}")

        # ---- train K teachers ----
        teachers, caps_list = train_K_teachers(train_ds, K, device, n_epochs=60)
        cap_means = [caps.mean().item() for caps in caps_list]
        print(f"  per-teacher mean cap: {[f'{c:.2f}' for c in cap_means]}")

        # ---- sensitivity Δ = 2/K (constant across channels in unit-norm space)
        deltas = torch.full((Cb_T,), 2.0 / K, device=device)

        results["sweep"][K] = {
            "partition_sizes": sizes,
            "cap_means":       cap_means,
            "epsilons":        {},
        }
        for eps in epsilons:
            rho   = eps_to_rho(eps)
            sigma = correct_uniform_sigma(deltas, rho)
            print(f"\n  ε={eps:>4.1f}   ρ={rho:.4f}   σ_uniform = {sigma[0].item():.4f}")

            cache = precompute_pate_cache(teachers, caps_list, train_ds,
                                           sigma, device,
                                           seed=42 + int(eps * 10))

            dices = []
            for s in student_seeds:
                t0 = time.time()
                best, _ = train_student_distill(
                    train_ds, val_loader, cache, device,
                    student_base=16, teacher_base=base_T,
                    n_epochs=40, lr=1e-3, lambda_feat=0.4, seed=s,
                )
                dices.append(best)
                print(f"    seed={s}:  best Dice = {best:.4f}   ({time.time()-t0:.1f}s)")
            mean_d = float(np.mean(dices))
            std_d  = float(np.std(dices))
            sem_d  = std_d / np.sqrt(len(dices))
            print(f"   → K={K} ε={eps}:  {mean_d:.4f} ± {std_d:.4f}  (SEM {sem_d:.4f})")
            results["sweep"][K]["epsilons"][eps] = {
                "dices": dices, "mean": mean_d, "std": std_d, "sem": sem_d,
                "sigma": float(sigma[0]),
            }

    # ---- summary ----
    print("\n" + "=" * 100)
    print(f"{'K':>3}  {'ε':>5}  {'σ_uni':>10}  {'Dice mean ± std':>22}  "
          f"{'vs K=1':>14}  {'paired vs K=1':>16}")
    print("-" * 100)
    for K in K_values:
        for eps in epsilons:
            r = results["sweep"][K]["epsilons"][eps]
            mean_d, std_d, sigma = r["mean"], r["std"], r["sigma"]
            if K == 1:
                vs1 = ""; paired = ""
            else:
                base_mean = results["sweep"][1]["epsilons"][eps]["mean"]
                base_dices = results["sweep"][1]["epsilons"][eps]["dices"]
                lift = mean_d - base_mean
                # paired per-seed
                paired_lifts = [k - b for k, b in zip(r["dices"], base_dices)]
                paired_mean = np.mean(paired_lifts)
                paired_sem  = np.std(paired_lifts) / np.sqrt(len(paired_lifts))
                vs1    = f"{lift:+.4f}"
                paired = f"{paired_mean:+.4f}±{paired_sem:.4f}"
            print(f"{K:>3}  {eps:>5.1f}  {sigma:>10.4f}  "
                  f"{mean_d:.4f} ± {std_d:.4f}    {vs1:>14}  {paired:>16}")
    print("=" * 100)

    # ---- save ----
    out_path = Path(__file__).parent / "drive_pate_poc_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved JSON: {out_path}")

    # ---- plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Panel a: Dice vs ε per K
    ax = axes[0]
    colors = {1: "#d62728", 3: "#ff7f0e", 5: "#2ca02c"}
    for K in K_values:
        means = [results["sweep"][K]["epsilons"][e]["mean"] for e in epsilons]
        stds  = [results["sweep"][K]["epsilons"][e]["std"]  for e in epsilons]
        ax.errorbar(epsilons, means, yerr=stds, fmt="o-", color=colors[K],
                    lw=2, ms=10, capsize=6,
                    label=f"K={K}  (Δ=2/{K}={2/K:.3f})")
    ax.set_xscale("log", base=2)
    ax.set_xticks(epsilons)
    ax.set_xticklabels([str(int(e)) for e in epsilons])
    ax.set_xlabel("privacy budget ε")
    ax.set_ylabel("vessel Dice")
    ax.set_title("(a) PATE multi-teacher: Dice vs ε per K  (3 seeds)")
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)

    # Panel b: paired lift K=3 vs K=1, K=5 vs K=1
    ax = axes[1]
    ax.axhline(0, color="black", lw=1)
    for K in [3, 5]:
        lifts_means, lifts_sems = [], []
        for eps in epsilons:
            base_dices = results["sweep"][1]["epsilons"][eps]["dices"]
            k_dices = results["sweep"][K]["epsilons"][eps]["dices"]
            paired = [k - b for k, b in zip(k_dices, base_dices)]
            lifts_means.append(np.mean(paired))
            lifts_sems.append(np.std(paired) / np.sqrt(len(paired)))
        ax.errorbar(epsilons, lifts_means, yerr=lifts_sems, fmt="o-",
                    color=colors[K], lw=2, ms=10, capsize=6,
                    label=f"K={K} − K=1")
    ax.fill_between([epsilons[0]*0.7, epsilons[-1]*1.3], -0.002, 0.002,
                    color="gray", alpha=0.15, label="±0.002 noise band")
    ax.set_xscale("log", base=2)
    ax.set_xticks(epsilons)
    ax.set_xticklabels([str(int(e)) for e in epsilons])
    ax.set_xlim(epsilons[0]*0.7, epsilons[-1]*1.3)
    ax.set_xlabel("privacy budget ε")
    ax.set_ylabel("Paired Dice lift vs K=1 (mean ± SEM)")
    ax.set_title("(b) Lift over K=1 baseline (paired per seed)")
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)

    fig.suptitle(f"M1 PoC — Honest PATE multi-teacher feature aggregation",
                 fontweight="bold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = Path(__file__).parent / "drive_pate_poc.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
