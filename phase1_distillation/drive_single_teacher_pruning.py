"""
Single-teacher channel-pruning to the extreme.

QUESTION
--------
Can a SINGLE teacher (no PATE, no channel-alignment problem) close most
of the gap to PATE K=10 purely by aggressive channel-pruning?

Channel-pruning is the only DP-clean single-teacher denoising lever:
unit-norm normalisation makes each channel an independent Δ=2/K
sensitivity unit, so dropping channels removes their sensitivity and
σ ∝ √(keep_fraction). (Transform-domain projections like PCA do NOT
denoise under worst-case DP — a replace-one perturbation can land
entirely in the kept subspace, so sensitivity is unchanged.)

This sweeps keep ∈ {100, 50, 25, 10, 5, 2}% on a SINGLE teacher and
compares against the PATE K=10 reference (0.648 at ε=16 from
drive_pate_K_saturation_results.json).

Sensitivity convention
----------------------
K=1 (true single teacher), Δ = 2/K = 2. This is the HONEST single-
teacher sensitivity. PATE K=10 gets Δ=2/10 — that 1/K linear factor is
what pruning's √f has to fight against.

GRID
----
6 keep-fractions × 3 ε × 5 seeds = 90 student trainings (~6-8 GPU-h).

OUTPUT
------
  drive_single_teacher_pruning_results.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from drive_local_demo import (
    DriveDataset, TinyUNet, train_teacher, compute_importance, collect_caps,
)
from synthetic_demo import (
    eps_to_rho, uniform_sigma, clip_and_normalise, denormalise,
)
from drive_student_distill import train_student_distill


def thresholded_uniform_sigma(deltas, rho, active_mask):
    """Uniform σ spread over the ACTIVE channels only; σ=0 for inactive."""
    sigma = torch.zeros_like(deltas)
    if active_mask.any():
        sigma[active_mask] = uniform_sigma(deltas[active_mask], rho)
    return sigma


@torch.no_grad()
def precompute_pruned_cache(teacher, train_ds, caps, sigma, active_mask,
                            device, seed):
    """
    Sample-once cache. Inactive channels are zeroed in the normalised
    space (data-independent constant → no release, no sensitivity);
    active channels get Gaussian noise with the given σ.
    Returns a list keyed by dataset index.
    """
    teacher.eval()
    torch.manual_seed(seed)
    cache = []
    loader = DataLoader(train_ds, batch_size=4, shuffle=False)
    C = active_mask.shape[0]
    for x, _ in loader:
        x = x.to(device)
        _, _, e3 = teacher.encode(x)
        bn_norm = clip_and_normalise(e3, caps)
        # zero out inactive channels (they carry no released signal)
        bn_norm = bn_norm * active_mask.view(1, C, 1, 1).float()
        B, Cc, H, W = bn_norm.shape
        noise = torch.randn(B, Cc, H, W, device=device) * sigma.view(1, Cc, 1, 1)
        bn_noisy = denormalise(bn_norm + noise, caps)
        for b in range(B):
            cache.append(bn_noisy[b].detach().cpu())
    return cache


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)
    print(f"Device: {device}")

    # --- single teacher ---------------------------------------------------
    print("\n[1/3] DRIVE + SINGLE teacher (60 epochs)...")
    train_ds = DriveDataset("train", size=96)
    val_ds   = DriveDataset("val",   size=96)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False)

    teacher = TinyUNet(in_ch=3, num_classes=2, base=32).to(device)
    train_teacher(teacher, train_loader, n_epochs=60, lr=1e-3, device=device)

    # --- importance (public-proxy stand-in) + caps ------------------------
    print("\n[2/3] Channel importance (for selection) + caps...")
    importance = compute_importance(teacher, train_loader, device).to(device)
    caps       = collect_caps(teacher, train_loader, device).to(device)
    C = importance.shape[0]
    rank_desc = torch.argsort(importance, descending=True)
    print(f"  C={C}  R={(importance.max()/importance.min()).item():.2f}")

    # --- TRUE single-teacher sensitivity ----------------------------------
    K = 1                                   # honest single teacher
    deltas = torch.full((C,), 2.0 / K, device=device)

    epsilons      = [2.0, 8.0, 16.0]
    keep_fracs    = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02]
    student_seeds = [100, 200, 300, 400, 500]
    base_T        = 32

    results = {
        "C": C, "K_sensitivity": K,
        "R_channel": float(importance.max() / importance.min()),
        "epsilons": epsilons, "keep_fractions": keep_fracs,
        "student_seeds": student_seeds,
        "pate_k10_reference": {"2.0": 0.607, "8.0": 0.628, "16.0": 0.648},
        "sweep": {},
    }

    print("\n[3/3] Sweep keep-fraction × ε × seed (single teacher)...")
    for eps in epsilons:
        rho = eps_to_rho(eps)
        results["sweep"][str(eps)] = {}
        print(f"\n{'='*72}\n  ε={eps}  ρ={rho:.4f}  (single teacher, Δ=2)\n{'='*72}")
        for f in keep_fracs:
            n_active = max(1, int(round(f * C)))
            active_mask = torch.zeros(C, dtype=torch.bool, device=device)
            active_mask[rank_desc[:n_active]] = True
            sigma = thresholded_uniform_sigma(deltas, rho, active_mask)
            sig_active = sigma[active_mask].mean().item()
            print(f"\n  keep={f*100:4.0f}%  n_active={n_active:3d}  "
                  f"σ_active={sig_active:.3f}")

            cache_seed = 42 + int(eps * 10) + int(f * 1000)
            cache = precompute_pruned_cache(teacher, train_ds, caps, sigma,
                                            active_mask, device, seed=cache_seed)
            dices = []
            for s in student_seeds:
                t0 = time.time()
                best, _ = train_student_distill(
                    train_ds, val_loader, cache, device,
                    student_base=16, teacher_base=base_T,
                    n_epochs=40, lr=1e-3, lambda_feat=0.4, seed=s,
                )
                dices.append(best)
                print(f"    seed={s}: Dice={best:.4f}  ({time.time()-t0:.1f}s)")
            results["sweep"][str(eps)][f"{f:.2f}"] = {
                "n_active": n_active,
                "sigma_active": sig_active,
                "dices": dices,
                "mean": float(np.mean(dices)),
                "std":  float(np.std(dices)),
                "sem":  float(np.std(dices) / np.sqrt(len(dices))),
            }
        # compare best single-teacher keep to PATE K=10 at this ε
        best_f = max(results["sweep"][str(eps)].items(),
                     key=lambda kv: kv[1]["mean"])
        pate_ref = results["pate_k10_reference"][str(eps)]
        print(f"\n  ε={eps}: best single-teacher = {best_f[1]['mean']:.4f} "
              f"(keep={best_f[0]})  vs  PATE K=10 = {pate_ref:.4f}  "
              f"(gap {best_f[1]['mean']-pate_ref:+.4f})")

    out_path = Path(__file__).parent / "drive_single_teacher_pruning_results.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote: {out_path}")
    print("\nDecision rule:")
    print("  If best single-teacher keep ≥ PATE K=10 − 0.01 at any ε:")
    print("    → single-teacher pruning is a viable PATE-free path; paper can")
    print("      drop the multi-teacher channel-alignment liability entirely.")
    print("  Else:")
    print("    → pruning helps but PATE's 1/K is needed; keep both, and the")
    print("      paper should fix PATE alignment (shared public init / logit-PATE).")


if __name__ == "__main__":
    main()
