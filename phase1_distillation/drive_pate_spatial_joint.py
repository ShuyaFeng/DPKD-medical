"""
EXP-5: PATE K=5 × spatial-WF — multiplicative composition.

PURPOSE
-------
Compose two orthogonal mechanism-level interventions:
  - PATE on the sample axis: K teachers reduce sensitivity Δ→2/K
  - spatial-WF on the spatial axis: reshape σ_{i,j} by saliency

If both work alone, this should give multiplicative (not additive)
benefit because they touch independent dimensions of the noise tensor.

ONLY RUN THIS IF EXP-4 (drive_spatial_wf) shows spatial-WF gives a
significant lift over uniform. Otherwise wasted compute.

GRID
----
2 conditions × 3 ε × 5 seeds = 30 student trainings.

  P_uniform.   PATE K=5 + uniform noise on aggregate    (baseline = drive_pate_5seed result)
  P_spatial.   PATE K=5 + spatial-WF on aggregate       (NEW)

DEPENDS ON
----------
- spatial_saliency.pt   (from EXP-3)
- The drive_pate_poc helper functions
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from drive_local_demo import DriveDataset
from synthetic_demo import (
    eps_to_rho, waterfilling_sigma, uniform_sigma,
    clip_and_normalise, denormalise,
)
from drive_pate_poc import (
    correct_uniform_sigma, partition_dataset, train_K_teachers,
)
from drive_student_distill import train_student_distill


def precompute_pate_cache_with_sigma(teachers, caps_list, train_ds,
                                       sigma, device, seed, broadcast):
    """
    Same idea as precompute_pate_cache, but takes an arbitrary sigma
    and broadcast mode (channel | spatial | joint).
    """
    K = len(teachers)
    g = torch.Generator(device=device).manual_seed(seed)
    cache = {}
    for t in teachers: t.eval()
    with torch.no_grad():
        for i in range(len(train_ds)):
            x, _ = train_ds[i]
            x = x.unsqueeze(0).to(device)
            # average K teacher bottlenecks (clipped + normalised), each clipped to its own caps
            agg = None
            for k_idx, (t, caps) in enumerate(zip(teachers, caps_list)):
                z = t.encoder(x) if hasattr(t, "encoder") else t(x, return_bottleneck=True)
                z_norm = clip_and_normalise(z, caps.to(device))
                agg = z_norm if agg is None else agg + z_norm
            agg = agg / K
            B, C, H, W = agg.shape
            noise_base = torch.randn(B, C, H, W, generator=g, device=device)
            if broadcast == "channel":
                noise = noise_base * sigma.view(1, C, 1, 1)
            elif broadcast == "spatial":
                noise = noise_base * sigma.view(1, 1, H, W)
            elif broadcast == "joint":
                noise = noise_base * sigma.view(1, C, H, W)
            else:
                raise ValueError(broadcast)
            # de-normalise using mean caps across K teachers (post-processing)
            mean_caps = torch.stack([c.to(device) for c in caps_list]).mean(dim=0)
            z_noisy = denormalise(agg + noise, mean_caps)
            cache[i] = z_noisy[0].detach().cpu().clone()
    return cache


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)
    print(f"Device: {device}")

    sal_path = Path(__file__).parent / "spatial_saliency.pt"
    if not sal_path.exists():
        raise FileNotFoundError("Run drive_spatial_saliency.py first to produce "
                                f"{sal_path}")
    sal_blob = torch.load(sal_path)
    spatial_importance = sal_blob["saliency"].to(device)
    H_b, W_b = sal_blob["H_b"], sal_blob["W_b"]
    print(f"  spatial saliency: ({H_b},{W_b})  R_spatial={sal_blob['R_spatial']:.2f}")

    train_ds = DriveDataset("train", size=96)
    val_ds   = DriveDataset("val",   size=96)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)
    N_train = len(train_ds)

    K = 5
    epsilons      = [2.0, 8.0, 16.0]
    student_seeds = [100, 200, 300, 400, 500]
    base_T        = 32
    Cb_T          = base_T * 4
    HW            = H_b * W_b

    print(f"\n[1/3] Train K={K} teachers on disjoint partitions...")
    partitions = partition_dataset(N_train, K)
    print(f"  partition sizes: {[len(p) for p in partitions]}")
    teachers, caps_list = train_K_teachers(train_ds, K, device, n_epochs=60)

    # Sensitivity halves with K (PATE) AND lives on the unit-norm bottleneck.
    deltas_channel = torch.full((Cb_T,), 2.0 / K, device=device)
    deltas_spatial = torch.full((HW,),   (2.0 / K) * np.sqrt(Cb_T), device=device)

    results = {
        "K": K, "C": Cb_T, "H_b": H_b, "W_b": W_b,
        "R_spatial": sal_blob["R_spatial"],
        "epsilons": epsilons, "student_seeds": student_seeds,
        "conditions": ["P_uniform", "P_spatial_WF"],
        "sweep": {},
    }

    for eps in epsilons:
        rho = eps_to_rho(eps)
        print(f"\n{'='*72}\n  ε={eps}  ρ={rho:.4f}  (K={K} so sensitivity is 2/K)\n{'='*72}")

        sigma_U = correct_uniform_sigma(deltas_channel, rho)
        sigma_S = spatial_importance.flatten()
        sigma_S = waterfilling_sigma(deltas_spatial, sigma_S, rho).view(H_b, W_b)

        print(f"  σ_U (PATE+uniform)    mean={sigma_U.mean():.3f}")
        print(f"  σ_S (PATE+spatial-WF) min={sigma_S.min():.3f}  max={sigma_S.max():.3f}")

        results["sweep"][str(eps)] = {}
        for cond, (sigma, broadcast) in [
            ("P_uniform",    (sigma_U, "channel")),
            ("P_spatial_WF", (sigma_S, "spatial")),
        ]:
            print(f"\n  --- {cond} ---")
            cache_seed = 42 + int(eps * 10) + hash(cond) % 1000
            cache = precompute_pate_cache_with_sigma(
                teachers, caps_list, train_ds,
                sigma, device, seed=cache_seed, broadcast=broadcast,
            )
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
            results["sweep"][str(eps)][cond] = {
                "dices": dices,
                "mean": float(np.mean(dices)),
                "std":  float(np.std(dices)),
                "sem":  float(np.std(dices) / np.sqrt(len(dices))),
            }

        # paired test
        U_a = np.array(results["sweep"][str(eps)]["P_uniform"]["dices"])
        S_a = np.array(results["sweep"][str(eps)]["P_spatial_WF"]["dices"])
        d = S_a - U_a
        t_stat = d.mean() / max(d.std()/np.sqrt(len(d)), 1e-12)
        print(f"  paired PATE+spatial vs PATE+uniform: "
              f"Δ={d.mean():+.4f}  t={t_stat:.2f}")
        results["sweep"][str(eps)]["paired_PATE_spatial_vs_PATE_uniform"] = {
            "mean_diff": float(d.mean()),
            "sem":       float(d.std() / np.sqrt(len(d))),
            "t_stat":    float(t_stat),
        }

    out_path = Path(__file__).parent / "drive_pate_spatial_joint_results.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote: {out_path}")
    print("\nWHAT TO REPORT IN PAPER:")
    print("  Best (ε, cond) combination, paired t-stat over 5 seeds.")
    print("  If consistent positive Δ across ε → §3.5 'Joint composition' contribution.")


if __name__ == "__main__":
    main()
