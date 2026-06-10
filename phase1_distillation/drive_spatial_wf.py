"""
EXP-4: Spatial water-filling on student feature distillation.

PURPOSE
-------
Test whether Theorem 1's WF allocation, instantiated on the SPATIAL
axis (per-pixel σ_{i,j}) instead of the channel axis, gives an
empirical Dice lift on the student.

Channel-WF failed because R_channel ≈ 1.96 on DRIVE. Spatial saliency
should have a much bigger R because vessel pixels concentrate
information very differently from background pixels.

GRID
----
3 conditions × 3 ε × 5 seeds = 45 student trainings.

  U.  full + uniform spatial noise              (baseline)
  S.  full + spatial-WF                          (NEW — the test)
  J.  full + joint channel × spatial WF         (combination)

DEPENDS ON
----------
spatial_saliency.pt produced by drive_spatial_saliency.py (EXP-3).

OUTPUT
------
  drive_spatial_wf_results.json
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
    DriveDataset, TinyUNet, train_teacher, compute_importance,
    collect_caps,
)
from synthetic_demo import (
    eps_to_rho, waterfilling_sigma, uniform_sigma,
    clip_and_normalise, denormalise,
)
from drive_student_distill import train_student_distill


# ---------------------------------------------------------------------------
# Spatial-axis water-filling. Same math as channel-WF, different importance.
# ---------------------------------------------------------------------------

def spatial_waterfilling_sigma(deltas_spatial: torch.Tensor,
                                spatial_importance: torch.Tensor,
                                rho: float) -> torch.Tensor:
    """
    deltas_spatial : (H_b*W_b,)  sensitivity per spatial position
    spatial_importance : (H_b*W_b,)  importance s_{ij}
    rho : scalar zCDP budget
    Returns sigma per spatial position, shape (H_b*W_b,).
    Identical formula to waterfilling_sigma, just reused on spatial axis.
    """
    return waterfilling_sigma(deltas_spatial, spatial_importance, rho)


def joint_channel_spatial_wf_sigma(deltas_per_position: torch.Tensor,
                                    channel_importance: torch.Tensor,
                                    spatial_importance: torch.Tensor,
                                    rho: float) -> torch.Tensor:
    """
    Joint allocation: importance is the outer product s_c × s_{ij}.
    Returns sigma_{c, i, j}, shape (C, H_b, W_b).
    """
    C = channel_importance.shape[0]
    HW = spatial_importance.numel()
    joint_imp = (channel_importance.view(C, 1) *
                 spatial_importance.view(1, HW)).flatten()
    deltas_flat = deltas_per_position.flatten()
    sigma_flat = waterfilling_sigma(deltas_flat, joint_imp, rho)
    return sigma_flat.view(C, *spatial_importance.shape)


def build_spatial_cache(teacher, train_ds, caps, sigma, device, seed,
                         broadcast: str):
    """
    broadcast : "channel" -> sigma is (C,) broadcast over spatial
                "spatial" -> sigma is (H_b, W_b) broadcast over channels
                "joint"   -> sigma is (C, H_b, W_b), no broadcast
    """
    g = torch.Generator(device=device).manual_seed(seed)
    cache = {}
    teacher.eval()
    with torch.no_grad():
        for i in range(len(train_ds)):
            x, _ = train_ds[i]
            x = x.unsqueeze(0).to(device)
            z = teacher.encoder(x) if hasattr(teacher, "encoder") else teacher(x, return_bottleneck=True)
            B, C, H, W = z.shape
            z_norm = clip_and_normalise(z, caps)
            noise_base = torch.randn(B, C, H, W, generator=g, device=device)
            if broadcast == "channel":
                noise = noise_base * sigma.view(1, C, 1, 1)
            elif broadcast == "spatial":
                noise = noise_base * sigma.view(1, 1, H, W)
            elif broadcast == "joint":
                noise = noise_base * sigma.view(1, C, H, W)
            else:
                raise ValueError(broadcast)
            z_noisy = denormalise(z_norm + noise, caps)
            cache[i] = z_noisy[0].detach().cpu().clone()
    return cache


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)
    print(f"Device: {device}")

    # --- load spatial saliency from EXP-3 ----------------------------------
    sal_path = Path(__file__).parent / "spatial_saliency.pt"
    if not sal_path.exists():
        raise FileNotFoundError(
            f"{sal_path} not found. Run drive_spatial_saliency.py first."
        )
    sal_blob = torch.load(sal_path)
    spatial_importance = sal_blob["saliency"].to(device)         # (H_b, W_b)
    H_b, W_b = sal_blob["H_b"], sal_blob["W_b"]
    R_spatial = sal_blob["R_spatial"]
    print(f"  spatial saliency loaded: shape ({H_b},{W_b})  R_spatial={R_spatial:.2f}")

    # --- data + teacher ----------------------------------------------------
    print("\n[1/3] DRIVE + teacher (60 epochs)...")
    train_ds = DriveDataset("train", size=96)
    val_ds   = DriveDataset("val",   size=96)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False)

    teacher = TinyUNet(in_ch=3, num_classes=2, base=32).to(device)
    train_teacher(teacher, train_loader, n_epochs=60, lr=1e-3, device=device)

    # --- importance + caps -------------------------------------------------
    print("\n[2/3] Channel importance + caps...")
    channel_importance = compute_importance(teacher, train_loader, device).to(device)
    caps               = collect_caps(teacher, train_loader, device).to(device)
    C = channel_importance.shape[0]
    print(f"  C={C}  R_channel={(channel_importance.max()/channel_importance.min()).item():.2f}")
    assert spatial_importance.numel() == H_b * W_b
    HW = H_b * W_b

    K = 10
    # per-channel sensitivity Δ = 2/K (replace-one in unit-norm space)
    # When σ is per-pixel only, total ρ = Σ_{i,j} (C·Δ²)/(2σ²_{ij}) etc.
    # We absorb C in the deltas vector for spatial-only and joint cases.
    deltas_channel  = torch.full((C,),    2.0 / K, device=device)
    deltas_spatial  = torch.full((HW,),   (2.0 / K) * np.sqrt(C), device=device)  # absorb C-summing
    deltas_joint    = torch.full((C, H_b, W_b), 2.0 / K, device=device)

    # --- grid --------------------------------------------------------------
    epsilons      = [2.0, 8.0, 16.0]
    student_seeds = [100, 200, 300, 400, 500]
    base_T        = 32

    results = {
        "C": C, "K": K, "H_b": H_b, "W_b": W_b,
        "R_channel": float(channel_importance.max() / channel_importance.min()),
        "R_spatial": R_spatial,
        "epsilons": epsilons, "student_seeds": student_seeds,
        "conditions": ["U_uniform", "S_spatial_WF", "J_joint_WF"],
        "sweep": {},
    }

    for eps in epsilons:
        rho = eps_to_rho(eps)
        print(f"\n{'='*72}\n  ε={eps}  ρ={rho:.4f}\n{'='*72}")

        sigma_U = uniform_sigma(deltas_channel, rho)
        sigma_S = spatial_waterfilling_sigma(deltas_spatial,
                                              spatial_importance.flatten(),
                                              rho).view(H_b, W_b)
        sigma_J = joint_channel_spatial_wf_sigma(deltas_joint,
                                                  channel_importance,
                                                  spatial_importance,
                                                  rho)
        print(f"  σ_U (uniform)        mean={sigma_U.mean():.3f}")
        print(f"  σ_S (spatial-WF)     min={sigma_S.min():.3f}  max={sigma_S.max():.3f}")
        print(f"  σ_J (joint WF)       min={sigma_J.min():.3f}  max={sigma_J.max():.3f}")

        results["sweep"][str(eps)] = {}
        for cond, (sigma, broadcast) in [
            ("U_uniform",     (sigma_U, "channel")),
            ("S_spatial_WF",  (sigma_S, "spatial")),
            ("J_joint_WF",    (sigma_J, "joint")),
        ]:
            print(f"\n  --- {cond} ---")
            cache_seed = 42 + int(eps * 10) + hash(cond) % 1000
            cache = build_spatial_cache(teacher, train_ds, caps, sigma, device,
                                         seed=cache_seed, broadcast=broadcast)
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

        # paired tests at this eps
        U_a = np.array(results["sweep"][str(eps)]["U_uniform"]["dices"])
        S_a = np.array(results["sweep"][str(eps)]["S_spatial_WF"]["dices"])
        J_a = np.array(results["sweep"][str(eps)]["J_joint_WF"]["dices"])
        for name, x, y in [("S-U (spatial vs uniform)", S_a, U_a),
                            ("J-U (joint vs uniform)", J_a, U_a),
                            ("J-S (joint vs spatial-only)", J_a, S_a)]:
            d = x - y
            t_stat = d.mean() / max(d.std()/np.sqrt(len(d)), 1e-12)
            print(f"  paired {name}: Δ={d.mean():+.4f}  t={t_stat:.2f}")
            results["sweep"][str(eps)][f"paired_{name.split()[0]}"] = {
                "mean_diff": float(d.mean()),
                "sem":       float(d.std() / np.sqrt(len(d))),
                "t_stat":    float(t_stat),
            }

    out_path = Path(__file__).parent / "drive_spatial_wf_results.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote: {out_path}")
    print("\nDecision rule:")
    print("  IF S-U paired t>2 AND Δ>+0.005 at any ε:")
    print("    spatial-WF works -> proceed to EXP-5 (PATE × spatial-WF joint)")
    print("  ELSE:")
    print("    spatial-WF flat on student. Stop here.")


if __name__ == "__main__":
    main()
