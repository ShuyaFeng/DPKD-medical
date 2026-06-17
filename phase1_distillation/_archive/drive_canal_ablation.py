"""
EXP-2: CANAL rescue ablation — isolate WF's marginal value on pruned active set.

THE QUESTION
------------
M5 reports that "top-10%-pruning + WF" lifts Dice by +0.095 over
"full-1024 + uniform". But that lift conflates THREE effects:
  (i) dimension reduction (σ per active channel ∝ √C/C_active),
  (ii) importance-based selection (kept channels carry more signal),
  (iii) WF allocation on the active subset.

This script isolates (iii) by adding the missing baseline:
   "top-10% + UNIFORM" (the control E in the design table).

Then the marginal value of CANAL is exactly F − E.

GRID
----
4 conditions × 3 ε × 5 seeds = 60 student trainings.

  A. full 1024 channels + uniform noise          (vanilla DP baseline)
  B. full 1024 channels + WF allocation          (channel-WF only, known to fail)
  E. top 10% channels + uniform noise            (pruning only — THE NEW CONTROL)
  F. top 10% channels + WF on active set         (pruning + WF, M5 result)

PAPER OUTPUT
------------
Build table:

  vanilla DP       (A) ε=2 / 8 / 16
  channel-WF only  (B) ε=2 / 8 / 16
  pruning only     (E) ε=2 / 8 / 16   <- ISOLATES dim+selection effect
  pruning + WF     (F) ε=2 / 8 / 16   <- M5 (vs A) and CANAL-rescue (vs E)
  --
  Δ(F − E)            <- IF significant, CANAL has marginal value on active set

OUTPUT FILE
-----------
  drive_canal_ablation_results.json
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
from drive_wf_threshold import thresholded_wf_sigma


# ---------------------------------------------------------------------------
# Cache builders for each of the 4 conditions
# ---------------------------------------------------------------------------

def build_cache(teacher, train_ds, caps, sigma, device, seed,
                active_mask=None, keep_fraction=1.0):
    """
    Run teacher on every training image once, add noise, store.

    sigma         : (C,) per-channel noise stddev (already accounts for
                    active subset budget)
    active_mask   : (C,) boolean. False channels are released as 0 in
                    normalised space (i.e. pure noise, no signal).
    keep_fraction : record-only metadata.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    cache = {}
    teacher.eval()
    with torch.no_grad():
        for i in range(len(train_ds)):
            x, y = train_ds[i]
            x = x.unsqueeze(0).to(device)
            z = teacher.encoder(x) if hasattr(teacher, "encoder") else teacher(x, return_bottleneck=True)
            B, C, H, W = z.shape
            z_norm = clip_and_normalise(z, caps)
            # Build noise; if a channel is inactive, set its signal to 0 too
            if active_mask is not None:
                z_norm = z_norm * active_mask.view(1, C, 1, 1).float()
            noise = torch.randn(B, C, H, W, generator=g, device=device) * sigma.view(1, C, 1, 1)
            z_noisy = denormalise(z_norm + noise, caps)
            cache[i] = z_noisy[0].detach().cpu().clone()
    return cache


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)
    print(f"Device: {device}")

    # --- data + teacher --------------------------------------------------
    print("\n[1/4] DRIVE + teacher (60 epochs)...")
    train_ds = DriveDataset("train", size=96)
    val_ds   = DriveDataset("val",   size=96)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False)

    teacher = TinyUNet(in_ch=3, num_classes=2, base=32).to(device)
    train_teacher(teacher, train_loader, n_epochs=60, lr=1e-3, device=device)

    # --- importance + caps -----------------------------------------------
    print("\n[2/4] Importance + caps...")
    importance = compute_importance(teacher, train_loader, device).to(device)
    caps       = collect_caps(teacher, train_loader, device).to(device)
    C = importance.shape[0]
    print(f"  C={C}  R_global={(importance.max()/importance.min()).item():.2f}")

    # --- active set: top 10% by importance --------------------------------
    keep_fraction = 0.10
    n_active = max(1, int(round(keep_fraction * C)))
    rank_desc = torch.argsort(importance, descending=True)
    active_mask = torch.zeros(C, dtype=torch.bool, device=device)
    active_mask[rank_desc[:n_active]] = True

    R_active = float(importance[active_mask].max() / importance[active_mask].min())
    print(f"  Active set top-{keep_fraction*100:.0f}% : n={n_active}, R_active={R_active:.2f}")

    K = 10                                          # teachers count for PATE-style Δ
    deltas = torch.full((C,), 2.0 / K, device=device)

    # --- the experiment grid ---------------------------------------------
    epsilons      = [2.0, 8.0, 16.0]
    student_seeds = [100, 200, 300, 400, 500]
    base_T        = 32

    results = {
        "C": C, "K": K, "keep_fraction": keep_fraction, "n_active": n_active,
        "R_global": float(importance.max() / importance.min()),
        "R_active_subset": R_active,
        "epsilons": epsilons, "student_seeds": student_seeds,
        "conditions": ["A_full_uniform", "B_full_WF",
                       "E_top10_uniform", "F_top10_WF"],
        "sweep": {},
    }

    for eps in epsilons:
        rho = eps_to_rho(eps)
        print(f"\n{'='*72}\n  ε={eps}  ρ={rho:.4f}\n{'='*72}")

        # --- compute σ for each condition (privacy-fair at same ρ) -------
        sigma_A = uniform_sigma(deltas, rho)
        sigma_B = waterfilling_sigma(deltas, importance, rho)
        # E & F: only spend budget on active subset
        sigma_E = torch.zeros(C, device=device)
        sigma_E[active_mask] = uniform_sigma(deltas[active_mask], rho)
        sigma_F = thresholded_wf_sigma(deltas, importance, rho, active_mask)

        print(f"  σ_A (full+unif)   mean={sigma_A.mean():.3f}")
        print(f"  σ_B (full+WF)     min={sigma_B.min():.3f}  max={sigma_B.max():.3f}")
        print(f"  σ_E (top10+unif)  σ_active={sigma_E[active_mask].mean():.3f}  σ_inactive=0(masked)")
        print(f"  σ_F (top10+WF)    σ_active min={sigma_F[active_mask].min():.3f}  "
              f"max={sigma_F[active_mask].max():.3f}")

        results["sweep"][str(eps)] = {}
        for cond, (sigma, mask) in [
            ("A_full_uniform",   (sigma_A, None)),
            ("B_full_WF",        (sigma_B, None)),
            ("E_top10_uniform",  (sigma_E, active_mask)),
            ("F_top10_WF",       (sigma_F, active_mask)),
        ]:
            print(f"\n  --- {cond} ---")
            cache_seed = 42 + int(eps * 10) + hash(cond) % 1000
            cache = build_cache(teacher, train_ds, caps, sigma, device,
                                seed=cache_seed,
                                active_mask=mask,
                                keep_fraction=keep_fraction if mask is not None else 1.0)
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

        # --- paired tests at this ε ---------------------------------------
        A = np.array(results["sweep"][str(eps)]["A_full_uniform"]["dices"])
        E = np.array(results["sweep"][str(eps)]["E_top10_uniform"]["dices"])
        F = np.array(results["sweep"][str(eps)]["F_top10_WF"]["dices"])
        for name, x, y in [
            ("F-A (M5+CANAL vs vanilla)", F, A),
            ("F-E (CANAL marginal value on active set, THE KEY)", F, E),
            ("E-A (pruning-only vs vanilla)", E, A),
        ]:
            diff = x - y
            mean_diff = float(diff.mean())
            sem_diff  = float(diff.std() / np.sqrt(len(diff)))
            t_stat    = mean_diff / max(sem_diff, 1e-12)
            results["sweep"][str(eps)][f"paired_{name.split()[0]}"] = {
                "mean_diff": mean_diff,
                "sem":       sem_diff,
                "t_stat":    t_stat,
            }
            print(f"  paired {name}: Δ={mean_diff:+.4f}  sem={sem_diff:.4f}  t={t_stat:.2f}")

    out_path = Path(__file__).parent / "drive_canal_ablation_results.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote: {out_path}")
    print("\nDecision rule:")
    print("  IF any ε shows F−E paired t > 2.0  AND  Δ > +0.005:")
    print("    CANAL has marginal value on active set; include in paper §3.3")
    print("  ELSE:")
    print("    CANAL = ceiling analysis only; M5 alone is the contribution")


if __name__ == "__main__":
    main()
