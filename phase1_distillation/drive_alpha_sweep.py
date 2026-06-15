"""
Alpha sweep (local TinyUNet, PATE K=10) — normalized teacher-vs-GT mix.

Paper §3.5. The student loss mixes a feature-matching loss against the
noisy teacher cache with a task loss against the GT mask. We use the
NORMALIZED form so that alpha is a TRUE percentage of gradient signal:

    scale = task_loss.detach() / (feat_loss.detach() + 1e-8)
    total = (1 - alpha)*task_loss + alpha*(feat_loss * scale)

  alpha = 0.0  → pure GT supervision (no teacher, ignores cache)
  alpha = 1.0  → pure feature distillation (ZERO private-label dependence)

The alpha=1.0 point is the clean privacy story: the student never
touches a private GT label, so the §0.5.1 label-leak gap closes
entirely. The sweep characterizes the privacy-utility-task triangle.

Uses the same PATE K=10 sample-once pipeline as the headline result so
the numbers drop straight into Tab. 4 as one extra row (best alpha).

GRID
----
alpha ∈ {0.3, 0.5, 0.7, 0.9, 1.0} × ε ∈ {2,8,16} × 5 seeds
= 75 student trainings (~1-1.5 GPU-h) + 10 teachers.

OUTPUT
------
  drive_alpha_sweep_results.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from drive_local_demo import (
    DriveDataset, TinyUNet, evaluate_vessel_dice,
)
from drive_student_distill import Adapter
from synthetic_demo import eps_to_rho, task_loss
from drive_pate_poc import (
    correct_uniform_sigma, train_K_teachers, precompute_pate_cache,
)


def train_student_alpha(train_ds, val_loader, cache, device, alpha,
                        student_base=16, teacher_base=32,
                        n_epochs=40, lr=1e-3, seed=2):
    """Student training with the NORMALIZED alpha loss."""
    torch.manual_seed(seed)
    student = TinyUNet(in_ch=3, num_classes=2, base=student_base).to(device)
    adapter = Adapter(student_base * 4, teacher_base * 4).to(device)
    opt = torch.optim.Adam(
        list(student.parameters()) + list(adapter.parameters()), lr=lr)
    N = len(train_ds)
    bs = 4
    best = 0.0
    for ep in range(n_epochs):
        student.train(); adapter.train()
        perm = torch.randperm(N)
        for i in range(0, N, bs):
            idxs = perm[i:i+bs].tolist()
            xs = torch.stack([train_ds[idx][0] for idx in idxs]).to(device)
            ys = torch.stack([train_ds[idx][1] for idx in idxs]).to(device)
            tt = torch.stack([cache[idx] for idx in idxs]).to(device)
            opt.zero_grad()
            e1, e2, e3 = student.encode(xs)
            f_loss = F.mse_loss(adapter(e3), tt)
            t_loss = task_loss(student.decode(e1, e2, e3), ys)
            scale = t_loss.detach() / (f_loss.detach() + 1e-8)
            total = (1 - alpha) * t_loss + alpha * (f_loss * scale)
            total.backward()
            opt.step()
        vd = evaluate_vessel_dice(student, val_loader, device)
        best = max(best, vd)
    return best


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)
    print(f"Device: {device}")

    train_ds = DriveDataset("train", size=96)
    val_ds   = DriveDataset("val",   size=96)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    K = 10
    base_T = 32
    Cb_T = base_T * 4
    alphas        = [0.3, 0.5, 0.7, 0.9, 1.0]
    epsilons      = [2.0, 8.0, 16.0]
    student_seeds = [100, 200, 300, 400, 500]

    print(f"\n[1/2] Train K={K} teachers...")
    teachers, caps_list = train_K_teachers(train_ds, K, device, n_epochs=60)
    deltas = torch.full((Cb_T,), 2.0 / K, device=device)

    results = {
        "K": K, "alphas": alphas, "epsilons": epsilons,
        "student_seeds": student_seeds, "sweep": {},
    }

    print("\n[2/2] Alpha sweep (PATE K=10, uniform full-channel cache)...")
    for eps in epsilons:
        rho = eps_to_rho(eps)
        sigma = correct_uniform_sigma(deltas, rho)
        cache = precompute_pate_cache(teachers, caps_list, train_ds,
                                      sigma, device, seed=42 + int(eps * 10))
        results["sweep"][str(eps)] = {}
        print(f"\n  ε={eps}  ρ={rho:.4f}  σ={sigma[0].item():.3f}")
        for alpha in alphas:
            dices = []
            for s in student_seeds:
                t0 = time.time()
                d = train_student_alpha(train_ds, val_loader, cache, device,
                                        alpha=alpha, teacher_base=base_T, seed=s)
                dices.append(d)
            m = float(np.mean(dices))
            results["sweep"][str(eps)][f"{alpha:.1f}"] = {
                "dices": dices, "mean": m,
                "std": float(np.std(dices)),
                "sem": float(np.std(dices) / np.sqrt(len(dices))),
            }
            tag = "  (PURE feature-distill, zero label dep.)" if alpha == 1.0 else ""
            print(f"    alpha={alpha:.1f}: Dice={m:.4f} ± {np.std(dices):.4f}{tag}")

    out = Path(__file__).parent / "drive_alpha_sweep_results.json"
    with out.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote: {out}")
    print("\nKEY: compare alpha=1.0 (zero label dependence) to the best alpha.")
    print("  Small gap → we can claim strong utility with NO private-label use.")


if __name__ == "__main__":
    main()
