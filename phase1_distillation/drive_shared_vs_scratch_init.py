"""
Shared-init vs scratch-init teachers — does fixing channel alignment help?

THE HYPOTHESIS
--------------
PATE feature aggregation averages K independently-trained teachers'
bottlenecks element-wise. But independently-trained networks have NO
channel correspondence (permutation symmetry): teacher_1's channel-37
and teacher_2's channel-37 mean different things, so element-wise
averaging partially cancels signal. This is the channel-alignment
problem.

If all K teachers start from the SAME initialization and only fine-tune
on their disjoint cohort, their channels stay roughly aligned (no
permutation drift), so averaging should preserve more signal.

THE TEST
--------
Train K=10 teachers two ways and compare student Dice on the identical
PATE sample-once pipeline:

  scratch : each teacher seed = 1000 + k   (current behavior, misaligned)
  shared  : every teacher seed = 1000      (same init, fine-tune apart)

We also report the aggregate's channel-importance ratio R as a proxy
for how much structure survives averaging:
  - if shared gives larger R than scratch → alignment was destroying
    structure, and shared recovers it.

GRID
----
2 init-modes × 3 ε × 5 seeds = 30 student trainings + 2×10 teachers
(~1 GPU-h).

OUTPUT
------
  drive_shared_vs_scratch_init_results.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent))
from drive_local_demo import (
    DriveDataset, TinyUNet, train_teacher, collect_caps, compute_importance,
)
from synthetic_demo import eps_to_rho, clip_and_normalise
from drive_pate_poc import (
    correct_uniform_sigma, partition_dataset, precompute_pate_cache,
)
from drive_student_distill import train_student_distill


def train_K_teachers_init(train_ds, K, device, init_mode,
                          n_epochs=60, lr=1e-3, base=32):
    """
    Train K teachers on disjoint cohorts.
      init_mode='scratch' : seed = 1000 + k   (independent init)
      init_mode='shared'  : seed = 1000       (same init, all teachers)
    """
    partitions = partition_dataset(len(train_ds), K)
    teachers, caps_list = [], []
    for k, idxs in enumerate(partitions):
        seed = 1000 if init_mode == "shared" else 1000 + k
        torch.manual_seed(seed)
        teacher = TinyUNet(in_ch=3, num_classes=2, base=base).to(device)
        subset = Subset(train_ds, idxs)
        bs = min(4, len(idxs))
        train_teacher(teacher,
                      DataLoader(subset, batch_size=bs, shuffle=True),
                      n_epochs=n_epochs, lr=lr, device=device)
        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher.eval()
        caps = collect_caps(teacher,
                            DataLoader(subset, batch_size=bs, shuffle=False),
                            device).to(device)
        teachers.append(teacher); caps_list.append(caps)
    return teachers, caps_list


@torch.no_grad()
def aggregate_importance_R(teachers, caps_list, train_ds, device):
    """Channel-importance ratio R of the (clean) aggregate bottleneck —
    proxy for how much structure survives averaging."""
    loader = DataLoader(train_ds, batch_size=4, shuffle=False)
    energy = None  # per-channel mean squared activation of the aggregate
    n = 0
    for x, _ in loader:
        x = x.to(device)
        bns = []
        for t, caps in zip(teachers, caps_list):
            _, _, e3 = t.encode(x)
            bns.append(clip_and_normalise(e3, caps))
        agg = torch.stack(bns, dim=0).mean(dim=0)        # (B,C,H,W)
        e = agg.pow(2).mean(dim=(0, 2, 3))               # (C,)
        energy = e if energy is None else energy + e
        n += 1
    energy = (energy / n).clamp(min=1e-12)
    return float(energy.max() / energy.min())


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
    epsilons      = [2.0, 8.0, 16.0]
    student_seeds = [100, 200, 300, 400, 500]
    deltas = torch.full((Cb_T,), 2.0 / K, device=device)

    results = {
        "K": K, "epsilons": epsilons, "student_seeds": student_seeds,
        "aggregate_R": {}, "sweep": {},
    }

    for init_mode in ["scratch", "shared"]:
        print(f"\n{'#'*72}\n#  init_mode = {init_mode}\n{'#'*72}")
        teachers, caps_list = train_K_teachers_init(
            train_ds, K, device, init_mode, n_epochs=60)

        R = aggregate_importance_R(teachers, caps_list, train_ds, device)
        results["aggregate_R"][init_mode] = R
        print(f"  aggregate channel-importance R = {R:.2f}  "
              f"({'more structure survived' if R > 2 else 'flattened by averaging'})")

        results["sweep"][init_mode] = {}
        for eps in epsilons:
            rho = eps_to_rho(eps)
            sigma = correct_uniform_sigma(deltas, rho)
            cache = precompute_pate_cache(teachers, caps_list, train_ds,
                                          sigma, device,
                                          seed=42 + int(eps * 10))
            dices = []
            for s in student_seeds:
                t0 = time.time()
                d, _ = train_student_distill(
                    train_ds, val_loader, cache, device,
                    student_base=16, teacher_base=base_T,
                    n_epochs=40, lr=1e-3, lambda_feat=0.4, seed=s)
                dices.append(d)
            m = float(np.mean(dices))
            results["sweep"][init_mode][str(eps)] = {
                "dices": dices, "mean": m,
                "std": float(np.std(dices)),
                "sem": float(np.std(dices) / np.sqrt(len(dices))),
            }
            print(f"  ε={eps:>4}: Dice={m:.4f} ± {np.std(dices):.4f}")

    # paired comparison
    print("\nShared − scratch (paired over seeds):")
    for eps in epsilons:
        sc = np.array(results["sweep"]["scratch"][str(eps)]["dices"])
        sh = np.array(results["sweep"]["shared"][str(eps)]["dices"])
        d = sh - sc
        t = d.mean() / max(d.std() / np.sqrt(len(d)), 1e-12)
        print(f"  ε={eps:>4}: Δ={d.mean():+.4f}  t={t:.2f}")

    out = Path(__file__).parent / "drive_shared_vs_scratch_init_results.json"
    with out.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote: {out}")
    print("\nKEY: if shared > scratch (Δ>0, t>2) → channel alignment WAS a")
    print("  bottleneck and shared-init fixes it (a free method improvement).")
    print("  Compare aggregate_R too: shared should be larger if structure")
    print("  survives averaging better.")


if __name__ == "__main__":
    main()
