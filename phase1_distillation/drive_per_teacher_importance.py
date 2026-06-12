"""
Per-teacher importance vs shared importance for channel-pruned PATE.

THE REALISTIC SETTING
---------------------
In a true federated setup each hospital (teacher) trains independently
and judges channel importance on its OWN data. There is no central
public importance. This script tests what happens when every teacher
brings its own importance ranking instead of a single shared mask.

WHAT THIS REALLY MEASURES
-------------------------
The overlap between independently-trained teachers' top-k channel sets
is a DIRECT measure of the channel-alignment problem:
  - high overlap  → importances agree → channels roughly aligned
  - low overlap   → each teacher cares about different channels →
                    permutation/alignment problem is severe

We report the pairwise Jaccard overlap of teacher masks AND the Dice
under three ways of reconciling per-teacher masks into one released
channel set:

  shared : top-k of the AVERAGED importance (|mask| = k·C)        [baseline]
  union  : ∪ of per-teacher top-k masks    (|mask| ≥ k·C, MORE noise)
  vote   : channels picked by ≥⌈K/2⌉ teachers (|mask| ≤ k·C, LESS noise)

All three aggregate the SAME K teacher bottlenecks (mean of normalised),
then keep only the chosen channel set; σ is uniform over the active set
with Δ=2/K, so each strategy is accounted honestly at its own |mask|.

GRID
----
K∈{5,10} × keep_target∈{0.1,0.2} × strategy∈{shared,union,vote} ×
ε∈{2,8,16} × 5 seeds = 180 trainings (~2-3 GPU-h) + 1+5+10 teachers.
Mask-overlap analysis is free (no training).

OUTPUT
------
  drive_per_teacher_importance_results.json
"""

import json
import sys
import time
from itertools import combinations
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


def per_teacher_masks(teachers, train_loader, device, keep_frac, C):
    """Each teacher's own top-(keep_frac·C) channel mask. Returns
    (list_of_bool_masks, averaged_importance)."""
    n_active = max(1, int(round(keep_frac * C)))
    masks = []
    imps = []
    for t in teachers:
        imp = compute_importance(t, train_loader, device)
        imps.append(imp)
        rank = torch.argsort(imp, descending=True)
        m = torch.zeros(C, dtype=torch.bool, device=device)
        m[rank[:n_active]] = True
        masks.append(m)
    avg_imp = torch.stack(imps, dim=0).mean(dim=0)
    return masks, avg_imp, n_active


def mask_jaccard(masks):
    """Mean pairwise Jaccard overlap of a list of boolean masks."""
    if len(masks) < 2:
        return 1.0
    vals = []
    for a, b in combinations(masks, 2):
        inter = (a & b).sum().item()
        union = (a | b).sum().item()
        vals.append(inter / max(union, 1))
    return float(np.mean(vals))


def reconcile(masks, avg_imp, n_active, strategy, C, device):
    """Build the released channel mask from per-teacher masks."""
    if strategy == "shared":
        rank = torch.argsort(avg_imp, descending=True)
        m = torch.zeros(C, dtype=torch.bool, device=device)
        m[rank[:n_active]] = True
        return m
    if strategy == "union":
        m = torch.zeros(C, dtype=torch.bool, device=device)
        for mk in masks:
            m |= mk
        return m
    if strategy == "vote":
        votes = torch.stack([mk.int() for mk in masks], dim=0).sum(dim=0)
        thresh = (len(masks) + 1) // 2          # ceil(K/2)
        return votes >= thresh
    raise ValueError(strategy)


def thresholded_uniform_sigma(deltas, rho, active_mask):
    sigma = torch.zeros_like(deltas)
    if active_mask.any():
        sigma[active_mask] = uniform_sigma(deltas[active_mask], rho)
    return sigma


@torch.no_grad()
def precompute_cache(teachers, caps_list, train_ds, sigma, active_mask,
                     device, seed):
    for t in teachers:
        t.eval()
    g = torch.Generator(device=device).manual_seed(seed)
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
        agg = agg / K
        B, Cc, H, W = agg.shape
        agg = agg * active_mask.view(1, Cc, 1, 1).float()
        noise = torch.randn(B, Cc, H, W, generator=g, device=device) * sigma.view(1, Cc, 1, 1)
        z_noisy = denormalise(agg + noise, mean_caps)
        for b in range(B):
            cache.append(z_noisy[b].detach().cpu())
    return cache


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)
    print(f"Device: {device}")

    train_ds = DriveDataset("train", size=96)
    val_ds   = DriveDataset("val",   size=96)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=False)
    val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False)
    N_train = len(train_ds)

    K_values      = [5, 10]
    keep_targets  = [0.1, 0.2]
    strategies    = ["shared", "union", "vote"]
    epsilons      = [2.0, 8.0, 16.0]
    student_seeds = [100, 200, 300, 400, 500]
    base_T        = 32
    Cb_T          = base_T * 4

    results = {
        "K_values": K_values, "keep_targets": keep_targets,
        "strategies": strategies, "epsilons": epsilons,
        "student_seeds": student_seeds, "C": Cb_T,
        "mask_overlap": {},   # K → keep → mean pairwise Jaccard
        "sweep": {},
    }

    for K in K_values:
        print(f"\n{'#'*72}\n#  K = {K}\n{'#'*72}")
        partition_dataset(N_train, K)
        teachers, caps_list = train_K_teachers(train_ds, K, device, n_epochs=60)
        deltas = torch.full((Cb_T,), 2.0 / K, device=device)

        results["mask_overlap"][str(K)] = {}
        results["sweep"][str(K)] = {}

        for keep in keep_targets:
            masks, avg_imp, n_active = per_teacher_masks(
                teachers, train_loader, device, keep, Cb_T)
            jac = mask_jaccard(masks)
            results["mask_overlap"][str(K)][f"{keep:.2f}"] = jac
            print(f"\n  keep_target={keep*100:.0f}%  "
                  f"per-teacher mask Jaccard overlap = {jac:.3f}  "
                  f"({'HIGH=aligned' if jac > 0.5 else 'LOW=misaligned'})")

            results["sweep"][str(K)][f"{keep:.2f}"] = {}
            for strat in strategies:
                rel_mask = reconcile(masks, avg_imp, n_active, strat, Cb_T, device)
                n_rel = int(rel_mask.sum().item())
                print(f"\n    strategy={strat:6s}  released channels={n_rel:3d}")
                results["sweep"][str(K)][f"{keep:.2f}"][strat] = {
                    "n_released": n_rel, "eps": {},
                }
                for eps in epsilons:
                    rho = eps_to_rho(eps)
                    sigma = thresholded_uniform_sigma(deltas, rho, rel_mask)
                    sig_a = sigma[rel_mask].mean().item() if rel_mask.any() else 0.0
                    cache_seed = 42 + int(eps * 10) + int(keep * 1000) + K * 7 + hash(strat) % 100
                    cache = precompute_cache(teachers, caps_list, train_ds,
                                             sigma, rel_mask, device, seed=cache_seed)
                    dices = []
                    for s in student_seeds:
                        t0 = time.time()
                        best, _ = train_student_distill(
                            train_ds, val_loader, cache, device,
                            student_base=16, teacher_base=base_T,
                            n_epochs=40, lr=1e-3, lambda_feat=0.4, seed=s,
                        )
                        dices.append(best)
                    m = float(np.mean(dices))
                    results["sweep"][str(K)][f"{keep:.2f}"][strat]["eps"][str(eps)] = {
                        "sigma_active": sig_a, "dices": dices,
                        "mean": m, "std": float(np.std(dices)),
                        "sem": float(np.std(dices) / np.sqrt(len(dices))),
                    }
                    print(f"      ε={eps:>4}: σ={sig_a:6.3f}  Dice={m:.4f} "
                          f"± {np.std(dices):.4f}")

    out_path = Path(__file__).parent / "drive_per_teacher_importance_results.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote: {out_path}")

    print("\nKEY READOUTS:")
    print("  1. mask_overlap (Jaccard): how much independent teachers agree")
    print("     on which channels matter. LOW → channel-alignment is severe.")
    print("  2. shared vs union vs vote Dice: whether a central public mask")
    print("     beats letting each teacher bring its own importance.")
    print("  3. union releases MORE channels (more noise) → expect lower Dice")
    print("     if teachers disagree; vote releases fewer (less noise) but may")
    print("     drop channels some teachers needed.")


if __name__ == "__main__":
    main()
