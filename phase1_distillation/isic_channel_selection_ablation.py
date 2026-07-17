"""
Channel selection ablation for WACV paper Tables 4 and 5.

Table 4 (tab:selection): How many channels to release.
  ISIC 2018, K=3, drop_fracs=[0%, 10%, 30%, 50%, 70%, 90%],
  ε∈{0.5, 1, 2, 4, 6}, uniform allocation within kept channels,
  importance-based selection privatized under honest split.

Table 5 (tab:select-vs-alloc): Selection vs allocation at ε=2.
  Three conditions at drop=90% (keep top 10%):
    1. uniform all-channel (drop=0%, uniform)  → should match Table 1 uniform ε=2
    2. selection only (drop=90%, uniform within top 10%)
    3. selection + CANAL (drop=90%, WF within top 10%)

Budget split (matches isic_canal_k3_eps2.py):
  rho_imp = ALPHA_IMP * rho    (default 0.10 * rho)
  rho_rel = (1 - ALPHA_IMP) * rho

The shared noised importance vector is used for both channel ranking (selection)
and water-filling weights (allocation), spending rho_imp only once.

Output: results/isic_channel_selection_ablation_results.json
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from drive_pate_poc import train_K_teachers, correct_uniform_sigma
from drive_pate_pruning_joint import (
    shared_importance,
    precompute_joint_cache,
)
from drive_pate_canal_combined import (
    add_dp_noise_to_importance,
    correct_waterfilling_sigma,
)
from drive_student_distill import train_student_distill
from synthetic_demo import eps_to_rho

HERE = Path(__file__).parent
(HERE / "results").mkdir(exist_ok=True)

ALPHA_IMP = 0.1
CLIP_IMP = 1e-8
K = 3


def get_dataset(name, split, size):
    if name == "isic":
        from isic_dataset import ISICDataset
        return ISICDataset(split, size), 3
    if name == "kvasir":
        from kvasir_dataset import KvasirDataset
        return KvasirDataset(split, size), 3
    if name == "busi":
        from busi_dataset import BUSIDataset
        return BUSIDataset(split, size), 1
    raise ValueError(name)


def build_active_sigma_uniform(C, rank_desc, n_keep, deltas, rho_rel, device):
    """Active mask from top-n_keep channels; uniform sigma within active set."""
    active_mask = torch.zeros(C, dtype=torch.bool, device=device)
    active_mask[rank_desc[:n_keep]] = True
    sigma = torch.zeros(C, device=device)
    if active_mask.any():
        sigma[active_mask] = correct_uniform_sigma(deltas[active_mask], rho_rel)
    return active_mask, sigma


def build_active_sigma_canal(C, rank_desc, n_keep, deltas, imp_noisy, rho_rel, device):
    """Active mask from top-n_keep channels; WF sigma within active set."""
    active_mask = torch.zeros(C, dtype=torch.bool, device=device)
    active_mask[rank_desc[:n_keep]] = True
    sigma = torch.zeros(C, device=device)
    if active_mask.any():
        sigma[active_mask] = correct_waterfilling_sigma(
            deltas[active_mask], imp_noisy[active_mask], rho_rel
        )
    return active_mask, sigma


def run_students(train_ds, val_loader, teachers, caps_list, active_mask, sigma,
                 device, seeds, n_epochs_student, in_ch, cache_seed):
    cache = precompute_joint_cache(
        teachers, caps_list, train_ds, sigma, active_mask, device, seed=cache_seed
    )
    dices = []
    for s in seeds:
        best, _ = train_student_distill(
            train_ds, val_loader, cache, device,
            student_base=16, teacher_base=32,
            n_epochs=n_epochs_student, lr=1e-3, lambda_feat=0.4,
            seed=s, in_ch=in_ch,
        )
        dices.append(best)
    return dices


def cell_stats(dices):
    m = float(np.mean(dices))
    s = float(np.std(dices))
    sem = s / math.sqrt(len(dices))
    return {"dices": dices, "mean": m, "std": s, "sem": sem}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="isic", choices=["isic", "kvasir", "busi"])
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--te", type=int, default=60, help="teacher epochs")
    ap.add_argument("--se", type=int, default=40, help="student epochs")
    ap.add_argument("--epsilons", type=str, default="0.5,1,2,4,6")
    ap.add_argument("--drops", type=str, default="0.0,0.1,0.3,0.5,0.7,0.9",
                    help="fraction of channels to drop (lowest importance first)")
    args = ap.parse_args()

    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else "cpu")
    EPS = [float(e) for e in args.epsilons.split(",")]
    DROP_FRACS = [float(f) for f in args.drops.split(",")]
    SEEDS = list(range(100, 100 + args.seeds * 100, 100))

    print(f"[{args.dataset}] device={dev}  K={K}  ε={EPS}  drops={DROP_FRACS}  seeds={SEEDS}")

    train_ds, in_ch = get_dataset(args.dataset, "train", args.size)
    val_ds, _ = get_dataset(args.dataset, "val", args.size)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    N = len(train_ds)
    print(f"  train={N}  val={len(val_ds)}  in_ch={in_ch}")

    # Train K=3 teachers
    teachers, caps_list = train_K_teachers(train_ds, K, dev, n_epochs=args.te, in_ch=in_ch)
    Cb = teachers[0].base * 4  # bottleneck channels (e.g., 128 for base=32)
    print(f"  bottleneck channels C={Cb}")

    # Compute shared importance (clean, from training data)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=False)
    importance = shared_importance(teachers, train_loader, dev).to(dev)

    # Clipping sensitivity for importance release
    sens_imp = 2.0 * CLIP_IMP / float(N)

    deltas = torch.full((Cb,), 2.0 / K, device=dev)  # sensitivity = 2/K

    results = {
        "dataset": args.dataset,
        "K": K,
        "drop_fracs": DROP_FRACS,
        "epsilons": EPS,
        "seeds": SEEDS,
        "ALPHA_IMP": ALPHA_IMP,
        "C": Cb,
        "sweep": {},   # [eps][drop_frac] → uniform cell
        "tab5": {},    # eps=2 three-way comparison
    }

    for eps in EPS:
        rho = eps_to_rho(eps)
        rho_imp = ALPHA_IMP * rho
        rho_rel = (1.0 - ALPHA_IMP) * rho

        # Noise importance once → shared ranking for all drop fracs at this ε
        noise_seed = int(eps * 100) + 7
        imp_noisy = add_dp_noise_to_importance(importance, sens_imp, rho_imp, seed=noise_seed)
        rank_desc = torch.argsort(imp_noisy, descending=True)

        results["sweep"][str(eps)] = {}
        print(f"\n{'='*64}")
        print(f"ε={eps}  rho={rho:.4f}  rho_imp={rho_imp:.4f}  rho_rel={rho_rel:.4f}")

        for drop in DROP_FRACS:
            n_keep = max(1, int(round((1.0 - drop) * Cb)))
            cache_seed = int(eps * 1000) + int(drop * 100) + 42

            print(f"\n  drop={drop*100:.0f}%  n_keep={n_keep}", end="", flush=True)
            t0 = time.time()

            active_mask, sigma = build_active_sigma_uniform(
                Cb, rank_desc, n_keep, deltas, rho_rel, dev
            )
            sig_mean = sigma[active_mask].mean().item() if active_mask.any() else 0.0

            dices = run_students(
                train_ds, val_loader, teachers, caps_list,
                active_mask, sigma, dev, SEEDS, args.se, in_ch, cache_seed
            )
            cell = cell_stats(dices)
            cell["n_keep"] = n_keep
            cell["sigma_active_mean"] = sig_mean
            results["sweep"][str(eps)][f"{drop:.2f}"] = cell

            print(f"  σ={sig_mean:.3f}  Dice={cell['mean']:.4f}±{cell['sem']:.4f}"
                  f"  ({time.time()-t0:.1f}s)")

        # Table 5: three-way at ε=2 (or whichever eps, stored under tab5[str(eps)])
        # Condition 1 already computed above at drop=0.0 for this eps
        # Condition 2: drop=0.9 + uniform  (already computed above)
        # Condition 3: drop=0.9 + CANAL
        drop_sel = 0.9
        n_keep_sel = max(1, int(round((1.0 - drop_sel) * Cb)))
        cache_seed_canal = int(eps * 1000) + int(drop_sel * 100) + 99

        print(f"\n  [Tab5] drop=90% + CANAL  n_keep={n_keep_sel}", end="", flush=True)
        t0 = time.time()
        am_canal, sig_canal = build_active_sigma_canal(
            Cb, rank_desc, n_keep_sel, deltas, imp_noisy, rho_rel, dev
        )
        dices_canal = run_students(
            train_ds, val_loader, teachers, caps_list,
            am_canal, sig_canal, dev, SEEDS, args.se, in_ch, cache_seed_canal
        )
        canal_cell = cell_stats(dices_canal)
        canal_cell["n_keep"] = n_keep_sel
        canal_cell["sigma_active_mean"] = sig_canal[am_canal].mean().item() if am_canal.any() else 0.0

        results["tab5"][str(eps)] = {
            "uniform_all_ch": results["sweep"][str(eps)]["0.00"],
            "select_uniform":  results["sweep"][str(eps)][f"{drop_sel:.2f}"],
            "select_canal":    canal_cell,
        }
        print(f"  Dice={canal_cell['mean']:.4f}±{canal_cell['sem']:.4f}"
              f"  ({time.time()-t0:.1f}s)")

    out_path = HERE / "results" / f"{args.dataset}_channel_selection_ablation_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out_path}")

    # Print Table 4 summary
    print("\n" + "="*80)
    print(f"TABLE 4 — How many channels to release  (K={K}, {args.dataset.upper()})")
    header = f"{'drop':>6} | " + " | ".join(f"  ε={e:<4}" for e in EPS)
    print(header)
    print("-" * len(header))
    for drop in DROP_FRACS:
        row = f"{drop*100:5.0f}% | "
        for eps in EPS:
            m = results["sweep"][str(eps)][f"{drop:.2f}"]["mean"]
            row += f"  {m*100:.2f}   | "
        print(row)

    # Print Table 5 summary
    print("\nTABLE 5 — Selection vs Allocation  (ε=2 highlighted)")
    for eps in EPS:
        if abs(eps - 2.0) < 1e-6:
            t5 = results["tab5"][str(eps)]
            print(f"\nε={eps}")
            print(f"  uniform all-channel : {t5['uniform_all_ch']['mean']*100:.2f}%")
            print(f"  select + uniform    : {t5['select_uniform']['mean']*100:.2f}%")
            print(f"  select + CANAL      : {t5['select_canal']['mean']*100:.2f}%")


if __name__ == "__main__":
    main()
