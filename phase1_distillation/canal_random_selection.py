"""
Random vs importance-based channel selection (PRIORITY experiment).

Question: does the gain from channel selection come from picking IMPORTANT
channels, or just from concentrating budget on ANY subset of channels?

Four conditions at keep=10% (n_keep ≈ C/10 channels):
  1. all_uniform  — all C channels, uniform sigma  (full-budget baseline)
  2. top_uniform  — top-n_keep by importance, uniform sigma
  3. rand_uniform — N_RAND random subsets of n_keep channels, dice avg over subsets
  4. top_canal    — top-n_keep by importance, WF sigma

Interpretation:
  rand ≈ top  →  gain is from budget concentration alone (any K channels works)
  rand << top →  importance ranking is crucial for selection gain

Budget split: rho_imp=10% (importance privatization), rho_rel=90% (noise release).

Usage:
  python canal_random_selection.py --dataset isic --seeds 3 --te 60 --se 40
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
from drive_pate_pruning_joint import shared_importance, precompute_joint_cache
from drive_pate_canal_combined import add_dp_noise_to_importance, correct_waterfilling_sigma
from drive_student_distill import train_student_distill
from synthetic_demo import eps_to_rho

HERE = Path(__file__).parent
(HERE / "results").mkdir(exist_ok=True)

ALPHA_IMP = 0.1
CLIP_IMP = 1e-8
K = 3
N_RAND = 5      # number of random channel subsets to avg over
KEEP_FRAC = 0.1  # keep top 10% channels


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


def build_sigma(C, selected_indices, deltas, imp_noisy, rho_rel, mode, device):
    active_mask = torch.zeros(C, dtype=torch.bool, device=device)
    active_mask[selected_indices] = True
    sigma = torch.zeros(C, device=device)
    if mode == "uniform":
        sigma[active_mask] = correct_uniform_sigma(deltas[active_mask], rho_rel)
    elif mode == "canal":
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
    ap.add_argument("--te", type=int, default=60)
    ap.add_argument("--se", type=int, default=40)
    ap.add_argument("--epsilons", default="0.5,1,2,4,6")
    args = ap.parse_args()

    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else "cpu")
    EPS = [float(e) for e in args.epsilons.split(",")]
    SEEDS = list(range(100, 100 + args.seeds * 100, 100))

    print(f"[{args.dataset}] device={dev}  K={K}  eps={EPS}  seeds={SEEDS}")
    train_ds, in_ch = get_dataset(args.dataset, "train", args.size)
    val_ds, _ = get_dataset(args.dataset, "val", args.size)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    N = len(train_ds)
    print(f"  train={N}  val={len(val_ds)}  in_ch={in_ch}")

    teachers, caps_list = train_K_teachers(train_ds, K, dev, n_epochs=args.te, in_ch=in_ch)
    Cb = teachers[0].base * 4
    n_keep = max(1, int(round(KEEP_FRAC * Cb)))
    print(f"  bottleneck C={Cb}  n_keep={n_keep} ({KEEP_FRAC*100:.0f}%)")

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=False)
    importance = shared_importance(teachers, train_loader, dev).to(dev)
    sens_imp = 2.0 * CLIP_IMP / float(N)
    deltas = torch.full((Cb,), 2.0 / K, device=dev)

    results = {
        "dataset": args.dataset, "K": K, "keep_frac": KEEP_FRAC,
        "n_keep": n_keep, "N_RAND": N_RAND, "epsilons": EPS,
        "seeds": SEEDS, "ALPHA_IMP": ALPHA_IMP, "C": Cb, "sweep": {},
    }

    for eps in EPS:
        rho = eps_to_rho(eps)
        rho_imp = ALPHA_IMP * rho
        rho_rel = (1.0 - ALPHA_IMP) * rho
        noise_seed = int(eps * 100) + 7
        imp_noisy = add_dp_noise_to_importance(importance, sens_imp, rho_imp, seed=noise_seed)
        rank_desc = torch.argsort(imp_noisy, descending=True)
        top_indices = rank_desc[:n_keep]

        print(f"\n{'='*64}\neps={eps}  rho_rel={rho_rel:.4f}  n_keep={n_keep}")
        ep_res = {}

        # --- 1. all channels + uniform ---
        print("  [1/4] all_uniform ...", end="", flush=True)
        t0 = time.time()
        all_mask = torch.ones(Cb, dtype=torch.bool, device=dev)
        sigma_all = correct_uniform_sigma(deltas, rho_rel)
        dices = run_students(train_ds, val_loader, teachers, caps_list,
                             all_mask, sigma_all, dev, SEEDS, args.se, in_ch,
                             cache_seed=int(eps * 1000) + 1)
        ep_res["all_uniform"] = cell_stats(dices)
        print(f"  Dice={ep_res['all_uniform']['mean']:.4f}  ({time.time()-t0:.1f}s)")

        # --- 2. top-n_keep + uniform ---
        print("  [2/4] top_uniform ...", end="", flush=True)
        t0 = time.time()
        top_mask, sigma_top_uni = build_sigma(Cb, top_indices, deltas, imp_noisy, rho_rel, "uniform", dev)
        dices = run_students(train_ds, val_loader, teachers, caps_list,
                             top_mask, sigma_top_uni, dev, SEEDS, args.se, in_ch,
                             cache_seed=int(eps * 1000) + 2)
        ep_res["top_uniform"] = cell_stats(dices)
        print(f"  Dice={ep_res['top_uniform']['mean']:.4f}  ({time.time()-t0:.1f}s)")

        # --- 3. random-n_keep + uniform (avg over N_RAND configs) ---
        print(f"  [3/4] rand_uniform  ({N_RAND} random configs) ...")
        rand_all_dices = []
        for r in range(N_RAND):
            rand_perm = torch.randperm(
                Cb, generator=torch.Generator().manual_seed(r * 137 + int(eps * 1000))
            )
            rand_idx = rand_perm[:n_keep]
            rand_mask, sigma_rand = build_sigma(Cb, rand_idx, deltas, imp_noisy, rho_rel, "uniform", dev)
            dices = run_students(train_ds, val_loader, teachers, caps_list,
                                 rand_mask, sigma_rand, dev, SEEDS, args.se, in_ch,
                                 cache_seed=int(eps * 1000) + 30 + r)
            rand_all_dices.extend(dices)
            print(f"    rand_config {r+1}/{N_RAND}  Dice={np.mean(dices):.4f}")
        ep_res["rand_uniform"] = cell_stats(rand_all_dices)
        print(f"  rand_uniform mean={ep_res['rand_uniform']['mean']:.4f}±{ep_res['rand_uniform']['sem']:.4f}")

        # --- 4. top-n_keep + CANAL ---
        print("  [4/4] top_canal ...", end="", flush=True)
        t0 = time.time()
        top_mask_c, sigma_canal = build_sigma(Cb, top_indices, deltas, imp_noisy, rho_rel, "canal", dev)
        dices = run_students(train_ds, val_loader, teachers, caps_list,
                             top_mask_c, sigma_canal, dev, SEEDS, args.se, in_ch,
                             cache_seed=int(eps * 1000) + 4)
        ep_res["top_canal"] = cell_stats(dices)
        print(f"  Dice={ep_res['top_canal']['mean']:.4f}  ({time.time()-t0:.1f}s)")

        results["sweep"][str(eps)] = ep_res

    # Summary
    print("\n" + "="*88)
    print(f"RANDOM vs IMPORTANCE SELECTION  ({args.dataset.upper()}, K={K}, keep={KEEP_FRAC*100:.0f}%)")
    print(f"{'eps':>5} | {'all_uni':>9} | {'top_uni':>9} | {'rand_uni':>9} | {'top_canal':>10} | "
          f"{'top-rand':>9} | {'canal-top':>10}")
    print("-"*88)
    for eps in EPS:
        r = results["sweep"][str(eps)]
        au = r["all_uniform"]["mean"]
        tu = r["top_uniform"]["mean"]
        ru = r["rand_uniform"]["mean"]
        tc = r["top_canal"]["mean"]
        print(f"{eps:>5.1f} | {au:>9.4f} | {tu:>9.4f} | {ru:>9.4f} | {tc:>10.4f} | "
              f"{tu-ru:>+9.4f} | {tc-tu:>+10.4f}")
    print("="*88)

    out = HERE / "results" / f"{args.dataset}_random_selection_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
