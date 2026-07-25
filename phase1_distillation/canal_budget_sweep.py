# -*- coding: utf-8 -*-
"""
CANAL vs uniform comparison — fair budget split.

Budget split:
  rho_caps = f_caps * rho          (caps privatization, paid by both)
  rho_rel  = (1 - f_caps) * rho   (channel noise, same for both)

Importance is computed cleanly and used internally by CANAL only to
set per-channel noise levels via water-filling. It is never released,
so it requires no privacy budget. Uniform ignores importance entirely
and applies the same sigma to every channel.

Usage:
    python canal_budget_sweep.py \\
        --dataset isic \\
        --f_caps 0.10 --clip_caps_type avg \\
        --epsilons "1,2,4,8,16,32" \\
        --seeds 7 --te 60 --se 60

Output:
    results/budget_sweep_{dataset}_fc{f_caps}_{clip_caps_type}caps.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent))

from drive_local_demo import task_loss
from drive_pate_canal_combined import correct_waterfilling_sigma
from drive_pate_poc import partition_dataset, train_K_teachers
from drive_pate_pruning_joint import (
    precompute_joint_cache,
    shared_importance,
    thresholded_uniform_sigma,
)
from drive_student_distill import train_student_distill
from synthetic_demo import eps_to_rho

HERE = Path(__file__).parent
(HERE / "results").mkdir(exist_ok=True)
CKPT_DIR = HERE / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)

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
    raise ValueError("Unknown dataset: {}".format(name))


def measure_per_sample_cap_norms(teachers, train_ds, device):
    partitions = partition_dataset(len(train_ds), len(teachers))
    all_norms = []
    for teacher, idxs in zip(teachers, partitions):
        teacher.eval()
        subset = Subset(train_ds, idxs)
        for i in range(len(subset)):
            x, _ = subset[i]
            x = x.unsqueeze(0).to(device)
            with torch.no_grad():
                _, _, e3 = teacher.encode(x)
            per_ch = e3[0].flatten(1).norm(dim=1)
            all_norms.append(per_ch.max().item())
    norms = torch.tensor(all_norms)
    return norms.mean().item(), norms.quantile(0.99).item()


def privatize_caps(caps_list, clip_caps, n_train, rho_caps, seed=0):
    if rho_caps <= 0:
        return caps_list
    sens  = 2.0 * clip_caps / float(n_train)
    sigma = sens / math.sqrt(2.0 * rho_caps)
    noisy = []
    for k, caps in enumerate(caps_list):
        g = torch.Generator()
        g.manual_seed(seed + k)
        noise = torch.randn(caps.shape, generator=g) * sigma
        noisy.append((caps.detach().cpu() + noise).clamp(min=1e-6))
    return noisy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",        default="isic",
                    choices=["isic", "kvasir", "busi"])
    ap.add_argument("--f_caps",         type=float, default=0.10,
                    help="Fraction of rho for caps privatization.")
    ap.add_argument("--clip_caps_type", default="avg", choices=["avg", "p99"],
                    help="Use mean or p99 of per-sample cap norms as CLIP_CAPS.")
    ap.add_argument("--size",           type=int,   default=96)
    ap.add_argument("--seeds",          type=int,   default=7)
    ap.add_argument("--epsilons",       type=str,   default="1,2,4,8,16,32")
    ap.add_argument("--te",             type=int,   default=60,
                    help="Teacher training epochs.")
    ap.add_argument("--se",             type=int,   default=60,
                    help="Student training epochs.")
    args = ap.parse_args()

    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else "cpu")
    EPS   = [float(e) for e in args.epsilons.split(",")]
    SEEDS = list(range(100, 100 + args.seeds * 100, 100))

    train_ds, in_ch = get_dataset(args.dataset, "train", args.size)
    val_ds,   _     = get_dataset(args.dataset, "val",   args.size)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    N = len(train_ds)

    print("[{}] device={}  N={}  in_ch={}".format(args.dataset, dev, N, in_ch))
    print("  f_caps={}  clip_caps={}".format(args.f_caps, args.clip_caps_type))
    print("  epsilons={}  seeds={}  te={}  se={}".format(
        EPS, SEEDS, args.te, args.se))

    # --- Train teachers ---
    print("\nTraining {} teachers ({} epochs)...".format(K, args.te))
    teachers, caps_list = train_K_teachers(
        train_ds, K, dev, n_epochs=args.te, in_ch=in_ch
    )
    Cb = teachers[0].base * 4

    # --- Measure per-sample cap norms ---
    print("\nMeasuring per-sample cap norms...")
    clip_caps_avg, clip_caps_p99 = measure_per_sample_cap_norms(
        teachers, train_ds, dev
    )
    clip_caps = clip_caps_avg if args.clip_caps_type == "avg" else clip_caps_p99
    print("  avg={:.4e}  p99={:.4e}  using ({})={:.4e}".format(
        clip_caps_avg, clip_caps_p99, args.clip_caps_type, clip_caps))

    # --- Compute clean importance (internal only, no budget charged) ---
    print("\nComputing importance (clean, internal)...")
    imp_clean = shared_importance(
        teachers, DataLoader(train_ds, batch_size=8), dev
    ).to(dev)
    print("  imp mean={:.4e}  max={:.4e}".format(
        imp_clean.mean().item(), imp_clean.max().item()))

    deltas = torch.full((Cb,), 2.0 / K, device=dev)
    am     = torch.ones(Cb, dtype=torch.bool, device=dev)

    # --- Results container ---
    R = {
        "dataset":        args.dataset,
        "in_ch":          in_ch,
        "n_train":        N,
        "f_caps":         args.f_caps,
        "clip_caps_type": args.clip_caps_type,
        "clip_caps_avg":  clip_caps_avg,
        "clip_caps_p99":  clip_caps_p99,
        "clip_caps_used": clip_caps,
        "imp_clean_mean": imp_clean.mean().item(),
        "imp_clean_max":  imp_clean.max().item(),
        "epsilons":       EPS,
        "seeds":          SEEDS,
        "series":         {},
    }

    uniform_results = {}
    canal_results   = {}

    def run_students(cache, label, e):
        return [
            train_student_distill(
                train_ds, val_loader, cache, dev,
                student_base=16, teacher_base=32,
                n_epochs=args.se, lr=1e-3, lambda_feat=0.4,
                seed=s, in_ch=in_ch,
                save_path=CKPT_DIR / "{}_{}_{}_seed{}.pt".format(
                    args.dataset, label, e, s)
            )[0]
            for s in SEEDS
        ]

    print("\n--- Experiments ---")
    for e in EPS:
        rho     = eps_to_rho(e)
        rho_caps = args.f_caps * rho
        rho_rel  = (1.0 - args.f_caps) * rho   # same for both methods

        # Privatize caps — same noisy caps used by both methods
        caps_noisy = privatize_caps(
            caps_list, clip_caps, N, rho_caps, seed=int(e * 100)
        )

        # -- UNIFORM: same sigma on every channel --
        sigma_uni  = thresholded_uniform_sigma(deltas, rho_rel, am)
        cache_uni  = precompute_joint_cache(
            teachers, caps_noisy, train_ds, sigma_uni, am, dev, seed=42
        )
        dices_uni  = run_students(cache_uni, "uni_e{}".format(e), e)
        uniform_results[str(e)] = {
            "dices":   dices_uni,
            "mean":    float(np.mean(dices_uni)),
            "sem":     float(np.std(dices_uni) / np.sqrt(len(dices_uni))),
            "rho_rel": rho_rel,
        }

        # -- CANAL: water-filling using clean importance, same rho_rel --
        sigma_canal = correct_waterfilling_sigma(
            deltas, imp_clean, rho_rel
        )
        cache_canal = precompute_joint_cache(
            teachers, caps_noisy, train_ds, sigma_canal, am, dev, seed=42
        )
        dices_canal = run_students(cache_canal, "canal_e{}".format(e), e)
        canal_results[str(e)] = {
            "dices":   dices_canal,
            "mean":    float(np.mean(dices_canal)),
            "sem":     float(np.std(dices_canal) / np.sqrt(len(dices_canal))),
            "rho_rel": rho_rel,
        }

        u = uniform_results[str(e)]["mean"]
        c = canal_results[str(e)]["mean"]
        print("  eps={:5.1f}: uniform={:.4f}  CANAL={:.4f}  diff={:+.4f}".format(
            e, u, c, c - u))

    R["series"]["uniform"] = uniform_results
    R["series"]["canal"]   = canal_results

    tag = "fc{:.2f}_{}caps".format(args.f_caps, args.clip_caps_type)
    out_path = HERE / "results" / "budget_sweep_{}_{}.json".format(
        args.dataset, tag
    )
    out_path.write_text(json.dumps(R, indent=2))
    print("\nSaved {}".format(out_path))

    print("\nSummary:")
    for e in EPS:
        u = uniform_results[str(e)]["mean"]
        c = canal_results[str(e)]["mean"]
        print("  eps={}: uniform={:.4f}  CANAL={:.4f}  diff={:+.4f}".format(
            e, u, c, c - u))


if __name__ == "__main__":
    main()
