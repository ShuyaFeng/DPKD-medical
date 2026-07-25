# -*- coding: utf-8 -*-
"""
CANAL vs uniform comparison across privacy budget split configurations.

Measures per-sample clipping bounds (CLIP_IMP, CLIP_CAPS) directly from
the data, privatizes caps and importance with their respective budget
fractions, and compares CANAL water-filling against uniform allocation.

Budget split: rho_tot = rho_caps + rho_imp + rho_rel
  - Both CANAL and uniform pay rho_caps for caps privatization.
  - Uniform receives (rho_caps + rho_imp) for release since it does not
    release importance; CANAL receives only rho_rel for release.
  - This is the honest apples-to-apples comparison at the same total epsilon.

Usage:
    python canal_budget_sweep.py \\
        --dataset isic \\
        --f_caps 0.10 --f_imp 0.45 --f_rel 0.45 \\
        --clip_imp_type avg --clip_caps_type avg \\
        --epsilons "1,2,4,8,16,32" \\
        --seeds 5 --te 60 --se 40

Output:
    results/budget_sweep_{dataset}_fc{f_caps}_fi{f_imp}_fr{f_rel}_{clip_imp_type}imp_{clip_caps_type}caps.json
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
from drive_pate_canal_combined import (
    add_dp_noise_to_importance,
    correct_waterfilling_sigma,
)
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


def measure_per_sample_imp_norms(teachers, train_ds, device):
    """
    Measure per-sample gradient-energy L2 norms across all teacher partitions.
    Each patient contributes one norm: the L2 norm of their per-channel
    mean squared gradient at the bottleneck.
    Returns (mean_norm, p99_norm).
    """
    partitions = partition_dataset(len(train_ds), len(teachers))
    all_norms = []
    for teacher, idxs in zip(teachers, partitions):
        teacher.eval()
        subset = Subset(train_ds, idxs)
        for i in range(len(subset)):
            x, y = subset[i]
            x = x.unsqueeze(0).to(device)
            y = y.unsqueeze(0).to(device)
            x.requires_grad_(True)
            teacher.zero_grad(set_to_none=True)
            e1, e2, e3 = teacher.encode(x)
            e3.retain_grad()
            loss = task_loss(teacher.decode(e1, e2, e3), y)
            loss.backward()
            g = e3.grad.detach().pow(2).mean(dim=(0, 2, 3))  # [C]
            all_norms.append(g.norm().item())
    norms = torch.tensor(all_norms)
    return norms.mean().item(), norms.quantile(0.99).item()


def measure_per_sample_cap_norms(teachers, train_ds, device):
    """
    Measure per-sample max per-channel activation L2 norm across all
    teacher partitions. For each patient, this is the maximum over all
    bottleneck channels of that channel's L2 spatial norm.
    Returns (mean_norm, p99_norm).
    """
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
            per_ch = e3[0].flatten(1).norm(dim=1)   # [C]
            all_norms.append(per_ch.max().item())
    norms = torch.tensor(all_norms)
    return norms.mean().item(), norms.quantile(0.99).item()


def privatize_caps(caps_list, clip_caps, n_train, rho_caps, seed=0):
    """
    Add Gaussian noise to each teacher's caps under rho_caps zCDP budget.
      sensitivity = 2 * clip_caps / n_train
      sigma_caps  = sensitivity / sqrt(2 * rho_caps)
    Returns noisy_caps_list (CPU tensors, clamped positive).
    If rho_caps <= 0, returns caps_list unchanged.
    """
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
    ap.add_argument("--dataset", default="isic",
                    choices=["isic", "kvasir", "busi"])
    ap.add_argument("--f_caps", type=float, default=0.10,
                    help="Fraction of rho for caps privatization.")
    ap.add_argument("--f_imp",  type=float, default=0.45,
                    help="Fraction of rho for importance privatization.")
    ap.add_argument("--f_rel",  type=float, default=0.45,
                    help="Fraction of rho for CANAL release.")
    ap.add_argument("--clip_imp_type",  default="avg", choices=["avg", "p99"],
                    help="Use mean or p99 of per-sample importance norms as CLIP_IMP.")
    ap.add_argument("--clip_caps_type", default="avg", choices=["avg", "p99"],
                    help="Use mean or p99 of per-sample cap norms as CLIP_CAPS.")
    ap.add_argument("--size",     type=int,   default=96)
    ap.add_argument("--seeds",    type=int,   default=5)
    ap.add_argument("--epsilons", type=str,   default="1,2,4,8,16,32")
    ap.add_argument("--te",       type=int,   default=60,
                    help="Teacher training epochs.")
    ap.add_argument("--se",       type=int,   default=40,
                    help="Student training epochs.")
    args = ap.parse_args()

    total = args.f_caps + args.f_imp + args.f_rel
    assert abs(total - 1.0) < 1e-5, \
        "f_caps + f_imp + f_rel must equal 1.0, got {:.6f}".format(total)

    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else "cpu")
    EPS   = [float(e) for e in args.epsilons.split(",")]
    SEEDS = list(range(100, 100 + args.seeds * 100, 100))

    train_ds, in_ch = get_dataset(args.dataset, "train", args.size)
    val_ds,   _     = get_dataset(args.dataset, "val",   args.size)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    N = len(train_ds)

    print("[{}] device={}  N={}  in_ch={}".format(args.dataset, dev, N, in_ch))
    print("  split: f_caps={} f_imp={} f_rel={}".format(
        args.f_caps, args.f_imp, args.f_rel))
    print("  clip_imp={}  clip_caps={}".format(
        args.clip_imp_type, args.clip_caps_type))
    print("  epsilons={}  seeds={}".format(EPS, SEEDS))

    # --- Train teachers ---
    print("\nTraining {} teachers ({} epochs)...".format(K, args.te))
    teachers, caps_list = train_K_teachers(
        train_ds, K, dev, n_epochs=args.te, in_ch=in_ch
    )
    Cb = teachers[0].base * 4

    # --- Measure per-sample importance norms ---
    print("\nMeasuring per-sample importance norms...")
    clip_imp_avg, clip_imp_p99 = measure_per_sample_imp_norms(
        teachers, train_ds, dev
    )
    clip_imp = clip_imp_avg if args.clip_imp_type == "avg" else clip_imp_p99
    print("  avg={:.4e}  p99={:.4e}  using ({})={:.4e}".format(
        clip_imp_avg, clip_imp_p99, args.clip_imp_type, clip_imp))

    # --- Measure per-sample cap norms ---
    print("\nMeasuring per-sample cap norms...")
    clip_caps_avg, clip_caps_p99 = measure_per_sample_cap_norms(
        teachers, train_ds, dev
    )
    clip_caps = clip_caps_avg if args.clip_caps_type == "avg" else clip_caps_p99
    print("  avg={:.4e}  p99={:.4e}  using ({})={:.4e}".format(
        clip_caps_avg, clip_caps_p99, args.clip_caps_type, clip_caps))

    # --- Compute shared (clean) importance ---
    print("\nComputing shared importance...")
    imp_clean = shared_importance(
        teachers, DataLoader(train_ds, batch_size=8), dev
    ).to(dev)
    print("  imp mean={:.4e}  max={:.4e}".format(
        imp_clean.mean().item(), imp_clean.max().item()))

    deltas   = torch.full((Cb,), 2.0 / K, device=dev)
    am       = torch.ones(Cb, dtype=torch.bool, device=dev)
    sens_imp = 2.0 * clip_imp / float(N)

    # --- Results container ---
    R = {
        "dataset":        args.dataset,
        "in_ch":          in_ch,
        "n_train":        N,
        "f_caps":         args.f_caps,
        "f_imp":          args.f_imp,
        "f_rel":          args.f_rel,
        "clip_imp_type":  args.clip_imp_type,
        "clip_imp_avg":   clip_imp_avg,
        "clip_imp_p99":   clip_imp_p99,
        "clip_imp_used":  clip_imp,
        "clip_caps_type": args.clip_caps_type,
        "clip_caps_avg":  clip_caps_avg,
        "clip_caps_p99":  clip_caps_p99,
        "clip_caps_used": clip_caps,
        "imp_clean_mean": imp_clean.mean().item(),
        "sens_imp":       sens_imp,
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
        rho          = eps_to_rho(e)
        rho_caps     = args.f_caps * rho
        rho_imp      = args.f_imp  * rho
        rho_rel_canal   = args.f_rel * rho
        # Uniform does not pay rho_imp; it receives that portion for release
        rho_rel_uniform = (1.0 - args.f_caps) * rho

        # Privatize caps — identical noisy caps used by both methods
        caps_noisy = privatize_caps(
            caps_list, clip_caps, N, rho_caps, seed=int(e * 100)
        )

        # -- UNIFORM --
        sigma_uni  = thresholded_uniform_sigma(deltas, rho_rel_uniform, am)
        cache_uni  = precompute_joint_cache(
            teachers, caps_noisy, train_ds, sigma_uni, am, dev, seed=42
        )
        dices_uni  = run_students(cache_uni, "uni_e{}".format(e), e)
        uniform_results[str(e)] = {
            "dices":    dices_uni,
            "mean":     float(np.mean(dices_uni)),
            "sem":      float(np.std(dices_uni) / np.sqrt(len(dices_uni))),
            "rho_rel":  rho_rel_uniform,
        }

        # -- CANAL --
        imp_noisy   = add_dp_noise_to_importance(
            imp_clean, sens_imp, rho_imp, seed=int(e * 100) + 1
        )
        sigma_canal = correct_waterfilling_sigma(
            deltas, imp_noisy.to(dev), rho_rel_canal
        )
        cache_canal = precompute_joint_cache(
            teachers, caps_noisy, train_ds, sigma_canal, am, dev, seed=42
        )
        dices_canal = run_students(cache_canal, "canal_e{}".format(e), e)
        canal_results[str(e)] = {
            "dices":   dices_canal,
            "mean":    float(np.mean(dices_canal)),
            "sem":     float(np.std(dices_canal) / np.sqrt(len(dices_canal))),
            "rho_rel": rho_rel_canal,
        }

        u = uniform_results[str(e)]["mean"]
        c = canal_results[str(e)]["mean"]
        print("  eps={:5.1f}: uniform={:.4f}  CANAL={:.4f}  diff={:+.4f}".format(
            e, u, c, c - u))

    R["series"]["uniform"] = uniform_results
    R["series"]["canal"]   = canal_results

    tag = "fc{:.2f}_fi{:.2f}_fr{:.2f}_{}imp_{}caps".format(
        args.f_caps, args.f_imp, args.f_rel,
        args.clip_imp_type, args.clip_caps_type
    )
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
