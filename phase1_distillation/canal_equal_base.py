"""
Equal base channels experiment: teacher base=16, student base=16.

Current setup: teacher base=32 (128-ch bottleneck), student base=16 (64-ch).
The student needs an adapter (1×1 conv 64→128) to match teacher's feature space.
The adapter introduces an extra learned mapping that may hurt or distort the
feature alignment.

This experiment: train teachers at base=16 (64-ch bottleneck). Student also
base=16. No channel mismatch → direct feature comparison in the same 64-ch space.
The adapter inside train_student_distill becomes 64→64 (identity-learnable 1×1 conv).

Hypothesis:
- Removing the bottleneck dimension mismatch may help the student copy the teacher
  more faithfully.
- Importance at 64ch may be more or less skewed than at 128ch.
- With 64ch (fewer channels), each channel is more "load-bearing" → may help CANAL.

Two conditions: uniform vs CANAL, all 3 datasets, seeds=3.
Also includes 128ch baseline (teacher base=32, student base=16) for direct comparison.

Budget split: rho_imp=10%, rho_rel=90%.

Usage:
  python canal_equal_base.py --dataset isic --seeds 3 --te 60 --se 40
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
KEEP_FRAC = 0.1


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


def run_one_config(label, teacher_base, student_base, train_ds, val_loader,
                   teachers, caps_list, importance, sens_imp, dev, EPS, SEEDS,
                   n_epochs_student, in_ch, ALPHA_IMP, CLIP_IMP, KEEP_FRAC):
    """Run uniform + CANAL at all epsilons for a given teacher/student base."""
    Cb = teacher_base * 4
    n_keep = max(1, int(round(KEEP_FRAC * Cb)))
    deltas = torch.full((Cb,), 2.0 / K, device=dev)
    imp = importance.to(dev)
    print(f"\n  [{label}] C={Cb}  n_keep={n_keep}")

    config_results = {"teacher_base": teacher_base, "student_base": student_base,
                      "C": Cb, "n_keep": n_keep, "sweep": {}}

    for eps in EPS:
        rho = eps_to_rho(eps)
        rho_imp = ALPHA_IMP * rho
        rho_rel = (1.0 - ALPHA_IMP) * rho
        noise_seed = int(eps * 100) + 7
        imp_noisy = add_dp_noise_to_importance(imp, sens_imp, rho_imp, seed=noise_seed)
        rank_desc = torch.argsort(imp_noisy, descending=True)
        top_indices = rank_desc[:n_keep]

        top_mask = torch.zeros(Cb, dtype=torch.bool, device=dev)
        top_mask[top_indices] = True

        # Uniform sigma on top channels
        sigma_uni = torch.zeros(Cb, device=dev)
        sigma_uni[top_mask] = correct_uniform_sigma(deltas[top_mask], rho_rel)

        # CANAL (WF) sigma on top channels
        sigma_canal = torch.zeros(Cb, device=dev)
        sigma_canal[top_mask] = correct_waterfilling_sigma(
            deltas[top_mask], imp_noisy[top_mask], rho_rel
        )

        ep_res = {}
        for cond, sigma, mask in [("uniform", sigma_uni, top_mask),
                                   ("canal", sigma_canal, top_mask)]:
            print(f"    eps={eps}  {cond} ...", end="", flush=True)
            t0 = time.time()
            cache = precompute_joint_cache(
                teachers, caps_list, train_ds, sigma, mask, dev,
                seed=int(eps * 1000) + (1 if cond == "uniform" else 2)
            )
            dices = []
            for s in SEEDS:
                best, _ = train_student_distill(
                    train_ds, val_loader, cache, dev,
                    student_base=student_base, teacher_base=teacher_base,
                    n_epochs=n_epochs_student, lr=1e-3, lambda_feat=0.4,
                    seed=s, in_ch=in_ch,
                )
                dices.append(best)
            m = float(np.mean(dices))
            sem = float(np.std(dices)) / math.sqrt(len(dices))
            ep_res[cond] = {"dices": dices, "mean": m, "std": float(np.std(dices)), "sem": sem}
            print(f"  Dice={m:.4f}±{sem:.4f}  ({time.time()-t0:.1f}s)")

        ep_res["canal_minus_uniform"] = ep_res["canal"]["mean"] - ep_res["uniform"]["mean"]
        config_results["sweep"][str(eps)] = ep_res

    return config_results


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

    results = {"dataset": args.dataset, "K": K, "ALPHA_IMP": ALPHA_IMP,
               "epsilons": EPS, "seeds": SEEDS}

    # --- Config A: teacher base=32, student base=16 (standard, for comparison) ---
    print("\n" + "="*64)
    print("Config A: teacher_base=32, student_base=16 (standard setup)")
    teachers_32, caps_32 = train_K_teachers(train_ds, K, dev, n_epochs=args.te,
                                             in_ch=in_ch, base=32)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=False)
    importance_32 = shared_importance(teachers_32, train_loader, dev)
    sens_imp_32 = 2.0 * CLIP_IMP / float(N)

    imp_32_cpu = importance_32.cpu()
    print(f"  importance (128ch): max/mean={float(imp_32_cpu.max()/imp_32_cpu.mean()):.2f}x  "
          f"max/min={float(imp_32_cpu.max()/imp_32_cpu.min()):.1f}x")

    results["config_A_base32"] = run_one_config(
        "A: teacher32/student16", 32, 16,
        train_ds, val_loader, teachers_32, caps_32,
        importance_32, sens_imp_32, dev, EPS, SEEDS, args.se, in_ch,
        ALPHA_IMP, CLIP_IMP, KEEP_FRAC
    )

    # --- Config B: teacher base=16, student base=16 (equal base) ---
    print("\n" + "="*64)
    print("Config B: teacher_base=16, student_base=16 (equal base, no channel mismatch)")
    teachers_16, caps_16 = train_K_teachers(train_ds, K, dev, n_epochs=args.te,
                                             in_ch=in_ch, base=16)
    importance_16 = shared_importance(teachers_16, train_loader, dev)
    sens_imp_16 = 2.0 * CLIP_IMP / float(N)

    imp_16_cpu = importance_16.cpu()
    print(f"  importance (64ch): max/mean={float(imp_16_cpu.max()/imp_16_cpu.mean()):.2f}x  "
          f"max/min={float(imp_16_cpu.max()/imp_16_cpu.min()):.1f}x")

    results["config_B_base16"] = run_one_config(
        "B: teacher16/student16", 16, 16,
        train_ds, val_loader, teachers_16, caps_16,
        importance_16, sens_imp_16, dev, EPS, SEEDS, args.se, in_ch,
        ALPHA_IMP, CLIP_IMP, KEEP_FRAC
    )

    # Summary
    print("\n" + "="*80)
    print(f"EQUAL BASE COMPARISON  ({args.dataset.upper()}, K={K}, keep={KEEP_FRAC*100:.0f}%)")
    print(f"{'eps':>5} | {'A_uni(b32)':>11} | {'A_canal':>9} | {'A_diff':>7} | "
          f"{'B_uni(b16)':>11} | {'B_canal':>9} | {'B_diff':>7}")
    print("-"*80)
    for eps in EPS:
        ra = results["config_A_base32"]["sweep"][str(eps)]
        rb = results["config_B_base16"]["sweep"][str(eps)]
        print(f"{eps:>5.1f} | {ra['uniform']['mean']:>11.4f} | {ra['canal']['mean']:>9.4f} | "
              f"{ra['canal_minus_uniform']:>+7.4f} | {rb['uniform']['mean']:>11.4f} | "
              f"{rb['canal']['mean']:>9.4f} | {rb['canal_minus_uniform']:>+7.4f}")

    out = HERE / "results" / f"{args.dataset}_equal_base_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
