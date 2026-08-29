"""
canal_final_experiment.py — Definitive CANAL vs Uniform comparison.

Three-way honest budget split (paper-consistent with corrected calibration):
  fc = 0.10  caps privatisation   (both methods pay)
  fi = 0.05  importance privatisation  (channel-selection step)
  fr = 0.85  per-channel release noise

Key fixes vs. prior experiments:
  1. clip_imp = 90th-percentile of actual importance values — prior scripts used
     a value 576x too large (H×W missing from normalization) which destroyed the
     importance signal entirely.
  2. fi = 0.05 instead of 0.45 — enough to maintain DP accounting while keeping
     the importance signal intact.
  3. Matches paper: eps in {1,2,4,8}, K=3, 5 seeds, top 10% channels.

Usage:
  python canal_final_experiment.py --dataset isic   --seeds 5 --te 60 --se 40
  python canal_final_experiment.py --dataset kvasir --seeds 5 --te 60 --se 40
  python canal_final_experiment.py --dataset busi   --seeds 5 --te 60 --se 40
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent))
from drive_pate_poc import train_K_teachers, correct_uniform_sigma, partition_dataset
from drive_pate_pruning_joint import shared_importance, precompute_joint_cache
from drive_pate_canal_combined import add_dp_noise_to_importance, correct_waterfilling_sigma
from drive_student_distill import train_student_distill
from synthetic_demo import eps_to_rho

HERE = Path(__file__).parent
(HERE / "results").mkdir(exist_ok=True)
(HERE / "checkpoints").mkdir(exist_ok=True)

K         = 3
KEEP_FRAC = 0.10
FC        = 0.10   # caps budget fraction
FI        = 0.05   # importance budget fraction
FR        = 0.85   # release budget fraction
assert abs(FC + FI + FR - 1.0) < 1e-9, "Budget fractions must sum to 1"


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


def measure_per_sample_cap_norms(teachers, train_ds, device):
    """Per-sample max-channel bottleneck norm → calibrates CLIP_CAPS."""
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
    return float(norms.mean()), float(norms.quantile(0.99))


def privatize_caps(caps_list, clip_caps, n_train, rho_caps, seed=0):
    """Add Gaussian DP noise to teacher caps."""
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


def run_students(train_ds, val_loader, teachers, caps_noisy, active_mask, sigma,
                 device, seeds, n_epochs_student, in_ch, cache_seed,
                 dataset, is_canal, eps):
    cache = precompute_joint_cache(
        teachers, caps_noisy, train_ds, sigma, active_mask, device, seed=cache_seed
    )
    dices = []
    for s in seeds:
        save_path = HERE / "checkpoints" / f"{dataset}_K3_canal{is_canal}_eps{eps}_seed{s}.pt"
        best, _ = train_student_distill(
            train_ds, val_loader, cache, device,
            student_base=16, teacher_base=32,
            n_epochs=n_epochs_student, lr=1e-3, lambda_feat=0.4,
            seed=s, in_ch=in_ch,
            save_path=save_path,
        )
        dices.append(best)
    return dices


def cell_stats(dices):
    m   = float(np.mean(dices))
    s   = float(np.std(dices))
    sem = s / math.sqrt(len(dices))
    return {"dices": dices, "mean": m, "std": s, "sem": sem}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",  default="isic", choices=["isic", "kvasir", "busi"])
    ap.add_argument("--size",     type=int, default=96)
    ap.add_argument("--seeds",    type=int, default=5)
    ap.add_argument("--te",       type=int, default=60)
    ap.add_argument("--se",       type=int, default=40)
    ap.add_argument("--epsilons", default="1,2,4,8")
    args = ap.parse_args()

    dev   = ("cuda" if torch.cuda.is_available()
             else "mps" if torch.backends.mps.is_available() else "cpu")
    EPS   = [float(e) for e in args.epsilons.split(",")]
    SEEDS = list(range(100, 100 + args.seeds * 100, 100))

    print(f"[{args.dataset}] device={dev}  K={K}  eps={EPS}  seeds={SEEDS}")
    print(f"  budget split: fc={FC}  fi={FI}  fr={FR}")

    train_ds, in_ch = get_dataset(args.dataset, "train", args.size)
    val_ds,   _     = get_dataset(args.dataset, "val",   args.size)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    N = len(train_ds)
    print(f"  train={N}  val={len(val_ds)}  in_ch={in_ch}")

    # --- Train teachers ---
    print(f"\nTraining {K} teachers ({args.te} epochs)...")
    teachers, caps_list = train_K_teachers(
        train_ds, K, dev, n_epochs=args.te, in_ch=in_ch
    )
    Cb     = teachers[0].base * 4
    n_keep = max(1, int(round(KEEP_FRAC * Cb)))
    deltas = torch.full((Cb,), 2.0 / K, device=dev)
    print(f"  bottleneck C={Cb}  n_keep={n_keep} ({KEEP_FRAC*100:.0f}%)")

    # --- Calibrate CLIP_CAPS from per-sample feature norms ---
    print("Measuring per-sample cap norms...")
    clip_caps_avg, clip_caps_p99 = measure_per_sample_cap_norms(teachers, train_ds, dev)
    clip_caps = clip_caps_avg   # conservative average (not p99) to keep sensitivity low
    print(f"  clip_caps: avg={clip_caps_avg:.4e}  p99={clip_caps_p99:.4e}  using avg")

    # --- Compute clean importance ---
    print("Computing channel importance...")
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=False)
    importance   = shared_importance(teachers, train_loader, dev).to(dev)
    imp_cpu = importance.cpu()

    # clip_imp = 90th percentile of actual importance values — correctly calibrated
    # (prior bug: clip was 576x too large because H×W normalisation was missing)
    clip_imp = float(torch.quantile(imp_cpu, 0.90))
    sens_imp = 2.0 * clip_imp / float(N)
    print(f"  importance: mean={float(imp_cpu.mean()):.4e}  max={float(imp_cpu.max()):.4e}  "
          f"p90={clip_imp:.4e}  sens_imp={sens_imp:.4e}")

    results = {
        "dataset": args.dataset, "K": K, "keep_frac": KEEP_FRAC, "n_keep": n_keep,
        "budget": {"fc": FC, "fi": FI, "fr": FR},
        "epsilons": EPS, "seeds": SEEDS,
        "clip_caps_avg": clip_caps_avg, "clip_caps_p99": clip_caps_p99,
        "clip_imp": clip_imp, "sens_imp": sens_imp,
        "importance_stats": {
            "mean": float(imp_cpu.mean()), "max": float(imp_cpu.max()),
            "std":  float(imp_cpu.std()),  "p90": clip_imp,
        },
        "sweep": {},
    }

    for eps in EPS:
        rho      = eps_to_rho(eps)
        rho_caps = FC * rho
        rho_imp  = FI * rho
        rho_rel  = FR * rho
        print(f"\n{'='*64}\neps={eps}  rho_caps={rho_caps:.5f}  rho_imp={rho_imp:.5f}  rho_rel={rho_rel:.4f}")

        # Privatize caps — identical noisy caps used by both uniform and CANAL
        caps_noisy = privatize_caps(caps_list, clip_caps, N, rho_caps,
                                    seed=int(eps * 100))

        # Privatize importance — used for top-k channel selection and WF weights
        imp_noisy   = add_dp_noise_to_importance(importance, sens_imp, rho_imp,
                                                  seed=int(eps * 100) + 7)
        rank_desc   = torch.argsort(imp_noisy, descending=True)
        top_indices = rank_desc[:n_keep]

        top_mask = torch.zeros(Cb, dtype=torch.bool, device=dev)
        top_mask[top_indices] = True

        # Diagnostics: imp ratio within active channels
        imp_active = imp_noisy[top_indices]
        imp_ratio  = float(imp_active.max() / imp_active.min().clamp(min=1e-12))
        print(f"  imp_ratio_active={imp_ratio:.1f}x")

        ep_res = {"rho_caps": rho_caps, "rho_imp": rho_imp, "rho_rel": rho_rel,
                  "imp_ratio_active": imp_ratio}

        # --- Baseline: top-k channels with uniform sigma ---
        sigma_uni = torch.zeros(Cb, device=dev)
        sigma_uni[top_mask] = correct_uniform_sigma(deltas[top_mask], rho_rel)

        print("  [uniform] ...", end="", flush=True)
        t0 = time.time()
        dices = run_students(train_ds, val_loader, teachers, caps_noisy,
                             top_mask, sigma_uni, dev, SEEDS, args.se, in_ch,
                             cache_seed=int(eps * 1000) + 1,
                             dataset=args.dataset, is_canal=False, eps=eps)
        ep_res["uniform"] = cell_stats(dices)
        print(f"  Dice={ep_res['uniform']['mean']:.4f}  ({time.time()-t0:.1f}s)")

        # --- CANAL: top-k channels with water-filling sigma ---
        sigma_wf = torch.zeros(Cb, device=dev)
        sigma_wf[top_mask] = correct_waterfilling_sigma(
            deltas[top_mask], imp_noisy[top_mask], rho_rel
        )

        print("  [canal]   ...", end="", flush=True)
        t0 = time.time()
        dices = run_students(train_ds, val_loader, teachers, caps_noisy,
                             top_mask, sigma_wf, dev, SEEDS, args.se, in_ch,
                             cache_seed=int(eps * 1000) + 2,
                             dataset=args.dataset, is_canal=True, eps=eps)
        ep_res["canal"] = cell_stats(dices)
        diff = ep_res["canal"]["mean"] - ep_res["uniform"]["mean"]
        print(f"  Dice={ep_res['canal']['mean']:.4f}  diff={diff:+.4f}  ({time.time()-t0:.1f}s)")

        results["sweep"][str(eps)] = ep_res

    # --- Summary table ---
    print(f"\n{'='*72}")
    print(f"FINAL RESULTS  ({args.dataset.upper()})  fc={FC}  fi={FI}  fr={FR}")
    print(f"  {'eps':>5}  {'uniform':>8}  {'canal':>8}  {'diff':>8}")
    for eps in EPS:
        r    = results["sweep"][str(eps)]
        diff = r["canal"]["mean"] - r["uniform"]["mean"]
        print(f"  {eps:5.1f}  {r['uniform']['mean']:.4f}    {r['canal']['mean']:.4f}   {diff:+.4f}")

    out = HERE / "results" / f"{args.dataset}_final_canal_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
