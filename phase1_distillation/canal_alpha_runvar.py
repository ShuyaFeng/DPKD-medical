"""
Power-scaling running_var importance for CANAL — BUSI pilot.

Problem: WF formula sigma ∝ 1/importance^(1/4) means even a 3.14x
importance ratio only produces 1.109x sigma ratio — too small to
consistently beat uniform noise allocation.

Fix: apply importance^alpha AFTER DP noise to amplify the differences
before feeding into WF. Since running_var survives DP noise intact
(ratio 3.14x preserved), scaling it amplifies real signal, not noise.

Expected sigma_ratio per alpha (top-13 channels, importance_ratio≈1.51x):
  alpha=1  →  1.109x  (current, already tested)
  alpha=2  →  1.229x
  alpha=4  →  1.510x
  alpha=8  →  2.280x
  alpha=16 →  5.190x

Privacy accounting: DP noise is added to raw running_var first (same
rho_imp=10% budget). Power scaling is applied post-noise and does not
affect the privacy guarantee of the importance privatisation step.

Conditions per alpha:
  top_canal_alphaA  — top-10% by running_var^A, WF sigma
  top_uniform_alphaA — top-10% by running_var^A, uniform sigma

Shared baselines (run once):
  all_uniform
  rand_uniform + rand_canal

Usage:
  python canal_alpha_runvar.py --dataset busi --seeds 3 --te 60 --se 40
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
from drive_pate_pruning_joint import precompute_joint_cache
from drive_pate_canal_combined import add_dp_noise_to_importance, correct_waterfilling_sigma
from drive_student_distill import train_student_distill
from synthetic_demo import eps_to_rho

HERE = Path(__file__).parent
(HERE / "results").mkdir(exist_ok=True)

ALPHA_IMP = 0.1
CLIP_IMP  = 1e-8
K         = 3
N_RAND    = 5
KEEP_FRAC = 0.1
ALPHAS    = [1, 2, 4, 8, 16]


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


def compute_running_var(teachers):
    """Pre-BN variance from last BN in enc3 — survives DP noise intact."""
    rvs = [t.enc3[4].running_var.detach().cpu() for t in teachers]
    return torch.stack(rvs).mean(dim=0)


def build_sigma(C, selected_idx, deltas, imp, rho_rel, mode, device):
    mask  = torch.zeros(C, dtype=torch.bool, device=device)
    mask[selected_idx] = True
    sigma = torch.zeros(C, device=device)
    if mode == "uniform":
        sigma[mask] = correct_uniform_sigma(deltas[mask], rho_rel)
    else:
        sigma[mask] = correct_waterfilling_sigma(deltas[mask], imp[mask], rho_rel)
    return mask, sigma


def sigma_ratio(sigma, mask):
    a = sigma[mask]
    if a.min() < 1e-12:
        return float("inf")
    return float(a.max() / a.min())


def run_students(train_ds, val_loader, teachers, caps_list, mask, sigma,
                 device, seeds, n_ep, in_ch, cache_seed):
    cache = precompute_joint_cache(
        teachers, caps_list, train_ds, sigma, mask, device, seed=cache_seed
    )
    dices = []
    for s in seeds:
        best, _ = train_student_distill(
            train_ds, val_loader, cache, device,
            student_base=16, teacher_base=32,
            n_epochs=n_ep, lr=1e-3, lambda_feat=0.4,
            seed=s, in_ch=in_ch,
        )
        dices.append(best)
    return dices


def cell(dices):
    m = float(np.mean(dices)); s = float(np.std(dices))
    return {"dices": dices, "mean": m, "std": s, "sem": s / math.sqrt(len(dices))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",  default="busi", choices=["isic", "kvasir", "busi"])
    ap.add_argument("--size",     type=int, default=96)
    ap.add_argument("--seeds",    type=int, default=3)
    ap.add_argument("--te",       type=int, default=60)
    ap.add_argument("--se",       type=int, default=40)
    ap.add_argument("--epsilons", default="0.5,1,2,4,6")
    args = ap.parse_args()

    dev   = ("cuda" if torch.cuda.is_available()
             else "mps" if torch.backends.mps.is_available() else "cpu")
    EPS   = [float(e) for e in args.epsilons.split(",")]
    SEEDS = list(range(100, 100 + args.seeds * 100, 100))
    print(f"[{args.dataset}] device={dev}  K={K}  eps={EPS}  seeds={SEEDS}")

    train_ds, in_ch = get_dataset(args.dataset, "train", args.size)
    val_ds,   _     = get_dataset(args.dataset, "val",   args.size)
    val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=False)
    N = len(train_ds)
    print(f"  train={N}  val={len(val_ds)}  in_ch={in_ch}")

    teachers, caps_list = train_K_teachers(train_ds, K, dev, n_epochs=args.te, in_ch=in_ch)
    Cb     = teachers[0].base * 4
    n_keep = max(1, int(round(KEEP_FRAC * Cb)))
    print(f"  bottleneck C={Cb}  n_keep={n_keep} ({KEEP_FRAC*100:.0f}%)")

    imp_raw  = compute_running_var(teachers).to(dev)
    sens_imp = 2.0 * CLIP_IMP / float(N)
    deltas   = torch.full((Cb,), 2.0 / K, device=dev)
    all_mask = torch.ones(Cb, dtype=torch.bool, device=dev)

    rv_min = float(imp_raw.min()); rv_max = float(imp_raw.max())
    print(f"  running_var: min={rv_min:.3f}  max={rv_max:.3f}  ratio={rv_max/rv_min:.2f}x")

    results = {
        "dataset": args.dataset, "K": K, "keep_frac": KEEP_FRAC,
        "n_keep": n_keep, "N_RAND": N_RAND, "epsilons": EPS,
        "seeds": SEEDS, "ALPHA_IMP": ALPHA_IMP, "C": Cb,
        "alphas": ALPHAS,
        "running_var_raw_ratio": rv_max / rv_min,
        "sweep": {},
    }
    out = HERE / "results" / f"{args.dataset}_alpha_runvar_results.json"

    for eps in EPS:
        rho     = eps_to_rho(eps)
        rho_imp = ALPHA_IMP * rho
        rho_rel = (1.0 - ALPHA_IMP) * rho
        print(f"\n{'='*72}")
        print(f"eps={eps}  rho_imp={rho_imp:.5f}  rho_rel={rho_rel:.4f}")

        # Add DP noise to raw running_var (once per epsilon)
        imp_noisy = add_dp_noise_to_importance(
            imp_raw, sens_imp, rho_imp, seed=int(eps * 100) + 7
        )
        noisy_ratio = float(imp_noisy.max() / imp_noisy.clamp(min=1e-12).min())
        print(f"  running_var noisy ratio: {noisy_ratio:.2f}x")

        # Print sigma_ratio diagnostic for all alphas before training
        print(f"\n  {'alpha':>6} {'imp_ratio(top)':>15} {'sigma_ratio':>12}")
        print(f"  {'-'*36}")
        for a in ALPHAS:
            imp_scaled = imp_noisy.clamp(min=1e-12).pow(a)
            top_idx_a  = torch.argsort(imp_scaled, descending=True)[:n_keep]
            _, sig_t   = build_sigma(Cb, top_idx_a, deltas, imp_scaled, rho_rel, "canal", dev)
            sr         = sigma_ratio(sig_t, sig_t > 0)
            top_ratio  = float(imp_scaled[top_idx_a].max() / imp_scaled[top_idx_a].min())
            print(f"  {a:>6}  {top_ratio:>15.2f}x  {sr:>12.3f}x")

        ep_res = {}

        # ── shared: all_uniform ────────────────────────────────────────────
        print(f"\n  [shared] all_uniform ...", end="", flush=True)
        t0 = time.time()
        sigma_au = correct_uniform_sigma(deltas, rho_rel)
        dices = run_students(train_ds, val_loader, teachers, caps_list,
                             all_mask, sigma_au, dev, SEEDS, args.se, in_ch,
                             cache_seed=int(eps * 1000) + 1)
        ep_res["all_uniform"] = cell(dices)
        print(f"  Dice={ep_res['all_uniform']['mean']:.4f}  ({time.time()-t0:.1f}s)")

        # ── shared: rand_uniform + rand_canal ─────────────────────────────
        print(f"  [shared] rand_uniform + rand_canal ({N_RAND} configs) ...")
        imp_ge_noisy = imp_noisy  # use running_var noisy imp for rand_canal WF
        ru_dices = []; rc_dices = []
        for r in range(N_RAND):
            rng = torch.Generator().manual_seed(r * 137 + int(eps * 1000))
            rand_idx = torch.randperm(Cb, generator=rng)[:n_keep]
            rm_u, sig_ru = build_sigma(Cb, rand_idx, deltas, imp_ge_noisy, rho_rel, "uniform", dev)
            rm_c, sig_rc = build_sigma(Cb, rand_idx, deltas, imp_ge_noisy, rho_rel, "canal",   dev)
            du = run_students(train_ds, val_loader, teachers, caps_list,
                              rm_u, sig_ru, dev, SEEDS, args.se, in_ch,
                              cache_seed=int(eps * 1000) + 30 + r)
            dc = run_students(train_ds, val_loader, teachers, caps_list,
                              rm_c, sig_rc, dev, SEEDS, args.se, in_ch,
                              cache_seed=int(eps * 1000) + 60 + r)
            ru_dices.extend(du); rc_dices.extend(dc)
            print(f"    config {r+1}/{N_RAND}  rand_uni={np.mean(du):.4f}  rand_canal={np.mean(dc):.4f}")
        ep_res["rand_uniform"] = cell(ru_dices)
        ep_res["rand_canal"]   = cell(rc_dices)

        # ── per alpha: top_uniform + top_canal ────────────────────────────
        for ai, a in enumerate(ALPHAS):
            imp_scaled = imp_noisy.clamp(min=1e-12).pow(a)
            top_idx    = torch.argsort(imp_scaled, descending=True)[:n_keep]
            base_seed  = int(eps * 1000) + 100 + ai * 10

            # top_uniform
            print(f"  [alpha={a}] top_uniform ...", end="", flush=True)
            t0 = time.time()
            mask_u, sig_u = build_sigma(Cb, top_idx, deltas, imp_scaled, rho_rel, "uniform", dev)
            dices = run_students(train_ds, val_loader, teachers, caps_list,
                                 mask_u, sig_u, dev, SEEDS, args.se, in_ch,
                                 cache_seed=base_seed + 1)
            ep_res[f"top_uniform_a{a}"] = cell(dices)
            print(f"  Dice={ep_res[f'top_uniform_a{a}']['mean']:.4f}  ({time.time()-t0:.1f}s)")

            # top_canal
            print(f"  [alpha={a}] top_canal   ...", end="", flush=True)
            t0 = time.time()
            mask_c, sig_c = build_sigma(Cb, top_idx, deltas, imp_scaled, rho_rel, "canal", dev)
            sr = sigma_ratio(sig_c, mask_c)
            dices = run_students(train_ds, val_loader, teachers, caps_list,
                                 mask_c, sig_c, dev, SEEDS, args.se, in_ch,
                                 cache_seed=base_seed + 2)
            ep_res[f"top_canal_a{a}"] = cell(dices)
            ep_res[f"sigma_ratio_a{a}"] = sr
            diff = ep_res[f"top_canal_a{a}"]["mean"] - ep_res[f"top_uniform_a{a}"]["mean"]
            print(f"  Dice={ep_res[f'top_canal_a{a}']['mean']:.4f}  "
                  f"sigma_ratio={sr:.3f}x  canal-uni={diff:+.4f}  ({time.time()-t0:.1f}s)")

        results["sweep"][str(eps)] = ep_res
        out.write_text(json.dumps(results, indent=2))
        print(f"  [checkpoint saved after eps={eps}]")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*110)
    print(f"RUNNING_VAR POWER SCALING  ({args.dataset.upper()}, K={K}, keep={KEEP_FRAC*100:.0f}%)")
    hdr = f"{'eps':>5} | {'all_uni':>8} | {'rand_uni':>8} | {'rand_c':>8}"
    for a in ALPHAS:
        hdr += f" | {'tu_a'+str(a):>9} | {'tc_a'+str(a):>9} | {'diff':>7} | {'sig_r':>6}"
    print(hdr)
    print("-"*110)
    for eps in EPS:
        r = results["sweep"].get(str(eps), {})
        if not r:
            continue
        row = (f"{eps:>5.1f} | {r['all_uniform']['mean']:>8.4f} | "
               f"{r['rand_uniform']['mean']:>8.4f} | {r['rand_canal']['mean']:>8.4f}")
        for a in ALPHAS:
            tu   = r.get(f"top_uniform_a{a}", {}).get("mean", float("nan"))
            tc   = r.get(f"top_canal_a{a}",   {}).get("mean", float("nan"))
            sr   = r.get(f"sigma_ratio_a{a}",  float("nan"))
            diff = tc - tu
            row += f" | {tu:>9.4f} | {tc:>9.4f} | {diff:>+7.4f} | {sr:>6.3f}x"
        print(row)
    print("="*110)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
