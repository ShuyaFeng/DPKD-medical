"""
Importance metric ablation for CANAL — BUSI pilot.

Root problem: WF requires channels with genuinely different importance.
Both gradient_energy and bn_gamma collapse to flat after DP noise because
BatchNorm equalized them before we even measure.

Three metrics tested — each with a different reason to be non-flat:
  1. gradient_energy  — existing baseline (post-BN, equalized by BN)
  2. running_var      — BN's own running_var = pre-BN variance per channel.
                        This is what BN divides out, so it is NOT equalized.
  3. weight_magnitude — L1 norm of enc3 last-conv output weights per channel.
                        Completely independent of BN normalization.

Conditions per epsilon (all with budget split: rho_imp=10%, rho_rel=90%):

  Shared baselines (run once):
    all_uniform          — all C channels, uniform sigma
    rand_uniform         — random 10%, uniform sigma  (avg N_RAND configs)
    rand_canal           — random 10%, WF sigma       (avg N_RAND configs)

  Per importance metric M in {gradient_energy, running_var, weight_magnitude}:
    top_uniform_{M}      — top-10% by M, uniform sigma
    top_canal_{M}        — top-10% by M, WF sigma
    all_canal_{M}        — ALL channels,  WF sigma

Key diagnostic per metric:
  raw max/min, noisy max/min, sigma_canal_ratio
  → sigma_ratio >> 1.0x means WF is doing real differentiation.

Usage:
  python canal_importance_methods.py --dataset busi --seeds 3 --te 60 --se 40
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
METRICS   = ["gradient_energy", "running_var", "weight_magnitude"]


# ── dataset ──────────────────────────────────────────────────────────────────

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


# ── importance metrics ────────────────────────────────────────────────────────

def compute_gradient_energy(teacher, train_loader, device):
    """Post-BN gradient energy (existing baseline). Flat due to BN equalization."""
    from drive_local_demo import compute_importance
    return compute_importance(teacher, train_loader, device).cpu()


def compute_running_var(teachers):
    """Per-channel pre-BN variance stored in the last BN of enc3.
    BN normalises by dividing by sqrt(running_var), making gradient energy flat.
    running_var itself captures the raw channel variation before that step.
    enc3[4] = second BN in conv_block(base*2, base*4).
    """
    rvs = [t.enc3[4].running_var.detach().cpu() for t in teachers]
    return torch.stack(rvs).mean(dim=0)


def compute_weight_magnitude(teachers):
    """L1 norm of output-channel weights in the last Conv2d of enc3.
    enc3[3] = second Conv2d; weight shape [out_ch, in_ch, kH, kW].
    Completely unaffected by BN normalization.
    """
    mags = [t.enc3[3].weight.detach().abs().mean(dim=[1, 2, 3]).cpu()
            for t in teachers]
    return torch.stack(mags).mean(dim=0)


def get_all_importances(teachers, train_loader, device):
    print("  Computing importance metrics ...", flush=True)
    return {
        "gradient_energy": compute_gradient_energy(teachers[0], train_loader, device),
        "running_var":     compute_running_var(teachers),
        "weight_magnitude": compute_weight_magnitude(teachers),
    }


# ── helpers ───────────────────────────────────────────────────────────────────

def imp_stats(imp):
    mn = float(imp.min()); mx = float(imp.max()); mu = float(imp.mean())
    return {
        "min": mn, "max": mx, "mean": mu, "std": float(imp.std()),
        "max_over_mean": mx / mu if mu > 0 else float("inf"),
        "max_over_min":  mx / mn  if mn > 0 else float("inf"),
    }


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


# ── main ──────────────────────────────────────────────────────────────────────

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

    raw_imps = get_all_importances(teachers, train_loader, dev)

    print(f"\n  Raw importance stats (before DP noise):")
    print(f"  {'Metric':<20} {'max/mean':>10} {'max/min':>10}")
    print(f"  {'-'*42}")
    for m, imp in raw_imps.items():
        s = imp_stats(imp)
        print(f"  {m:<20} {s['max_over_mean']:>10.2f}x {s['max_over_min']:>10.1f}x")

    sens_imp  = 2.0 * CLIP_IMP / float(N)
    deltas    = torch.full((Cb,), 2.0 / K, device=dev)
    all_idx   = torch.arange(Cb, device=dev)
    all_mask  = torch.ones(Cb, dtype=torch.bool, device=dev)

    results = {
        "dataset": args.dataset, "K": K, "keep_frac": KEEP_FRAC,
        "n_keep": n_keep, "N_RAND": N_RAND, "epsilons": EPS,
        "seeds": SEEDS, "ALPHA_IMP": ALPHA_IMP, "C": Cb,
        "metrics": METRICS,
        "raw_importance_stats": {m: imp_stats(v) for m, v in raw_imps.items()},
        "sweep": {},
    }
    out = HERE / "results" / f"{args.dataset}_importance_methods_results.json"

    for eps in EPS:
        rho     = eps_to_rho(eps)
        rho_imp = ALPHA_IMP * rho
        rho_rel = (1.0 - ALPHA_IMP) * rho
        print(f"\n{'='*72}")
        print(f"eps={eps}  rho_imp={rho_imp:.5f}  rho_rel={rho_rel:.4f}")

        # ── Add DP noise to each metric and print diagnostics ──────────────
        noisy_imps  = {}
        top_idx_per = {}
        print(f"\n  {'Metric':<20} {'raw max/min':>12} {'noisy max/min':>14} {'sigma_ratio(top)':>17}")
        print(f"  {'-'*67}")
        for mi, m in enumerate(METRICS):
            imp_d      = raw_imps[m].to(dev)
            nseed      = int(eps * 100) + 7 + mi * 31
            imp_noisy  = add_dp_noise_to_importance(imp_d, sens_imp, rho_imp, seed=nseed)
            noisy_imps[m] = imp_noisy

            top_idx_per[m] = torch.argsort(imp_noisy, descending=True)[:n_keep]

            _, sig_test = build_sigma(Cb, top_idx_per[m], deltas, imp_noisy, rho_rel, "canal", dev)
            sr = sigma_ratio(sig_test, sig_test > 0)

            raw_r   = float(raw_imps[m].max()   / raw_imps[m].clamp(min=1e-12).min())
            noisy_r = float(imp_noisy.max() / imp_noisy.clamp(min=1e-12).min())
            print(f"  {m:<20} {raw_r:>12.2f}x {noisy_r:>14.2f}x {sr:>17.3f}x")

        ep_res = {}

        # ── 1. all_uniform (shared baseline) ──────────────────────────────
        print(f"\n  [shared] all_uniform ...", end="", flush=True)
        t0 = time.time()
        sigma_au = correct_uniform_sigma(deltas, rho_rel)
        dices = run_students(train_ds, val_loader, teachers, caps_list,
                             all_mask, sigma_au, dev, SEEDS, args.se, in_ch,
                             cache_seed=int(eps * 1000) + 1)
        ep_res["all_uniform"] = cell(dices)
        print(f"  Dice={ep_res['all_uniform']['mean']:.4f}  ({time.time()-t0:.1f}s)")

        # ── 2. rand_uniform + rand_canal (shared, N_RAND configs) ──────────
        print(f"  [shared] rand_uniform + rand_canal ({N_RAND} configs) ...")
        # random conditions use gradient_energy noisy imp for WF (standard)
        imp_ge = noisy_imps["gradient_energy"]
        ru_dices = []; rc_dices = []
        for r in range(N_RAND):
            rng = torch.Generator().manual_seed(r * 137 + int(eps * 1000))
            rand_idx = torch.randperm(Cb, generator=rng)[:n_keep]
            rm_u, sig_ru = build_sigma(Cb, rand_idx, deltas, imp_ge, rho_rel, "uniform", dev)
            rm_c, sig_rc = build_sigma(Cb, rand_idx, deltas, imp_ge, rho_rel, "canal",   dev)
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

        # ── 3. Per-metric: top_uniform, top_canal, all_canal ───────────────
        for mi, m in enumerate(METRICS):
            imp_n   = noisy_imps[m]
            top_idx = top_idx_per[m]
            base    = int(eps * 1000) + 100 + mi * 10

            # top_uniform_{m}
            print(f"  [{m}] top_uniform ...", end="", flush=True)
            t0 = time.time()
            mask_tu, sig_tu = build_sigma(Cb, top_idx, deltas, imp_n, rho_rel, "uniform", dev)
            dices = run_students(train_ds, val_loader, teachers, caps_list,
                                 mask_tu, sig_tu, dev, SEEDS, args.se, in_ch,
                                 cache_seed=base + 1)
            ep_res[f"top_uniform_{m}"] = cell(dices)
            print(f"  Dice={ep_res[f'top_uniform_{m}']['mean']:.4f}  ({time.time()-t0:.1f}s)")

            # top_canal_{m}
            print(f"  [{m}] top_canal   ...", end="", flush=True)
            t0 = time.time()
            mask_tc, sig_tc = build_sigma(Cb, top_idx, deltas, imp_n, rho_rel, "canal", dev)
            sr_top = sigma_ratio(sig_tc, mask_tc)
            dices = run_students(train_ds, val_loader, teachers, caps_list,
                                 mask_tc, sig_tc, dev, SEEDS, args.se, in_ch,
                                 cache_seed=base + 2)
            ep_res[f"top_canal_{m}"] = cell(dices)
            print(f"  Dice={ep_res[f'top_canal_{m}']['mean']:.4f}  "
                  f"sigma_ratio={sr_top:.3f}x  ({time.time()-t0:.1f}s)")

            # all_canal_{m}
            print(f"  [{m}] all_canal   ...", end="", flush=True)
            t0 = time.time()
            _, sig_ac = build_sigma(Cb, all_idx, deltas, imp_n, rho_rel, "canal", dev)
            sr_all = sigma_ratio(sig_ac, all_mask)
            dices = run_students(train_ds, val_loader, teachers, caps_list,
                                 all_mask, sig_ac, dev, SEEDS, args.se, in_ch,
                                 cache_seed=base + 3)
            ep_res[f"all_canal_{m}"] = cell(dices)
            print(f"  Dice={ep_res[f'all_canal_{m}']['mean']:.4f}  "
                  f"sigma_ratio={sr_all:.3f}x  ({time.time()-t0:.1f}s)")

            ep_res[f"sigma_stats_{m}"] = {
                "top_canal_ratio": sr_top,
                "all_canal_ratio": sr_all,
            }

        results["sweep"][str(eps)] = ep_res
        out.write_text(json.dumps(results, indent=2))
        print(f"  [checkpoint saved after eps={eps}]")

    # ── Summary ───────────────────────────────────────────────────────────────
    W = 14
    print("\n" + "="*120)
    print(f"IMPORTANCE METHODS  ({args.dataset.upper()}, K={K}, keep={KEEP_FRAC*100:.0f}%)")
    hdr = f"{'eps':>5} | {'all_uni':>8} | {'rand_uni':>8} | {'rand_c':>8}"
    for m in METRICS:
        s = m[:5]
        hdr += f" | {'tu_'+s:>{W}} | {'tc_'+s:>{W}} | {'ac_'+s:>{W}}"
    print(hdr)
    print("-"*120)
    for eps in EPS:
        r = results["sweep"].get(str(eps), {})
        if not r:
            continue
        row = (f"{eps:>5.1f} | {r['all_uniform']['mean']:>8.4f} | "
               f"{r['rand_uniform']['mean']:>8.4f} | {r['rand_canal']['mean']:>8.4f}")
        for m in METRICS:
            tu = r.get(f"top_uniform_{m}", {}).get("mean", float("nan"))
            tc = r.get(f"top_canal_{m}",   {}).get("mean", float("nan"))
            ac = r.get(f"all_canal_{m}",   {}).get("mean", float("nan"))
            row += f" | {tu:>{W}.4f} | {tc:>{W}.4f} | {ac:>{W}.4f}"
        print(row)
    print("="*120)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
