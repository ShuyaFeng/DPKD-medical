"""
BN Gamma vs Gradient Energy importance — BUSI pilot experiment.

Problem with gradient energy: BatchNorm normalizes all channels to unit
variance before the gradient flows back, so gradient energy ends up nearly
flat across channels (max/mean ≈ 2–5x). After DP noise is added to
importance, the signal is completely destroyed → CANAL ≈ Uniform.

Proposed fix: use the BatchNorm gamma (learned scale) from the last BN
layer in enc3 as the importance signal instead. Gamma captures how much
the network has learned to rely on each channel AFTER normalization, so
it is not flattened by BatchNorm's normalization step. Max/mean ratio
is typically 5–20x, which survives DP noise much better.

The DP noise on importance is kept identical to the current CANAL
pipeline (same rho_imp budget, same sensitivity formula) so the
comparison is fair.

Same 5 conditions as canal_random_selection v2:
  1. all_uniform  — all C channels, uniform sigma
  2. top_uniform  — top-10% by gamma importance, uniform sigma
  3. rand_uniform — random 10%, uniform sigma (avg 5 configs)
  4. top_canal    — top-10% by gamma importance, WF sigma
  5. rand_canal   — random 10%, WF sigma (avg 5 configs)

Key comparison vs canal_random_selection_v2_results.json:
  Does gamma importance give top_canal > rand_canal by more than
  gradient energy did? If yes → better importance metric rescues CANAL.

Usage:
  python canal_gamma_importance.py --dataset busi --seeds 3 --te 60 --se 40
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
from drive_local_demo import collect_caps
from synthetic_demo import eps_to_rho

HERE = Path(__file__).parent
(HERE / "results").mkdir(exist_ok=True)

ALPHA_IMP = 0.1   # fraction of rho spent privatising importance (same as current CANAL)
CLIP_IMP  = 1e-8  # sensitivity clip (kept same for fair comparison)
K         = 3
N_RAND    = 5
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


def gamma_importance(teachers):
    """
    Per-channel importance from the last BatchNorm in enc3 (bottleneck).

    enc3 = conv_block(base*2, base*4):
      [0] Conv2d  [1] BN  [2] ReLU  [3] Conv2d  [4] BN  [5] ReLU
                                                  ^^^^ gamma lives here

    Average |gamma_c| across K teachers.
    Gamma encodes how much the network scaled each channel after
    normalisation — not flattened by BN's /std_c operation.
    """
    gammas = []
    for t in teachers:
        gamma = t.enc3[4].weight.detach().abs().cpu()
        gammas.append(gamma)
    return torch.stack(gammas).mean(dim=0)


def build_sigma(C, selected_indices, deltas, imp, rho_rel, mode, device):
    active_mask = torch.zeros(C, dtype=torch.bool, device=device)
    active_mask[selected_indices] = True
    sigma = torch.zeros(C, device=device)
    if mode == "uniform":
        sigma[active_mask] = correct_uniform_sigma(deltas[active_mask], rho_rel)
    else:  # canal
        sigma[active_mask] = correct_waterfilling_sigma(
            deltas[active_mask], imp[active_mask], rho_rel
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
    return {"dices": dices, "mean": m, "std": s, "sem": s / math.sqrt(len(dices))}


def importance_stats(imp):
    mn = float(imp.min()); mx = float(imp.max())
    mu = float(imp.mean()); sd = float(imp.std())
    return {
        "min": mn, "max": mx, "mean": mu, "std": sd,
        "max_over_mean": mx / mu,
        "max_over_min": mx / mn if mn > 0 else float("inf"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="busi", choices=["isic", "kvasir", "busi"])
    ap.add_argument("--size",    type=int, default=96)
    ap.add_argument("--seeds",   type=int, default=3)
    ap.add_argument("--te",      type=int, default=60)
    ap.add_argument("--se",      type=int, default=40)
    ap.add_argument("--epsilons", default="0.5,1,2,4,6")
    args = ap.parse_args()

    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else "cpu")
    EPS   = [float(e) for e in args.epsilons.split(",")]
    SEEDS = list(range(100, 100 + args.seeds * 100, 100))

    print(f"[{args.dataset}] device={dev}  K={K}  eps={EPS}  seeds={SEEDS}")

    train_ds, in_ch = get_dataset(args.dataset, "train", args.size)
    val_ds,   _     = get_dataset(args.dataset, "val",   args.size)
    val_loader  = DataLoader(val_ds,   batch_size=8, shuffle=False)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=False)
    N = len(train_ds)
    print(f"  train={N}  val={len(val_ds)}  in_ch={in_ch}")

    teachers, caps_list = train_K_teachers(train_ds, K, dev, n_epochs=args.te, in_ch=in_ch)
    Cb     = teachers[0].base * 4
    n_keep = max(1, int(round(KEEP_FRAC * Cb)))
    print(f"  bottleneck C={Cb}  n_keep={n_keep} ({KEEP_FRAC*100:.0f}%)")

    # -----------------------------------------------------------------
    # Compute BOTH importance metrics and print stats for comparison
    # -----------------------------------------------------------------
    from drive_local_demo import compute_importance as grad_importance
    imp_grad  = grad_importance(teachers[0], train_loader, dev)
    imp_gamma = gamma_importance(teachers).to(dev)

    gs = importance_stats(imp_grad.cpu())
    ys = importance_stats(imp_gamma.cpu())
    print(f"\n  Importance comparison:")
    print(f"    gradient_energy: max/mean={gs['max_over_mean']:.2f}x  max/min={gs['max_over_min']:.1f}x")
    print(f"    bn_gamma:        max/mean={ys['max_over_mean']:.2f}x  max/min={ys['max_over_min']:.1f}x")
    print(f"  (higher ratio = more variation = WF has more to exploit)")

    sens_imp = 2.0 * CLIP_IMP / float(N)
    deltas   = torch.full((Cb,), 2.0 / K, device=dev)

    results = {
        "dataset": args.dataset, "K": K, "keep_frac": KEEP_FRAC,
        "n_keep": n_keep, "N_RAND": N_RAND, "epsilons": EPS,
        "seeds": SEEDS, "ALPHA_IMP": ALPHA_IMP, "C": Cb,
        "importance_method": "bn_gamma",
        "importance_stats": {
            "gradient_energy": gs,
            "bn_gamma": ys,
        },
        "sweep": {},
    }

    for eps in EPS:
        rho     = eps_to_rho(eps)
        rho_imp = ALPHA_IMP * rho
        rho_rel = (1.0 - ALPHA_IMP) * rho

        # Add DP noise to gamma importance (same formula as current CANAL)
        noise_seed = int(eps * 100) + 7
        imp_noisy = add_dp_noise_to_importance(imp_gamma, sens_imp, rho_imp, seed=noise_seed)

        # Check how much variation survives the DP noise
        active_ratio = float(imp_noisy.max() / imp_noisy.clamp(min=1e-12).min())
        print(f"\n{'='*64}")
        print(f"eps={eps}  rho_rel={rho_rel:.4f}")
        print(f"  imp_noisy max/min after DP noise: {active_ratio:.2f}x")

        rank_desc  = torch.argsort(imp_noisy, descending=True)
        top_indices = rank_desc[:n_keep]

        ep_res = {}

        # 1. all channels + uniform
        print("  [1/5] all_uniform ...", end="", flush=True)
        t0 = time.time()
        all_mask = torch.ones(Cb, dtype=torch.bool, device=dev)
        sigma_all = correct_uniform_sigma(deltas, rho_rel)
        dices = run_students(train_ds, val_loader, teachers, caps_list,
                             all_mask, sigma_all, dev, SEEDS, args.se, in_ch,
                             cache_seed=int(eps * 1000) + 1)
        ep_res["all_uniform"] = cell_stats(dices)
        print(f"  Dice={ep_res['all_uniform']['mean']:.4f}  ({time.time()-t0:.1f}s)")

        # 2. top-n_keep + uniform
        print("  [2/5] top_uniform ...", end="", flush=True)
        t0 = time.time()
        top_mask_u, sigma_top_u = build_sigma(Cb, top_indices, deltas, imp_noisy, rho_rel, "uniform", dev)
        dices = run_students(train_ds, val_loader, teachers, caps_list,
                             top_mask_u, sigma_top_u, dev, SEEDS, args.se, in_ch,
                             cache_seed=int(eps * 1000) + 2)
        ep_res["top_uniform"] = cell_stats(dices)
        print(f"  Dice={ep_res['top_uniform']['mean']:.4f}  ({time.time()-t0:.1f}s)")

        # 3. random + uniform AND canal (avg over N_RAND configs)
        print(f"  [3/5] rand_uniform + rand_canal ({N_RAND} configs) ...")
        rand_uni_dices = []; rand_canal_dices = []
        for r in range(N_RAND):
            rand_perm = torch.randperm(
                Cb, generator=torch.Generator().manual_seed(r * 137 + int(eps * 1000))
            )
            rand_idx = rand_perm[:n_keep]
            rm_u, sig_ru = build_sigma(Cb, rand_idx, deltas, imp_noisy, rho_rel, "uniform", dev)
            rm_c, sig_rc = build_sigma(Cb, rand_idx, deltas, imp_noisy, rho_rel, "canal",   dev)
            du = run_students(train_ds, val_loader, teachers, caps_list,
                              rm_u, sig_ru, dev, SEEDS, args.se, in_ch,
                              cache_seed=int(eps * 1000) + 30 + r)
            dc = run_students(train_ds, val_loader, teachers, caps_list,
                              rm_c, sig_rc, dev, SEEDS, args.se, in_ch,
                              cache_seed=int(eps * 1000) + 60 + r)
            rand_uni_dices.extend(du); rand_canal_dices.extend(dc)
            print(f"    config {r+1}/{N_RAND}  rand_uni={np.mean(du):.4f}  rand_canal={np.mean(dc):.4f}")
        ep_res["rand_uniform"] = cell_stats(rand_uni_dices)
        ep_res["rand_canal"]   = cell_stats(rand_canal_dices)

        # 4. top-n_keep + canal (WF with gamma importance)
        print("  [4/5] top_canal ...", end="", flush=True)
        t0 = time.time()
        top_mask_c, sigma_canal = build_sigma(Cb, top_indices, deltas, imp_noisy, rho_rel, "canal", dev)

        # Log sigma variation within active channels
        active_sigma_u = sigma_top_u[top_mask_u]
        active_sigma_c = sigma_canal[top_mask_c]
        print(f"\n    sigma_uni[active]  : min={active_sigma_u.min():.4f}  max={active_sigma_u.max():.4f}  ratio={active_sigma_u.max()/active_sigma_u.min():.3f}x")
        print(f"    sigma_canal[active]: min={active_sigma_c.min():.4f}  max={active_sigma_c.max():.4f}  ratio={active_sigma_c.max()/active_sigma_c.min():.3f}x")

        dices = run_students(train_ds, val_loader, teachers, caps_list,
                             top_mask_c, sigma_canal, dev, SEEDS, args.se, in_ch,
                             cache_seed=int(eps * 1000) + 4)
        ep_res["top_canal"] = cell_stats(dices)
        ep_res["sigma_stats"] = {
            "uni_min":   float(active_sigma_u.min()),
            "uni_max":   float(active_sigma_u.max()),
            "canal_min": float(active_sigma_c.min()),
            "canal_max": float(active_sigma_c.max()),
            "canal_ratio": float(active_sigma_c.max() / active_sigma_c.min()),
        }
        print(f"  top_canal Dice={ep_res['top_canal']['mean']:.4f}  ({time.time()-t0:.1f}s)")

        results["sweep"][str(eps)] = ep_res

        # Incremental save
        out = HERE / "results" / f"{args.dataset}_gamma_importance_results.json"
        out.write_text(json.dumps(results, indent=2))
        print(f"  [checkpoint saved after eps={eps}]")

    # Summary
    print("\n" + "="*90)
    print(f"BN GAMMA IMPORTANCE  ({args.dataset.upper()}, K={K}, keep={KEEP_FRAC*100:.0f}%)")
    print(f"{'eps':>5} | {'all_uni':>8} | {'top_uni':>8} | {'rand_uni':>9} | {'top_canal':>9} | {'rand_canal':>10} | {'topc-randc':>11}")
    print("-"*90)
    for eps in EPS:
        r = results["sweep"][str(eps)]
        print(f"{eps:>5.1f} | {r['all_uniform']['mean']:>8.4f} | {r['top_uniform']['mean']:>8.4f} | "
              f"{r['rand_uniform']['mean']:>9.4f} | {r['top_canal']['mean']:>9.4f} | "
              f"{r['rand_canal']['mean']:>10.4f} | "
              f"{r['top_canal']['mean']-r['rand_canal']['mean']:>+11.4f}")
    print("="*90)

    out = HERE / "results" / f"{args.dataset}_gamma_importance_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
