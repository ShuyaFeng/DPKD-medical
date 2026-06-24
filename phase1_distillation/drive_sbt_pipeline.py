# -*- coding: utf-8 -*-
"""
SBT - Sparsity-induced Bottleneck Training - pipeline.

Goal: make TinyUNet's bottleneck channel importance heterogeneous (R >> 2),
so CANAL's R^{1/4} advantage has real room.

Stage 1: sweep lambda_sbt on K=1 teacher, report
   (a) clean val Dice  vs  (b) act_norm importance ratio R  vs  (c) near-zero channel fraction.
   Pick highest lambda whose Dice drop is within tolerance.

Stage 2: train K=1 + K=10 SBT teachers with the chosen lambda, then run the
   3-method sweep (PATE K=1 / PATE+uniform K=10 / PATE+CANAL K=10) over the
   user's epsilons, paired across seeds.

Server usage (no push needed; runs from working tree):
    python drive_sbt_pipeline.py --dataset drive --stage 1
    python drive_sbt_pipeline.py --dataset drive --stage 2 --lambda_sbt 1e-3
    python drive_sbt_pipeline.py --dataset isic  --stage 1
    python drive_sbt_pipeline.py --dataset isic  --stage 2 --lambda_sbt 1e-3
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent))
from drive_local_demo import (
    TinyUNet, evaluate_vessel_dice, compute_importance_actnorm, collect_caps,
)
from drive_pate_poc import partition_dataset, correct_uniform_sigma, precompute_pate_cache
from drive_pate_canal_combined import correct_waterfilling_sigma
from drive_student_distill import train_student_distill
from synthetic_demo import eps_to_rho, task_loss


HERE = Path(__file__).parent
(HERE / "results").mkdir(exist_ok=True)


def cleanup(device: str):
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def load_dataset(name: str, size: int):
    if name == "drive":
        from drive_local_demo import DriveDataset
        return DriveDataset("train", size=size), DriveDataset("val", size=size)
    if name == "isic":
        from isic_dataset import ISICDataset
        return ISICDataset(split="train", size=size), ISICDataset(split="val", size=size)
    raise ValueError(f"unknown dataset: {name}")


# ----------------------------------------------------------------------------
# SBT training - task loss + lambda_sbt * mean(|e3|)
# ----------------------------------------------------------------------------

def train_teacher_sbt(train_ds, n_epochs, lr, device, in_ch, lambda_sbt,
                      seed, base=32, val_loader=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    teacher = TinyUNet(in_ch=in_ch, num_classes=2, base=base).to(device)
    loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    opt = torch.optim.Adam(teacher.parameters(), lr=lr)
    for ep in range(n_epochs):
        teacher.train()
        ep_task = ep_l1 = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            e1, e2, e3 = teacher.encode(x)
            logits = teacher.decode(e1, e2, e3)
            tl = task_loss(logits, y)
            l1 = e3.abs().mean()
            loss = tl + lambda_sbt * l1
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_task += tl.item()
            ep_l1 += l1.item()
        if (ep + 1) % 10 == 0 or ep == 0:
            tag = ""
            if val_loader is not None:
                tag = f"  val_dice={evaluate_vessel_dice(teacher, val_loader, device):.4f}"
                teacher.train()
            print(f"    ep {ep+1:3d}/{n_epochs} task={ep_task/len(loader):.4f} "
                  f"l1={ep_l1/len(loader):.4f}{tag}")
    return teacher


def train_K_teachers_sbt(train_ds, K, n_epochs, lr, device, in_ch, lambda_sbt, base=32):
    cohorts = partition_dataset(len(train_ds), K)
    teachers, caps_list = [], []
    for k in range(K):
        print(f"  [SBT teacher {k+1}/{K}]")
        sub = Subset(train_ds, cohorts[k])
        t = train_teacher_sbt(
            sub, n_epochs, lr, device, in_ch, lambda_sbt,
            seed=1000 + k, base=base,
        )
        t.eval()
        for p in t.parameters():
            p.requires_grad_(False)
        caps = collect_caps(t, DataLoader(sub, batch_size=4), device).to(device)
        teachers.append(t)
        caps_list.append(caps)
        cleanup(device)
    return teachers, caps_list


def shared_actnorm_importance(teachers, loader, device):
    imps = [compute_importance_actnorm(t, loader, device) for t in teachers]
    return torch.stack(imps, dim=0).mean(dim=0)


# ----------------------------------------------------------------------------
# Stage 1 - lambda sweep on K=1
# ----------------------------------------------------------------------------

def stage_1(args, device):
    LAMBDAS = [float(x) for x in args.lambdas.split(",")]
    train_ds, val_ds = load_dataset(args.dataset, args.size)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    imp_loader = DataLoader(train_ds, batch_size=8, shuffle=False)
    print(f"[stage 1] dataset={args.dataset}  N_train={len(train_ds)}  N_val={len(val_ds)}")
    print(f"          lambdas = {LAMBDAS}")

    rows = []
    for lam in LAMBDAS:
        print(f"\n[lambda={lam:.0e}]")
        t0 = time.time()
        t = train_teacher_sbt(
            train_ds, args.te, 1e-3, device, args.in_ch, lam,
            seed=1000, base=32, val_loader=None,
        )
        t.eval()
        for p in t.parameters():
            p.requires_grad_(False)
        dice = evaluate_vessel_dice(t, val_loader, device)
        imp = compute_importance_actnorm(t, imp_loader, device)
        R = (imp.max() / imp.min()).item()
        near_zero = (imp < 0.01 * imp.max()).float().mean().item()
        rows.append({
            "lambda": lam,
            "dice": float(dice),
            "R": float(R),
            "near_zero_frac": float(near_zero),
            "imp_min": float(imp.min()),
            "imp_max": float(imp.max()),
            "elapsed_sec": time.time() - t0,
        })
        print(f"  Dice={dice:.4f}  R={R:.2f}  near_zero={near_zero:.1%}  "
              f"({rows[-1]['elapsed_sec']:.0f}s)")
        del t
        cleanup(device)

    out = HERE / "results" / f"sbt_lambda_sweep_{args.dataset}_results.json"
    out.write_text(json.dumps({
        "dataset": args.dataset, "te": args.te,
        "lambdas": LAMBDAS, "rows": rows,
    }, indent=2))
    print(f"\nSaved: {out}")

    # ---- summary table + recommendation ----
    print("\n" + "=" * 72)
    print(f"{'lambda':>10}  {'Dice':>8}  {'R':>8}  {'near_zero':>10}")
    print("-" * 72)
    for r in rows:
        print(f"{r['lambda']:>10.0e}  {r['dice']:>8.4f}  {r['R']:>8.2f}  "
              f"{r['near_zero_frac']:>9.1%}")
    print("=" * 72)

    base = rows[0]  # lambda=0 row (or smallest)
    tol = args.dice_tol
    cands = [r for r in rows if r["dice"] >= base["dice"] - tol]
    if cands:
        best = max(cands, key=lambda r: r["R"])
        print(f"\n[recommend] lambda = {best['lambda']:.0e}  "
              f"(R = {best['R']:.2f} vs baseline {base['R']:.2f}, "
              f"Dice {best['dice']:.4f} vs baseline {base['dice']:.4f})")
        print(f"\nStage 2 command:")
        print(f"  python drive_sbt_pipeline.py --dataset {args.dataset} "
              f"--stage 2 --lambda_sbt {best['lambda']:.0e}")
    else:
        print(f"\n[recommend] no lambda within {tol} Dice drop. Relax --dice_tol "
              "or extend --lambdas with smaller values.")


# ----------------------------------------------------------------------------
# Stage 2 - 3-method sweep on SBT teachers
# ----------------------------------------------------------------------------

def stage_2(args, device):
    EPS = [float(e) for e in args.epsilons.split(",")]
    SEEDS = list(range(100, 100 + args.seeds * 100, 100))
    train_ds, val_ds = load_dataset(args.dataset, args.size)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    imp_loader = DataLoader(train_ds, batch_size=8, shuffle=False)
    print(f"[stage 2] dataset={args.dataset}  lambda_sbt={args.lambda_sbt}  "
          f"eps={EPS}  seeds={SEEDS}")

    # ---- K=1 teacher ----
    print("\n[teachers] K=1 with SBT")
    t0 = time.time()
    teacher_k1 = train_teacher_sbt(
        train_ds, args.te, 1e-3, device, args.in_ch, args.lambda_sbt,
        seed=1000, base=32, val_loader=None,
    )
    teacher_k1.eval()
    for p in teacher_k1.parameters():
        p.requires_grad_(False)
    caps_k1 = [collect_caps(teacher_k1, DataLoader(train_ds, batch_size=4), device).to(device)]
    teachers_k1 = [teacher_k1]
    print(f"  K=1 done in {time.time() - t0:.0f}s, "
          f"clean Dice = {evaluate_vessel_dice(teacher_k1, val_loader, device):.4f}")
    cleanup(device)

    # ---- K=10 teachers ----
    print("\n[teachers] K=10 with SBT")
    t0 = time.time()
    teachers_k10, caps_k10 = train_K_teachers_sbt(
        train_ds, K=10, n_epochs=args.te, lr=1e-3, device=device,
        in_ch=args.in_ch, lambda_sbt=args.lambda_sbt, base=32,
    )
    print(f"  K=10 done in {time.time() - t0:.0f}s")

    # ---- shared importance over K=10 ----
    Cb = teacher_k1.base * 4
    imp_k10 = shared_actnorm_importance(teachers_k10, imp_loader, device).to(device)
    R10 = (imp_k10.max() / imp_k10.min()).item()
    nz10 = (imp_k10 < 0.01 * imp_k10.max()).float().mean().item()
    print(f"\n[importance] shared act_norm  R={R10:.2f}  near_zero={nz10:.1%}  C={Cb}")
    cleanup(device)

    results = {
        "dataset": args.dataset, "lambda_sbt": args.lambda_sbt,
        "epsilons": EPS, "seeds": SEEDS, "C": Cb,
        "R_K10_sbt": R10, "near_zero_K10_sbt": nz10,
        "series": {
            "PATE (K=1 SBT)":             {},
            "PATE+uniform (K=10 SBT)":    {},
            "PATE+CANAL (K=10 SBT)":      {},
        },
    }

    deltas_k1  = torch.full((Cb,), 2.0,        device=device)
    deltas_k10 = torch.full((Cb,), 2.0 / 10.0, device=device)

    def run_cell(teachers, caps_list, sigma, label, eps):
        cache = precompute_pate_cache(teachers, caps_list, train_ds, sigma,
                                      device, seed=42 + int(eps * 10))
        dices = []
        for s in SEEDS:
            t0 = time.time()
            best, _ = train_student_distill(
                train_ds, val_loader, cache, device,
                student_base=16, teacher_base=32,
                n_epochs=args.se, lr=1e-3, lambda_feat=0.4, seed=s, in_ch=args.in_ch,
            )
            dices.append(best)
            print(f"      {label} eps={eps} seed={s}: {best:.4f}  ({time.time() - t0:.0f}s)")
            cleanup(device)
        del cache
        cleanup(device)
        return {
            "dices": dices,
            "mean": float(np.mean(dices)),
            "std":  float(np.std(dices)),
            "sem":  float(np.std(dices) / np.sqrt(len(dices))),
        }

    save_path = HERE / "results" / (
        f"sbt_3method_{args.dataset}_lam{args.lambda_sbt:.0e}_results.json"
    )

    total = 3 * len(EPS)
    done = 0
    job_t0 = time.time()
    for eps in EPS:
        rho = eps_to_rho(eps)
        print(f"\n========== eps = {eps}  (rho = {rho:.5f}) ==========")

        # 1) PATE K=1 SBT
        sigma = correct_uniform_sigma(deltas_k1, rho)
        print(f"   [K=1]        sigma = {sigma[0].item():.4f}")
        results["series"]["PATE (K=1 SBT)"][str(eps)] = run_cell(
            teachers_k1, caps_k1, sigma, "PATE-K1", eps,
        )
        done += 1
        elapsed = time.time() - job_t0
        eta = elapsed / done * (total - done)
        print(f"   [progress] {done}/{total}  elapsed={elapsed/60:.1f}min  ETA={eta/60:.1f}min")
        save_path.write_text(json.dumps(results, indent=2))

        # 2) PATE+uniform K=10 SBT
        sigma = correct_uniform_sigma(deltas_k10, rho)
        print(f"   [K=10 uni]   sigma = {sigma[0].item():.4f}")
        results["series"]["PATE+uniform (K=10 SBT)"][str(eps)] = run_cell(
            teachers_k10, caps_k10, sigma, "PATE-K10-uni", eps,
        )
        done += 1
        elapsed = time.time() - job_t0
        eta = elapsed / done * (total - done)
        print(f"   [progress] {done}/{total}  elapsed={elapsed/60:.1f}min  ETA={eta/60:.1f}min")
        save_path.write_text(json.dumps(results, indent=2))

        # 3) PATE+CANAL K=10 SBT
        sigma = correct_waterfilling_sigma(deltas_k10, imp_k10, rho)
        top10 = torch.argsort(imp_k10, descending=True)[:10]
        bot10 = torch.argsort(imp_k10, descending=False)[:10]
        ratio = (sigma[bot10].mean() / sigma[top10].mean()).item()
        print(f"   [K=10 CANAL] sigma top10={sigma[top10].mean().item():.4f}  "
              f"bot10={sigma[bot10].mean().item():.4f}  ratio={ratio:.3f}x")
        results["series"]["PATE+CANAL (K=10 SBT)"][str(eps)] = run_cell(
            teachers_k10, caps_k10, sigma, "PATE-K10-CANAL", eps,
        )
        done += 1
        elapsed = time.time() - job_t0
        eta = elapsed / done * (total - done)
        print(f"   [progress] {done}/{total}  elapsed={elapsed/60:.1f}min  ETA={eta/60:.1f}min")
        save_path.write_text(json.dumps(results, indent=2))

    # ---- final summary + plot ----
    print("\n" + "=" * 96)
    print(f"SBT 3-method sweep  -  {args.dataset}  -  lambda={args.lambda_sbt:.0e}  "
          f"R(K=10)={R10:.2f}")
    print("=" * 96)
    hdr = f"{'eps':>6}  " + "  ".join(f"{name[:24]:>24s}" for name in results["series"])
    print(hdr)
    print("-" * len(hdr))
    for eps in EPS:
        row = f"{eps:>6.2f}  "
        for name in results["series"]:
            m = results["series"][name][str(eps)]
            row += f"{m['mean']:>10.4f} +/- {m['std']:.4f}    "
        print(row)
    print("=" * 96)

    save_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved JSON: {save_path}")

    # plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 6.5))
        colors  = {"PATE (K=1 SBT)": "#d62728",
                   "PATE+uniform (K=10 SBT)": "#2ca02c",
                   "PATE+CANAL (K=10 SBT)": "#9467bd"}
        markers = {"PATE (K=1 SBT)": "s",
                   "PATE+uniform (K=10 SBT)": "o",
                   "PATE+CANAL (K=10 SBT)": "^"}
        for name, data in results["series"].items():
            means = [data[str(e)]["mean"] for e in EPS]
            stds  = [data[str(e)]["std"]  for e in EPS]
            ax.errorbar(EPS, means, yerr=stds, fmt=f"{markers[name]}-",
                        color=colors[name], lw=2.2, ms=10, capsize=6, label=name)
        ax.set_xscale("log")
        ax.set_xticks(EPS)
        ax.set_xticklabels([str(e) for e in EPS])
        ax.set_xlabel("privacy budget  eps (user-level, sample-once)")
        ax.set_ylabel(f"Dice ({args.dataset} val)")
        ax.set_title(f"{args.dataset.upper()} SBT 3-method  (lambda={args.lambda_sbt:.0e}, "
                     f"R(K=10)={R10:.2f}, {args.seeds} seeds)")
        ax.legend(loc="best", fontsize=10)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        plot_path = HERE / f"fig_sbt_3method_{args.dataset}_lam{args.lambda_sbt:.0e}.png"
        fig.savefig(plot_path, dpi=150)
        print(f"Saved plot: {plot_path}")
    except Exception as e:
        print(f"(plot skipped: {e})")

    cleanup(device)


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["drive", "isic"], required=True)
    ap.add_argument("--stage", type=int, choices=[1, 2], required=True)
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--te", type=int, default=50, help="teacher epochs")
    ap.add_argument("--se", type=int, default=40, help="student epochs")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--lambda_sbt", type=float, default=None,
                    help="stage 2: the chosen lambda from stage 1")
    ap.add_argument("--lambdas", type=str,
                    default="0,1e-4,5e-4,1e-3,5e-3,1e-2",
                    help="stage 1: lambdas to sweep")
    ap.add_argument("--dice_tol", type=float, default=0.02,
                    help="stage 1: max allowed Dice drop vs baseline")
    ap.add_argument("--epsilons", type=str, default="0.1,0.5,1,2,4,6,8")
    ap.add_argument("--in_ch", type=int, default=3)
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[sbt_pipeline] device={device}")

    if args.stage == 1:
        stage_1(args, device)
    else:
        if args.lambda_sbt is None:
            raise SystemExit("stage 2 requires --lambda_sbt VALUE")
        stage_2(args, device)


if __name__ == "__main__":
    main()
