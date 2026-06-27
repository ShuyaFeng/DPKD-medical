"""
Targeted comparison: K=3, uniform vs CANAL (water-filling), at epsilon=2.0,
on Kvasir-SEG.

This replicates the single standout result from the full ISIC sweep
(K=3 at eps=2 showed the largest CANAL-vs-uniform improvement: +0.0113
Dice) on the new Kvasir-SEG dataset, before committing to the full
96-cell comparison matrix.

Reuses the exact same functions and calling conventions as
run_comparison.py / drive_canal_small_eps.py (train_K_teachers,
shared_importance, thresholded_uniform_sigma, correct_waterfilling_sigma,
precompute_joint_cache, train_student_distill) for full consistency
with the existing ISIC results.

Saves results/kvasir_canal_k3_eps2_results.json
"""
import argparse, sys, json, time
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from drive_pate_poc import train_K_teachers
from drive_pate_pruning_joint import shared_importance, thresholded_uniform_sigma, precompute_joint_cache
from drive_pate_canal_combined import (
    correct_waterfilling_sigma, correct_waterfilling_sigma_honest,
)

# Honest CANAL accounting: charge rho_imp = ALPHA_IMP * rho for releasing the
# (data-dependent) importance vector. See `correct_waterfilling_sigma_honest`.
ALPHA_IMP = 0.1
CLIP_IMP = 5e-8  
from drive_student_distill import train_student_distill
from synthetic_demo import eps_to_rho

HERE = Path(__file__).parent
(HERE / "results").mkdir(exist_ok=True)

CKPT_DIR = HERE / "checkpoints"        # <-- ADD THIS LINE
CKPT_DIR.mkdir(exist_ok=True)   

K_VALUES = (3,)


def get_dataset(name, split, size):
    if name == "drive":
        from drive_local_demo import DriveDataset; return DriveDataset(split, size), 3
    if name == "isic":
        from isic_dataset import ISICDataset; return ISICDataset(split, size), 3
    if name == "kvasir":
        from kvasir_dataset import KvasirDataset; return KvasirDataset(split, size), 3
    if name == "brats":
        from brats_dataset import BRATSDataset; return BRATSDataset(split, size), 4
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="kvasir", choices=["drive", "isic", "kvasir", "brats"])
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epsilons", type=str, default="2")
    ap.add_argument("--te", type=int, default=60, help="teacher epochs")
    ap.add_argument("--se", type=int, default=40, help="student epochs")
    args = ap.parse_args()

    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else "cpu")
    EPS = [float(e) for e in args.epsilons.split(",")]
    SEEDS = list(range(100, 100 + args.seeds * 100, 100))

    train_ds, in_ch = get_dataset(args.dataset, "train", args.size)
    val_ds, _ = get_dataset(args.dataset, "val", args.size)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    print(f"[{args.dataset}] device={dev} in_ch={in_ch}  train={len(train_ds)} val={len(val_ds)}  "
          f"ε={EPS} seeds={SEEDS}  K={K_VALUES}")

    # teachers for K=3 only (trained once, reused)
    tk = {K: train_K_teachers(train_ds, K, dev, n_epochs=args.te, in_ch=in_ch) for K in K_VALUES}
    Cb = tk[K_VALUES[0]][0][0].base * 4

    # importance ranking (needed for CANAL water-filling)
    imp = {K: shared_importance(tk[K][0], DataLoader(train_ds, batch_size=8), dev).to(dev)
           for K in K_VALUES}

    def students(cache, canal, e):
        return [train_student_distill(
                    train_ds, val_loader, cache, dev, student_base=16,
                    teacher_base=32, n_epochs=args.se, lr=1e-3,
                    lambda_feat=0.4, seed=s, in_ch=in_ch,
                    save_path=CKPT_DIR / f"{args.dataset}_K{K_VALUES[0]}_canal{canal}_eps{e}_seed{s}.pt"
                )[0] for s in SEEDS]
    
    def cell(K, am, sigma, canal, e):
        cache = precompute_joint_cache(tk[K][0], tk[K][1], train_ds, sigma, am, dev, seed=42)
        d = students(cache, canal, e)
        return {"dices": d, "mean": float(np.mean(d)), "sem": float(np.std(d) / np.sqrt(len(d)))}

    def sweep(K, canal=False):
        deltas = torch.full((Cb,), 2.0 / K, device=dev)
        out = {}
        for e in EPS:
            rho = eps_to_rho(e)
            am = torch.ones(Cb, dtype=torch.bool, device=dev)
            if canal:
                sens_imp = 2.0 * CLIP_IMP / float(len(train_ds))
                sigma = correct_waterfilling_sigma_honest(
                    deltas, imp[K], rho,
                    sensitivity=sens_imp, alpha=ALPHA_IMP,
                    noise_seed=int(e * 10) + 1,
                )
            else:
                sigma = thresholded_uniform_sigma(deltas, rho, am)
            out[f"{e}"] = cell(K, am, sigma, canal, e)
            print(f"   {args.dataset} K={K} canal={canal} ε={e}: {out[f'{e}']['mean']:.4f}")
        return out

    R = {"dataset": args.dataset, "in_ch": in_ch, "train": len(train_ds), "epsilons": EPS,
         "seeds": SEEDS, "K_values": K_VALUES, "series": {}}

    print("\n[K=3: uniform vs CANAL at eps=2]")
    R["series"]["PATE K=3 uniform"] = sweep(3, canal=False)
    R["series"]["PATE K=3 + CANAL"] = sweep(3, canal=True)

    out_path = HERE / "results" / f"{args.dataset}_canal_k3_eps2_results.json"
    out_path.write_text(json.dumps(R, indent=2))
    print(f"\nsaved {out_path}")

    print("\nSummary:")
    for K in K_VALUES:
        for e in EPS:
            u = R["series"][f"PATE K={K} uniform"][f"{e}"]["mean"]
            c = R["series"][f"PATE K={K} + CANAL"][f"{e}"]["mean"]
            print(f"  K={K} ε={e}: uniform={u:.4f}  CANAL={c:.4f}  diff={c-u:+.4f}")


if __name__ == "__main__":
    main()