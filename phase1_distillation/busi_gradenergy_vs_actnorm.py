"""
Ablation requested by Dr. Tian: CANAL with gradient-energy importance vs.
CANAL with activation-norm importance, on ONE dataset (BUSI), across a
range of epsilons, to find where the gap is clearest. (He confirmed: one
dataset is enough, epsilon range is our choice; main result is uniform vs
grad-energy -- this is a secondary ablation only.)

Saves results/busi_gradenergy_vs_actnorm_results.json
"""
import argparse, sys, json
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from drive_pate_poc import train_K_teachers
from drive_pate_pruning_joint import (
    shared_importance, shared_importance_actnorm, precompute_joint_cache,
)
from drive_pate_canal_combined import correct_waterfilling_sigma
from drive_student_distill import train_student_distill
from synthetic_demo import eps_to_rho

HERE = Path(__file__).parent
(HERE / "results").mkdir(exist_ok=True)
CKPT_DIR = HERE / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)

K_VALUES = (3,)


def get_dataset(name, split, size):
    if name == "busi":
        from busi_dataset import BUSIDataset; return BUSIDataset(split, size), 1
    if name == "kvasir":
        from kvasir_dataset import KvasirDataset; return KvasirDataset(split, size), 3
    if name == "isic":
        from isic_dataset import ISICDataset; return ISICDataset(split, size), 3
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="busi", choices=["busi", "kvasir", "isic"])
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epsilons", type=str, default="0.5,1,2,4,8")
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

    K = K_VALUES[0]
    tk = {K: train_K_teachers(train_ds, K, dev, n_epochs=args.te, in_ch=in_ch)}
    Cb = tk[K][0][0].base * 4

    imp_grad = shared_importance(tk[K][0], DataLoader(train_ds, batch_size=8), dev).to(dev)
    imp_act = shared_importance_actnorm(tk[K][0], DataLoader(train_ds, batch_size=8), dev).to(dev)

    def students(cache, tag, e):
        return [train_student_distill(
                    train_ds, val_loader, cache, dev, student_base=16,
                    teacher_base=32, n_epochs=args.se, lr=1e-3,
                    lambda_feat=0.4, seed=s, in_ch=in_ch,
                    save_path=CKPT_DIR / f"{args.dataset}_K{K}_{tag}_eps{e}_seed{s}.pt"
                )[0] for s in SEEDS]

    def cell(am, sigma, tag, e):
        cache = precompute_joint_cache(tk[K][0], tk[K][1], train_ds, sigma, am, dev, seed=42)
        d = students(cache, tag, e)
        return {"dices": d, "mean": float(np.mean(d)), "sem": float(np.std(d) / np.sqrt(len(d)))}

    deltas = torch.full((Cb,), 2.0 / K, device=dev)
    am = torch.ones(Cb, dtype=torch.bool, device=dev)

    R = {"dataset": args.dataset, "in_ch": in_ch, "train": len(train_ds), "epsilons": EPS,
         "seeds": SEEDS, "K": K, "series": {}}

    print("\n[grad-energy CANAL vs act-norm CANAL]")
    grad_out, act_out = {}, {}
    for e in EPS:
        rho = eps_to_rho(e)
        sigma_grad = correct_waterfilling_sigma(deltas, imp_grad, rho)
        sigma_act = correct_waterfilling_sigma(deltas, imp_act, rho)

        grad_out[f"{e}"] = cell(am, sigma_grad, "gradenergy", e)
        print(f"   {args.dataset} grad-energy ε={e}: {grad_out[f'{e}']['mean']:.4f}")

        act_out[f"{e}"] = cell(am, sigma_act, "actnorm", e)
        print(f"   {args.dataset} act-norm    ε={e}: {act_out[f'{e}']['mean']:.4f}")

    R["series"]["CANAL grad-energy"] = grad_out
    R["series"]["CANAL act-norm"] = act_out

    out_path = HERE / "results" / f"{args.dataset}_gradenergy_vs_actnorm_results.json"
    out_path.write_text(json.dumps(R, indent=2))
    print(f"\nsaved {out_path}")

    print("\nSummary:")
    for e in EPS:
        g = R["series"]["CANAL grad-energy"][f"{e}"]["mean"]
        a = R["series"]["CANAL act-norm"][f"{e}"]["mean"]
        print(f"  ε={e}: grad-energy={g:.4f}  act-norm={a:.4f}  diff={g-a:+.4f}")


if __name__ == "__main__":
    main()
