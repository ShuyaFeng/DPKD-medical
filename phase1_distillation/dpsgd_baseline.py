"""
DP-SGD baseline (Abadi et al. 2016) — faithful re-implementation.

Algorithm 1 from the paper:
  1. Sample a random minibatch of size L.
  2. Compute per-example gradients.
  3. Clip each per-example gradient to norm C.
  4. Average clipped gradients and add Gaussian noise N(0, sigma^2*C^2*I).
  5. Update model with the noisy gradient.

Privacy accounting: RDP (Moments Accountant) via Opacus, equivalent to the
Moments Accountant from the original paper. Opacus requires BatchNorm ->
GroupNorm and inplace=False on all ReLU layers.

Train directly on private data — NO teacher — so this is the plain DP baseline.
Same TinyUNet base=16 student, same val Dice metric, same 5 seeds,
same epsilons as CANAL: [1, 2, 4, 8]. Directly comparable.

Usage:
  python dpsgd_baseline.py --dataset isic
  python dpsgd_baseline.py --dataset kvasir
  python dpsgd_baseline.py --dataset busi
  python dpsgd_baseline.py --dataset isic --smoke
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from drive_local_demo import TinyUNet, evaluate_vessel_dice
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

HERE = Path(__file__).parent
(HERE / "results").mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPSILONS = [1.0, 2.0, 4.0, 8.0]
SEEDS    = [100, 200, 300, 400, 500]
DELTA    = 1e-5

# Per-dataset hyperparameters following Abadi 2016 Section 3.3:
#   bs = √N (paper recommendation for lot size)
#   max_grad_norm = median of unclipped gradient norms (measured via measure_grad_norms.py)
#   lr = 0.05 (paper Section 3.3: accuracy peaks at 0.05)
#   epochs = 200 (same as CANAL for fair comparison)
DATASET_CFG = {
    "isic":   dict(epochs=200, bs=45, lr=0.05, max_grad_norm=4.86),
    "kvasir": dict(epochs=200, bs=28, lr=0.05, max_grad_norm=3.46),
    "busi":   dict(epochs=200, bs=23, lr=0.05, max_grad_norm=2.28),
}


def get_dataset(name, split):
    if name == "isic":
        from isic_dataset import ISICDataset
        return ISICDataset(split, 96), 3
    if name == "kvasir":
        from kvasir_dataset import KvasirDataset
        return KvasirDataset(split, 96), 3
    if name == "busi":
        from busi_dataset import BUSIDataset
        return BUSIDataset(split, 96), 1
    raise ValueError(f"Unknown dataset: {name}. Choose from isic, kvasir, busi.")


def seg_loss(logits, y):
    """CE + 3*Dice. Per-sample Dice (no cross-sample sum) keeps the loss DP-safe."""
    ce = F.cross_entropy(logits, y, reduction="mean")
    probs = logits.softmax(1)[:, 1]
    yf = (y == 1).float()
    inter = (probs * yf).sum(dim=(1, 2))
    denom = probs.sum(dim=(1, 2)) + yf.sum(dim=(1, 2))
    dice = 1.0 - (2 * inter + 1.0) / (denom + 1.0)
    return ce + 3.0 * dice.mean()


def train_dpsgd(train_ds, val_loader, eps, seed, in_ch,
                epochs, bs, lr, max_grad_norm):
    """Train one (eps, seed) run with DP-SGD (Abadi 2016, Algorithm 1)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = TinyUNet(in_ch=in_ch, num_classes=2, base=16)
    # Opacus requirement 1: replace BatchNorm with GroupNorm
    model = ModuleValidator.fix(model)
    # Opacus requirement 2: disable inplace ReLU (breaks per-example grad hooks)
    for mod in model.modules():
        if isinstance(mod, torch.nn.ReLU):
            mod.inplace = False
    model = model.to(DEVICE)

    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True)

    # make_private_with_epsilon automatically sets noise_multiplier (sigma) so
    # that the RDP accountant certifies (eps, delta)-DP after `epochs` passes.
    # This is the direct equivalent of the Moments Accountant in Abadi 2016.
    pe = PrivacyEngine(accountant="rdp")
    model, opt, loader = pe.make_private_with_epsilon(
        module=model,
        optimizer=opt,
        data_loader=loader,
        target_epsilon=eps,
        target_delta=DELTA,
        epochs=epochs,
        max_grad_norm=max_grad_norm,
    )

    model.train()
    for _ in range(epochs):
        for x, y in loader:
            opt.zero_grad()
            loss = seg_loss(model(x.to(DEVICE)), y.to(DEVICE))
            loss.backward()
            opt.step()          # Opacus clips + adds noise here (Algorithm 1, step 3-4)

    eps_spent = pe.get_epsilon(DELTA)
    noise_mult = float(opt.noise_multiplier)
    val_dice = evaluate_vessel_dice(model, val_loader, DEVICE)
    return val_dice, eps_spent, noise_mult


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["isic", "kvasir", "busi"])
    parser.add_argument("--smoke", action="store_true",
                        help="Quick test: 1 epsilon, 1 seed, fewer epochs")
    args = parser.parse_args()

    train_ds, in_ch = get_dataset(args.dataset, "train")
    val_ds,   _     = get_dataset(args.dataset, "val")
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    print(f"DP-SGD baseline on {args.dataset} | in_ch={in_ch} | device={DEVICE}")
    print(f"Train size: {len(train_ds)}  Val size: {len(val_ds)}")

    epsilons = [8.0] if args.smoke else EPSILONS
    seeds    = [100]  if args.smoke else SEEDS
    cfg = dict(DATASET_CFG[args.dataset])
    if args.smoke:
        cfg["epochs"] = 10

    results = {
        "method":   "DP-SGD (Abadi 2016) — no teacher",
        "dataset":  args.dataset,
        "config":   cfg,
        "delta":    DELTA,
        "epsilons": epsilons,
        "seeds":    seeds,
        "sweep":    {},
    }

    for eps in epsilons:
        dices, eps_checks, noise_mults = [], [], []
        for s in seeds:
            dice, eps_sp, nm = train_dpsgd(
                train_ds, val_loader, eps, s, in_ch, **cfg
            )
            dices.append(dice)
            eps_checks.append(eps_sp)
            noise_mults.append(nm)
            print(f"  eps={eps}  seed={s}:  Dice={dice:.4f}  "
                  f"(accounted eps={eps_sp:.3f}, sigma={nm:.3f})")

        mean = float(np.mean(dices))
        std  = float(np.std(dices))
        results["sweep"][str(eps)] = {
            "dices":            [float(d) for d in dices],
            "mean":             mean,
            "std":              std,
            "sem":              std / np.sqrt(len(dices)),
            "noise_multiplier": float(np.mean(noise_mults)),
            "eps_accounted":    float(np.mean(eps_checks)),
        }
        print(f"  -> eps={eps}: {mean:.4f} ± {std:.4f}\n")

    if not args.smoke:
        out = HERE / "results" / f"{args.dataset}_dpsgd_results.json"
        out.write_text(json.dumps(results, indent=2))
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
