"""
DP-SGD hyperparameter grid search — BUSI dataset, ε=8.0.

Searches over:
  C      (max_grad_norm): [1.0, 2.28, 3.5, 5.0]
  lr:                     [0.01, 0.05, 0.1]
  epochs:                 [100, 200, 300]

Fixed:
  dataset = BUSI
  epsilon = 8.0
  seeds   = [100, 200, 300]
  delta   = 1e-5
  bs      = 23 (sqrt(530) — same as full experiment)

Goal: find (lr, C, epochs) that gives best mean Dice on BUSI val set.
Results saved to results/busi_dpsgd_gridsearch.json.

Usage:
  python dpsgd_gridsearch.py
"""
import itertools
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

EPSILON = 8.0
DELTA   = 1e-5
SEEDS   = [100, 200, 300]
BS      = 23   # sqrt(530) — BUSI train size

C_VALUES      = [1.0, 2.28, 3.5, 5.0]
LR_VALUES     = [0.01, 0.05, 0.1]
EPOCH_VALUES  = [100, 200, 300]


def seg_loss(logits, y):
    ce = F.cross_entropy(logits, y, reduction="mean")
    probs = logits.softmax(1)[:, 1]
    yf = (y == 1).float()
    inter = (probs * yf).sum(dim=(1, 2))
    denom = probs.sum(dim=(1, 2)) + yf.sum(dim=(1, 2))
    dice = 1.0 - (2 * inter + 1.0) / (denom + 1.0)
    return ce + 3.0 * dice.mean()


def train_one(train_ds, val_loader, seed, lr, epochs, max_grad_norm):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = TinyUNet(in_ch=1, num_classes=2, base=16)
    model = ModuleValidator.fix(model)
    for mod in model.modules():
        if isinstance(mod, torch.nn.ReLU):
            mod.inplace = False
    model = model.to(DEVICE)

    opt    = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loader = DataLoader(train_ds, batch_size=BS, shuffle=True, drop_last=True)

    pe = PrivacyEngine(accountant="rdp")
    model, opt, loader = pe.make_private_with_epsilon(
        module=model,
        optimizer=opt,
        data_loader=loader,
        target_epsilon=EPSILON,
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
            opt.step()

    dice = evaluate_vessel_dice(model, val_loader, DEVICE)
    return dice


def main():
    from busi_dataset import BUSIDataset

    train_ds   = BUSIDataset("train", 96)
    val_ds     = BUSIDataset("val",   96)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    print(f"BUSI grid search | ε={EPSILON} | device={DEVICE}")
    print(f"Train={len(train_ds)}  Val={len(val_ds)}")
    print(f"Grid: C={C_VALUES}  lr={LR_VALUES}  epochs={EPOCH_VALUES}")
    print(f"Seeds={SEEDS}  Total runs={len(C_VALUES)*len(LR_VALUES)*len(EPOCH_VALUES)*len(SEEDS)}\n")

    results = {
        "dataset": "busi", "epsilon": EPSILON, "delta": DELTA,
        "seeds": SEEDS, "bs": BS,
        "grid": {"C": C_VALUES, "lr": LR_VALUES, "epochs": EPOCH_VALUES},
        "runs": [],
    }

    best_mean = -1
    best_cfg  = None

    combos = list(itertools.product(C_VALUES, LR_VALUES, EPOCH_VALUES))
    for i, (C, lr, epochs) in enumerate(combos):
        dices = []
        for s in SEEDS:
            dice = train_one(train_ds, val_loader, s, lr, epochs, C)
            dices.append(dice)
            print(f"  [{i+1}/{len(combos)}] C={C}  lr={lr}  epochs={epochs}  "
                  f"seed={s}  Dice={dice:.4f}")

        mean = float(np.mean(dices))
        std  = float(np.std(dices))
        print(f"  → mean={mean:.4f}  std={std:.4f}\n")

        run = {"C": C, "lr": lr, "epochs": epochs,
               "dices": [float(d) for d in dices],
               "mean": mean, "std": std}
        results["runs"].append(run)

        if mean > best_mean:
            best_mean = mean
            best_cfg  = run

    print("=" * 60)
    print(f"BEST: C={best_cfg['C']}  lr={best_cfg['lr']}  "
          f"epochs={best_cfg['epochs']}  mean Dice={best_mean:.4f}")
    print("=" * 60)

    results["best"] = best_cfg

    out = HERE / "results" / "busi_dpsgd_gridsearch.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
