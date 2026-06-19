"""
DP-SGD baseline: train the student DIRECTLY on the private DRIVE images
with DP-SGD (Abadi 2016), no teacher. The standard DP baseline.

Per-patient = per-example (DRIVE k=1), so Opacus example-level accounting
gives exactly the per-patient epsilon we report elsewhere. Same TinyUNet
base=16 student, same val Dice metric, 5 seeds — directly comparable to
the feature-distillation pipeline. BatchNorm is auto-replaced by GroupNorm
(BN is not DP-compatible).
"""
import json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from drive_local_demo import DriveDataset, TinyUNet, evaluate_vessel_dice
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

HERE = Path(__file__).parent
DEVICE = "cpu"   # Opacus grad-sample hooks: CPU is the safe path for tiny data


def seg_loss(logits, y):
    """CE + 3*Dice, computed PER-SAMPLE (no cross-sample mixing -> DP-safe)."""
    ce = F.cross_entropy(logits, y, reduction="mean")
    probs = logits.softmax(1)[:, 1]
    yf = (y == 1).float()
    inter = (probs * yf).sum(dim=(1, 2))
    denom = probs.sum(dim=(1, 2)) + yf.sum(dim=(1, 2))
    dice = 1.0 - (2 * inter + 1.0) / (denom + 1.0)
    return ce + 3.0 * dice.mean()


def train_dpsgd(train_ds, val_loader, eps, seed, epochs=60, bs=8,
                max_grad_norm=1.0, lr=5e-3, delta=1e-5):
    torch.manual_seed(seed); np.random.seed(seed)
    model = TinyUNet(in_ch=3, num_classes=2, base=16).to(DEVICE)
    model = ModuleValidator.fix(model)            # BatchNorm -> GroupNorm
    for mod in model.modules():                   # inplace ReLU breaks Opacus hooks
        if isinstance(mod, torch.nn.ReLU):
            mod.inplace = False
    model = model.to(DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    pe = PrivacyEngine(accountant="rdp")
    model, opt, loader = pe.make_private_with_epsilon(
        module=model, optimizer=opt, data_loader=loader,
        target_epsilon=eps, target_delta=delta, epochs=epochs,
        max_grad_norm=max_grad_norm,
    )
    model.train()
    for ep in range(epochs):
        for x, y in loader:
            opt.zero_grad()
            loss = seg_loss(model(x.to(DEVICE)), y.to(DEVICE))
            loss.backward()
            opt.step()
    eps_spent = pe.get_epsilon(delta)
    return evaluate_vessel_dice(model, val_loader, DEVICE), eps_spent, float(opt.noise_multiplier)


def main():
    smoke = "--smoke" in sys.argv
    train_ds = DriveDataset("train", size=96)
    val_loader = DataLoader(DriveDataset("val", size=96), batch_size=4, shuffle=False)
    epsilons = [1.0, 2.0, 3.0, 4.0, 5.0]
    seeds = [100, 200, 300, 400, 500]
    if smoke:
        epsilons, seeds = [8.0], [100]

    # best config from the ε=8 tuning sweep (epochs/bs/lr/C); more epochs only
    # forces higher σ at fixed ε and hurts, so 200 is near-optimal here.
    cfg = dict(epochs=200, bs=20, lr=0.05, max_grad_norm=0.5)
    results = {"method": "DP-SGD (student, no teacher)", "config": cfg,
               "epsilons": epsilons, "seeds": seeds, "sweep": {}}
    for eps in epsilons:
        dices, eps_check, nm = [], None, None
        for s in seeds:
            d, e_sp, nmul = train_dpsgd(train_ds, val_loader, eps, s, **cfg)
            dices.append(d); eps_check = e_sp; nm = nmul
            print(f"  ε={eps} seed={s}: Dice={d:.4f}  (acct ε={e_sp:.3f}, σ_noise={nmul:.3f})")
        m, sd = float(np.mean(dices)), float(np.std(dices))
        results["sweep"][str(eps)] = {
            "dices": dices, "mean": m, "std": sd, "sem": sd/np.sqrt(len(dices)),
            "noise_multiplier": nm, "eps_accounted": eps_check,
        }
        print(f" → ε={eps}: {m:.4f} ± {sd:.4f}\n")
    if not smoke:
        (HERE/"drive_dpsgd_baseline_results.json").write_text(json.dumps(results, indent=2))
        print("saved drive_dpsgd_baseline_results.json")


if __name__ == "__main__":
    main()
