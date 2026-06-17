"""
Input-space Gaussian DP release baseline (the 'DP image release' family).

Symmetric to our feature-release pipeline, but the Gaussian noise is added
to the 3 RAW INPUT channels instead of the teacher's 128 bottleneck channels:
  per-channel L2 clip + normalize (Δ = 2, K=1)  ->  add N(0, σ²),
  σ = sqrt(Σ Δ²/(2ρ)) = sqrt(6/ρ)  ->  denormalize  ->  release ONCE.
Then a student is trained directly on the noisy images + GT (no teacher),
and evaluated on CLEAN val images. Same TinyUNet base-16, 5 seeds,
per-patient ε (k=1) — directly comparable to the feature-release table.
"""
import json, sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent))
from drive_local_demo import DriveDataset, TinyUNet, train_teacher, evaluate_vessel_dice
from synthetic_demo import eps_to_rho, clip_and_normalise, denormalise
from drive_pate_poc import correct_uniform_sigma

HERE = Path(__file__).parent


def input_caps(train_ds, q=0.95):
    """Per-channel L2-norm cap over the (private) training images."""
    norms = []
    for x, _ in DataLoader(train_ds, batch_size=4):
        norms.append(x.flatten(2).norm(dim=2))        # (B, C)
    return torch.quantile(torch.cat(norms), q, dim=0)  # (C,)


def make_noisy_release(train_ds, caps, sigma, device, seed):
    """Per-channel clip+normalize, add Gaussian noise once, denormalize."""
    torch.manual_seed(seed)
    xs, ys = [], []
    for x, y in DataLoader(train_ds, batch_size=4):
        x = x.to(device)
        xn = clip_and_normalise(x, caps.to(device))
        B, C, H, W = xn.shape
        xn = xn + torch.randn(B, C, H, W, device=device) * sigma.view(1, C, 1, 1)
        xnoisy = denormalise(xn, caps.to(device)).clamp(0, 1).cpu()
        xs.append(xnoisy); ys.append(y)
    return TensorDataset(torch.cat(xs), torch.cat(ys))


def main():
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    train_ds = DriveDataset("train", size=96)
    val_loader = DataLoader(DriveDataset("val", size=96), batch_size=4, shuffle=False)
    caps = input_caps(train_ds)
    print(f"input per-channel caps (RGB): {[f'{c:.2f}' for c in caps.tolist()]}")

    epsilons = [2.0, 8.0, 16.0]
    seeds = [100, 200, 300, 400, 500]
    deltas = torch.full((3,), 2.0, device=device)     # Δ = 2/K, K=1, 3 input channels

    results = {"method": "input-space Gaussian DP release (no teacher)",
               "epsilons": epsilons, "seeds": seeds,
               "input_caps": caps.tolist(), "sweep": {}}
    for eps in epsilons:
        rho = eps_to_rho(eps)
        sigma = correct_uniform_sigma(deltas, rho)
        noisy_ds = make_noisy_release(train_ds, caps, sigma, device, seed=42 + int(eps))
        loader = DataLoader(noisy_ds, batch_size=4, shuffle=True)
        dices = []
        for s in seeds:
            torch.manual_seed(s)
            stu = TinyUNet(in_ch=3, num_classes=2, base=16).to(device)
            train_teacher(stu, loader, n_epochs=40, lr=1e-3, device=device)
            d = evaluate_vessel_dice(stu, val_loader, device)
            dices.append(d)
            print(f"  ε={eps} seed={s}: Dice={d:.4f}")
        m, sd = float(np.mean(dices)), float(np.std(dices))
        results["sweep"][str(eps)] = {"dices": dices, "mean": m, "std": sd,
                                      "sem": sd/np.sqrt(len(dices)), "sigma": float(sigma[0])}
        print(f" → ε={eps}: {m:.4f} ± {sd:.4f}  (σ={sigma[0]:.3f})\n")
    (HERE/"drive_input_dp_baseline_results.json").write_text(json.dumps(results, indent=2))
    print("saved drive_input_dp_baseline_results.json")


if __name__ == "__main__":
    main()
