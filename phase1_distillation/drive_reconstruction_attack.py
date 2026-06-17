"""
Empirical privacy audit #2 — Reconstruction attack on the feature release.

Worst-case adversary: trains a decoder (released-feature -> input image) on the
clean (feature, image) pairs of the training set itself (an upper bound on
inversion power), then inverts the NOISY released features back to images.
Compares NO NOISE vs ε∈{128,32,16,8,2}. Reports PSNR/SSIM vs ε plus a visual
panel — at σ=0 the vessels are recoverable (leak); noise destroys them.
"""
import json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, str(Path(__file__).parent))
from drive_local_demo import DriveDataset, TinyUNet, train_teacher, collect_caps
from synthetic_demo import eps_to_rho, clip_and_normalise
from drive_pate_poc import correct_uniform_sigma

HERE = Path(__file__).parent


class Recon(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, 64, 3, padding=1), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 2, 2), nn.ReLU(True),     # 24 -> 48
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 2, 2), nn.ReLU(True),     # 48 -> 96
            nn.Conv2d(16, 3, 3, padding=1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def main():
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    train_ds = DriveDataset("train", 96)
    teacher = TinyUNet(in_ch=3, num_classes=2, base=32).to(device)
    train_teacher(teacher, DataLoader(train_ds, batch_size=4, shuffle=True),
                  n_epochs=60, lr=1e-3, device=device)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    caps = collect_caps(teacher, DataLoader(train_ds, batch_size=4), device).to(device)
    C = teacher.base * 4

    # clean normalized features + images
    imgs, feats = [], []
    for x, _ in DataLoader(train_ds, batch_size=4):
        x = x.to(device)
        _, _, e3 = teacher.encode(x)
        feats.append(clip_and_normalise(e3, caps)); imgs.append(x)
    F = torch.cat(feats); X = torch.cat(imgs)              # (N,C,24,24), (N,3,96,96)

    # worst-case adversary decoder: clean feature -> image
    dec = Recon(C).to(device)
    opt = torch.optim.Adam(dec.parameters(), lr=2e-3)
    for ep in range(400):
        opt.zero_grad()
        loss = ((dec(F) - X) ** 2).mean()
        loss.backward(); opt.step()
    print(f"decoder trained, clean recon MSE={loss.item():.5f}")

    deltas = torch.full((C,), 2.0)
    sig = lambda e: float(correct_uniform_sigma(deltas, eps_to_rho(e))[0])
    epsilons = [32.0, 16.0, 8.0, 2.0]
    configs = [("no-noise", None, 0.0)] + [(f"ε={e:g}", e, sig(e)) for e in epsilons]

    torch.manual_seed(0)
    results, recons = {}, {}
    print(f"\n{'config':>10} {'σ':>9} {'PSNR':>7} {'SSIM':>7}")
    Xn = X.cpu().numpy().transpose(0, 2, 3, 1)             # (N,96,96,3) in [0,1]
    for name, eps, sigma in configs:
        rel = F + (torch.randn_like(F) * sigma if sigma > 0 else 0.0)
        with torch.no_grad():
            xhat = dec(rel).clamp(0, 1).cpu().numpy().transpose(0, 2, 3, 1)
        mse = ((xhat - Xn) ** 2).reshape(len(Xn), -1).mean(1)
        psnr = float(np.mean(10 * np.log10(1.0 / np.maximum(mse, 1e-10))))
        ss = float(np.mean([ssim(Xn[i], xhat[i], channel_axis=2, data_range=1.0)
                            for i in range(len(Xn))]))
        results[name] = {"epsilon": eps, "sigma": sigma, "psnr": psnr, "ssim": ss}
        recons[name] = xhat
        print(f"{name:>10} {sigma:>9.3f} {psnr:>7.2f} {ss:>7.3f}")

    (HERE / "drive_reconstruction_attack_results.json").write_text(json.dumps(results, indent=2))

    # ---- visual panel: a few images across configs ----
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = [0, 5, 10]
    panel = ["original", "no-noise", "ε=16", "ε=8", "ε=2"]
    fig, axes = plt.subplots(len(show), len(panel), figsize=(2.1 * len(panel), 2.1 * len(show)))
    for r, idx in enumerate(show):
        for c, key in enumerate(panel):
            ax = axes[r, c]
            ax.imshow(Xn[idx] if key == "original" else recons[key][idx])
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                t = key if key == "original" else (
                    key if key == "no-noise" else f"{key} (PSNR {results[key]['psnr']:.0f})")
                ax.set_title(t, fontsize=9)
    fig.suptitle("Reconstruction from released features — noise destroys recoverable structure",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(HERE / "fig_reconstruction.png", dpi=140)

    # ---- PSNR/SSIM vs ε ----
    fig2, ax = plt.subplots(figsize=(7, 5))
    es = epsilons
    ax.plot(es, [results[f"ε={e:g}"]["psnr"] for e in es], "o-", color="#d62728", lw=2, ms=8, label="PSNR (dB)")
    ax.axhline(results["no-noise"]["psnr"], color="black", ls="--", lw=1.3,
               label=f"no-noise PSNR={results['no-noise']['psnr']:.1f}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(es); ax.set_xticklabels([f"{int(e)}" for e in es]); ax.minorticks_off()
    ax.set_xlabel("privacy budget ε"); ax.set_ylabel("reconstruction PSNR (dB)")
    ax.set_title("Reconstruction fidelity vs ε"); ax.legend(); ax.grid(alpha=0.3)
    fig2.tight_layout(); fig2.savefig(HERE / "fig_reconstruction_psnr.png", dpi=140)
    print("saved fig_reconstruction.png, fig_reconstruction_psnr.png + results json")


if __name__ == "__main__":
    main()
