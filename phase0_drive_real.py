"""
Phase-0 utility validation on the REAL DRIVE dataset (40 images,
20 train + 20 val, 584x565 RGB TIFFs with expert vessel masks).

Unlike `phase0_drive.py` (which used scikit-image's single retina sample
with Frangi-derived labels), this script uses the actual DRIVE images and
expert annotations sourced from the Hugging Face mirror
`Zomba/DRIVE-digital-retinal-images-for-vessel-extraction`.

Pipeline per image:
  1. resize to 256x256, normalize to [0, 1]
  2. expert vessel mask is the spatial saliency v(i, j)
  3. per-channel μ+ (vessel mean) and μ- (background mean) → contrast δμ_c
  4. within-class std → β
  5. rank-1 mu field: μ(c, i, j) = |δμ_c| · v(i, j)
  6. Bayes accuracy of each mechanism at every ε in the dense sweep

Then aggregate across the 40 images: report mean ± std curves.
"""

import glob
import os
import numpy as np
from PIL import Image
from scipy.stats import norm
from skimage import transform
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phase0_validation import (
    eps_to_rho,
    bayes_acc,
    mech_uniform,
    mech_channel_wf,
    mech_spatial_wf,
    mech_joint_wf,
    mech_joint_wf_threshold,
    mech_bayes_optimal,
    mech_adversarial,
)


DRIVE_ROOT = "data/DRIVE"


def load_drive(target_size=256):
    """Yield (image_id, img[H,W,3] in [0,1], vessel_mask[H,W] bool)."""
    pairs = []
    for split in ["train", "val"]:
        in_dir = os.path.join(DRIVE_ROOT, split, "input")
        lab_dir = os.path.join(DRIVE_ROOT, split, "label")
        for tif in sorted(glob.glob(os.path.join(in_dir, "*.tif"))):
            stem = os.path.splitext(os.path.basename(tif))[0]
            # train labels use "{stem}.png", val labels use "{stem}_manual1.png"
            cands = [
                os.path.join(lab_dir, f"{stem}.png"),
                os.path.join(lab_dir, f"{stem}_manual1.png"),
            ]
            png = next((p for p in cands if os.path.exists(p)), None)
            if png is None:
                continue
            img = np.array(Image.open(tif), dtype=np.uint8)
            lab = np.array(Image.open(png), dtype=np.uint8)
            img_rs = transform.resize(
                img, (target_size, target_size, 3),
                preserve_range=True, anti_aliasing=True,
            ).astype(float) / 255.0
            lab_rs = transform.resize(
                lab.astype(float), (target_size, target_size),
                preserve_range=True, anti_aliasing=False, order=0,
            )
            pairs.append((f"{split}/{stem}", img_rs, lab_rs > 127))
    return pairs


def per_image_problem(img, vessel_mask):
    """Build the (mu, Delta, beta, w_c, w_ij) tuple for one DRIVE image."""
    img_chw = img.transpose(2, 0, 1)                                     # (3, H, W)
    background = ~vessel_mask
    if vessel_mask.sum() < 50 or background.sum() < 50:
        return None

    mu_plus = np.array([img_chw[c][vessel_mask].mean() for c in range(3)])
    mu_minus = np.array([img_chw[c][background].mean() for c in range(3)])
    delta_mu = np.abs(mu_plus - mu_minus)                                # (3,)

    sigmas_within = []
    for c in range(3):
        s_pos = img_chw[c][vessel_mask].std()
        s_neg = img_chw[c][background].std()
        sigmas_within.append(0.5 * (s_pos + s_neg))
    beta = float(np.mean(sigmas_within))

    v_map = vessel_mask.astype(float)                                    # (H, W) in {0, 1}
    mu_field = delta_mu[:, None, None] * v_map[None, :, :]               # (3, H, W)
    Delta = np.ones_like(mu_field)
    w_c = (mu_field ** 2).sum(axis=(1, 2))
    w_ij = (mu_field ** 2).sum(axis=0)

    return dict(
        mu=mu_field, Delta=Delta, beta=beta, w_c=w_c, w_ij=w_ij,
        delta_mu=delta_mu, mu_plus=mu_plus, mu_minus=mu_minus,
    )


def main():
    delta_dp = 1e-5
    eps_dense = np.geomspace(0.5, 10.0, 25)
    eps_table = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 10.0])
    epsilons = np.unique(np.round(np.concatenate([eps_dense, eps_table]), 6))
    rhos = [eps_to_rho(e, delta_dp) for e in epsilons]

    print("Loading DRIVE (40 images, 256x256)...")
    pairs = load_drive(target_size=256)
    print(f"  loaded {len(pairs)} (image, mask) pairs")

    # Per-image stats
    delta_mu_all = []
    beta_all = []
    vessel_frac = []
    for name, img, mask in pairs:
        prob = per_image_problem(img, mask)
        if prob is None:
            continue
        delta_mu_all.append(prob["delta_mu"])
        beta_all.append(prob["beta"])
        vessel_frac.append(float(mask.mean()))
    delta_mu_all = np.stack(delta_mu_all)                                # (N, 3)
    beta_all = np.array(beta_all)
    vessel_frac = np.array(vessel_frac)

    print()
    print("Per-channel |contrast| |μ+ - μ-| across 40 DRIVE images:")
    for c, ch in enumerate("RGB"):
        v = delta_mu_all[:, c]
        print(f"  {ch}: mean={v.mean():.4f}  std={v.std():.4f}  "
              f"min={v.min():.4f}  max={v.max():.4f}")
    mean_delta_mu = delta_mu_all.mean(axis=0)
    print(f"  channel ratio (mean): max/min = "
          f"{mean_delta_mu.max() / max(mean_delta_mu.min(), 1e-12):.2f}")
    print(f"  per-image max/min ratios: mean = "
          f"{(delta_mu_all.max(axis=1) / np.maximum(delta_mu_all.min(axis=1), 1e-6)).mean():.2f}")
    print()
    print(f"Within-class std β: mean={beta_all.mean():.4f}  std={beta_all.std():.4f}")
    print(f"Vessel pixel fraction: mean={vessel_frac.mean():.4f} (literature ~0.075)")
    print()

    mechanisms = [
        ("uniform",       lambda r, p: mech_uniform(r, p["mu"], p["Delta"], p["beta"])),
        ("channel-WF",    lambda r, p: mech_channel_wf(r, p["mu"], p["Delta"], p["beta"], p["w_c"])),
        ("spatial-WF",    lambda r, p: mech_spatial_wf(r, p["mu"], p["Delta"], p["beta"], p["w_ij"])),
        ("joint-WF",      lambda r, p: mech_joint_wf(r, p["mu"], p["Delta"], p["beta"], p["w_c"], p["w_ij"])),
        ("joint-WF+thr",  lambda r, p: mech_joint_wf_threshold(r, p["mu"], p["Delta"], p["beta"], p["w_c"], p["w_ij"])),
        ("Bayes-optimal", lambda r, p: mech_bayes_optimal(r, p["mu"], p["Delta"], p["beta"])),
        ("adversarial",   lambda r, p: mech_adversarial(r, p["mu"], p["Delta"], p["beta"], p["w_c"], p["w_ij"])),
    ]

    # accs[name] is shape (N_images, N_eps)
    print("Running mechanisms over 40 images x", len(epsilons), "epsilons ...")
    accs = {name: np.zeros((len(pairs), len(epsilons))) for name, _ in mechanisms}
    for img_idx, (name_img, img, mask) in enumerate(pairs):
        prob = per_image_problem(img, mask)
        for name, fn in mechanisms:
            for j, rho in enumerate(rhos):
                sigma2 = fn(rho, prob)
                accs[name][img_idx, j] = bayes_acc(prob["mu"], sigma2, prob["beta"])
        if (img_idx + 1) % 10 == 0:
            print(f"  done {img_idx+1}/{len(pairs)}")

    # Aggregate: mean and std across images
    table_idx = [int(np.argmin(np.abs(epsilons - e))) for e in eps_table]
    header = f"{'mechanism':<15s}" + "".join(f"  eps={e:>4.1f}" for e in eps_table)

    print()
    print("Mean Bayes accuracy across 40 DRIVE images (whole-image classification):")
    print(header)
    print("-" * len(header))
    for name, _ in mechanisms:
        row_mean = accs[name].mean(axis=0)
        row = f"{name:<15s}" + "".join(f"  {row_mean[i]:>8.4f}" for i in table_idx)
        print(row)
    print()
    print("Lift over uniform (pp, mean across images):")
    print(header)
    print("-" * len(header))
    uniform_mean = accs["uniform"].mean(axis=0)
    for name, _ in mechanisms:
        if name == "uniform":
            continue
        lifts_per_img = 100 * (accs[name] - accs["uniform"])
        lifts_mean = lifts_per_img.mean(axis=0)
        lifts_std = lifts_per_img.std(axis=0)
        row = f"{name:<15s}" + "".join(
            f"  {lifts_mean[i]:>+5.2f}±{lifts_std[i]:>4.2f}" for i in table_idx
        )
        print(row)

    # Plot: mean curve with shaded ±1 std band, log-x in [0.5, 10]
    styles = {
        "uniform":       dict(color="gray",     linestyle="--", marker="o", markersize=4),
        "channel-WF":    dict(color="tab:blue",                marker="s",  markersize=4),
        "spatial-WF":    dict(color="tab:green",               marker="^",  markersize=4),
        "joint-WF":      dict(color="tab:red",                 marker="D",  markersize=4, linewidth=2.0),
        "joint-WF+thr":  dict(color="tab:orange",              marker="P",  markersize=5, linewidth=2.5),
        "Bayes-optimal": dict(color="black",    linestyle=":",  marker="*", markersize=5),
        "adversarial":   dict(color="tab:purple", linestyle="-.", marker="x", markersize=4),
    }

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for name, _ in mechanisms:
        m = accs[name].mean(axis=0)
        s = accs[name].std(axis=0)
        kw = styles[name]
        ax.plot(epsilons, m, label=name, **kw)
        ax.fill_between(epsilons, m - s, m + s, color=kw["color"], alpha=0.10)
    ax.axhline(0.5, color="red", linestyle=":", alpha=0.3, label="chance")
    ax.set_xscale("log")
    ax.set_xticks(eps_table)
    ax.set_xticklabels([f"{e:g}" for e in eps_table])
    ax.set_xlim(0.45, 11.0)
    ax.set_xlabel(r"DP $\varepsilon$ (at $\delta=10^{-5}$)")
    ax.set_ylabel(r"Whole-image Bayes accuracy $\Phi(\sqrt{V})$")
    ax.set_title(f"Phase-0 on real DRIVE ({len(pairs)} images, 256x256, expert vessel masks)\n"
                 r"$\varepsilon \in [0.5, 10]$, mean $\pm$ 1 std band over images")
    ax.legend(loc="upper left", fontsize=9, ncol=1)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = "phase0_drive_real.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved curve plot to {out}")

    # Companion: per-channel contrast distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    bp = ax.boxplot(
        [delta_mu_all[:, c] for c in range(3)],
        tick_labels=["R", "G", "B"],
        patch_artist=True,
    )
    for patch, color in zip(bp["boxes"], ["tab:red", "tab:green", "tab:blue"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.set_ylabel(r"|$\mu^+_c$ - $\mu^-_c$| (per-image)")
    ax.set_title("DRIVE per-channel vessel-vs-background contrast (40 images)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig("phase0_drive_real_channels.png", dpi=150)
    plt.close(fig)
    print(f"Saved channel-contrast boxplot to phase0_drive_real_channels.png")


if __name__ == "__main__":
    main()
