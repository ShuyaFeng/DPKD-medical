"""
Phase-0 utility validation on a real fundus image (DRIVE-style).

DRIVE itself is gated behind registration (no working open mirror found),
so as a stand-in we use scikit-image's public-domain retinal photograph and
derive vessel labels with the Frangi vesselness filter — the same kind of
classical operator that sits at the core of many automated DRIVE baselines.
This gives us:

  * real RGB channel statistics (the green-channel-best phenomenon is
    physical, not synthetic),
  * realistic spatial vessel sparsity (Frangi tree on the actual fundus),
  * a real FOV mask (everything outside the retinal disc is background).

We then plug the empirical channel and spatial weight maps into the same
closed-form Bayes-accuracy framework as `phase0_validation.py`, so the
numbers are directly comparable to the synthetic 4-channel toy.

This is NOT a substitute for a U-Net-on-DRIVE Phase-1 run; it is the
real-image analogue of Phase-0 to check that the mechanism's headline
behaviour (joint-WF beats uniform; +thr nearly closes the gap to
Bayes-optimal) survives when the channel and spatial weights come from
real fundus structure rather than from a controlled synthetic generator.
"""

import numpy as np
from scipy.stats import norm
from scipy.ndimage import binary_erosion
from skimage import data, filters, transform
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


# -----------------------------------------------------------------------------
# Build a DRIVE-style problem from the public retina image
# -----------------------------------------------------------------------------

def build_drive_like_problem(target_size=256):
    """Returns mu, Delta, beta, w_c, w_ij, plus diagnostic intermediates."""
    raw = data.retina()  # (1411, 1411, 3) uint8
    img = transform.resize(
        raw, (target_size, target_size), preserve_range=True, anti_aliasing=True
    ).astype(float) / 255.0                                              # (H, W, 3) in [0, 1]

    # FOV mask: pixels with non-trivial luminance — i.e. the actual retinal disc.
    luma = img.mean(axis=2)
    fov_raw = luma > 0.05
    # Erode by ~6 px to drop the bright crescent at the FOV boundary that
    # Frangi mistakes for vessels.
    fov = binary_erosion(fov_raw, iterations=6).astype(float)

    # Vessels are dark on a bright fundus → Frangi prefers bright tubes, so
    # we feed it the inverted green channel (which has the highest vessel
    # contrast in a fundus photograph).
    green = img[:, :, 1]
    inv_green = (1.0 - green) * fov
    v_map = filters.frangi(
        inv_green,
        sigmas=np.linspace(1.0, 4.0, 6),
        black_ridges=False,
    )
    v_map = v_map * fov                                                  # confine to eroded FOV
    v_map = v_map / max(v_map.max(), 1e-12)                              # normalize to [0, 1]

    # Binary vessel label by quantile threshold on v_map (within FOV).
    fov_pix = v_map[fov > 0.5]
    thr = float(np.quantile(fov_pix, 0.90))                              # ~10% of FOV labelled vessel
    vessel = (v_map > thr) & (fov > 0.5)
    background = (~vessel) & (fov > 0.5)

    # Per-channel global signal: mean RGB of vessels vs mean RGB of background.
    img_chw = img.transpose(2, 0, 1)                                     # (3, H, W)
    mu_plus = np.array([img_chw[c][vessel].mean() for c in range(3)])
    mu_minus = np.array([img_chw[c][background].mean() for c in range(3)])
    delta_mu = mu_plus - mu_minus                                        # signed; we use |.| later

    # Within-class std (averaged across the two classes) — used as beta.
    sigmas_within = []
    for c in range(3):
        s_pos = img_chw[c][vessel].std()
        s_neg = img_chw[c][background].std()
        sigmas_within.append(0.5 * (s_pos + s_neg))
    beta = float(np.mean(sigmas_within))

    # Rank-1 mu field: per-pixel signal strength = channel signal × spatial saliency.
    # We make μ_eff(c, i, j) = |delta_mu_c| * v_map(i, j) — vessels are where the
    # discriminative information lives, weighted by per-channel contrast.
    mu_field = np.abs(delta_mu)[:, None, None] * v_map[None, :, :]       # (3, H, W)

    # Per-pixel L2 sensitivity Delta. After [0,1] normalization with per-pixel
    # add-one-image neighbouring relation, treat as Δ = 1 element-wise.
    Delta = np.ones_like(mu_field)

    # Utility weights (oracle — from ground-truth μ field; see §3.4 robustness
    # in RESEARCH_PLAN.md §13.3 for what happens when the mask is corrupted).
    w_c = (mu_field ** 2).sum(axis=(1, 2))                               # per-channel
    w_ij = (mu_field ** 2).sum(axis=0)                                   # per-pixel

    diag = dict(
        img=img, fov=fov, v_map=v_map, vessel=vessel, background=background,
        mu_plus=mu_plus, mu_minus=mu_minus, delta_mu=delta_mu,
        sigmas_within=np.array(sigmas_within),
    )
    return mu_field, Delta, beta, w_c, w_ij, diag


def per_pixel_segmentation_acc(sigma2, delta_mu, beta, fov):
    """Mean per-pixel Bayes accuracy on the DRIVE segmentation task.

    For each pixel (i, j) in the FOV, the per-pixel LDA classifier on the
    released RGB triple has Bayes accuracy
        Φ( d_ij / 2 ) ,   d_ij² = Σ_c (Δμ_c)² / (β² + σ²_{c,i,j})
    The DRIVE-style metric is the FOV-average of these per-pixel accuracies.
    Unlike `bayes_acc` (which is whole-image classification), this measures
    how well a downstream model can label each pixel as vessel/background
    given the noisy released image — i.e. exactly what segmentation does.
    """
    # delta_mu has shape (3,); sigma2 has shape (3, H, W).
    dm2 = (delta_mu ** 2)[:, None, None]                                 # (3, 1, 1)
    d2 = (dm2 / (beta ** 2 + sigma2)).sum(axis=0)                        # (H, W)
    acc_pp = norm.cdf(np.sqrt(d2) / 2.0)                                 # (H, W)
    fov_mask = fov > 0.5
    return float(acc_pp[fov_mask].mean())


# -----------------------------------------------------------------------------
# Sweep
# -----------------------------------------------------------------------------

def main():
    target_size = 256
    delta_dp = 1e-5
    eps_dense = np.geomspace(0.5, 10.0, 25)
    eps_table = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 10.0])
    epsilons = np.unique(np.round(np.concatenate([eps_dense, eps_table]), 6))
    rhos = [eps_to_rho(e, delta_dp) for e in epsilons]

    print(f"Loading and processing retina at {target_size}x{target_size}...")
    mu, Delta, beta, w_c, w_ij, diag = build_drive_like_problem(target_size)
    C, H, W = mu.shape

    print()
    print(f"Image size:                          {H} x {W}, {C} channels")
    print(f"FOV pixels:                          {int(diag['fov'].sum())} / {H*W}")
    print(f"Labelled vessel pixels (Frangi):     {int(diag['vessel'].sum())}")
    print(f"Background pixels (FOV \\ vessel):    {int(diag['background'].sum())}")
    print()
    print(f"Per-channel vessel-mean μ+:          R={diag['mu_plus'][0]:.3f}  "
          f"G={diag['mu_plus'][1]:.3f}  B={diag['mu_plus'][2]:.3f}")
    print(f"Per-channel background-mean μ-:      R={diag['mu_minus'][0]:.3f}  "
          f"G={diag['mu_minus'][1]:.3f}  B={diag['mu_minus'][2]:.3f}")
    print(f"Per-channel signed contrast μ+ - μ-: R={diag['delta_mu'][0]:+.3f}  "
          f"G={diag['delta_mu'][1]:+.3f}  B={diag['delta_mu'][2]:+.3f}")
    print(f"Per-channel |contrast| (sorted):     "
          f"{dict(zip('RGB', np.round(np.abs(diag['delta_mu']), 4)))}")
    print(f"Within-class std β (averaged):       {beta:.4f}")
    print()
    print(f"Channel utility w_c (rank-1):        R={w_c[0]:.2f}  "
          f"G={w_c[1]:.2f}  B={w_c[2]:.2f}")
    print(f"  ratio max/min:                      {w_c.max() / max(w_c.min(), 1e-12):.1f}")
    print(f"Spatial utility w_ij sum-on-vessel / sum-off-vessel: "
          f"{w_ij[diag['vessel']].sum():.2f} / {w_ij[~diag['vessel']].sum():.2f}")
    print()

    no_dp_acc = bayes_acc(mu, np.zeros_like(mu), beta)
    print(f"No-DP Bayes accuracy (upper bound):  {no_dp_acc:.4f}")
    print()

    mechanisms = [
        ("uniform",       lambda r: mech_uniform(r, mu, Delta, beta)),
        ("channel-WF",    lambda r: mech_channel_wf(r, mu, Delta, beta, w_c)),
        ("spatial-WF",    lambda r: mech_spatial_wf(r, mu, Delta, beta, w_ij)),
        ("joint-WF",      lambda r: mech_joint_wf(r, mu, Delta, beta, w_c, w_ij)),
        ("joint-WF+thr",  lambda r: mech_joint_wf_threshold(r, mu, Delta, beta, w_c, w_ij)),
        ("Bayes-optimal", lambda r: mech_bayes_optimal(r, mu, Delta, beta)),
        ("adversarial",   lambda r: mech_adversarial(r, mu, Delta, beta, w_c, w_ij)),
    ]

    # Two metrics, both closed-form:
    #   results_cls[name] = whole-image binary classification (synthetic-comparable)
    #   results_seg[name] = mean per-pixel FOV accuracy (DRIVE-relevant for vessel seg)
    results_cls = {}
    results_seg = {}
    for name, fn in mechanisms:
        accs_cls, accs_seg = [], []
        for rho in rhos:
            sigma2 = fn(rho)
            accs_cls.append(bayes_acc(mu, sigma2, beta))
            accs_seg.append(per_pixel_segmentation_acc(
                sigma2, np.abs(diag["delta_mu"]), beta, diag["fov"]
            ))
        results_cls[name] = np.array(accs_cls)
        results_seg[name] = np.array(accs_seg)
        print(f"  {name:<14s} done")

    # Canonical-eps tables
    table_idx = [int(np.argmin(np.abs(epsilons - e))) for e in eps_table]
    header = f"{'mechanism':<15s}" + "".join(f"  eps={e:>4.1f}" for e in eps_table)

    for label, results in [
        ("Whole-image binary classification (synthetic-comparable):", results_cls),
        ("Mean per-pixel segmentation accuracy on FOV (DRIVE-relevant):", results_seg),
    ]:
        print()
        print(label)
        print(header)
        print("-" * len(header))
        for name, _ in mechanisms:
            row = f"{name:<15s}" + "".join(f"  {results[name][i]:>8.4f}" for i in table_idx)
            print(row)
        print()
        print("  Lift over uniform (pp):")
        for name, _ in mechanisms:
            if name == "uniform":
                continue
            lifts = [100 * (results[name][i] - results["uniform"][i]) for i in table_idx]
            row = f"  {name:<13s}" + "".join(f"  {l:>+8.2f}" for l in lifts)
            print(row)

    # Plot
    styles = {
        "uniform":       dict(color="gray",     linestyle="--", marker="o", markersize=4),
        "channel-WF":    dict(color="tab:blue",                marker="s",  markersize=4),
        "spatial-WF":    dict(color="tab:green",               marker="^",  markersize=4),
        "joint-WF":      dict(color="tab:red",                 marker="D",  markersize=4, linewidth=2.0),
        "joint-WF+thr":  dict(color="tab:orange",              marker="P",  markersize=5, linewidth=2.5),
        "Bayes-optimal": dict(color="black",    linestyle=":",  marker="*", markersize=5),
        "adversarial":   dict(color="tab:purple", linestyle="-.", marker="x", markersize=4),
    }

    # Single utility plot (whole-image classification, synthetic-comparable).
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for name, _ in mechanisms:
        ax.plot(epsilons, results_cls[name], label=name, **styles[name])
    ax.axhline(0.5, color="red", linestyle=":", alpha=0.3, label="chance")
    ax.set_xscale("log")
    ax.set_xticks(eps_table)
    ax.set_xticklabels([f"{e:g}" for e in eps_table])
    ax.set_xlim(0.45, 11.0)
    ax.set_xlabel(r"DP $\varepsilon$ (at $\delta=10^{-5}$)")
    ax.set_ylabel(r"Whole-image Bayes accuracy $\Phi(\sqrt{V})$")
    ax.set_title("Phase-0 on real fundus image (skimage retina + Frangi labels)\n"
                 r"$\varepsilon \in [0.5, 10]$, $\delta=10^{-5}$, 256×256×3 RGB")
    ax.legend(loc="upper left", fontsize=9, ncol=1)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig("phase0_drive.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved utility plot to phase0_drive.png")

    # Diagnostic: per-mechanism noise-allocation heatmap at eps=2.
    rho_target = eps_to_rho(2.0, delta_dp)
    show_mechs = ["uniform", "channel-WF", "spatial-WF", "joint-WF", "joint-WF+thr", "Bayes-optimal"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, name in zip(axes.ravel(), show_mechs):
        fn = dict(mechanisms)[name]
        sigma2 = fn(rho_target)
        # Show log10(sigma^2) averaged across channels (mask out FOV outside).
        mean_log = np.log10(np.maximum(sigma2.mean(axis=0), 1e-3))
        mean_log = np.where(diag["fov"] > 0.5, mean_log, np.nan)
        n_active = int((sigma2 < 1e10).all(axis=0).sum())
        im = ax.imshow(mean_log, cmap="viridis")
        ax.set_title(f"{name}\nactive pixels: {n_active}/{H*W}", fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$\log_{10}\bar\sigma^2$")
    fig.suptitle(r"Noise allocation $\bar\sigma^2_{ij}$ at $\varepsilon=2$ (channel-mean, log scale)", y=1.02)
    fig.tight_layout()
    fig.savefig("phase0_drive_alloc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved noise-allocation diagnostic to phase0_drive_alloc.png")

    # Companion figure: image + Frangi map + vessel mask + spatial utility
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(diag["img"]); axes[0].set_title("retina (RGB, 256×256)"); axes[0].axis("off")
    axes[1].imshow(diag["v_map"], cmap="hot"); axes[1].set_title("Frangi vesselness $v(i,j)$"); axes[1].axis("off")
    axes[2].imshow(diag["vessel"], cmap="gray"); axes[2].set_title(f"vessel label (top {int(100 * diag['vessel'].sum() / diag['fov'].sum())}% in FOV)"); axes[2].axis("off")
    axes[3].imshow(w_ij, cmap="viridis"); axes[3].set_title(r"spatial utility $w_{ij} = \sum_c \mu_{cij}^2$"); axes[3].axis("off")
    fig.tight_layout()
    fig.savefig("phase0_drive_inputs.png", dpi=150)
    plt.close(fig)
    print(f"Saved input-diagnostic figure to phase0_drive_inputs.png")


if __name__ == "__main__":
    main()
