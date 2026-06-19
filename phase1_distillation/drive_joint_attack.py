"""
Privacy audit of the ADVOCATED config: joint PATE K=10 + keep-2% release.
Faithfully replicates the joint release (10 teachers aggregated, top-2% shared
channels, Δ=2/K=0.2, low σ), then runs the same MIA + reconstruction attacks
as on the K=1 full release. Shows the high-utility config is also private.
"""
import json, sys
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, str(Path(__file__).parent))
from drive_local_demo import DriveDataset
from synthetic_demo import eps_to_rho, clip_and_normalise
from drive_pate_poc import train_K_teachers
from drive_pate_pruning_joint import shared_importance, thresholded_uniform_sigma
from drive_reconstruction_attack import Recon

HERE = Path(__file__).parent
K, KEEP = 10, 0.02


def auc_mw(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return ((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (pos.size * neg.size)


@torch.no_grad()
def agg_feats(teachers, caps_list, ds, device):
    """Mean of per-teacher clip+normalized bottlenecks (the unit-norm aggregate)."""
    out = []
    for x, _ in DataLoader(ds, batch_size=4):
        x = x.to(device); a = None
        for t, caps in zip(teachers, caps_list):
            _, _, e3 = t.encode(x)
            fn = clip_and_normalise(e3, caps.to(device))
            a = fn if a is None else a + fn
        out.append((a / len(teachers)).cpu())
    return torch.cat(out)                                    # (N,C,24,24)


def get_ds(name, split):
    if name == "isic":
        from isic_dataset import ISICDataset; return ISICDataset(split, 96), 3
    if name == "brats":
        from brats_dataset import BRATSDataset; return BRATSDataset(split, 96), 4
    return DriveDataset(split, 96), 3


def main():
    import sys
    ds_name = "drive"
    for i, a in enumerate(sys.argv):
        if a == "--dataset" and i + 1 < len(sys.argv):
            ds_name = sys.argv[i + 1]
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}  (auditing joint K={K}, keep={KEEP:.0%}, dataset={ds_name})")
    train_ds, in_ch = get_ds(ds_name, "train")
    val_ds, _ = get_ds(ds_name, "val")
    teachers, caps_list = train_K_teachers(train_ds, K, device, n_epochs=60, in_ch=in_ch)
    Cb = teachers[0].base * 4

    imp = shared_importance(teachers, DataLoader(train_ds, batch_size=4), device).to(device)
    rank = torch.argsort(imp, descending=True)
    n_active = max(1, int(round(KEEP * Cb)))
    active = torch.zeros(Cb, dtype=torch.bool, device=device); active[rank[:n_active]] = True
    deltas = torch.full((Cb,), 2.0 / K, device=device)
    act_idx = active.cpu().numpy().nonzero()[0]
    print(f"  C={Cb}  n_active={n_active}  Δ=2/K={2/K:.3f}")

    Fm = agg_feats(teachers, caps_list, train_ds, device)    # members
    Fn = agg_feats(teachers, caps_list, val_ds,   device)    # non-members
    Fm_a = Fm[:, act_idx].reshape(len(Fm), -1).numpy()       # active channels only
    Fn_a = Fn[:, act_idx].reshape(len(Fn), -1).numpy()

    epsilons = [16.0, 8.0, 2.0]
    sig_at = {e: thresholded_uniform_sigma(deltas, eps_to_rho(e), active)[active].mean().item()
              for e in epsilons}
    configs = [("no-noise", None, 0.0)] + [(f"ε={e:g}", e, sig_at[e]) for e in epsilons]

    # ---------- MIA ----------
    rng = np.random.default_rng(0); N_TRIAL = 200
    mia = {}
    print(f"\n[MIA]  {'config':>9} {'σ_act':>8} {'AUC':>7} {'TPR@.1':>7}")
    for name, eps, sigma in configs:
        aucs, tpr = [], []
        for _ in range(N_TRIAL):
            rel = Fm_a + (rng.normal(0, sigma, Fm_a.shape) if sigma > 0 else 0.0)
            d_m = ((rel[:, None] - Fm_a[None]) ** 2).sum(-1).min(0)
            d_n = ((rel[:, None] - Fn_a[None]) ** 2).sum(-1).min(0)
            s_m, s_n = -d_m, -d_n
            aucs.append(auc_mw(s_m, s_n)); tpr.append((s_m >= np.quantile(s_n, 0.9)).mean())
        mia[name] = {"epsilon": eps, "sigma_active": sigma,
                     "auc": float(np.mean(aucs)), "tpr_at_fpr0.1": float(np.mean(tpr))}
        print(f"        {name:>9} {sigma:>8.3f} {np.mean(aucs):>7.3f} {np.mean(tpr):>7.3f}")

    # ---------- reconstruction ----------
    Fm_masked = (Fm.to(device) * active.view(1, Cb, 1, 1).float())     # zero inactive
    imgs = torch.cat([x for x, _ in DataLoader(train_ds, batch_size=4)]).to(device)
    dec = Recon(Cb, out_ch=in_ch).to(device); opt = torch.optim.Adam(dec.parameters(), lr=2e-3)
    for _ in range(400):
        opt.zero_grad(); loss = ((dec(Fm_masked) - imgs) ** 2).mean(); loss.backward(); opt.step()
    Xn = imgs.cpu().numpy().transpose(0, 2, 3, 1)
    recon = {}; torch.manual_seed(0)
    print(f"\n[Recon] {'config':>9} {'PSNR':>7} {'SSIM':>7}")
    for name, eps, sigma in configs:
        noise = torch.zeros_like(Fm_masked)
        if sigma > 0:
            noise[:, active] = torch.randn_like(Fm_masked[:, active]) * sigma
        with torch.no_grad():
            xh = dec(Fm_masked + noise).clamp(0, 1).cpu().numpy().transpose(0, 2, 3, 1)
        mse = ((xh - Xn) ** 2).reshape(len(Xn), -1).mean(1)
        psnr = float(np.mean(10 * np.log10(1.0 / np.maximum(mse, 1e-10))))
        ss = float(np.mean([ssim(Xn[i], xh[i], channel_axis=2, data_range=1.0) for i in range(len(Xn))]))
        recon[name] = {"epsilon": eps, "psnr": psnr, "ssim": ss, "img": xh}
        print(f"        {name:>9} {psnr:>7.2f} {ss:>7.3f}")

    out = {"config": f"joint K={K} keep={KEEP}", "n_active": int(n_active),
           "mia": mia, "reconstruction": {k: {kk: v[kk] for kk in ("epsilon", "psnr", "ssim")}
                                          for k, v in recon.items()}}
    tag = "drive_joint_attack" if ds_name == "drive" else f"{ds_name}_joint_attack"
    (HERE / f"{tag}_results.json").write_text(json.dumps(out, indent=2))

    # panel (RGB datasets show 3ch; multi-modal shows modality 0)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    def disp(a): return a if a.shape[-1] == 3 else a[..., 0]
    show = [0, 5, 10]; panel = ["original", "no-noise", "ε=16", "ε=2"]
    fig, axes = plt.subplots(len(show), len(panel), figsize=(2.1 * len(panel), 2.1 * len(show)))
    for r, idx in enumerate(show):
        for c, key in enumerate(panel):
            ax = axes[r, c]; ax.imshow(disp(Xn[idx] if key == "original" else recon[key]["img"][idx]))
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(key if key in ("original", "no-noise") else
                             f"{key} (PSNR {recon[key]['psnr']:.0f})", fontsize=9)
    fig.suptitle(f"Reconstruction — advocated joint config (K={K}, keep {KEEP:.0%}, {n_active} ch)",
                 fontweight="bold")
    figname = "fig_joint_attack_recon.png" if ds_name == "drive" else f"fig_{ds_name}_joint_attack_recon.png"
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(HERE / figname, dpi=140)
    print(f"saved {tag}_results.json + {figname}")


if __name__ == "__main__":
    main()
