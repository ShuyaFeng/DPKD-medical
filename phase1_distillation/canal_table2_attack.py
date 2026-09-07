"""
canal_table2_attack.py — Table 2: MIA + reconstruction attacks, CANAL vs uniform.

Paper-consistent config: K=3, top-10% channels, budget split fc=0.10/fi=0.05/fr=0.85,
epsilon in {2, 8}, Emily's corrected importance sensitivity (teacher-level formula).

Attacks:
  MIA   — nearest-neighbour distance in active-channel feature space (200 noise trials).
  Recon — decoder trained on clean MASKED features (active channels only, inactive zeroed);
          SSIM measured on masked+noisy released features at each epsilon.

No-noise baseline included: proves the leak exists without DP (MIA AUC=1.0, SSIM high).
Both attacks run for uniform and CANAL (water-filling) sigma at each epsilon.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, str(Path(__file__).parent))
from synthetic_demo import eps_to_rho, clip_and_normalise
from drive_pate_poc import train_K_teachers, correct_uniform_sigma, partition_dataset
from drive_pate_pruning_joint import shared_importance
from drive_pate_canal_combined import (
    add_dp_noise_to_importance, correct_waterfilling_sigma, importance_sensitivity,
)
from drive_reconstruction_attack import Recon

HERE = Path(__file__).parent
(HERE / "results").mkdir(exist_ok=True)

K         = 3
KEEP_FRAC = 0.10
FC        = 0.10   # caps budget fraction
FI        = 0.05   # importance budget fraction
FR        = 0.85   # release budget fraction
assert abs(FC + FI + FR - 1.0) < 1e-9, "Budget fractions must sum to 1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def auc_mw(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return (
        (pos[:, None] > neg[None, :]).sum()
        + 0.5 * (pos[:, None] == neg[None, :]).sum()
    ) / (pos.size * neg.size)


@torch.no_grad()
def agg_feats(teachers, caps_list, ds, device):
    """Mean of per-teacher clip+normalize bottlenecks over the full dataset."""
    out = []
    for x, _ in DataLoader(ds, batch_size=4):
        x = x.to(device)
        a = None
        for t, caps in zip(teachers, caps_list):
            _, _, e3 = t.encode(x)
            fn = clip_and_normalise(e3, caps.to(device))
            a = fn if a is None else a + fn
        out.append((a / len(teachers)).cpu())
    return torch.cat(out)                                        # (N, C, H, W)


def measure_per_sample_cap_norms(teachers, train_ds, device):
    """Per-sample max-channel bottleneck norm → calibrates clip_caps."""
    partitions = partition_dataset(len(train_ds), len(teachers))
    all_norms = []
    for teacher, idxs in zip(teachers, partitions):
        teacher.eval()
        subset = Subset(train_ds, idxs)
        for i in range(len(subset)):
            x, _ = subset[i]
            x = x.unsqueeze(0).to(device)
            with torch.no_grad():
                _, _, e3 = teacher.encode(x)
            per_ch = e3[0].flatten(1).norm(dim=1)
            all_norms.append(per_ch.max().item())
    norms = torch.tensor(all_norms)
    return float(norms.mean()), float(norms.quantile(0.99))


def privatize_caps(caps_list, clip_caps, n_train, rho_caps, seed=0):
    """Gaussian DP noise on teacher caps."""
    if rho_caps <= 0:
        return caps_list
    sens  = 2.0 * clip_caps / float(n_train)
    sigma = sens / math.sqrt(2.0 * rho_caps)
    noisy = []
    for k, caps in enumerate(caps_list):
        g = torch.Generator()
        g.manual_seed(seed + k)
        noise = torch.randn(caps.shape, generator=g) * sigma
        noisy.append((caps.detach().cpu() + noise).clamp(min=1e-6))
    return noisy


def get_ds(name, split):
    if name == "isic":
        from isic_dataset import ISICDataset
        return ISICDataset(split, 96), 3
    if name == "kvasir":
        from kvasir_dataset import KvasirDataset
        return KvasirDataset(split, 96), 3
    if name == "busi":
        from busi_dataset import BUSIDataset
        return BUSIDataset(split, 96), 1
    raise ValueError(f"Unknown dataset: {name}")


def compute_ssim_mean(Xn, xh, in_ch):
    vals = []
    for i in range(len(Xn)):
        a, b = Xn[i], xh[i]
        if in_ch == 1:
            s = ssim(a[..., 0], b[..., 0], data_range=1.0)
        else:
            s = ssim(a, b, channel_axis=2, data_range=1.0)
        vals.append(s)
    return float(np.mean(vals))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",        default="isic", choices=["isic", "kvasir", "busi"])
    ap.add_argument("--epsilons",       default="2,8")
    ap.add_argument("--teacher-epochs", type=int, default=60)
    ap.add_argument("--n-eval",         type=int, default=200,
                    help="Max samples used in MIA distance computation (memory guard)")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    EPS = [float(e) for e in args.epsilons.split(",")]

    print(f"Device: {device}  dataset={args.dataset}  K={K}  keep={KEEP_FRAC:.0%}")
    print(f"Budget split: fc={FC}  fi={FI}  fr={FR}  eps={EPS}")

    train_ds, in_ch = get_ds(args.dataset, "train")
    val_ds,   _     = get_ds(args.dataset, "val")
    N = len(train_ds)
    print(f"  train={N}  val={len(val_ds)}  in_ch={in_ch}")

    # --- Train K=3 teachers (60 epochs) ---
    print(f"\nTraining {K} teachers ({args.teacher_epochs} epochs)...")
    teachers, caps_list = train_K_teachers(
        train_ds, K, device, n_epochs=args.teacher_epochs, in_ch=in_ch
    )
    Cb     = teachers[0].base * 4
    n_keep = max(1, int(round(KEEP_FRAC * Cb)))
    deltas = torch.full((Cb,), 2.0 / K, device=device)
    print(f"  C={Cb}  n_keep={n_keep}  delta=2/K={2.0/K:.4f}")

    # --- Calibrate clip_caps from per-sample feature norms ---
    print("Measuring per-sample cap norms...")
    clip_caps_avg, clip_caps_p99 = measure_per_sample_cap_norms(teachers, train_ds, device)
    clip_caps = clip_caps_avg
    print(f"  clip_caps avg={clip_caps_avg:.4e}  p99={clip_caps_p99:.4e}  using avg")

    # --- Compute importance with corrected sensitivity ---
    print("Computing channel importance...")
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=False)
    importance   = shared_importance(teachers, train_loader, device).to(device)
    imp_cpu      = importance.cpu()
    clip_imp     = float(torch.quantile(imp_cpu, 0.90))
    sens_imp     = importance_sensitivity(clip_imp, K, N)
    print(f"  mean={float(imp_cpu.mean()):.4e}  p90={clip_imp:.4e}  sens_imp={sens_imp:.4e}")

    # --- Channel mask from CLEAN (non-privatized) importance ---
    # Used for decoder training and attack evaluation (worst-case: attacker knows the mask).
    clean_rank        = torch.argsort(importance, descending=True)
    clean_top_indices = clean_rank[:n_keep]
    clean_top_mask    = torch.zeros(Cb, dtype=torch.bool, device=device)
    clean_top_mask[clean_top_indices] = True
    clean_act_idx     = clean_top_indices.cpu().numpy()
    print(f"  Clean channel mask: {n_keep} active channels")

    # --- Reference features (clean, mean of K teachers) ---
    print("Computing reference features (clean, all channels)...")
    Fm_all = agg_feats(teachers, caps_list, train_ds, device)    # (N, C, H, W)
    Fn_all = agg_feats(teachers, caps_list, val_ds,   device)
    H, W   = Fm_all.shape[2], Fm_all.shape[3]

    N_m = min(N, args.n_eval)
    N_n = min(len(val_ds), args.n_eval)
    Fm  = Fm_all[:N_m]
    Fn  = Fn_all[:N_n]
    print(f"  MIA: {N_m} members / {N_n} non-members  H={H} W={W}")

    # MIA feature matrices: clean active channels only, flattened
    Fm_a = Fm[:, clean_act_idx].reshape(N_m, -1).numpy().astype(np.float32)
    Fn_a = Fn[:, clean_act_idx].reshape(N_n, -1).numpy().astype(np.float32)

    # Masked features: inactive channels zeroed (same input the decoder sees at test time)
    Fm_dev    = Fm_all.to(device)
    Fm_masked = Fm_dev * clean_top_mask.float().view(1, Cb, 1, 1)

    # --- Train reconstruction decoder ONCE on masked clean features ---
    print("Training reconstruction decoder (400 iters, masked channels, clean)...")
    imgs_all = torch.cat([x for x, _ in DataLoader(train_ds, batch_size=4)]).to(device)
    dec = Recon(Cb, out_ch=in_ch).to(device)
    opt = torch.optim.Adam(dec.parameters(), lr=2e-3)
    for _ in range(400):
        opt.zero_grad()
        loss = ((dec(Fm_masked) - imgs_all) ** 2).mean()
        loss.backward()
        opt.step()
    Xn = imgs_all.cpu().numpy().transpose(0, 2, 3, 1)   # (N, H, W, C)
    print(f"  Decoder trained  final MSE={loss.item():.5f}")

    # --- No-noise baselines (prove the leak exists without DP) ---
    print("\nComputing no-noise baselines...")
    # MIA no-noise: released features == member features exactly → AUC = 1.0
    print(f"  No-noise MIA AUC = 1.000  (by construction: rel=Fm, every member self-matches)")
    # Reconstruction no-noise: decoder on clean masked features
    with torch.no_grad():
        xh_nn = dec(Fm_masked).clamp(0, 1).cpu().numpy().transpose(0, 2, 3, 1)
    ssim_nn = compute_ssim_mean(Xn, xh_nn, in_ch)
    print(f"  No-noise Recon SSIM = {ssim_nn:.3f}  (sanity: high = leak exists without DP)")

    results = {
        "dataset": args.dataset, "K": K,
        "keep_frac": KEEP_FRAC, "n_keep": n_keep,
        "budget": {"fc": FC, "fi": FI, "fr": FR},
        "clip_caps_avg": clip_caps_avg, "clip_caps_p99": clip_caps_p99,
        "clip_imp": clip_imp, "sens_imp": sens_imp,
        "epsilons": EPS,
        "no_noise": {
            "mia_auc":           1.0,
            "mia_tpr_at_fpr0.1": 1.0,
            "recon_ssim":        ssim_nn,
        },
        "sweep": {},
    }

    rng     = np.random.default_rng(0)
    N_TRIAL = 200

    for eps in EPS:
        rho      = eps_to_rho(eps)
        rho_caps = FC * rho
        rho_imp  = FI * rho
        rho_rel  = FR * rho
        print(f"\n{'='*64}\neps={eps}  rho_caps={rho_caps:.5f}  rho_imp={rho_imp:.5f}  rho_rel={rho_rel:.4f}")

        # Per-epsilon privatized caps and importance (for sigma computation only)
        caps_noisy = privatize_caps(caps_list, clip_caps, N, rho_caps,
                                    seed=int(eps * 100))
        imp_noisy  = add_dp_noise_to_importance(importance, sens_imp, rho_imp,
                                                seed=int(eps * 100) + 7)

        # Per-channel sigmas on clean-mask channels
        sigma_uni = torch.zeros(Cb, device=device)
        sigma_uni[clean_top_mask] = correct_uniform_sigma(deltas[clean_top_mask], rho_rel)

        sigma_wf = torch.zeros(Cb, device=device)
        sigma_wf[clean_top_mask] = correct_waterfilling_sigma(
            deltas[clean_top_mask], imp_noisy[clean_top_mask], rho_rel
        )

        sig_uni_mean = sigma_uni[clean_top_mask].mean().item()
        sig_wf_mean  = sigma_wf[clean_top_mask].mean().item()
        print(f"  sigma_uni active_mean={sig_uni_mean:.4f}")
        print(f"  sigma_wf  active_mean={sig_wf_mean:.4f}  "
              f"range=[{sigma_wf[clean_top_mask].min():.4f}, {sigma_wf[clean_top_mask].max():.4f}]")

        # Spatial sigma vectors for MIA (n_keep * H * W,)
        sig_uni_scalar  = float(sigma_uni[clean_top_mask][0].item())
        sig_wf_per_ch   = sigma_wf[clean_act_idx].cpu().numpy().astype(np.float32)
        sig_uni_spatial = np.full(n_keep * H * W, sig_uni_scalar, dtype=np.float32)
        sig_wf_spatial  = np.repeat(sig_wf_per_ch, H * W)

        ep_res = {"rho_caps": rho_caps, "rho_imp": rho_imp, "rho_rel": rho_rel,
                  "sigma_uni_active_mean": sig_uni_mean,
                  "sigma_wf_active_mean":  sig_wf_mean}

        # ── MIA ──────────────────────────────────────────────────────────────
        print(f"\n[MIA  eps={eps}]  {'config':>9}  {'AUC':>6}  {'TPR@FPR0.1':>10}")
        for label, sig_spatial in [("uniform", sig_uni_spatial), ("canal", sig_wf_spatial)]:
            aucs, tprs = [], []
            for _ in range(N_TRIAL):
                noise = rng.standard_normal(Fm_a.shape).astype(np.float32) * sig_spatial[None, :]
                rel   = Fm_a + noise
                d_m   = ((rel[:, None] - Fm_a[None]) ** 2).sum(-1).min(axis=0)
                d_n   = ((rel[:, None] - Fn_a[None]) ** 2).sum(-1).min(axis=0)
                s_m, s_n = -d_m, -d_n
                aucs.append(auc_mw(s_m, s_n))
                tprs.append(float((s_m >= np.quantile(s_n, 0.9)).mean()))
            ep_res[f"mia_{label}"] = {
                "auc":           float(np.mean(aucs)),
                "tpr_at_fpr0.1": float(np.mean(tprs)),
            }
            print(f"           {label:>9}  {np.mean(aucs):>6.3f}  {np.mean(tprs):>10.3f}")

        # ── Reconstruction attack ─────────────────────────────────────────────
        print(f"\n[Recon eps={eps}]  {'config':>9}  {'SSIM':>6}")

        for i_label, (label, sigma_per_ch) in enumerate([("uniform", sigma_uni), ("canal", sigma_wf)]):
            torch.manual_seed(int(eps * 100) + i_label)
            noise = torch.zeros_like(Fm_dev)
            sig_active = sigma_per_ch[clean_top_mask].view(1, -1, 1, 1)
            noise[:, clean_top_mask] = torch.randn_like(Fm_dev[:, clean_top_mask]) * sig_active
            with torch.no_grad():
                xh = dec(Fm_masked + noise).clamp(0, 1).cpu().numpy().transpose(0, 2, 3, 1)
            ss = compute_ssim_mean(Xn, xh, in_ch)
            ep_res[f"recon_{label}"] = {"ssim": ss}
            print(f"            {label:>9}  {ss:>6.3f}")

        results["sweep"][str(eps)] = ep_res

    # --- Summary table ---
    print(f"\n{'='*72}")
    print(f"TABLE 2  ({args.dataset.upper()})  K={K}  fc={FC}/fi={FI}/fr={FR}")
    print(f"  {'config':>10}  {'MI-AUC-uni':>11}  {'MI-AUC-canal':>13}  "
          f"{'SSIM-uni':>9}  {'SSIM-canal':>10}")
    print(f"  {'no-noise':>10}  {'1.000':>11}  {'1.000':>13}  "
          f"{results['no_noise']['recon_ssim']:>9.3f}  {'(same)':>10}")
    for eps in EPS:
        r = results["sweep"][str(eps)]
        print(f"  {eps:>10.1f}  "
              f"{r['mia_uniform']['auc']:>11.3f}  "
              f"{r['mia_canal']['auc']:>13.3f}  "
              f"{r['recon_uniform']['ssim']:>9.3f}  "
              f"{r['recon_canal']['ssim']:>10.3f}")

    out = HERE / "results" / f"{args.dataset}_table2_attack_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
