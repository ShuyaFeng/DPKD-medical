"""
canal_table2_attack.py — Table 2: membership-inference + feature-inversion attacks
on the released features, CANAL vs uniform, under the paper's exact mechanism.

Mechanism (identical to canal_final_experiment.py / paper Sec. 2.1-2.4, 3.1):
  K=3 teachers on disjoint cohorts, per-channel clip+normalize with privatized caps,
  mean over teachers, importance privatized with the teacher-level sensitivity
  importance_sensitivity(clip_imp, K, N), top-10% channels by the PRIVATIZED scores,
  budget split rho_tot = rho(eps) -> (fc, fi, fr) = (0.10, 0.05, 0.85), release noise
  calibrated to rho_rel on the active channels: uniform sigma or CANAL water-filling.
  5 seeds, teachers retrained per seed (same protocol as Table 1).

Membership inference (strong adversary):
  The attacker knows the clean aggregated feature f_x of every candidate x
  (members = teacher-training images, which are also the query images; non-members
  = validation images) and all N released vectors r_j = f_j + N(0, diag(sigma^2)).
  Bayes-optimal test for "is x a member": for every release j compute the posterior
  P(j came from x) over the candidate set under the Gaussian likelihood with the
  actual per-channel sigma, and score x by max_j P(j from x).  We also report the
  max log-likelihood-ratio statistic and the nearest-release distance statistic used
  previously; the latter is dominated by each release's own noise energy ||n_j||^2
  and reads ~0.5 regardless of eps.
  AUC and TPR@FPR=0.1 are averaged over N_TRIAL fresh noise draws.

Feature inversion:
  Decoder trained on clean masked (active channels) features -> images of the
  training set itself (upper-bound adversary), then applied to the noisy release.
  SSIM vs the true images; the no-noise row shows the leak without DP.

Usage:
  python canal_table2_attack.py --dataset isic                 # paper config
  python canal_table2_attack.py --dataset isic --seeds 100 --te 20 --n-trial 20   # smoke
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, str(Path(__file__).parent))
from synthetic_demo import eps_to_rho, clip_and_normalise
from drive_pate_poc import train_K_teachers, correct_uniform_sigma, partition_dataset
from drive_pate_pruning_joint import shared_importance
from drive_pate_canal_combined import correct_waterfilling_sigma, importance_sensitivity
from drive_reconstruction_attack import Recon


def add_dp_noise_to_importance(importance, sensitivity, rho_imp, seed=42):
    """Gaussian mechanism on the importance vector (same as canal_final_experiment)."""
    sigma_imp = sensitivity / math.sqrt(2.0 * rho_imp)
    g = torch.Generator().manual_seed(int(seed))
    return (importance.detach().cpu() + torch.randn(importance.shape, generator=g) * sigma_imp).clamp(min=1e-12)

HERE = Path(__file__).parent
(HERE / "results").mkdir(exist_ok=True)

K         = 3
KEEP_FRAC = 0.10
FC, FI, FR = 0.10, 0.05, 0.85
assert abs(FC + FI + FR - 1.0) < 1e-9
DELTA     = 1e-5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def auc_mw(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return ((pos[:, None] > neg[None, :]).sum()
            + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (pos.size * neg.size)


def mm(a, b):
    # einsum instead of @: Apple Accelerate emits spurious FP warnings on large matmuls
    return np.einsum("id,jd->ij", a, b)


@torch.no_grad()
def agg_feats(teachers, caps_list, ds, device):
    """Mean over teachers of per-channel clip+normalized bottlenecks. (N, C, H, W) on CPU."""
    out = []
    for x, _ in DataLoader(ds, batch_size=8):
        x = x.to(device)
        a = None
        for t, caps in zip(teachers, caps_list):
            _, _, e3 = t.encode(x)
            fn = clip_and_normalise(e3, caps.to(device))
            a = fn if a is None else a + fn
        out.append((a / len(teachers)).cpu())
    return torch.cat(out)


def measure_per_sample_cap_norms(teachers, train_ds, device):
    partitions = partition_dataset(len(train_ds), len(teachers))
    all_norms = []
    for teacher, idxs in zip(teachers, partitions):
        teacher.eval()
        for x, _ in DataLoader(Subset(train_ds, idxs), batch_size=8):
            with torch.no_grad():
                _, _, e3 = teacher.encode(x.to(device))
            all_norms.extend(e3.flatten(2).norm(dim=2).max(dim=1).values.tolist())
    norms = torch.tensor(all_norms)
    return float(norms.mean()), float(norms.quantile(0.99))


def privatize_caps(caps_list, clip_caps, n_train, rho_caps, seed=0):
    """Same mechanism as canal_final_experiment.privatize_caps."""
    if rho_caps <= 0:
        return caps_list
    sens  = 2.0 * clip_caps / float(n_train)
    sigma = sens / math.sqrt(2.0 * rho_caps)
    noisy = []
    for k, caps in enumerate(caps_list):
        g = torch.Generator().manual_seed(seed + k)
        noisy.append((caps.detach().cpu() + torch.randn(caps.shape, generator=g) * sigma).clamp(min=1e-6))
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
    raise ValueError(name)


def ssim_mean(Xn, xh, in_ch):
    if in_ch == 1:
        return float(np.mean([ssim(Xn[i][..., 0], xh[i][..., 0], data_range=1.0) for i in range(len(Xn))]))
    return float(np.mean([ssim(Xn[i], xh[i], channel_axis=2, data_range=1.0) for i in range(len(Xn))]))


# ---------------------------------------------------------------------------
# Membership-inference statistics
# ---------------------------------------------------------------------------

def mia_scores(rel, cand, sig2):
    """
    rel  : (N_rel, D) released vectors (members + noise)
    cand : (N_cand, D) clean candidate features, members first then non-members
    sig2 : (D,) per-coordinate noise variance
    Returns dict of per-candidate membership scores (higher = more likely member).
    """
    w_rel  = rel / sig2[None, :]                                   # rel / sigma^2
    loglik = mm(w_rel, cand) - 0.5 * ((cand ** 2) / sig2[None, :]).sum(1)[None, :]   # (N_rel, N_cand)
    llr    = loglik.max(0)
    L = loglik - loglik.max(1, keepdims=True)
    P = np.exp(L); P /= P.sum(1, keepdims=True)
    post   = P.max(0)
    d      = (rel ** 2).sum(1)[:, None] + (cand ** 2).sum(1)[None, :] - 2.0 * mm(rel, cand)
    dist   = -d.min(0)
    return {"posterior": post, "llr": llr, "distance": dist}


def run_mia(Fm_a, Fn_a, sig_per_coord, n_trial, rng):
    cand = np.concatenate([Fm_a, Fn_a]); nm = len(Fm_a)
    sig2 = (sig_per_coord ** 2).astype(np.float64)
    acc = {k: {"auc": [], "tpr": []} for k in ("posterior", "llr", "distance")}
    for _ in range(n_trial):
        rel = Fm_a + rng.normal(0.0, 1.0, Fm_a.shape) * sig_per_coord[None, :]
        for k, s in mia_scores(rel, cand, sig2).items():
            sm, sn = s[:nm], s[nm:]
            acc[k]["auc"].append(auc_mw(sm, sn))
            acc[k]["tpr"].append(float((sm >= np.quantile(sn, 0.9)).mean()))
    return {k: {"auc": float(np.mean(v["auc"])), "tpr_at_fpr0.1": float(np.mean(v["tpr"]))}
            for k, v in acc.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",  default="isic", choices=["isic", "kvasir", "busi"])
    ap.add_argument("--epsilons", default="2,8")
    ap.add_argument("--seeds",    type=int, nargs="+", default=[100, 200, 300, 400, 500])
    ap.add_argument("--te",       type=int, default=60, help="teacher epochs")
    ap.add_argument("--n-trial",  type=int, default=100, help="noise draws per MIA cell")
    ap.add_argument("--dec-iters", type=int, default=1500, help="inversion decoder iterations")
    ap.add_argument("--out",      default=None)
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    EPS = [float(e) for e in args.epsilons.split(",")]
    print(f"Device={device}  dataset={args.dataset}  K={K}  keep={KEEP_FRAC:.0%}  "
          f"split=({FC},{FI},{FR})  eps={EPS}  seeds={args.seeds}")

    train_ds, in_ch = get_ds(args.dataset, "train")
    val_ds,   _     = get_ds(args.dataset, "val")
    N = len(train_ds)
    X_all = torch.cat([x for x, _ in DataLoader(train_ds, batch_size=8)])
    Xn    = X_all.numpy().transpose(0, 2, 3, 1)
    print(f"  members(train)={N}  non-members(val)={len(val_ds)}")

    results = {"dataset": args.dataset, "K": K, "keep_frac": KEEP_FRAC,
               "budget": {"fc": FC, "fi": FI, "fr": FR}, "delta": DELTA,
               "epsilons": EPS, "seeds": args.seeds, "teacher_epochs": args.te,
               "n_trial": args.n_trial, "n_members": N, "n_nonmembers": len(val_ds),
               "mia_statistic": "posterior (Bayes-optimal Gaussian adversary knowing all candidate features)",
               "per_seed": [], "summary": {}}

    for seed in args.seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        rng = np.random.default_rng(seed)
        t0 = time.time()
        teachers, caps_list = train_K_teachers(train_ds, K, device, n_epochs=args.te, in_ch=in_ch)
        Cb     = teachers[0].base * 4
        n_keep = max(1, int(round(KEEP_FRAC * Cb)))
        deltas = torch.full((Cb,), 2.0 / K)
        clip_caps, _ = measure_per_sample_cap_norms(teachers, train_ds, device)
        importance = shared_importance(teachers, DataLoader(train_ds, batch_size=8, shuffle=False), device).cpu()
        clip_imp   = float(torch.quantile(importance, 0.90))
        sens_imp   = importance_sensitivity(clip_imp, K, N)
        print(f"\n[seed {seed}] teachers trained ({time.time()-t0:.0f}s)  C={Cb} n_keep={n_keep}  "
              f"clip_caps={clip_caps:.3f}  sens_imp={sens_imp:.3e}")
        seed_res = {"seed": seed, "sweep": {}}

        # no-noise reference: clean caps, clean ranking
        rank = torch.argsort(importance, descending=True)
        mask0 = torch.zeros(Cb, dtype=torch.bool); mask0[rank[:n_keep]] = True
        F0 = agg_feats(teachers, caps_list, train_ds, device)
        dec0, nn_ssim = train_decoder(F0, mask0, X_all, Cb, in_ch, device, args.dec_iters)
        mean_img = np.broadcast_to(Xn.mean(0, keepdims=True), Xn.shape)
        ssim_meanimg = ssim_mean(Xn, mean_img, in_ch)
        seed_res["no_noise"] = {"mia_auc": 1.0, "recon_ssim": nn_ssim, "ssim_mean_image": ssim_meanimg}
        print(f"  no-noise: MIA AUC=1.000 (release == feature)   recon SSIM={nn_ssim:.3f}   "
              f"[SSIM of dataset-mean image = {ssim_meanimg:.3f}: no-information floor]")

        for eps in EPS:
            rho = eps_to_rho(eps, DELTA)
            rho_caps, rho_imp, rho_rel = FC * rho, FI * rho, FR * rho
            caps_noisy = privatize_caps(caps_list, clip_caps, N, rho_caps, seed=seed + int(eps * 100))
            imp_noisy  = add_dp_noise_to_importance(importance, sens_imp, rho_imp, seed=seed + int(eps * 100) + 7)
            mask = torch.zeros(Cb, dtype=torch.bool)
            mask[torch.argsort(imp_noisy, descending=True)[:n_keep]] = True
            act = mask.nonzero().flatten()

            sig_uni = torch.zeros(Cb); sig_uni[mask] = correct_uniform_sigma(deltas[mask], rho_rel)
            sig_wf  = torch.zeros(Cb); sig_wf[mask]  = correct_waterfilling_sigma(deltas[mask], imp_noisy[mask], rho_rel)

            Fm = agg_feats(teachers, caps_noisy, train_ds, device)
            Fn = agg_feats(teachers, caps_noisy, val_ds, device)
            HW = Fm.shape[2] * Fm.shape[3]
            Fm_a = Fm[:, act].reshape(N, -1).numpy().astype(np.float64)
            Fn_a = Fn[:, act].reshape(len(Fn), -1).numpy().astype(np.float64)
            dec, _ = train_decoder(Fm, mask, X_all, Cb, in_ch, device, args.dec_iters)

            ep = {"rho_rel": rho_rel, "sigma_uni": float(sig_uni[mask].mean()),
                  "sigma_wf_mean": float(sig_wf[mask].mean()),
                  "sigma_wf_range": [float(sig_wf[mask].min()), float(sig_wf[mask].max())]}
            for label, sig in (("uniform", sig_uni), ("canal", sig_wf)):
                sig_coord = np.repeat(sig[act].numpy().astype(np.float64), HW)
                ep[f"mia_{label}"] = run_mia(Fm_a, Fn_a, sig_coord, args.n_trial, rng)
                g = torch.Generator().manual_seed(seed + int(eps * 1000) + (label == "canal"))
                noise = torch.randn(Fm.shape, generator=g) * sig.view(1, Cb, 1, 1)
                with torch.no_grad():
                    xh = dec((Fm * mask.view(1, Cb, 1, 1).float() + noise).to(device)).clamp(0, 1).cpu().numpy().transpose(0, 2, 3, 1)
                # shuffled pairing: reconstruction of image i scored against image i+1.
                # Equal to the unshuffled SSIM means the decoder output carries no image-specific information.
                ep[f"recon_{label}"] = {"ssim": ssim_mean(Xn, xh, in_ch),
                                        "ssim_shuffled": ssim_mean(np.roll(Xn, 1, axis=0), xh, in_ch)}
                m = ep[f"mia_{label}"]
                print(f"  eps={eps:g} {label:8s} sigma={float(sig[mask].mean()):6.3f}  "
                      f"MI-AUC posterior={m['posterior']['auc']:.3f} llr={m['llr']['auc']:.3f} "
                      f"distance={m['distance']['auc']:.3f}  SSIM={ep[f'recon_{label}']['ssim']:.3f} "
                      f"(shuffled {ep[f'recon_{label}']['ssim_shuffled']:.3f})", flush=True)
            seed_res["sweep"][str(eps)] = ep
        results["per_seed"].append(seed_res)
        (Path(args.out) if args.out else HERE / "results" / f"{args.dataset}_table2_attack_results.json"
         ).write_text(json.dumps(results, indent=2))

    # --- summary over seeds ---
    def agg(path):
        vals = []
        for s in results["per_seed"]:
            v = s
            for p in path: v = v[p]
            vals.append(v)
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "values": [float(v) for v in vals]}
    summ = {"no_noise_ssim": agg(["no_noise", "recon_ssim"]),
            "ssim_mean_image_floor": agg(["no_noise", "ssim_mean_image"])}
    for eps in EPS:
        for label in ("uniform", "canal"):
            for stat in ("posterior", "llr", "distance"):
                summ[f"{label}_eps{eps:g}_auc_{stat}"] = agg(["sweep", str(eps), f"mia_{label}", stat, "auc"])
            summ[f"{label}_eps{eps:g}_ssim"] = agg(["sweep", str(eps), f"recon_{label}", "ssim"])
            summ[f"{label}_eps{eps:g}_ssim_shuffled"] = agg(["sweep", str(eps), f"recon_{label}", "ssim_shuffled"])
    results["summary"] = summ
    out = Path(args.out) if args.out else HERE / "results" / f"{args.dataset}_table2_attack_results.json"
    out.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*78}\nTable 2 ({args.dataset}, K={K}, {len(args.seeds)} seeds)  MI-AUC = posterior adversary; "
          f"[distance] = old statistic")
    print(f"{'Mechanism':<24s}" + "".join(f"{'AUC e'+format(e,'g'):>12s}" for e in EPS) + "".join(f"{'SSIM e'+format(e,'g'):>10s}" for e in EPS))
    print(f"{'No noise':<24s}" + "".join(f"{'1.000':>12s}" for _ in EPS) + "".join(f"{summ['no_noise_ssim']['mean']:>10.3f}" for _ in EPS))
    for label, name in (("uniform", "Uniform feature noise"), ("canal", "CANAL")):
        row = f"{name:<24s}"
        for e in EPS:
            row += f"{summ[f'{label}_eps{e:g}_auc_posterior']['mean']:>6.3f}[{summ[f'{label}_eps{e:g}_auc_distance']['mean']:.3f}]"
        for e in EPS:
            row += f"{summ[f'{label}_eps{e:g}_ssim']['mean']:>10.3f}"
        print(row)
    print(f"SSIM floors: dataset-mean image {summ['ssim_mean_image_floor']['mean']:.3f}; shuffled pairing "
          + ", ".join(f"{l} e{e:g} {summ[f'{l}_eps{e:g}_ssim_shuffled']['mean']:.3f}" for e in EPS for l in ('uniform', 'canal')))
    print(f"{'='*78}\nSaved: {out}")


def train_decoder(F, mask, X_all, Cb, in_ch, device, iters, bs=64):
    """Inversion decoder on clean masked features -> images (upper-bound adversary)."""
    Fm_masked = (F * mask.view(1, Cb, 1, 1).float()).to(device)
    X = X_all.to(device)
    dec = Recon(Cb, out_ch=in_ch).to(device)
    opt = torch.optim.Adam(dec.parameters(), lr=2e-3)
    g = torch.Generator().manual_seed(0)
    for _ in range(iters):
        idx = torch.randint(0, len(X), (min(bs, len(X)),), generator=g).to(device)
        opt.zero_grad()
        ((dec(Fm_masked[idx]) - X[idx]) ** 2).mean().backward()
        opt.step()
    with torch.no_grad():
        xh = dec(Fm_masked).clamp(0, 1).cpu().numpy().transpose(0, 2, 3, 1)
    return dec, ssim_mean(X_all.numpy().transpose(0, 2, 3, 1), xh, in_ch)


if __name__ == "__main__":
    main()
