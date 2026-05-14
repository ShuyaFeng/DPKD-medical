"""
Phase 1 - U-Net vessel segmentation on DRIVE, with and without DP image release.

This script implements the Phase 1 plan from RESEARCH_PLAN.md §8:

  1. Resize the 40 DRIVE images to 96 x 96 RGB.
  2. Apply one of {no-DP, uniform, channel-WF, spatial-WF, joint-WF, joint-WF+thr}
     to the training images, producing the released set X_tilde.
  3. Train a small U-Net on (X_tilde, y) where y is the expert mask.
  4. Evaluate Dice on the held-out val set (also DP-released under the same mechanism).
  5. Report and plot.

The mechanism implementations match phase0_validation.py (closed-form WF allocations).
Saliency mask is the expert vessel mask (oracle setting, same as Phase 0 §13.7).
"""

import argparse
import glob
import json
import os
import time

import numpy as np
from PIL import Image
from skimage import transform
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from phase0_validation import (
    eps_to_rho,
    mech_uniform,
    mech_channel_wf,
    mech_spatial_wf,
    mech_joint_wf,
    mech_joint_wf_threshold,
)


DRIVE_ROOT = "data/DRIVE"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else
                      ("cuda" if torch.cuda.is_available() else "cpu"))


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_drive_pairs(target_size=96):
    """Return list of (image_id, x[3,H,W], y[H,W]) numpy tensors."""
    pairs = []
    for split in ["train", "val"]:
        in_dir = os.path.join(DRIVE_ROOT, split, "input")
        lab_dir = os.path.join(DRIVE_ROOT, split, "label")
        for tif in sorted(glob.glob(os.path.join(in_dir, "*.tif"))):
            stem = os.path.splitext(os.path.basename(tif))[0]
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
            ).astype(np.float32) / 255.0
            lab_rs = transform.resize(
                lab.astype(float), (target_size, target_size),
                preserve_range=True, anti_aliasing=False, order=0,
            )
            x = img_rs.transpose(2, 0, 1)                                   # (3, H, W)
            y = (lab_rs > 127).astype(np.float32)                           # (H, W)
            pairs.append((f"{split}/{stem}", x, y))
    return pairs


# -----------------------------------------------------------------------------
# DP image release — applied lazily per epoch so the model sees fresh noise
# -----------------------------------------------------------------------------

_PROXY_CACHE = {}


def public_proxy_prior(pairs, target_size, n_public=10):
    """Build a strictly data-independent spatial saliency by averaging the
    first `n_public` training masks. Mimics what an external public atlas
    or proxy-trained saliency model would give us.

    The result is a single (H, W) heatmap used for EVERY released image —
    so σ² is the same function regardless of which specific image we are
    releasing → no per-image data dependency.

    To make this strict for the held-out part of the train set + val:
    only the first `n_public` masks contribute; those `n_public` images
    are EXCLUDED from U-Net training/eval downstream.
    """
    if "v" in _PROXY_CACHE:
        return _PROXY_CACHE["v"], _PROXY_CACHE["chan"]

    train_pairs = [p for p in pairs if p[0].startswith("train/")]
    pub = train_pairs[:n_public]
    masks = np.stack([p[2] for p in pub], axis=0)                            # (n_public, H, W)
    v_proxy = masks.mean(axis=0).astype(np.float64)                          # (H, W) in [0, 1]
    v_proxy = v_proxy / max(v_proxy.max(), 1e-12)
    v_proxy = v_proxy + 1e-3                                                 # avoid divide-by-zero
    chan_prior = np.array([0.4, 1.0, 0.2])                                   # green-channel-best-for-vessels textbook
    _PROXY_CACHE["v"] = v_proxy
    _PROXY_CACHE["chan"] = chan_prior
    _PROXY_CACHE["public_ids"] = set(p[0] for p in pub)
    return v_proxy, chan_prior


def data_independent_prior(pairs, target_size, n_public=10, _shape=None):
    """Returns mu_field, v_proxy, chan_prior — all data-independent w.r.t.
    the released set (the held-out part of train + val)."""
    v_proxy, chan_prior = public_proxy_prior(pairs, target_size, n_public)
    C, H, W = len(chan_prior), v_proxy.shape[0], v_proxy.shape[1]
    mu_field = chan_prior[:, None, None] * v_proxy[None, :, :]
    return mu_field, v_proxy, chan_prior


def build_sigma2_per_image(mechanism, x, y, eps_val, pairs=None, delta_dp=1e-5):
    """Return sigma^2 array of shape (3, H, W) for a single (x, y) pair.

    Saliency mask is a public-proxy prior (average of first n_public train
    masks) — fixed across all released images, data-independent of the
    specific image being released. We do NOT touch this image's y.
    """
    if mechanism == "no-dp":
        return np.zeros_like(x, dtype=np.float64)

    C, H, W = x.shape
    rho = eps_to_rho(eps_val, delta_dp)
    Delta = np.ones((C, H, W), dtype=np.float64)

    assert pairs is not None
    mu_field, _, _ = data_independent_prior(pairs, H)
    beta = 0.1

    w_c = (mu_field ** 2).sum(axis=(1, 2)) + 1e-9
    w_ij = (mu_field ** 2).sum(axis=0)

    if mechanism == "uniform":
        return mech_uniform(rho, mu_field, Delta, beta)
    if mechanism == "channel-WF":
        return mech_channel_wf(rho, mu_field, Delta, beta, w_c)
    if mechanism == "spatial-WF":
        return mech_spatial_wf(rho, mu_field, Delta, beta, w_ij)
    if mechanism == "joint-WF":
        return mech_joint_wf(rho, mu_field, Delta, beta, w_c, w_ij)
    if mechanism == "joint-WF+thr":
        return mech_joint_wf_threshold(rho, mu_field, Delta, beta, w_c, w_ij)
    raise ValueError(f"unknown mechanism: {mechanism}")


# -----------------------------------------------------------------------------
# Dataset (applies DP noise on the fly each call)
# -----------------------------------------------------------------------------

class DriveDPDataset(Dataset):
    """Adds Gaussian noise per the chosen mechanism's sigma^2 map.

    A noise multiplier reduces the noise scale uniformly. This is purely a
    knob for getting numerically meaningful results at the resolution we
    can train at — Phase 0 already documented that input-space DP at standard
    eps and full resolution buries the signal entirely. The multiplier
    rescales the noise variance by a fixed factor so that the relative
    behaviour of the mechanisms is visible at this resolution.
    """
    def __init__(self, pairs, all_pairs, mechanism, eps_val, noise_multiplier=1.0,
                 augment=False, seed=0):
        self.pairs = pairs
        self.mechanism = mechanism
        self.eps_val = eps_val
        self.noise_multiplier = noise_multiplier
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self.sigma2_cache = {}
        for name, x, y in pairs:
            self.sigma2_cache[name] = build_sigma2_per_image(
                mechanism, x, y, eps_val, pairs=all_pairs,
            )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        name, x, y = self.pairs[idx]
        sigma2 = self.sigma2_cache[name] * (self.noise_multiplier ** 2)
        sigma2 = np.clip(sigma2, 0.0, 1e12)
        noise = self.rng.standard_normal(x.shape).astype(np.float32) * np.sqrt(sigma2).astype(np.float32)
        if self.mechanism == "no-dp":
            x_tilde = x
        else:
            x_tilde = np.clip(x + noise, -3.0, 3.0)                          # let model see noise, no hard clip
        if self.augment and self.rng.random() < 0.5:
            x_tilde = x_tilde[..., ::-1].copy()
            y = y[..., ::-1].copy()
        return torch.from_numpy(x_tilde.astype(np.float32)), torch.from_numpy(y.astype(np.float32))


# -----------------------------------------------------------------------------
# Tiny U-Net
# -----------------------------------------------------------------------------

def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class TinyUNet(nn.Module):
    def __init__(self, in_ch=3, base=16):
        super().__init__()
        self.e1 = conv_block(in_ch, base)
        self.e2 = conv_block(base, base * 2)
        self.e3 = conv_block(base * 2, base * 4)
        self.bot = conv_block(base * 4, base * 8)
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.d3 = conv_block(base * 8, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d2 = conv_block(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d1 = conv_block(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.e1(x); p1 = F.max_pool2d(e1, 2)
        e2 = self.e2(p1); p2 = F.max_pool2d(e2, 2)
        e3 = self.e3(p2); p3 = F.max_pool2d(e3, 2)
        b = self.bot(p3)
        u3 = self.u3(b); d3 = self.d3(torch.cat([u3, e3], dim=1))
        u2 = self.u2(d3); d2 = self.d2(torch.cat([u2, e2], dim=1))
        u1 = self.u1(d2); d1 = self.d1(torch.cat([u1, e1], dim=1))
        return self.out(d1)


# -----------------------------------------------------------------------------
# Loss / metrics
# -----------------------------------------------------------------------------

def dice_score(logits, target, smooth=1.0):
    pred = (torch.sigmoid(logits) > 0.5).float().reshape(logits.size(0), -1)
    target_flat = target.reshape(target.size(0), -1)
    inter = (pred * target_flat).sum(dim=1)
    union = pred.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2 * inter + smooth) / (union + smooth)
    return dice.mean().item()


def dice_loss(logits, target, smooth=1.0):
    pred = torch.sigmoid(logits)
    pred_flat = pred.reshape(pred.size(0), -1)
    target_flat = target.reshape(target.size(0), -1)
    inter = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    return 1 - ((2 * inter + smooth) / (union + smooth)).mean()


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------

def train_one(pairs, mechanism, eps_val, *, epochs=80, lr=1e-3,
              batch_size=4, noise_multiplier=1.0, seed=0, verbose=False,
              n_public=10):
    """Train a single U-Net under the given mechanism / eps and return Dice on val.

    The first `n_public` training images are reserved for building the
    data-independent saliency prior — they are EXCLUDED from U-Net training
    so that the prior is strictly data-independent of the training set
    the U-Net sees.
    """
    torch.manual_seed(seed)
    # Force the proxy prior to be built once (uses first n_public)
    public_proxy_prior(pairs, target_size=None, n_public=n_public)
    public_ids = _PROXY_CACHE.get("public_ids", set())

    train_pairs = [p for p in pairs if p[0].startswith("train/") and p[0] not in public_ids]
    val_pairs = [p for p in pairs if p[0].startswith("val/")]
    train_ds = DriveDPDataset(train_pairs, pairs, mechanism, eps_val,
                              noise_multiplier=noise_multiplier, augment=True, seed=seed)
    val_ds = DriveDPDataset(val_pairs, pairs, mechanism, eps_val,
                            noise_multiplier=noise_multiplier, augment=False, seed=seed + 100)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = TinyUNet().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    best_val_dice = 0.0
    history = []
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(DEVICE); y = y.to(DEVICE)
            logits = model(x).squeeze(1)
            loss = 0.5 * bce(logits, y) + 0.5 * dice_loss(logits.unsqueeze(1), y.unsqueeze(1))
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            val_dice = np.mean([
                dice_score(model(x.to(DEVICE)).squeeze(1).unsqueeze(1), y.to(DEVICE).unsqueeze(1))
                for x, y in val_loader
            ])
        history.append(val_dice)
        if val_dice > best_val_dice:
            best_val_dice = val_dice
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"    epoch {epoch:3d}  val_dice={val_dice:.4f}  best={best_val_dice:.4f}")

    return dict(best_val_dice=best_val_dice, final_val_dice=val_dice, history=history)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_size", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--noise_multipliers", nargs="+", type=float, default=[1.0, 0.05])
    parser.add_argument("--eps", nargs="+", type=float, default=[8.0, 32.0])
    parser.add_argument("--mechanisms", nargs="+",
                        default=["no-dp", "uniform", "channel-WF", "spatial-WF", "joint-WF", "joint-WF+thr"])
    parser.add_argument("--smoke_only", action="store_true",
                        help="Train only the no-DP baseline as a smoke test.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--out", default="phase1_results.json")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Loading DRIVE at {args.target_size}x{args.target_size}...")
    pairs = load_drive_pairs(target_size=args.target_size)
    print(f"  {len(pairs)} pairs loaded "
          f"(train: {sum(1 for p in pairs if p[0].startswith('train/'))}, "
          f"val: {sum(1 for p in pairs if p[0].startswith('val/'))})")
    print()

    results = []
    t0 = time.time()

    # No-DP smoke test (per seed)
    if "no-dp" in args.mechanisms or args.smoke_only:
        for seed in args.seeds:
            print(f"[no-DP baseline, seed={seed}]")
            out = train_one(pairs, "no-dp", eps_val=1.0, epochs=args.epochs,
                            batch_size=args.batch_size, seed=seed,
                            verbose=(seed == args.seeds[0]))
            out.update(mechanism="no-dp", eps=None, noise_multiplier=0.0, seed=seed)
            results.append(out)
            print(f"  → best val Dice: {out['best_val_dice']:.4f}")
        print()

    if args.smoke_only:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved smoke-test result to {args.out}")
        return

    dp_mechs = [m for m in args.mechanisms if m != "no-dp"]
    for nm in args.noise_multipliers:
        for eps_val in args.eps:
            for mech in dp_mechs:
                for seed in args.seeds:
                    tag = f"{mech} eps={eps_val} nm={nm} seed={seed}"
                    print(f"[{tag}]")
                    out = train_one(pairs, mech, eps_val, epochs=args.epochs,
                                    batch_size=args.batch_size, seed=seed,
                                    noise_multiplier=nm, verbose=False)
                    out.update(mechanism=mech, eps=eps_val,
                               noise_multiplier=nm, seed=seed)
                    results.append(out)
                    print(f"  → best val Dice: {out['best_val_dice']:.4f}  "
                          f"(elapsed {time.time() - t0:.0f}s)")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nTotal time: {time.time() - t0:.0f}s")
    print(f"Saved full results to {args.out}")


if __name__ == "__main__":
    main()
