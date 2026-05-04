"""
Demo: Channel-wise PATE feature aggregation and noise injection.

This script simulates one step of the CW-PATE-FD release mechanism for a
single query image. It is a teaching demo, not the production pipeline.

Pipeline:
  1. Simulate K teacher feature maps for a query x.
  2. Per-channel L2 clipping on each teacher's output.
  3. Aggregate teachers per channel (mean).
  4. Compute per-channel cross-teacher variance (stability check).
  5. Allocate per-channel noise sigma_c via utility-weighted water-filling.
  6. Add channel-wise Gaussian noise; selectively release stable channels.
  7. Report SNR, release rate, and per-channel noise.

Run:
  python demo_channel_noise.py
"""

import torch


def clip_channels(features: torch.Tensor, caps: torch.Tensor) -> torch.Tensor:
    """
    Per-channel L2 clipping.

    Each channel slice f_c (shape: H x W) is rescaled so its Frobenius norm
    is at most caps[c]. Bounds per-channel sensitivity to caps[c].

    Args:
        features: shape (K, C, H, W) — K teachers' feature maps for one query.
        caps:     shape (C,)         — per-channel cap C_c.

    Returns:
        Clipped features of the same shape.
    """
    K, C, H, W = features.shape
    norms = features.flatten(2).norm(dim=2)               # (K, C)
    scale = (caps.view(1, C) / norms.clamp(min=1e-12)).clamp(max=1.0)
    return features * scale.view(K, C, 1, 1)


def aggregate_mean(clipped: torch.Tensor) -> torch.Tensor:
    """Mean over the teacher axis. Returns shape (C, H, W)."""
    return clipped.mean(dim=0)


def cross_teacher_variance(clipped: torch.Tensor) -> torch.Tensor:
    """
    Per-channel cross-teacher variance (single scalar per channel).

    V_c = (1/K) * sum_k || F_c^k - mean_k F_c^k ||_2^2

    Returns shape (C,).
    """
    mean = clipped.mean(dim=0, keepdim=True)              # (1, C, H, W)
    diff = clipped - mean                                 # (K, C, H, W)
    return diff.flatten(2).pow(2).sum(dim=2).mean(dim=0)  # (C,)


def waterfilling_sigma(
    deltas: torch.Tensor,
    importances: torch.Tensor,
    rdp_budget: float,
) -> torch.Tensor:
    """
    Closed-form water-filling allocation.

    Solves:
        min_{sigma}  sum_c s_c * sigma_c^2
        s.t.         sum_c (delta_c^2 / sigma_c^2) = B

    Lagrangian KKT gives  sigma_c^4 = mu * delta_c^2 / s_c, hence
        sigma_c = kappa * sqrt(delta_c) / s_c^{1/4}
    with kappa = mu^{1/4}. Substituting into the constraint:
        kappa^2 = (1 / B) * sum_c delta_c * sqrt(s_c).

    Args:
        deltas:      shape (C,) — per-channel L2 sensitivity Δ_c.
        importances: shape (C,) — per-channel utility weight s_c (>0).
        rdp_budget:  scalar     — total per-query RDP budget B (sum of Δ²/σ²).

    Returns:
        sigma per channel, shape (C,).
    """
    numerator = (deltas * importances.sqrt()).sum()       # sum_c Δ_c * √s_c
    kappa = (numerator / rdp_budget).sqrt()               # kappa^2 = num / B
    return kappa * deltas.sqrt() / importances.pow(0.25)


def selective_release(
    aggregated: torch.Tensor,
    variance: torch.Tensor,
    sigma: torch.Tensor,
    tau: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Add per-channel Gaussian noise; mark channels where variance > tau as withheld.

    Args:
        aggregated: (C, H, W) — mean over teachers.
        variance:   (C,)      — cross-teacher variance.
        sigma:      (C,)      — per-channel noise std.
        tau:        (C,)      — per-channel stability threshold.

    Returns:
        released: (C, H, W) — noisy aggregated features (zeros where withheld).
        mask:     (C,)      — 1.0 where released, 0.0 where withheld.
    """
    C, H, W = aggregated.shape
    noise = torch.randn_like(aggregated) * sigma.view(C, 1, 1)
    noisy = aggregated + noise
    mask = (variance <= tau).float()                      # (C,)
    released = noisy * mask.view(C, 1, 1)
    return released, mask


def demo():
    torch.manual_seed(0)

    # ---- problem setup ----
    K = 10           # number of teachers
    C = 64           # channels (small for demo; real models have 256+)
    H, W = 14, 14    # spatial dims (e.g., last conv feature map of ResNet)
    rdp_budget = 0.5  # per-query RDP budget; in practice set by accountant

    # ---- simulate K teachers' feature maps for one query ----
    # In real code, this comes from forwarding the query through K teachers.
    # We simulate teacher disagreement by giving each teacher a small offset
    # plus a per-teacher noise. Half the channels are "agreed" (low variance),
    # half are "disputed" (high variance) — to demonstrate selective release.
    base = torch.randn(C, H, W)
    teacher_features = base.unsqueeze(0).repeat(K, 1, 1, 1)  # all start equal
    disagreement = torch.zeros(C)
    disagreement[C // 2 :] = 1.0   # second half is "disputed"
    teacher_features += torch.randn(K, C, H, W) * disagreement.view(1, C, 1, 1) * 0.8

    # ---- per-channel calibration (in practice: precomputed from public probe) ----
    # Clipping cap: 90th percentile of per-channel norm across teachers
    norms = teacher_features.flatten(2).norm(dim=2)        # (K, C)
    caps = norms.quantile(0.9, dim=0)                      # (C,)

    # Importance s_c: in practice, derivative magnitude of task loss w.r.t. each
    # channel on a public probe set. Here we synthesize: first half "important",
    # second half less so.
    importances = torch.ones(C)
    importances[: C // 2] = 4.0    # important channels
    importances[C // 2 :] = 1.0    # less important channels

    # Stability threshold tau: precomputed on public probe data; one value here
    # for simplicity, set so that the "agreed" channels pass and "disputed" don't.
    tau = torch.full((C,), 0.5 * caps.median().pow(2).item())

    # ---- step 1: per-teacher per-channel clipping ----
    clipped = clip_channels(teacher_features, caps)

    # ---- step 2: aggregate ----
    aggregated = aggregate_mean(clipped)                   # (C, H, W)

    # ---- step 3: stability test (per-channel cross-teacher variance) ----
    variance = cross_teacher_variance(clipped)             # (C,)

    # ---- step 4: per-channel sensitivity and noise allocation ----
    # For mean aggregation under add/remove-one neighbor: Δ_c = 2 * C_c / K
    deltas = 2.0 * caps / K                                # (C,)
    sigma = waterfilling_sigma(deltas, importances, rdp_budget)

    # ---- step 5: add noise and selectively release ----
    released, release_mask = selective_release(aggregated, variance, sigma, tau)

    # ---- diagnostics ----
    signal_per_channel = aggregated.flatten(1).norm(dim=1) / (H * W) ** 0.5
    snr_per_channel = signal_per_channel / sigma.clamp(min=1e-12)

    print(f"Number of teachers K       : {K}")
    print(f"Number of channels C       : {C}")
    print(f"Spatial size               : {H}x{W}")
    print(f"Total per-query RDP budget : {rdp_budget}")
    print()
    print(f"Per-channel cap C_c        : mean={caps.mean():.3f}  median={caps.median():.3f}")
    print(f"Per-channel sensitivity Δ_c: mean={deltas.mean():.3f}")
    print(f"Per-channel noise sigma_c  : mean={sigma.mean():.3f}  range=[{sigma.min():.3f}, {sigma.max():.3f}]")
    print()
    print(f"Important-channel sigma    : mean={sigma[: C // 2].mean():.3f}")
    print(f"Less-important-channel sig : mean={sigma[C // 2 :].mean():.3f}")
    print(f"  (important channels get less noise — water-filling working)")
    print()
    print(f"Mean SNR (signal / sigma)  : {snr_per_channel.mean():.3f}")
    print(f"Released channels          : {int(release_mask.sum().item())} / {C}")
    print(f"  (low-variance 'agreed' channels passed; disputed ones withheld)")
    print()
    print(f"Released tensor shape      : {tuple(released.shape)}")
    print(f"Withheld channels are zero : "
          f"{(released[release_mask == 0].abs().sum().item() == 0.0)}")

    # ---- sanity check: budget constraint should be satisfied ----
    used_budget = (deltas.pow(2) / sigma.pow(2)).sum().item()
    print()
    print(f"Budget check: used = {used_budget:.4f}, target B = {rdp_budget:.4f}")
    assert abs(used_budget - rdp_budget) < 1e-3, \
        "Water-filling did not saturate the constraint — math bug."
    print("Budget constraint saturated correctly.")


if __name__ == "__main__":
    demo()
