"""
Phase-0 utility validation for the proposed structure-aware DP image-release
mechanism (channel x spatial water-filling).

We use a controlled synthetic generative model so that the Bayes-optimal
classifier has a closed form, and the *Bayes accuracy* under any noise
allocation can be computed analytically. This removes training noise and
isolates the question we actually care about for Phase 0:

    "At a matched per-image rho-zCDP budget, does structure-aware allocation
     give materially higher downstream Bayes accuracy than uniform Gaussian?"

If structure-aware does not beat uniform here, real medical data is hopeless.
If it does, we have green light to spend weeks on dataset access + training.

Generative model
----------------
Multi-channel image x in R^{C x H x W}, label y in {0, 1} with prior 1/2.
For each (c, i, j):
    x_{c,i,j} | y = +1  ~  N( +mu_{c,i,j}, beta^2 )
    x_{c,i,j} | y = -1  ~  N( -mu_{c,i,j}, beta^2 )
where mu_{c,i,j} = s_c * a_{i,j}, with
    s_c            per-channel signal strength (mocks T1ce >> FLAIR > T1 >> T2)
    a_{i,j}        per-pixel signal strength (Gaussian bump inside diagnostic ROI)

Mechanism: tilde_x = x + N(0, sigma_{c,i,j}^2 I).

Bayes accuracy under released noise (closed form):
    V       = sum_{c,i,j} mu_{c,i,j}^2 / ( beta^2 + sigma_{c,i,j}^2 )
    Acc(V)  = Phi( sqrt(V) )

Privacy: per-element zCDP cost rho_{c,i,j} = Delta_{c,i,j}^2 / (2 sigma_{c,i,j}^2).
Total budget: sum_{c,i,j} rho_{c,i,j} <= rho.
"""

import numpy as np
from scipy.stats import norm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Generative model
# -----------------------------------------------------------------------------

def build_generative_model(C=4, H=16, W=16, beta=1.0, seed=0):
    """Return (mu, Delta, beta, w_c, w_ij) for the synthetic problem."""
    # Diagnostic ROI: 6x6 square in the center
    cy, cx = H // 2, W // 2
    roi = np.zeros((H, W))
    roi[cy - 3 : cy + 3, cx - 3 : cx + 3] = 1.0

    # Spatial bump inside ROI (Gaussian falloff)
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    a = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 8.0) * roi  # (H, W)

    # Per-channel signal strength: T1=0.3, T1ce=1.0, FLAIR=0.7, T2=0.0
    s = np.array([0.3, 1.0, 0.7, 0.0])
    assert s.shape[0] == C

    mu = s[:, None, None] * a[None, :, :]                       # (C, H, W)

    # Per-element sensitivity: assume per-pixel L2 clipping with cap 1.0.
    Delta = np.ones((C, H, W))

    # Utility weights used by the plan's water-filling allocations.
    # Plan §3.5: w_c is a public-proxy estimate of channel utility.
    # Plan §3.4: w_ij comes from a public-data saliency map.
    # Here we feed in the ground-truth mu^2 — i.e. an *oracle* utility
    # estimate. Real masks will be noisier; this is the best case for the
    # plan's WF, and a fair head-to-head against the Bayes-optimal.
    w_c = (mu ** 2).sum(axis=(1, 2))            # (C,)
    w_ij = (mu ** 2).sum(axis=0)                # (H, W)

    return mu, Delta, beta, w_c, w_ij


# -----------------------------------------------------------------------------
# Privacy accounting
# -----------------------------------------------------------------------------

def eps_to_rho(eps, delta):
    """Bun-Steinke zCDP -> (eps, delta)-DP conversion, inverted.

    eps = rho + 2*sqrt(rho * log(1/delta))
    """
    log_inv_d = np.log(1.0 / delta)
    a = 1.0
    b = 2.0 * np.sqrt(log_inv_d)
    c = -eps
    r = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
    return r * r


def rho_used(Delta, sigma2):
    return float(np.sum(Delta ** 2 / (2.0 * sigma2)))


def bayes_acc(mu, sigma2, beta):
    """Closed-form Bayes accuracy under released-image LDA."""
    V = np.sum(mu ** 2 / (beta ** 2 + sigma2))
    return float(norm.cdf(np.sqrt(V)))


# -----------------------------------------------------------------------------
# Mechanisms
# -----------------------------------------------------------------------------

def _renormalize_to_rho(sigma2, Delta, rho):
    """Scale sigma2 so the constraint is exactly saturated."""
    used = rho_used(Delta, sigma2)
    return sigma2 * (used / rho)


def mech_uniform(rho, mu, Delta, beta):
    """sigma_{cij}^2 constant for all (c,i,j)."""
    sigma2 = np.full(Delta.shape, np.sum(Delta ** 2) / (2.0 * rho))
    return sigma2


def mech_channel_wf(rho, mu, Delta, beta, w_c):
    """Plan §3.1: sigma_c^2 ∝ Delta_c / sqrt(w_c), constant across pixels."""
    Delta_c = Delta.mean(axis=(1, 2))                       # (C,)
    eps_w = 1e-6
    profile = Delta_c / np.sqrt(w_c + eps_w)                # (C,)
    sigma2 = np.broadcast_to(profile[:, None, None], Delta.shape).copy()
    return _renormalize_to_rho(sigma2, Delta, rho)


def mech_spatial_wf(rho, mu, Delta, beta, w_ij):
    """Plan §3.2: sigma_ij^2 ∝ Delta_ij / sqrt(w_ij), constant across channels."""
    Delta_ij = Delta.mean(axis=0)                           # (H, W)
    eps_w = 1e-6
    profile = Delta_ij / np.sqrt(w_ij + eps_w)              # (H, W)
    sigma2 = np.broadcast_to(profile[None, :, :], Delta.shape).copy()
    return _renormalize_to_rho(sigma2, Delta, rho)


def mech_joint_wf(rho, mu, Delta, beta, w_c, w_ij):
    """Plan §3.3 factorized form: sigma_{c,ij}^2 = sigma_c^2 * pi(m_{ij})."""
    Delta_c = Delta.mean(axis=(1, 2))
    Delta_ij = Delta.mean(axis=0)
    eps_w = 1e-6
    profile_c = Delta_c / np.sqrt(w_c + eps_w)
    profile_ij = Delta_ij / np.sqrt(w_ij + eps_w)
    sigma2 = profile_c[:, None, None] * profile_ij[None, :, :]
    return _renormalize_to_rho(sigma2, Delta, rho)


def mech_bayes_optimal(rho, mu, Delta, beta):
    """Numerical solution to:
        max_{sigma2 >= 0}  sum mu^2 / (beta^2 + sigma2)
        s.t.  sum Delta^2 / (2 sigma2) <= rho

    KKT: |mu| / (beta^2 + sigma2) = sqrt(lambda) * Delta / sigma2.
    Substituting v = sigma2:
        v |mu| = sqrt(lambda) Delta (beta^2 + v)
        v ( |mu| - sqrt(lambda) Delta ) = sqrt(lambda) Delta beta^2
    Active set: |mu| > sqrt(lambda) Delta. Inactive elements get sigma -> infinity
    (i.e. the mechanism declines to release them — consistent with releasing
    the diagnostic content and not bothering with pure-background pixels).
    """
    mu_abs = np.abs(mu)
    big = 1e18

    def total_rho(sqrt_lam):
        active = mu_abs > sqrt_lam * Delta
        denom = np.maximum(mu_abs - sqrt_lam * Delta, 1e-15)
        v = np.where(active, sqrt_lam * Delta * beta ** 2 / denom, big)
        return float(np.sum(np.where(active, Delta ** 2 / (2.0 * v), 0.0)))

    # total_rho is monotonically decreasing in sqrt_lam.
    lo, hi = 1e-15, 1e6
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        if total_rho(mid) > rho:
            lo = mid
        else:
            hi = mid
    sqrt_lam = 0.5 * (lo + hi)
    active = mu_abs > sqrt_lam * Delta
    denom = np.maximum(mu_abs - sqrt_lam * Delta, 1e-15)
    sigma2 = np.where(active, sqrt_lam * Delta * beta ** 2 / denom, big)
    return sigma2


def mech_joint_wf_threshold(rho, mu, Delta, beta, w_c, w_ij, tau=None):
    """Joint factorized WF restricted to an active set.

    Defines an active set A = { (c,i,j) : |mu_{c,i,j}| / Delta_{c,i,j} > tau }.
    Inactive elements are dropped (sigma -> infinity, not released). On A we
    run the plan's factorized WF with utility weights restricted to A:

        w_c'   = sum over (i,j) such that (c,i,j) in A of mu^2_{cij}
        w_ij'  = sum over c     such that (c,i,j) in A of mu^2_{cij}
        sigma_{c,ij}^2 ∝ (Delta_c'/sqrt(w_c')) * (Delta_ij'/sqrt(w_ij'))   on A
                       = inf                                              off A

    If tau is None, sweep over a grid and pick the value that maximizes
    Bayes accuracy at this rho.
    """
    big = 1e18
    Delta_c = Delta.mean(axis=(1, 2))
    Delta_ij = Delta.mean(axis=0)

    def acc_for_tau(t):
        active = (np.abs(mu) / np.maximum(Delta, 1e-12)) > t
        if active.sum() == 0:
            return 0.5, np.full_like(Delta, big)
        # Restricted utility weights computed on the active set
        mu2 = mu ** 2
        w_c_a = (mu2 * active).sum(axis=(1, 2))
        w_ij_a = (mu2 * active).sum(axis=0)
        eps_w = 1e-9
        profile_c = Delta_c / np.sqrt(w_c_a + eps_w)
        profile_ij = Delta_ij / np.sqrt(w_ij_a + eps_w)
        sigma2_dense = profile_c[:, None, None] * profile_ij[None, :, :]
        sigma2 = np.where(active, sigma2_dense, big)
        used = float(np.sum((Delta ** 2 * active) / (2.0 * np.where(active, sigma2_dense, 1.0))))
        if used <= 0:
            return 0.5, sigma2
        sigma2 = np.where(active, sigma2_dense * (used / rho), big)
        return bayes_acc(mu, sigma2, beta), sigma2

    if tau is not None:
        return acc_for_tau(tau)[1]

    # Sweep tau over a grid spanning [0, max(|mu|/Delta)]
    ratios = (np.abs(mu) / np.maximum(Delta, 1e-12)).ravel()
    grid = np.unique(np.concatenate([[0.0], np.quantile(ratios, np.linspace(0.0, 0.99, 50))]))
    best_acc, best_sigma2 = -1.0, None
    for t in grid:
        acc, s2 = acc_for_tau(t)
        if acc > best_acc:
            best_acc, best_sigma2 = acc, s2
    return best_sigma2


def mech_adversarial(rho, mu, Delta, beta, w_c, w_ij):
    """Inverse of joint-WF: put HIGH noise on diagnostic, LOW on background.
    A *lower* bound on what a misallocated mechanism can do — makes the
    structure-aware win look less like a numerical fluke.
    """
    Delta_c = Delta.mean(axis=(1, 2))
    Delta_ij = Delta.mean(axis=0)
    eps_w = 1e-6
    profile_c = Delta_c * np.sqrt(w_c + eps_w)
    profile_ij = Delta_ij * np.sqrt(w_ij + eps_w)
    sigma2 = profile_c[:, None, None] * profile_ij[None, :, :] + 1e-6
    return _renormalize_to_rho(sigma2, Delta, rho)


# -----------------------------------------------------------------------------
# Sweep
# -----------------------------------------------------------------------------

def main():
    delta_dp = 1e-5
    # Dense sweep for the plot, log-spaced from 0.5 to 10 at delta=1e-5.
    eps_dense = np.geomspace(0.5, 10.0, 25)
    # Canonical reporting values for the table.
    eps_table = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 10.0])
    # Combine for one sweep so we don't double-compute mechanisms.
    epsilons = np.unique(np.round(np.concatenate([eps_dense, eps_table]), 6))
    rhos = [eps_to_rho(e, delta_dp) for e in epsilons]

    mu, Delta, beta, w_c, w_ij = build_generative_model()
    C, H, W = mu.shape
    N = C * H * W

    # Sanity: no-DP Bayes accuracy
    no_dp_acc = bayes_acc(mu, np.zeros_like(mu), beta)
    print(f"Synthetic problem: C={C}, H={H}, W={W}, N={N}, beta={beta}")
    print(f"Channel signal strengths s = {[0.3, 1.0, 0.7, 0.0]}")
    print(f"Diagnostic ROI: 6x6 center bump")
    print(f"No-DP Bayes accuracy (upper bound): {no_dp_acc:.4f}")
    print()
    print(f"Per-channel utility w_c = {w_c.round(3)}")
    print(f"Spatial utility w_ij sum-in-ROI / sum-out-of-ROI: "
          f"{w_ij[5:11, 5:11].sum():.2f} / {w_ij.sum() - w_ij[5:11, 5:11].sum():.2f}")
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

    # Compute Bayes accuracy for every (mechanism, epsilon) pair once.
    results = {}
    for name, fn in mechanisms:
        accs = []
        for rho in rhos:
            sigma2 = fn(rho)
            accs.append(bayes_acc(mu, sigma2, beta))
        results[name] = np.array(accs)

    # Look up the canonical eps values inside the dense sweep.
    table_idx = [int(np.argmin(np.abs(epsilons - e))) for e in eps_table]

    header = f"{'mechanism':<15s}" + "".join(f"  eps={e:>4.1f}" for e in eps_table)
    print(header)
    print("-" * len(header))
    for name, _ in mechanisms:
        row = f"{name:<15s}" + "".join(f"  {results[name][i]:>8.4f}" for i in table_idx)
        print(row)

    # Lift table: how much does each mechanism beat uniform?
    print()
    print("Lift over uniform (percentage points):")
    print(header)
    print("-" * len(header))
    for name, _ in mechanisms:
        if name == "uniform":
            continue
        lifts = [
            100 * (results[name][i] - results["uniform"][i]) for i in table_idx
        ]
        row = f"{name:<15s}" + "".join(f"  {l:>+8.2f}" for l in lifts)
        print(row)

    # Two plots: (1) canonical 5-point sparse, (2) dense [0.5, 10].
    styles = {
        "uniform":       dict(color="gray",     linestyle="--", marker="o", markersize=4),
        "channel-WF":    dict(color="tab:blue",                marker="s",  markersize=4),
        "spatial-WF":    dict(color="tab:green",               marker="^",  markersize=4),
        "joint-WF":      dict(color="tab:red",                 marker="D",  markersize=4, linewidth=2.0),
        "joint-WF+thr":  dict(color="tab:orange",              marker="P",  markersize=5, linewidth=2.5),
        "Bayes-optimal": dict(color="black",    linestyle=":",  marker="*", markersize=5),
        "adversarial":   dict(color="tab:purple", linestyle="-.", marker="x", markersize=4),
    }

    # ---- (1) sparse canonical [0.5, 1, 2, 4, 8] ----
    eps_sparse = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
    sparse_idx = [int(np.argmin(np.abs(epsilons - e))) for e in eps_sparse]
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, _ in mechanisms:
        ax.plot(eps_sparse, results[name][sparse_idx], label=name, **styles[name])
    ax.axhline(no_dp_acc, color="green", linestyle=":", alpha=0.5,
               label=f"no-DP ({no_dp_acc:.3f})")
    ax.axhline(0.5, color="red", linestyle=":", alpha=0.3, label="chance")
    ax.set_xscale("log")
    ax.set_xticks(eps_sparse)
    ax.set_xticklabels([f"{e:g}" for e in eps_sparse])
    ax.set_xlabel(r"DP $\varepsilon$ (at $\delta=10^{-5}$)")
    ax.set_ylabel("Bayes accuracy (closed-form)")
    ax.set_title("Phase-0 utility validation:\n"
                 "structure-aware allocation vs uniform on synthetic 4-channel images")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("phase0_validation_canonical.png", dpi=150)
    plt.close(fig)
    print("\nSaved canonical-points plot to phase0_validation_canonical.png")

    # ---- (2) dense sweep eps in [0.5, 10] ----
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for name, _ in mechanisms:
        ax.plot(epsilons, results[name], label=name, **styles[name])
    ax.axhline(no_dp_acc, color="green", linestyle=":", alpha=0.5,
               label=f"no-DP ({no_dp_acc:.3f})")
    ax.axhline(0.5, color="red", linestyle=":", alpha=0.3, label="chance")
    ax.set_xscale("log")
    ax.set_xticks(eps_table)
    ax.set_xticklabels([f"{e:g}" for e in eps_table])
    ax.set_xlim(0.45, 11.0)
    ax.set_xlabel(r"DP $\varepsilon$ (at $\delta=10^{-5}$)")
    ax.set_ylabel("Bayes accuracy (closed-form)")
    ax.set_title("Phase-0 utility validation: privacy-utility curve\n"
                 r"structure-aware DP image release on synthetic 4-channel images, $\varepsilon \in [0.5, 10]$")
    ax.legend(loc="lower right", fontsize=9, ncol=1)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig("phase0_validation.png", dpi=150)
    plt.close(fig)
    print("Saved dense-sweep plot to phase0_validation.png")

    # Save the noise allocations at eps=2 for inspection
    rho_target = eps_to_rho(2.0, delta_dp)
    print(f"\n--- Noise allocation diagnostics at eps=2 (rho={rho_target:.4f}) ---")
    roi_mask = np.zeros((H, W), dtype=bool)
    roi_mask[5:11, 5:11] = True
    for name, fn in mechanisms:
        sigma2 = fn(rho_target)
        # Treat sigma^2 > 1e10 as "suppressed" — averaging includes infs as
        # noise, which would obscure the diagnostic. Report active-set stats.
        active = sigma2 < 1e10
        n_active = int(active.sum())
        roi_active = active & roi_mask[None, :, :]
        bg_active = active & ~roi_mask[None, :, :]
        active_mean = float(sigma2[active].mean()) if n_active else float("nan")
        roi_mean = float(sigma2[roi_active].mean()) if roi_active.any() else float("nan")
        bg_mean = float(sigma2[bg_active].mean()) if bg_active.any() else float("nan")
        print(f"{name:<15s} active={n_active:>4d}/{sigma2.size}   "
              f"active-mean sigma^2={active_mean:>10.2f}   "
              f"ROI-mean={roi_mean:>10.2f}   bg-mean={bg_mean:>10.2f}")


if __name__ == "__main__":
    main()
