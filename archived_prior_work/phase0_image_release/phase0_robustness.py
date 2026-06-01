"""
Phase-0 robustness check: how does joint-WF utility degrade as the mask
(public-proxy saliency) becomes inaccurate?

In the main validation we fed in oracle utility weights w_c, w_ij = ground-truth
mu^2. In practice the mask comes from a public-proxy model and will be biased
and noisy. If a small amount of mask noise wipes out the win over uniform,
the method is fragile.

We test three corruptions:
  1. Multiplicative noise on w_ij:   w_ij' = w_ij * exp(eta * z),  z ~ N(0,1)
  2. Spatial misregistration:        shift the diagnostic-region prior by k pixels
  3. Channel misranking:             permute the channel utility weights

For each corruption we measure the lift of joint-WF over uniform at eps=2,
and compare to the oracle-mask win (+6.5 pp) measured in phase0_validation.py.
"""

import numpy as np
from scipy.stats import norm

from phase0_validation import (
    build_generative_model,
    eps_to_rho,
    bayes_acc,
    mech_uniform,
    mech_joint_wf,
    mech_joint_wf_threshold,
)


def corrupt_multiplicative(w_ij, eta, rng):
    """Multiplicative log-normal noise."""
    z = rng.standard_normal(w_ij.shape)
    return w_ij * np.exp(eta * z)


def corrupt_shift(w_ij, dy, dx):
    """Shift the saliency map by (dy, dx) pixels (mask misregistered)."""
    return np.roll(w_ij, shift=(dy, dx), axis=(0, 1))


def corrupt_permute_channels(w_c, perm):
    return w_c[perm]


def _corrupted_mu(mu, w_c_used, w_c_true, w_ij_used, w_ij_true):
    """Simulate the practical setting where the WF and threshold both use the
    corrupted mask. We rebuild a 'mu_proxy' by tensor-decomposing
    w_c x w_ij — this is what the mechanism would actually see when its only
    information is the corrupted public proxy.
    """
    # Reconstruct an mu-proxy that matches the corrupted weights.
    # mu_proxy_{c,ij}^2 ~ (w_c / sum w_c) * (w_ij / sum w_ij) * (sum_c sum_ij mu_true^2)
    total_signal = float((mu ** 2).sum())
    norm_c = w_c_used / max(w_c_used.sum(), 1e-12)
    norm_ij = w_ij_used / max(w_ij_used.sum(), 1e-12)
    mu_proxy_sq = total_signal * norm_c[:, None, None] * norm_ij[None, :, :]
    mu_proxy = np.sqrt(np.maximum(mu_proxy_sq, 0.0))
    return mu_proxy


def lift_at_eps2(w_c_used, w_ij_used, mu, Delta, beta, *, mechanism="joint-WF",
                 w_c_true=None, w_ij_true=None):
    """Lift over uniform at eps=2 for either joint-WF or joint-WF+thr.

    Crucially, the mechanism only sees the CORRUPTED weights. The Bayes
    accuracy is then evaluated against the TRUE mu (so we measure real
    downstream utility, not utility under the wrong model).
    """
    rho = eps_to_rho(2.0, 1e-5)
    s2_uni = mech_uniform(rho, mu, Delta, beta)

    if mechanism == "joint-WF":
        s2_mech = mech_joint_wf(rho, mu, Delta, beta, w_c_used, w_ij_used)
    elif mechanism == "joint-WF+thr":
        # The threshold mechanism uses |mu|/Delta to pick the active set, so
        # under mask corruption we must feed it a 'mu_proxy' built from the
        # corrupted weights — otherwise we'd be cheating with oracle mu.
        mu_proxy = _corrupted_mu(mu, w_c_used, w_c_true, w_ij_used, w_ij_true)
        s2_mech = mech_joint_wf_threshold(rho, mu_proxy, Delta, beta,
                                          w_c_used, w_ij_used)
    else:
        raise ValueError(mechanism)

    return bayes_acc(mu, s2_mech, beta) - bayes_acc(mu, s2_uni, beta)


def main():
    mu, Delta, beta, w_c_true, w_ij_true = build_generative_model()

    def lift(w_c, w_ij, mech):
        return lift_at_eps2(w_c, w_ij, mu, Delta, beta, mechanism=mech,
                            w_c_true=w_c_true, w_ij_true=w_ij_true)

    rng = np.random.default_rng(42)
    mechs = ["joint-WF", "joint-WF+thr"]

    # Oracle-mask lifts
    print("Oracle-mask lift over uniform at eps=2 (pp):")
    for m in mechs:
        l = 100 * lift(w_c_true, w_ij_true, m)
        print(f"  {m:<14s} +{l:.2f}")
    print()

    # 1. Multiplicative noise on spatial mask
    print("(1) Multiplicative log-normal noise on w_ij  (mean lift in pp over 50 seeds)")
    print(f"{'eta':>6s}  " + "  ".join(f"{m:<14s}" for m in mechs))
    for eta in [0.0, 0.25, 0.5, 1.0, 2.0]:
        row = f"{eta:>6.2f}  "
        for m in mechs:
            lifts_arr = []
            for _ in range(50):
                w_ij_c = corrupt_multiplicative(w_ij_true, eta, rng)
                lifts_arr.append(lift(w_c_true, w_ij_c, m))
            row += f"{100*np.mean(lifts_arr):>+8.2f} (min{100*np.min(lifts_arr):>+5.1f})  "
        print(row)
    print()

    # 2. Spatial shift
    print("(2) Spatial shift of saliency map (lift in pp)")
    print(f"{'shift':>6s}  " + "  ".join(f"{m:<14s}" for m in mechs))
    for dx in [0, 1, 2, 3, 4, 6]:
        w_ij_shift = corrupt_shift(w_ij_true, dx, dx)
        row = f"{dx:>6d}  "
        for m in mechs:
            row += f"{100*lift(w_c_true, w_ij_shift, m):>+8.2f}        "
        print(row)
    print()

    # 3. Channel misranking
    print("(3) Channel utility permutations (lift in pp)")
    perms = [
        ("identity",        np.array([0, 1, 2, 3])),
        ("swap top-2",      np.array([0, 2, 1, 3])),
        ("dead chan first", np.array([3, 1, 2, 0])),
        ("full reverse",    np.array([3, 2, 1, 0])),
    ]
    print(f"  {'permutation':<18s}  " + "  ".join(f"{m:<14s}" for m in mechs))
    for name, perm in perms:
        w_c_c = corrupt_permute_channels(w_c_true, perm)
        row = f"  {name:<18s}  "
        for m in mechs:
            row += f"{100*lift(w_c_c, w_ij_true, m):>+8.2f}        "
        print(row)
    print()

    # 4. Combined corruption
    print("(4) Combined: shift=2px AND swap top-2 channels")
    w_ij_c = corrupt_shift(w_ij_true, 2, 2)
    w_c_c = corrupt_permute_channels(w_c_true, np.array([0, 2, 1, 3]))
    for m in mechs:
        l = 100 * lift(w_c_c, w_ij_c, m)
        print(f"  {m:<14s} +{l:.2f} pp")
    print()

    # 5. Degenerate mask
    print("(5) Degenerate mask: uniform w_c and w_ij (no information)")
    w_ij_flat = np.ones_like(w_ij_true)
    w_c_flat = np.ones_like(w_c_true)
    for m in mechs:
        l = 100 * lift(w_c_flat, w_ij_flat, m)
        print(f"  {m:<14s} +{l:.2f} pp")


if __name__ == "__main__":
    main()
