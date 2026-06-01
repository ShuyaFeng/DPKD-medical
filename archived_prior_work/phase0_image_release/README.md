# Phase 0 — Archived prior work (image-level DP release)

**Status:** Archived. NOT part of the current WACV submission.
**Date archived:** 2026-05-26
**Reason:** Different threat model from the algorithm we now publish.

---

## What this directory contains

Eleven files implementing **closed-form Bayes-accuracy analysis of
structure-aware differential privacy for direct image release**. They
predate the pivot to feature-level distillation and analyze a
fundamentally different mechanism.

| File | Role |
|------|------|
| `phase0_validation.py` | Synthetic 4-channel generative model (T1ce / FLAIR / T1 / T2 mock). Closed-form Bayes accuracy under joint channel-x-spatial water-filling. |
| `phase0_drive.py` | Real retinal-photograph variant using scikit-image's sample fundus + Frangi-derived vessel mask (superseded by `phase0_drive_real.py`). |
| `phase0_drive_real.py` | Closed-form analysis on the actual DRIVE dataset (40 RGB fundus images, expert vessel masks). |
| `phase0_robustness.py` | Sensitivity of joint-WF lift to corrupted importance masks (multiplicative noise, spatial shift, channel permutation). |
| `phase0_*.png` | Figures produced by the above scripts. |

---

## Why this is NOT used in the WACV paper

The WACV paper is about **per-channel noise allocation for the
bottleneck of a feature-distillation pipeline**. Phase 0 is about
**per-pixel x per-channel noise allocation on raw images**. The two
differ in essentially every methodologically relevant way:

|  | Phase 0 (this directory) | WACV paper |
|---|---|---|
| **What gets noised** | Raw image `x` in `R^(C x H x W)` | U-Net bottleneck features `z = phi_T(x)` in `R^(1024 x 37 x 36)` |
| **"Channel" means** | RGB component or imaging modality (`C` = 3–4) | Encoder output channel (`C` = 1024) |
| **Noise granularity** | Per-pixel x per-channel `sigma(c,i,j)` | Per-channel `sigma_c` (shared across spatial dims) |
| **Threat model** | DP image release: site publishes a noisy image | DP feature distillation: site publishes noisy bottleneck features, student trains on them |
| **Downstream utility** | Bayes accuracy of post-release classifier (closed form) | Student segmentation mDice (empirical) |

Because the threat model is different, **the numerical results in Phase 0
do not validate, motivate, or upper-bound the channel-WF allocation in
the WACV paper.** Treating them as a closed-form analog would be
methodologically incorrect — a reviewer would catch this immediately.

In particular, the joint channel-x-spatial mechanism analyzed here
collapses to roughly uniform when one of the two axes carries no
information (see Test 5 in `phase0_robustness.py` output, where joint-WF
gives +0.00 pp lift when both `w_c` and `w_ij` are flat). The
WACV paper's algorithm is the per-channel projection of this richer
mechanism; its theoretical analysis lives in Theorem 1 of the WACV
paper's `section_3_privacy.tex`, not here.

---

## Why we keep it

1. **Methodology is reusable.** The closed-form Bayes-accuracy
   framework and the noisy-importance robustness protocol generalize
   to other DP release mechanisms.
2. **Possible future paper.** A standalone paper on DP medical-image
   release (rather than feature distillation) would build directly on
   this code.
3. **Audit trail.** Documents the original line of inquiry and the
   point at which the pivot to feature distillation happened.

---

## Reproducing Phase 0 results (for future reference)

```bash
# from repo root
python3 archived_prior_work/phase0_image_release/phase0_validation.py
python3 archived_prior_work/phase0_image_release/phase0_robustness.py
# phase0_drive_real.py needs the actual DRIVE images (40 RGB TIFFs)
```

Dependencies: `numpy`, `scipy`, `matplotlib`, `scikit-image`, `PIL`.

---

## Related archived work

The empirical companion to Phase 0 — a small U-Net trained on the
same noisy images — lives in
`../phase1_image_release_empirical/`. It was originally called
"Phase 1" but has nothing to do with the current
`phase1_distillation/` (feature distillation) effort that the WACV
paper is built on.

---

## What the WACV paper uses instead

| Goal | Phase 0 file (archived) | WACV paper's replacement |
|------|-------------------------|--------------------------|
| Closed-form optimality of WF allocation | `phase0_validation.py` derivation | **Theorem 1** in `paper_draft/section_3_privacy.tex` (channel-only, no spatial) |
| Sensitivity to noisy importance | `phase0_robustness.py` | **Section 3.4 + Table 2** in `paper_draft/section_3_privacy.tex` and `phase1_distillation/scripts/budget_split_analysis.py` |
| Empirical validation | (none — Phase 0 was analytical only) | `phase1_distillation/` distillation experiments |
