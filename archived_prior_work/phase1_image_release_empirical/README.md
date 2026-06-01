# Phase 1 (old) — Archived prior work (image-release empirical validation)

**Status:** Archived. NOT part of the current WACV submission.
**Date archived:** 2026-05-26
**Reason:** Same threat model as Phase 0 (image-level DP release). The
"Phase 1" name now refers to the feature-distillation effort in
`phase1_distillation/` instead.

---

## What this directory contains

The empirical companion to the closed-form analysis in
`archived_prior_work/phase0_image_release/`. These scripts trained a
small U-Net on DRIVE images that had been noised by the same
mechanisms that Phase 0 analyzed in closed form.

| File | Role |
|------|------|
| `phase1_unet.py` | Trains a small U-Net on (noisy image, mask) pairs. Supports six mechanisms: `no-dp`, `uniform`, `channel-WF`, `spatial-WF`, `joint-WF`, `joint-WF+thr`. Mechanism implementations match `phase0_validation.py`. |
| `phase1_analyze.py` | Aggregates and plots the JSON sweep output by mechanism / noise multiplier. |
| `phase1_results_sweep.json` | 63 training-run records produced by `phase1_unet.py`. Each record carries `mechanism`, `eps`, `noise_multiplier`, `seed`, `best_val_dice`, `final_val_dice`, full validation history. |
| `phase1_dice.png` | The summary plot produced by `phase1_analyze.py`. |

---

## Why this is NOT used in the WACV paper

The same reason Phase 0 was archived: this work releases **noisy
images** and trains a network on them. The WACV paper releases
**noisy U-Net bottleneck features** and uses them as distillation
targets for a student. Different mechanism, different threat model,
different downstream pipeline — the empirical numbers in
`phase1_results_sweep.json` do not transfer.

See `../phase0_image_release/README.md` for the full comparison table
of what is different between the image-release track (this) and the
feature-distillation track (the WACV paper, in
`phase1_distillation/`).

---

## What did get carried forward

A few things from this directory survived the pivot into the current
`phase1_distillation/` codebase:

- **Per-channel L2 clipping + normalisation** of the released vector
  before adding Gaussian noise — same idea, different vector
  (bottleneck features instead of pixel channels).
- **zCDP <-> (eps, delta)-DP conversion** via Bun-Steinke — kept
  verbatim and is now in
  `phase1_distillation/scripts/privacy_accounting.py`.
- **Channel water-filling closed form** — extracted into Theorem 1
  of the WACV paper (`paper_draft/section_3_body.tex`).

---

## Reproducing for future reference

```bash
# from repo root
python3 archived_prior_work/phase1_image_release_empirical/phase1_unet.py \
    --mechanism joint-WF --eps 2 --seed 0
python3 archived_prior_work/phase1_image_release_empirical/phase1_analyze.py \
    archived_prior_work/phase1_image_release_empirical/phase1_results_sweep.json
```

Dependencies: `numpy`, `scipy`, `matplotlib`, `Pillow`, `scikit-image`,
`torch`. Needs the full DRIVE dataset (40 RGB TIFFs) on disk.

---

## Cross-reference

| Track | Directory | Status |
|-------|-----------|--------|
| Image release — closed-form analysis | `archived_prior_work/phase0_image_release/` | archived |
| Image release — empirical U-Net | `archived_prior_work/phase1_image_release_empirical/` (this dir) | archived |
| **Feature distillation — current direction** | **`phase1_distillation/`** | **active (WACV)** |
