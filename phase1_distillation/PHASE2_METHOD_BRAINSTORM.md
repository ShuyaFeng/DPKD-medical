# Phase 2 — Brainstorm: Methods to Replace or Augment CANAL as Contribution (ii)

**Last updated:** 2026-06-02
**Owner:** Emily
**Status:** brainstorm + execution plan

---

## 0. The decision we are making

Sample-once-per-image release (paper contribution (i)) **is locked**. It is the
honest-accounting framework, validated by Table 1 (per-iteration ε = 812 vs.
sample-once ε = 2 at the same per-release noise). This is independent of which
noise-shaping mechanism we plug into it.

CANAL (paper contribution (ii)) is the closed-form channel-wise water-filling
allocation. On real DRIVE we have demonstrated, across **probe (single forward),
3-seed distillation, 5-seed paired distillation, importance-ratio sweep
R∈[1, 10⁵], sensitivity sweep K∈[1, 1000], per-channel Δ analysis**:

> **WF − uniform Dice gap is statistically indistinguishable from zero
> across every setting tested.**

The structural reason is the ¼-power compression of importance in Theorem 1:
on RGB fundus the bottleneck importance ratio is ~2×, so the σ ratio is ~1.13×,
which is below any decoder's Dice sensitivity. The CANAL closed form is still
a valid mathematical contribution (Theorem 1 + proof), but the empirical claim
"CANAL beats uniform on DRIVE" cannot be defended.

**This document brainstorms candidate replacements/additions for contribution
(ii) that DO deliver a measurable, defensible empirical lift over a clean
baseline, within the WACV timeline.**

---

## 1. Acceptance criteria for each method

To qualify as a paper contribution, a method must satisfy ALL of:

1. **Mechanism-level novelty in this setting.** Either a new DP mechanism, a
   tighter accounting, or a non-trivial application of a known mechanism to
   DP feature distillation.
2. **A clean, defensible baseline.** The comparison must be at MATCHED user-
   level ε. We cannot compare a method at ε=2 to a method at ε=812 and call
   the lift legitimate.
3. **A measurable empirical lift.** ≥ +0.005 vessel Dice over baseline on real
   DRIVE, statistically significant under 5-seed paired test (p < 0.05).
4. **Compatible with sample-once framework.** No method that requires
   per-iteration noise re-sampling.
5. **Honest privacy story.** Where ρ is paid is explicit; no privacy theatre
   (e.g. simulating K teachers without training them).

---

## 2. Tier-1 candidates — highest priority for empirical validation

These are the methods most likely to deliver lift AND tell a clean DP story.

### M1: Honest PATE multi-teacher feature aggregation
- **Mechanism**: Train K teachers on disjoint patient cohorts (k=1 each).
  For each patient image, query all K teachers' bottlenecks. Aggregate via
  robust statistic (mean or median). Add Gaussian noise to the aggregate.
- **Privacy story**: replace-one a patient → affects exactly 1 of K
  teachers → aggregate L₂ sensitivity = 2/K (under unit-norm clipping).
  σ scales as √(1/K) at fixed ρ → utility ↑ as K grows.
- **Baseline**: K=1 single-teacher sample-once release at the same per-
  release ε. (Our entire current setup IS K=1.)
- **Expected lift**: Substantial — Dice projected to climb from 0.62 (K=1)
  to 0.70-0.80 at K=5 based on σ scaling. This is the strongest paper
  story because it is a *true* sensitivity reduction.
- **Implementation effort**: 1 day local PoC (5-10 teachers on TinyUNet);
  cluster reproduction with mmseg = 1-2 days.
- **Paper home**: Reference [12, 13] for PATE; § paper.5 already mentions
  this as future work — we just deliver it.
- **Privacy comparison literature**: Papernot et al. 2017/2018 PATE; Bagdasaryan
  et al. on federated DP-PATE; PATE-G (Jordon et al.) for image generation.

### M2: Logit-level distillation (PATE-G adapted to segmentation)
- **Mechanism**: Instead of releasing noisy bottleneck features, release
  noisy soft-label predictions per pixel (or per super-pixel). Student
  trains on these as soft labels.
- **Privacy story**: Same sample-once framework, but the released object
  is the C×H×W softmax instead of the C×H_b×W_b feature.
  Sensitivity depends on softmax L2-bound. Easier to bound: ‖softmax(z) -
  softmax(z')‖₂ ≤ √2 always.
- **Baseline**: Feature-level distillation (current CANAL or uniform).
- **Expected lift**: Mixed. Logits are lower-dimensional than bottleneck
  (2 vs 1024 channels), so per-coordinate noise is smaller. But logit
  release is also less informative for the student than feature release.
- **Implementation effort**: 0.5 day. The release object changes from
  cache[i] = (1024,H,W) features to cache[i] = (2,H,W) soft labels.
  Student trains via KL divergence on soft labels (no feature loss).
- **Paper home**: This is exactly PATE-style for segmentation. Reference
  [12, 13] PATE and adapt to per-pixel granularity. The "noisy soft labels
  as supervisory signal" idea is from PATE-G (Jordon 2019).
- **Comparison gotcha**: Logit release fundamentally changes what is
  protected. Per-pixel logit ↔ per-pixel decision. Different threat model
  from feature release.

### M3: Subsampled-Gaussian privacy amplification
- **Mechanism**: At precompute time, randomly subsample fraction p of the
  N patients (e.g., p=0.5 → 10 of 20 DRIVE patients). Only the sampled
  patients' features get released and cached. Student trains on the
  reduced set.
- **Privacy story**: By Mironov 2017 / Wang-Balle-Kasiviswanathan 2019,
  Gaussian + subsampling has tighter RDP. User-level ε reduces by
  approximately a factor of p (or √p for Gaussian, depends on regime).
- **Baseline**: Full p=1 release at the same per-release ε.
- **Expected lift**: Lower noise per release at the same FINAL ε. But
  less data for the student. The optimal p depends on dataset size; for
  N=20 DRIVE patients, subsampling is risky.
- **Implementation effort**: Trivial (4 lines: random sample N patients).
  Accounting needs RDP moments accountant (~50 lines, can borrow from
  Opacus).
- **Paper home**: Mironov 2017 RDP; Wang et al. 2019 subsampled-Gaussian
  RDP; Abadi et al. 2016 DP-SGD moments accountant (which is exactly
  subsampled-Gaussian).

### M4: Public-proxy denoising adapter (already partially done)
- **Mechanism**: Train a small adapter on HRF public proxy that learns
  to map a thresholded-and-noised feature release back toward the clean
  feature space. Apply adapter at inference.
- **Privacy story**: Adapter trains entirely on public HRF → contributes
  0 to DRIVE's user-level ε. Pure post-processing on the DRIVE release.
- **Baseline**: No adapter (decode directly from noisy release).
- **Status**: We have local data on this. `drive_student_adapter.py`
  shows +19% gap closure on threshold-released features.
- **Expected lift**: +0.05-0.08 vessel Dice over no-adapter baseline.
  Strong, replicated across 3 ε values in single-seed experiments.
- **Paper home**: Post-processing immunity argument. Reference: Dwork-
  Roth book on DP post-processing. Architecturally similar to image
  denoising literature (DnCNN, etc.).

### M5: Channel-pruning threshold release (the local-commit backup)
- **Mechanism**: For each released image, drop the bottom (1-k)·C channels
  by public-proxy importance. Distribute ρ budget only across the top kC
  channels. Inactive channels released as zero.
- **Privacy story**: Same sample-once framework. ρ allocated only on
  active set. By post-processing, downstream consumption is free.
- **Baseline**: Full-channel release at same per-release ε.
- **Status**: We have local data (commit 2034c7e). At ε=32, keep-10%
  gives +0.17 Dice over uniform full-channel release.
- **Critique (user's)**: This is pre-processing (channel selection) not
  a new DP mechanism. Counter: the privacy amplification from releasing
  fewer dimensions IS a mechanism-level effect via the constraint
  Σ Δ²/(2σ²)=ρ scaling with |active|, not C.
- **Paper home**: RESEARCH_PLAN §13.2 sketches Bayes-optimality of +thr
  variant. Sparsity-aware DP literature (Smith et al., Asi et al.).

---

## 3. Tier-2 candidates — interesting, worth a half-day each

### M6: Channel aggregation via linear projection (user's suggestion #1)
- **Mechanism**: Apply a learned linear projection P : ℝ^C → ℝ^K with
  K << C BEFORE adding noise. Release P(z) + η where η ~ N(0,σ²I_K).
  P trained on HRF public proxy (e.g., PCA top-K components, or a 1x1
  conv learned to preserve task-loss gradient).
- **Privacy story**: P is a function of public data only (P trained on
  HRF). Applying P to private z is a deterministic linear map. L2-
  sensitivity of P(z) is ‖P‖_op · Δ_z = ‖P‖_op · 2/K (post-clip).
- **Baseline**: Identity P (full C-channel release) at same ρ.
- **Expected lift**: Modest. Lower dim → lower σ per dim, but lossy
  projection discards info. Net effect depends on importance
  concentration in top-K subspace.
- **Implementation effort**: 0.5 day. Compute PCA from HRF, project
  before noise, modify student adapter to receive K-dim targets.
- **Paper home**: Random-projection DP (Kenthapadi et al.); PCA-DP
  release (Dwork-Talwar projections).

### M7: Spatial water-filling allocation
- **Mechanism**: Allocate the privacy budget across spatial positions
  rather than channels. σ_ij computed from a spatial saliency map
  (HRF average vessel mask), more important pixels get less noise.
- **Privacy story**: Spatial mask from HRF is data-independent (the team
  caught this exact issue in WORKFLOW §14.2). Saliency comes from public
  proxy. Per-pixel sensitivity Δ_ij = 2 / K under clipping.
- **Baseline**: Uniform spatial noise at same ρ.
- **Caveat**: RESEARCH_PLAN §13.6 #2: lung-field-wide abnormalities on
  CXR may make spatial allocation flat. DRIVE vessels are spatially
  concentrated though, so this should work.
- **Implementation effort**: 1 day. Requires generating HRF spatial
  saliency map + modifying noise broadcast.
- **Paper home**: RESEARCH_PLAN §13.7 already showed spatial-WF lift =
  +1.64 pp at ε=2 on closed-form DRIVE — that was the strongest
  closed-form signal we ever got. Worth re-running on the U-Net student.

### M8: DP-SGD on student task-loss path (closing the §0.5.1 gap)
- **Mechanism**: Apply Opacus DP-SGD to the student's task-loss against
  GT labels (current pipeline uses these labels in violation of pure
  post-processing). Compose: ε_total = ε_features + ε_DPSGD.
- **Privacy story**: Properly closes the WORKFLOW §0.5.1 open gap. The
  student model now IS DP at ε_total user-level. No more "ε of released
  features only" disclaimer.
- **Baseline**: Feature-DP only (current; ignores label-side leak).
- **Caveat**: DRIVE has N=20; DP-SGD will be brutal at small ε.
- **Implementation effort**: 1 day. Integrate Opacus, tune.
- **Paper home**: Abadi et al. 2016 DP-SGD; Papernot et al. on combined
  DP-SGD + KD.

### M9: DP-SGD trained teacher (replace noise-on-release with noise-in-training)
- **Mechanism**: Train the teacher with DP-SGD so its weights are DP.
  Then release teacher features clean — they're DP by post-processing
  of DP weights.
- **Privacy story**: Different threat model: protect against weight
  release rather than feature release. Cleaner story for "teacher
  model deployment".
- **Baseline**: Clean-trained teacher + DP noise on released features.
- **Caveat**: DP-SGD on UNet for segmentation is hard; little prior
  work shows it converging well at small ε.
- **Implementation effort**: 2 days.
- **Paper home**: This crosses into prior DP-SGD literature; need
  positioning vs. their results.

---

## 4. Tier-3 candidates — research depth, harder to deliver in WACV timeline

### M10: Smooth sensitivity / instance-optimal noise
- Per-image local sensitivity instead of worst-case Δ.
- Hard math (Nissim-Raskhodnikova-Smith 2007 + propose-test-release).
- Major paper if it works; high risk.

### M11: Wavelet/frequency-domain release
- Release features after wavelet decomposition; noise in frequency.
- Similar to UESTC 2024 wavelet-DP (RESEARCH_PLAN §6 baseline #5).
- Worth re-implementing as comparison.

### M12: Multi-resolution feature pyramid release
- Release features at multiple resolutions; student gets coarse-to-fine.
- More release objects → more ρ; not obviously a win.

### M13: Coreset-based release
- Find K representative patients, release only their features.
- Like dataset distillation under DP.
- Reduces N effective; could amplify privacy.

### M14: Federated multi-hospital aggregation (PATE in distributed form)
- Same as M1 but framed as multi-site federation.
- More realistic for "medical DP" story but mostly a framing change.

### M15: Sparse coding / dictionary release
- Learn a dictionary on HRF; release noisy sparse codes per image.
- Each image is a few non-zero coefficients.
- Privacy on the sparse codes only.

---

## 5. Method × baseline matrix

| # | Method | Baseline | Estimated effort | Risk | Expected lift |
|---|---|---|---|---|---|
| M1 | PATE multi-teacher | K=1 single teacher | 1 day | low | 🟢 high (+0.05-0.15 Dice) |
| M2 | Logit distillation | Feature distillation | 0.5 day | medium | 🟡 mixed |
| M3 | Subsampled-Gaussian | Full N at same final ε | 0.5 day | low | 🟡 modest (+0.01-0.02) |
| M4 | Public-proxy adapter | No adapter | 0 day (done) | low | 🟢 +0.05-0.08 (already measured) |
| M5 | Channel-pruning thr | Full channels | 0 day (done) | low | 🟢 +0.17 at ε=32 (already measured) |
| M6 | Channel aggregation | Identity projection | 0.5 day | medium | 🟡 modest |
| M7 | Spatial-WF | Uniform spatial | 1 day | medium | 🟢 +1.64 pp closed-form predicted |
| M8 | DP-SGD on student task loss | Feature-DP only | 1 day | high | 🔴 brutal at small ε |
| M9 | DP-SGD trained teacher | Clean teacher + feat noise | 2 days | high | 🔴 likely brutal |
| M10 | Smooth sensitivity | Worst-case sensitivity | 1 week | very high | unknown |
| M11 | Wavelet-DP | Pixel/feature noise | 1 day | medium | 🟡 (matches UESTC baseline) |
| M12 | Multi-resolution | Single resolution | 1 day | medium | 🔴 more ρ to pay |
| M13 | Coreset release | Full release | 1.5 days | high | 🟡 unknown |
| M14 | Federated framing | Single site | 0.5 day (framing) | low | 🟡 reframe of M1 |
| M15 | Sparse coding | Dense release | 2 days | high | 🟡 unknown |

---

## 6. Execution priority

Goal: validate **2-3 methods** that deliver lift over their respective
baselines, within ~1 week of compute.

**Day 1 — sample sensitivity reduction (the biggest lever)**
- M1: PATE K=3 and K=5 PoC. Train K small teachers on disjoint patient
  cohorts; aggregate via median. Run sample-once + WF or uniform on the
  aggregate. Compare to K=1 single-teacher baseline at same per-release ε.

**Day 2 — accounting tightening**
- M3: subsampled-Gaussian via RDP moments accountant. Sweep p ∈
  {0.25, 0.5, 0.75, 1.0}. Compare to p=1 baseline.

**Day 3 — release-object variation (user's two suggestions)**
- M2: logit distillation PoC. Replace bottleneck cache with logit cache.
  Student trains via KL divergence on noisy soft labels.
- M6: linear channel aggregation. PCA on HRF, project to K=64 dims.

**Day 4 — confirm what we already have**
- M4 + M5: re-run with multi-seed paired tests to confirm earlier
  single-seed numbers from drive_student_adapter.py and
  drive_wf_threshold.py.

**Day 5 — spatial axis**
- M7: spatial-WF with HRF saliency. (Highest closed-form signal of any
  Phase-0 result.)

**Day 6+ — write-up**

---

## 7. Decision rule

A method enters the paper as contribution (ii) iff:
- (a) it shows ≥ +0.005 Dice lift over its CLEAN baseline at the same
  user-level ε,
- (b) p < 0.05 under 5-seed paired test,
- (c) the privacy story is defensible (we can write the ε derivation in
  one paragraph without hand-waving),
- (d) the comparison is at SAME user-level ε (not "same per-release ε
  with different release-count").

We currently have provisional positive data for M4 (adapter) and M5
(channel-pruning) from single-seed runs. PATE (M1) and subsampled-
Gaussian (M3) are the highest-priority new experiments because they are
real DP-mechanism advances with strong theoretical grounding.

---

## 8. References (for paper introduction / related work)

DP mechanisms:
- Abadi et al. 2016 — DP-SGD
- Bun & Steinke 2016 — zCDP
- Mironov 2017 — Rényi DP
- Wang, Balle, Kasiviswanathan 2019 — subsampled Gaussian RDP
- Balle & Wang 2018 — analytical Gaussian mechanism
- Nissim, Raskhodnikova, Smith 2007 — smooth sensitivity
- Dwork et al. 2014 — DP foundations (algorithmic foundations book)

PATE family:
- Papernot et al. 2017 — semi-supervised PATE
- Papernot et al. 2018 — scalable PATE
- Jordon et al. 2019 — PATE-GAN
- Bagdasaryan et al. 2019 — federated PATE

DP knowledge distillation specifically:
- Anonymous "DP feature distillation" [paper ref [2]] — per-iter approach
- Lan & Tian 2024 — GKD (non-DP gradient-weighted distillation)

Privacy attacks:
- Carlini et al. 2022 — LiRA membership inference
- Dosovitskiy & Brox 2016 — feature inversion
- Yin et al. 2021 — gradient inversion
- Zhu et al. 2019 — deep leakage

DP for medical imaging:
- Kaissis et al. 2020 — DP/federated for medical AI
- Ziller et al. 2021 — PriMIA
- Fan 2018 — DP-Pix
- UESTC 2024 — wavelet heterogeneous DP

---

## 9. Open questions before starting

1. **For M1 PATE**: how to partition DRIVE patients into K disjoint
   cohorts when N=20 and we want K=5? → 4 patients per teacher,
   data-poor. Possibly use HRF-pretrained backbone + fine-tune on each
   DRIVE subset. Or accept lower per-teacher accuracy and let
   aggregation recover.
2. **For M2 logit**: per-pixel softmax has dimension 2 (vessel / bg) ×
   H × W. Is this small enough that per-position sensitivity is
   tractable? L2 sensitivity per position is bounded by √2.
3. **For M7 spatial-WF**: how to generate HRF average vessel saliency
   if we do not have HRF labels? Use Frangi filter (vesselness) as
   data-independent saliency proxy.

---

## 10. Status snapshot

- Methods already validated (need multi-seed confirmation): M4, M5
- Methods to run this week: M1, M2, M3, M6, M7
- Methods deferred to future work or rejected: M8-M15
- Plan committed to git as `phase1_distillation/PHASE2_METHOD_BRAINSTORM.md`
