# Wu et al. 2025 — Implementation Departures to Disclose in Paper

Reference: "A multi-view privacy-preserving knowledge distillation method with adversarial
training and differential privacy", Wu et al., Information Sciences 721 (2025) 122588.

---

## 1. Corrections to the Paper's Algorithm (Departures That Strengthen Wu et al.)

These are places where the paper's pseudocode has a flaw. We implemented the
corrected version, which makes Wu et al. *stronger* as a baseline. This must be
disclosed so reviewers cannot accuse us of a weak reimplementation.

### 1a. z.detach() removed — Algorithm 2, line 11

**What the paper says:**
```
11:  p_mia ← D_MIA(z.detach())
```

**What our code does:**
```python
l_adv = bce(mia(feat), torch.zeros(...))   # no detach
```

**Why we changed it:**
With `z.detach()` the computation graph is cut at the feature tensor z. The gradient
of l_adv = BCE(p_mia, 0) therefore contributes *zero gradient* to the teacher
parameters. The teacher only receives gradient from l_task — the adversarial defence
term λ_adv·l_adv is numerically included in the total loss but has no effect on
training. The teacher never learns to fool the MIA discriminator.

This contradicts the paper's stated training objective (Section 5.4.3, p.10):
"the teacher model minimizes adversarial loss to prevent the MIA discriminator from
extracting meaningful membership information from its outputs", and Eq. 6:
L_adv = −E_{x∼D}[log(1−D(f(x)))].

Evidence the authors themselves did not use detach: their own training loss curves
(Figs. 4, 5, 6) show Adv Loss decreasing steadily over epochs. A detached gradient
cannot produce this behaviour — the teacher's features would not change adversarially
and the frozen MIA would produce a flat Adv Loss. The pseudocode line is a typo.

**Paper text to use:**
"Algorithm 2 of Wu et al. contains z.detach() at line 11, which prevents the
adversarial gradient from reaching the teacher and makes the adversarial defence
inactive. We remove this detach so that the gradient flows from the MIA loss back
to the teacher, consistent with the method's stated objective (Eq. 6) and with
the decreasing adversarial loss curves reported in their paper."

---

### 1b. Per-sample L2 clipping before noise — Algorithm 1, line 7

**What the paper says:**
```
6:  Sample noise ∼ N(0, σ²)
7:  x̃ ← x + noise
```
No clipping is shown before noise addition.

**What our code does:**
```python
# Per-sample L2 clip to sensitivity bound, then add noise
norms = flat.norm(2, dim=1, keepdim=True)
scale = (sensitivity / norms.clamp(min=1e-8)).clamp(max=1.0)
clipped = (flat * scale).view_as(x)
return clipped + torch.randn_like(clipped) * sigma
```

**Why we added it:**
The Gaussian mechanism's privacy guarantee requires the L2 sensitivity of the
randomised function to be bounded by S. Without clipping, neural network outputs
are unbounded — the actual sensitivity can be arbitrarily large, invalidating the
formal (ε, δ)-DP guarantee. Per-sample L2 clipping to S enforces the sensitivity
assumption, making the privacy accounting mathematically valid.

**Paper text to use:**
"Wu et al. Algorithm 1 adds Gaussian noise directly to teacher outputs without
bounding their L2 norm. We add per-sample L2 clipping to the stated sensitivity
bound S prior to noise injection, which is required for the Gaussian mechanism's
(ε, δ)-DP guarantee to hold [Dwork & Roth, 2014]. This correction strengthens
Wu et al.'s formal privacy guarantee."

---

## 2. Necessary Adaptations for Medical Image Segmentation

Wu et al. was designed and evaluated on image classification (MNIST, SVHN, CIFAR-10).
We adapt it to medical image segmentation. These are not corrections — they are
domain-required changes, all of which are disclosed.

| Aspect | Wu et al. (original) | Our adaptation | Reason |
|---|---|---|---|
| Sub-model architecture | ResNet18 + GAP → class vector | TinyUNet bottleneck → spatial map | Segmentation requires dense spatial predictions, not a pooled class vector |
| Task loss (L_task, Eq. 4) | Cross-entropy (classification) | CE + 3×Dice | Standard for medical segmentation |
| KD loss (L_KD, Eq. 9) | KL divergence over C class logits | Pixel-wise KL over H×W×C logit map | Segmentation logits are spatial, not image-level |
| Batch size | 128 (stated in Section 6.1.1) | √N per dataset (BUSI=23, ISIC=45, Kvasir=28) | Matches CANAL and DP-SGD batch size for fair comparison |
| Training epochs | Not stated in paper | Teacher=60, Student=40 | Matches CANAL (--te 60 --se 40) for fair comparison |
| n_views for grayscale (BUSI) | N/A (all datasets RGB) | n_views=1, single sub-model | Paper itself states MNIST (grayscale) uses one sub-model; same rule applied to BUSI |

---

## 3. What Is Unchanged (Faithful to the Paper)

For completeness — these match the paper exactly and require no disclosure:

- σ = S√(2ln(1.25/δ)) / (ε/T) — Algorithm 1, lines 2–3
- Sensitivity values: S_feat=1.0, S_logit=2.0 — Section 6.1.1
- Privacy budget split: 50% for features, 50% for logits — Section 6.1.1
- δ = 1e-5 — Section 6.1.1
- ε ∈ {1, 2, 4, 8} — Section 6.2.3
- Learning rates: 1e-3 (teacher/student), 1e-4 (MIA) — Section 6.1.1
- MIA warmup: extract all train features (label=1) + val features (label=0) — Alg 2 lines 3–4
- MIA warmup: normalize to zero mean / unit variance — Algorithm 2, line 5
- MIA warmup: 2 inner training epochs — Algorithm 2, line 6
- Teacher adversarial loss direction: BCE(p_mia, 0) — Eq. 6
- Total teacher loss: L = l_task + λ_adv·l_adv — Eq. 3
- Student loss: L_student = α·L_KD + β·L_task + γ·L_feature + δ·L_contrastive — Eq. 8
- L_KD: τ²·KL(σ(z_t/τ) ‖ σ(z_s/τ)) — Eq. 9
- Multi-view channel split (R/G/B → separate sub-models) — Algorithm 3
- Channel attention at feature fusion — Fig. 3
- DP noise applied to teacher outputs (not gradients) — Algorithm 1

---

## 4. How These Departures Affect the Comparison

Both corrections in Section 1 make Wu et al. **stronger**:
- Correction 1a (no detach): teacher's adversarial defence actually works → better privacy resistance
- Correction 1b (L2 clip): formal DP guarantee is valid → honest privacy accounting

If CANAL outperforms this strengthened version of Wu et al., the result is more
credible — we cannot be accused of weakening Wu et al. to make CANAL look better.
