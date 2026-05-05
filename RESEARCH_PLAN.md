# Research Plan — Structure-Aware Differentially Private Image Release for Medical Imaging

**Target venue:** WACV 2026 (algorithms or applications track — to be decided)
**Owner:** Emily
**Last updated:** 2026-04-30

---

## 1. Headline contribution

A single differentially private (DP) mechanism for medical image release that allocates the privacy budget jointly across **channel** (modality) and **spatial** (region) axes via a unified water-filling formulation, with formal $(\varepsilon, \delta)$-DP / zCDP guarantees and a downstream knowledge-distillation pipeline.

**One-sentence pitch.** Uniform Gaussian noise on medical images wastes budget on uninformative channels and background pixels; a structure-aware allocation gives the same $\varepsilon$ with materially better downstream utility, and our mechanism is the first to do this jointly with a data-independent saliency map and end-to-end DP composition.

**Three contributions:**
1. **Mechanism.** A channel × spatial Gaussian mechanism with closed-form water-filling allocation, generalizing DP-Pix (uniform pixel) and wavelet-domain heterogeneous DP (frequency-only).
2. **Theory.** zCDP composition bound, sensitivity analysis under per-patient neighboring relation, attack-aware noise calibration in the spirit of Nasr et al.
3. **Empirical.** On BraTS (multi-modal MRI segmentation), NIH/MIMIC-CXR (classification), and ISIC (skin), the mechanism beats uniform DP image release and matches DP-SGD at $\varepsilon \le 2$ — with empirical membership-inference resistance.

---

## 2. Threat model & privacy definition (commit early)

| Choice | Decision | Rationale |
|---|---|---|
| Privacy unit | **Per-patient** | Only meaningful unit clinically. |
| Neighboring relation | $D \sim D'$ if they differ in all images of one patient (with $\le k$ images per patient, $k$ = small constant per dataset). | Cleanly extends DP-Pix's $m$-neighborhood. |
| Adversary | Has $\tilde x$ released, may have auxiliary medical data; aims at membership inference / re-identification. | Standard. |
| Mechanism placement | **Input-space release**: publish $\tilde x$, then any downstream training/inference is post-processing — DP guarantee survives. | Lets us bound utility without cycle-by-cycle privacy bookkeeping. |
| Privacy accountant | Rényi DP / zCDP | Tightest available; standard at top venues. |

**Sensitivity bound.** Each image is clipped to $\|x_c\|_2 \le \Delta_c$ per channel. Per-patient sensitivity is $k \cdot \Delta_c$ (clipping × cap on images-per-patient). State this as a numbered assumption.

---

## 3. Method: structure-aware DP image release

### 3.1 Channel-wise allocation (Idea 2 component)

For a $C$-channel image $x \in \mathbb{R}^{C \times H \times W}$, release per channel:
$$
\tilde x_c = x_c + \mathcal{N}(0, \sigma_c^2 \mathbf{I}).
$$
Optimization:
$$
\min_{\{\sigma_c\}} \; \sum_{c=1}^C \frac{w_c}{\sigma_c^2} \quad \text{s.t.} \quad \sum_{c=1}^C \frac{(k\Delta_c)^2}{2 \sigma_c^2} \le \rho.
$$
Closed-form (water-filling): $\sigma_c^{\star 2} \propto k\Delta_c \sqrt{w_c} / \sqrt{\rho}$.
Weights $w_c$ encode per-channel utility (computed on a public proxy, see §3.4).

### 3.2 Spatial allocation (Idea 3 component)

Per-pixel variance modulated by a saliency mask $m \in [0,1]^{H\times W}$:
$$
\sigma^2(i,j) = \sigma_{\min}^2 + (\sigma_{\max}^2 - \sigma_{\min}^2)\cdot s(m_{ij}),
$$
where $s$ is monotone (linear or sigmoid). High $m_{ij}$ → diagnostically important → less noise.

### 3.3 Joint mechanism

Combine: $\sigma^2_{c,ij} = \sigma_c^2 \cdot \pi(m_{ij})$ for a normalized spatial profile $\pi$. Privacy accounting composes per-pixel Gaussians via zCDP additivity. Total privacy cost:
$$
\rho_{\text{total}} = \sum_{c,i,j} \frac{(k\Delta_{c,ij})^2}{2 \sigma_{c,ij}^2}.
$$

### 3.4 Data-independent saliency / weighting

This is the technical crux — make it bulletproof.

Two acceptable mask sources, both keep $m$ independent of the private data:

a. **Anatomical-atlas mask** (X-ray, head CT). Register a public lung-field / brain atlas to image space; $m$ is a deterministic function of image dimensions.
b. **Public-data saliency proxy.** Train a Grad-CAM / attention model on a *separate public* dataset (NIH-CXR), apply to the private dataset (MIMIC-CXR). The mask depends only on the public model's parameters and on $x$ at inference time, so per-patient neighboring noise calibration still holds.

**Reject:** computing $m$ from gradients on the private dataset — this is the trap the CSI 2025 paper falls into. We will explicitly contrast.

### 3.5 Channel weights $w_c$

Compute on the public proxy: per-channel input-gradient $L_2$ norm averaged over a held-out public split, normalized to sum to one. Data-independent w.r.t. the private set.

---

## 4. Theory & analysis to write

- **Theorem 1 (zCDP composition).** Joint mechanism is $\rho$-zCDP under per-patient neighboring with $k$-image cap.
- **Theorem 2 (optimal allocation).** The water-filling solution minimizes a quadratic Bayes-risk surrogate of downstream MSE under fixed $\rho$.
- **Proposition (post-processing).** Any function of $\tilde x$ inherits the privacy guarantee.
- **Conversion.** $(\rho)$-zCDP $\to (\varepsilon, \delta)$-DP via Bun-Steinke conversion.
- **Tightness check.** Compare to analytical Gaussian mechanism (Balle-Wang 2018) at the per-pixel level.

---

## 5. Datasets & tasks

| Dataset | Modality / channels | Task | Role |
|---|---|---|---|
| BraTS 2021 | Multi-parametric MRI (T1, T1ce, T2, FLAIR) | Tumor segmentation | Channel allocation showcase |
| NIH-CXR14 | Single-channel X-ray | 14-label classification | **Public proxy** for MIMIC |
| MIMIC-CXR | Single-channel X-ray | Multi-label classification | Private set |
| ISIC 2019 | RGB dermoscopy | 8-class classification | Both axes (RGB + spatial) |
| CheXpert | Single-channel X-ray | Multi-label classification | Optional second private set |

**Per-patient cap $k$.** BraTS: $k=1$ (one volume per subject). MIMIC-CXR: $k \le 4$ (most patients). ISIC: $k=1$.

---

## 6. Baselines

Each must run end-to-end in our pipeline:

1. **No privacy** — upper bound.
2. **DP-SGD** (Opacus) — the dominant medical-DP baseline; report at $\varepsilon \in \{1,2,4,8\}$.
3. **DP-Pix** (Fan 2018) — uniform-noise image release.
4. **DP-Image** (feature-space DP) — alt input-space mechanism.
5. **Wavelet-domain heterogeneous DP** (UESTC 2024) — closest method-level competitor; reimplement.
6. **ADP-FL** (arxiv 2604.06518) — federated adaptive DP; for context.
7. **Ours (channel-only)** — ablation.
8. **Ours (spatial-only)** — ablation.
9. **Ours (joint)** — full method.

---

## 7. Evaluation protocol

| Axis | Metric |
|---|---|
| Utility | Dice (BraTS), AUROC / mAP (CXR, ISIC) at $\varepsilon \in \{1,2,4,8\}$ |
| Privacy (formal) | $(\varepsilon, \delta)$ via RDP accountant; report $\varepsilon$ at $\delta = 1/n$ |
| Privacy (empirical) | **LiRA membership inference** (Carlini et al.) attack accuracy / TPR@FPR=0.1% on released images |
| Reconstruction | Off-the-shelf inversion attempt (e.g., diffusion-prior reconstruction); report SSIM / LPIPS gap |
| Visual | Side-by-side DP-Pix vs. ours at matched $\varepsilon$ |
| Robustness | Sensitivity to mask quality (corrupt $m$, measure utility drop) |

**Pre-registered claim:** at $\varepsilon = 2$, joint mechanism beats DP-Pix by $\ge 5$ Dice points on BraTS and $\ge 3$ AUROC points on MIMIC-CXR.

---

## 8. Implementation plan & milestones

Working backwards from a mid-July 2026 WACV deadline.

### Phase 0 — Setup (Week 1, May 5–11)
- [ ] Read the 5 most threatening prior-art papers cover-to-cover (Fischer 2020, UESTC wavelet 2024, CSI 2025, DP-Pix, ADP-FL).
- [ ] Pin environment: PyTorch + Opacus + MONAI + albumentations.
- [ ] Get data access: BraTS 2021, NIH-CXR (public), MIMIC-CXR (PhysioNet credentialed), ISIC 2019.
- [ ] Decide track (algorithms vs. applications) and update `\confYear` to 2026 in `PaperForReview.tex` once 2026 style file is released.

### Phase 1 — Baselines (Weeks 2–3, May 12–25)
- [ ] DP-SGD on NIH-CXR with Opacus; replicate published $\varepsilon$/AUROC numbers.
- [ ] DP-Pix implementation; verify $\varepsilon$ accounting matches Fan 2018.
- [ ] LiRA attack pipeline; verify it can break a model trained without DP.

### Phase 2 — Core mechanism (Weeks 4–6, May 26 – June 15)
- [ ] Channel-wise mechanism: closed-form water-filling, end-to-end on BraTS.
- [ ] Spatial mechanism: atlas mask for CXR; public-proxy saliency for MIMIC.
- [ ] Joint mechanism on ISIC.
- [ ] Privacy accountant unit tests (RDP composition matches simulated $(\varepsilon, \delta)$).

### Phase 3 — Theory + writing (Weeks 7–8, June 16–29)
- [ ] Write zCDP composition theorem + post-processing corollary.
- [ ] Write data-independence argument for $m$ formally.
- [ ] Draft method, theory, and intro sections.

### Phase 4 — Experiments at scale (Weeks 9–10, June 30 – July 13)
- [ ] Sweeps: $\varepsilon \in \{1,2,4,8\}$, all baselines, all datasets, all ablations.
- [ ] LiRA-MIA on every released dataset.
- [ ] Robustness ablation (corrupt $m$, vary $k$, vary $\Delta_c$).

### Phase 5 — Polish & submit (Week 11, July 14–19)
- [ ] Final figures, tables, bibtex.
- [ ] Internal review pass.
- [ ] Submit.

### Phase 6 — Post-submission
- [ ] Code release prep (anonymized for review).
- [ ] Supplementary: full proofs, additional ablations.

---

## 9. Risks & mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| DP-SGD with pretraining matches our utility at all $\varepsilon$ | Medium | High | Target the low-$\varepsilon$ regime ($\varepsilon \le 2$) where input-space mechanisms have the advantage; report as our "win zone." |
| Mask data-independence argument is shaky on review | Medium | High | Pick atlas-based mask as the *primary* setting (cleanest); public-proxy saliency as the secondary; never gradient-on-private. |
| LiRA-MIA shows our method is more attackable than DP-Pix | Low | Critical | If empirical attack succeeds, mechanism has a bug — fix in Phase 2, not Phase 5. |
| MIMIC-CXR access takes >2 weeks | Medium | Medium | Start credentialing in Phase 0; have CheXpert as fallback private set. |
| Wavelet-DP (UESTC 2024) reproduction fails | Medium | Medium | If their code is unavailable, implement our best-faith version and report that. |
| BraTS 3D compute exceeds budget | Low | Medium | Default to 2D slice models; 3D as a stretch goal. |
| Reviewer claims "this is just the CSI paper for medical" | High | Medium | Explicit comparison table in related work; emphasize (a) joint channel × spatial, (b) data-independent mask, (c) downstream distillation. |

---

## 10. Deliverables

- LaTeX paper (`PaperForReview.tex`) — 8 pages + references.
- Method figure: channel × spatial mechanism diagram.
- Qualitative figure: DP-Pix vs. ours at matched $\varepsilon$ on BraTS and CXR.
- Results table: utility × privacy at 4 $\varepsilon$ levels, 9 methods, 3+ datasets.
- LiRA-MIA results table.
- Code repo (anonymous for review): mechanism, baselines, eval scripts.

---

## 11. Open decisions

These are not yet pinned; flag them for the next sync.

1. **Track:** algorithms vs. applications. My lean is **applications** — narrative is method-for-medical, not novel core DP theory.
2. **Mask source priority:** atlas-first vs. public-proxy-first. Probably atlas for CXR, proxy for ISIC, both for BraTS.
3. **Federated extension:** include or defer to follow-up? If included, position vs. ADP-FL is delicate.
4. **3D vs. 2D BraTS:** start 2D, escalate only if 2D doesn't differentiate.
5. **Whether to include a PATE-distilled student as a downstream consumer.** Currently descoped; can be added if Phase 4 has time.

---

## 12. Reading list (priority order)

1. Fischer et al. 2020 — *Decentralized DP Segmentation with PATE.* (closest to PATE-segmentation gap)
2. UESTC 2024 — *DP for Medical Image Big Data with Multi-resolution Analysis.* (wavelet heterogeneous DP)
3. arxiv 2512.20323 — *Adaptive Privacy Budget on CSI Spectrograms.* (closest to our spatial allocation)
4. Fan 2018 — *Image Pixelization with DP (DP-Pix).* (defines image-DP neighboring)
5. Liu et al. 2021 — *DP-Image (feature-space DP).*
6. Balle & Wang 2018 — *Analytical Gaussian Mechanism.* (tight calibration)
7. Bun & Steinke 2016 — *Concentrated Differential Privacy.* (zCDP composition)
8. Carlini et al. 2022 — *LiRA: Membership Inference Attacks From First Principles.*
9. npj Digital Medicine 2025 — DP-medical scoping review.
10. Ziller et al. 2021 — *Medical imaging deep learning with DP / PriMIA.*
