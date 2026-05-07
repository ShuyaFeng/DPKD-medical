# DPKD-medical — structure-aware DP image release for medical imaging

Working repository for a WACV 2026 submission on **differentially private
image release with channel × spatial water-filling allocation**, targeting
medical imaging benchmarks (DRIVE, BraTS, MIMIC-CXR, ISIC).

The headline mechanism is a closed-form Gaussian noise allocation that
splits a per-image $\rho$-zCDP budget jointly across channel and spatial
axes, and then suppresses low-utility pixels via a release threshold
("**joint-WF+thr**"). On real DRIVE (40 images, expert vessel masks,
$\delta=10^{-5}$), it lifts whole-image Bayes accuracy by **+3.05 ± 0.58 pp
at $\varepsilon=2$** over uniform Gaussian release, matching the numerical
Bayes-optimal upper bound under binary saliency.

The full design, threat model, baselines, milestones, and Phase 0 results
are in [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md). Section 13 is the most
up-to-date evidence for the mechanism.

## Status

- [x] Phase 0 — closed-form utility validation on synthetic 4-channel
      images and on real DRIVE (40 imgs).
- [ ] Phase 1 — DRIVE U-Net baseline + DP-Pix / DP-SGD baselines, then
      BraTS / NIH-CXR.
- [ ] Phase 2 — full mechanism, theory, sweeps.
- [ ] Phase 3+ — writing, polish, submission.

See [`RESEARCH_PLAN.md` §8](RESEARCH_PLAN.md) for the timeline.

## Phase 0 results at a glance

Whole-image Bayes accuracy at $\varepsilon=2$, $\delta=10^{-5}$, mean over
40 real DRIVE images. "Ours" is in **bold**.

| Mechanism | Bayes acc | lift over uniform |
|---|---|---|
| uniform Gaussian | 0.5071 | — |
| channel-WF (ablation) | 0.5090 | +0.19 ± 0.06 pp |
| spatial-WF (ablation) | 0.5235 | +1.64 ± 0.31 pp |
| **joint-WF (ours)** | 0.5300 | **+2.29 ± 0.47 pp** |
| **joint-WF + release threshold (ours)** | **0.5375** | **+3.05 ± 0.58 pp** |
| Bayes-optimal (ceiling) | 0.5375 | +3.05 ± 0.58 pp |
| adversarial (sanity check) | 0.5003 | −0.67 ± 0.14 pp |

Curves over $\varepsilon \in [0.5, 10]$:

- Synthetic toy: [`phase0_validation.png`](phase0_validation.png)
- Real DRIVE (40 imgs, mean ± std): [`phase0_drive_real.png`](phase0_drive_real.png)
- Per-channel contrast on DRIVE: [`phase0_drive_real_channels.png`](phase0_drive_real_channels.png)
- Noise-allocation heatmap (single-image): [`phase0_drive_alloc.png`](phase0_drive_alloc.png)

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Data

DRIVE is gated behind registration; the data directory is intentionally
not committed (license + size). The `phase0_drive_real.py` script expects:

```
data/DRIVE/
  train/input/*.tif         # 20 RGB fundus images (584x565)
  train/label/*.png         # 20 expert vessel masks
  val/input/*.tif           # 20 RGB fundus images
  val/label/*_manual1.png   # 20 expert vessel masks
```

A working open mirror is the Hugging Face dataset
`Zomba/DRIVE-digital-retinal-images-for-vessel-extraction`. Inspect the
license terms before redistributing the data.

## Running the Phase 0 scripts

All scripts are pure NumPy/SciPy/scikit-image — no GPU, no training. Each
is self-contained and writes its plots into the repo root.

```bash
# Synthetic 4-channel toy: dense epsilon sweep + canonical 5-point plot.
python phase0_validation.py
# → phase0_validation.png, phase0_validation_canonical.png

# Mask-corruption robustness for joint-WF and joint-WF+thr.
python phase0_robustness.py

# Single-image stand-in using skimage's public-domain retina sample.
python phase0_drive.py
# → phase0_drive.png, phase0_drive_alloc.png, phase0_drive_inputs.png

# Real DRIVE (40 images, expert masks). Requires data/DRIVE/ above.
python phase0_drive_real.py
# → phase0_drive_real.png, phase0_drive_real_channels.png
```

## Repository layout

```
.
├── RESEARCH_PLAN.md             # full plan; §13 has Phase 0 results
├── phase0_validation.py         # synthetic toy validation
├── phase0_robustness.py         # mask-corruption robustness
├── phase0_drive.py              # single-image fundus stand-in
├── phase0_drive_real.py         # 40-image real DRIVE validation
├── phase0_*.png                 # generated plots
├── demo_noise.py                # PATE-style teaching demo (uses torch)
├── PaperForReview.tex           # WACV LaTeX manuscript (template)
├── RebuttalTemplate.tex         # rebuttal template
├── egbib.bib, ieee_fullname.bst, wacv.sty
├── requirements.txt
└── .gitignore                   # excludes .venv/, data/, LaTeX builds
```

## Building the LaTeX paper

```bash
pdflatex PaperForReview.tex && bibtex PaperForReview && pdflatex PaperForReview.tex && pdflatex PaperForReview.tex
```

The combined `PaperForReview.tex` is initially set up for review
submission. See the comments at the top of the file for how to toggle
between review and camera-ready, set the CMT paper ID, and pick the
track (Applications vs Algorithms).

## License & attribution

- Code: MIT (TBD — confirm before publication).
- LaTeX templates: WACV author kit.
- DRIVE data: governed by the original DRIVE dataset license; not
  included in this repository.
