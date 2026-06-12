# WORKFLOW — From DRIVE images to a publishable DP result

This is the **single source of truth** for what to run, in what order, and
why. Read top to bottom on a first pass; later use the table of contents
as a jump table.

---

## 0. The 30-second mental model

The paper pipeline has FIVE layers. The first (sample-once + public
proxy) is the anchor that makes user-level ε meaningful; PATE and
channel-pruning sit on top to boost utility; noise is added once; the
student learns from the cache. See `paper_draft/WACV_STRATEGY.md` for
the contribution-level map; this file is the operational how-to.

```
   PUBLIC HRF          PRIVATE DRIVE (N=20 patients, k=1 image each)
   (caps + importance)         │
        │                      │
        ▼                      ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ Phase A — One-time setup (~30 min, once)                    │
   │   - Public proxy: caps + importance on HRF (zero ε cost)    │
   │   - ANCHOR of the whole pipeline (contribution 1.1)         │
   └────────────────────────────┬─────────────────────────────────┘
                                ↓
   ┌──────────────────────────────────────────────────────────────┐
   │ Phase B — DP mechanism (per ε × seed). FOUR sub-steps:      │
   │                                                              │
   │   B1. PATE aggregate (contribution 1.2)                     │
   │       - K teachers on disjoint cohorts                      │
   │       - mean of K normalised bottlenecks → Δ = 2/K          │
   │                                                              │
   │   B2. Channel-pruning (contribution 1.3)                    │
   │       - drop bottom-(1−k)C channels by public importance    │
   │       - keep only top-kC "active" channels                  │
   │                                                              │
   │   B3. Allocate σ over the ACTIVE channels                   │
   │       - uniform (default) OR water-filling (Theorem 1, 1.4) │
   │       - whole ρ budget spent on active set only             │
   │                                                              │
   │   B4. ★ ADD GAUSSIAN NOISE ★  ONCE per patient, cache it    │
   │       - the only DP-relevant operation in the pipeline      │
   └────────────────────────────┬─────────────────────────────────┘
                                ↓
   ┌──────────────────────────────────────────────────────────────┐
   │ Phase C — Student training (no DP work, post-processing)    │
   │   - Student matches cached noisy features (feat loss)       │
   │   - + GT task loss, mixed via normalized α (1.5)            │
   └────────────────────────────┬─────────────────────────────────┘
                                ↓
   ┌──────────────────────────────────────────────────────────────┐
   │ Phase D — Privacy accounting / reporting                   │
   │   - Re-derive user-level ε (= reported ε by construction)   │
   │   - Tables 3/4 for paper                                    │
   └──────────────────────────────────────────────────────────────┘
```

**Phase A** runs once for the entire project.
**Phase B + Phase C** are bundled into **one** job per (ε, seed).
**Phase D** is a single python invocation when results are in.

The only place **noise is actually added** is **Phase B4**. See §6 for
the exact math and code path.

> **Two codebases, same pipeline.** The cluster path uses mmseg
> (`scripts/phase1_gkd_distill_v2.py`); the paper's empirical results
> come from the lighter local TinyUNet path
> (`drive_pate_*.py`, `drive_*_results.json`). Both implement the same
> five layers; the local path is what produced Tab. 3 / Tab. 4.

---

## 0.5 Threat model — the assumptions every later phase rests on

Everything that follows assumes:

| Property | Value | Why it matters |
|---|---|---|
| Privacy unit | **Per-patient** | The only clinically meaningful unit. |
| Per-patient image cap on DRIVE | **k = 1** (one image per patient) | Lets `per-image release = per-patient release` collapse in §6.4. Breaks on datasets with k > 1 (e.g. MIMIC-CXR, k ≤ 4). |
| Private data | DRIVE **images AND vessel masks** | The teacher was trained on DRIVE image-label pairs, so both encode per-patient information. Whatever protects DRIVE has to protect both. |
| Public data | HRF (or STARE / CHASE-DB1) | Used in Phase A to compute caps + importance. No DRIVE byte ever enters Phase A. |
| Mechanism placement | **Feature release** — add Gaussian noise to teacher bottleneck features of each DRIVE image | The "released artifact" is the cached noisy feature tensor. Anything downstream is post-processing **of that tensor**. |
| Sample-once | Each DRIVE image released **exactly once**, cached, reused for all 40k iterations | Avoids per-iteration composition blow-up. With k = 1 this is also exactly one release per patient. |
| Accountant | Rényi DP / zCDP, converted to (ε, δ)-DP via Bun-Steinke at report time | Tightest for Gaussian. |

### 0.5.1 The GT-label caveat (open question, not yet closed)

DRIVE's vessel masks were curated from DRIVE images by expert annotators, so logically they belong to the private set — anything that learns from them learns from the private data. Phase C's student computes a **task loss against the ground-truth mask** (`CE + 3·Dice`), which is **not** a function of the cached noisy features; it is a direct data-dependent operation on the private labels.

What this means for the ε we report:

- The cached noisy features are **(ε, δ)-DP** by §6.4 — that part is solid.
- The student model, as a function of (cached features) **and** (private labels), is **not** automatically (ε, δ)-DP at the same ε. The label-side gradient flow needs its own accounting.

Three ways to close this gap, none committed:

1. **Treat labels as public auxiliary.** Defensible-ish on DRIVE (vessel annotations are a clinical interpretation, not raw patient identifiers) but weakens fast when extending to BraTS / MIMIC where labels are diagnoses.
2. **DP-SGD on the task-loss path** for the student, on top of the feature release. Adds a second ε that has to compose with the feature-release ε.
3. **Drop the task loss entirely**, train the student purely on cached noisy features (pure feature-matching). Cleanest privacy story; gives an upper bound on what "pure post-processing" can achieve, at the cost of forgoing GT supervision.

**Until one is picked**, the safest framing in writeups is: report ε as "ε of the released features", and disclose the label-side data dependence as a methodological limitation — **not** as the student model's user-level ε.

---

## 1. "I want to do X" — jump table

| What you want to do | Run this | Section |
|---------------------|----------|---------|
| Set up the project from scratch | sbatch `run_compute_public_proxy.sh HRF` | §3 |
| Run one DP+training experiment | sbatch `run_lambda_sweep.sh 20 channel_WF 8 0 yes` | §4–5 |
| Run the full WACV experiment matrix | bash `submit_all_wacv.sh` | §4–5 |
| Re-verify the privacy claim of all results | python3 `privacy_accounting.py` | §7 |
| Analyse budget-split trade-offs (for §3.4 ablation) | python3 `budget_split_analysis.py --total-eps 8` | §7 |
| Smoke-test the codebase locally | python3 `smoke_test_diagnostics.py` | §10 |
| Understand exactly how noise is added | nothing to run; read | §6 |

---

## 2. File / script inventory

### Python scripts (`phase1_distillation/scripts/`)

| File | What it does | When you run it |
|------|--------------|-----------------|
| `compute_public_proxy.py` | Forward teacher on public data; output caps + importance CSV | Phase A.3 |
| `phase1_gkd_distill_v2.py` | Phase B (DP precompute) + Phase C (student training) in one job | per (ε, seed) |
| `privacy_accounting.py` | Re-derive user-level ε from the (eps, threat_model) you reported | after results land |
| `budget_split_analysis.py` | Analytical sweep for §3.4 ablation table | once, when writing paper |
| `smoke_test_diagnostics.py` | Local unit tests (20+ assertions) | anytime you change code |

### sbatch wrappers (`phase1_distillation/slurm/`)

| Script | Wraps | When |
|--------|-------|------|
| `run_compute_public_proxy.sh` | `compute_public_proxy.py` | Phase A.3 |
| `run_lambda_sweep.sh` | `phase1_gkd_distill_v2.py` | one DP run per call |
| `run_diag_{A,B,C}.sh` | `phase1_gkd_distill_v2.py` w/ ablation flags | diagnostic ablations |
| `submit_all_wacv.sh` | All of the above, in priority order | one-shot to submit everything |

### LaTeX (`paper_draft/`)

| File | Purpose |
|------|---------|
| `section_3_body.tex` | §3 of the paper — gets `\input` into `PaperForReview.tex` line 303 |
| `section_3_appendix.tex` | Appendix A (pseudocode) + B (full proof) — `\input` at line 481 |
| `section_3_privacy.tex` | Thin wrapper for the standalone `main.tex` preview |
| `main.tex` | Standalone preview document (compile to PDF for review) |

---

## 3. PHASE A — One-time setup

### A.1 — Teacher checkpoint (already done)

| Quantity | Value |
|----------|-------|
| Config | `unet-s5-d16_fcn_4xb4-ce-1.0-dice-3.0-40k_drive-64x64.py` |
| Checkpoint | `fcn_unet_s5-d16_..._drive_20211210_785de5c2.pth` |
| Location on cluster | `/data/user/home/ialam/mmseg_models/unet_teacher/` |

If this is missing on a new machine, you'll need to either:
- Copy the existing checkpoint, or
- Retrain via mmseg with the config above on DRIVE.

### A.2 — Download a public retinal dataset

Pick **one** of these (HRF is easiest to download, no registration):

| Dataset | URL | Images | Notes |
|---------|-----|--------|-------|
| **HRF** (recommended) | https://www5.cs.fau.de/research/data/fundus-images/ | 45 | Single zip, no auth |
| STARE | https://cecas.clemson.edu/~ahoover/stare/ | 20 | Per-image downloads |
| CHASE-DB1 | https://blogs.kingston.ac.uk/retinal/chasedb1/ | 28 | Child fundus, different distribution |

```bash
# On the cluster:
mkdir -p ~/public_retinal/HRF
cd ~/public_retinal/HRF
wget https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/all.zip
unzip all.zip
# Verify: should have 45 .jpg files
ls *.jpg | wc -l
```

### A.3 — Compute public caps + importance (~30 min)

This is the step that produces the two CSVs every later phase will read.

```bash
sbatch run_compute_public_proxy.sh HRF
```

This calls `compute_public_proxy.py` which:
1. Loads the teacher (frozen)
2. Iterates each public image:
   - Resize to 584×565 (DRIVE's native size)
   - Forward through teacher encoder → bottleneck `z ∈ R^(1024×37×36)`
   - **Caps**: record per-channel L2 norm `‖z_c‖_2`
   - **Importance**: forward through decoder, take teacher's argmax as
     pseudo-label, compute `|∂L_CE/∂z_c|` averaged over spatial dims
3. Aggregates across all 45 images:
   - `cap_c   = quantile_0.95(‖z_c‖_2 across images)`
   - `imp_c   = mean over images of mean_{ij} |∂L/∂z_{c,i,j}|`
4. Writes:
   - `work_dirs/PUBLIC_PROXY_HRF/public_caps.csv`        (1024 rows)
   - `work_dirs/PUBLIC_PROXY_HRF/public_importance.csv`  (1024 rows)

**Privacy cost on DRIVE: ZERO** — none of DRIVE was touched here.

**Sanity check after the job finishes**:
```bash
head work_dirs/PUBLIC_PROXY_HRF/public_caps.csv
head work_dirs/PUBLIC_PROXY_HRF/public_importance.csv
# Look at the job's stdout — it prints importance max/min ratio.
# Ratio > 50 → importance is skewed → WF will help.
# Ratio < 5  → importance is flat → WF won't help much.
```

---

## 4. PHASE B — DP mechanism (the *only* place real noise is added)

This phase is **bundled into the same script as Phase C** (student
training) because we use `--precompute-noise`. Below I describe what
the precompute pass does conceptually; §6 has the exact math.

For each (ε, noise_type, seed):

### B.1 — Pick the privacy budget

The `--epsilon` CLI argument is the **per-release ε**, which in the
public-proxy + sample-once threat model **equals the user-level ε**.

| Want user-level ε = | Pass `--epsilon` |
|---------------------|------------------|
| 2 | 2 |
| 8 | 8 |
| 16 | 16 |

(Compare: in the per-iteration release model, you'd need
`--epsilon 0.021` to get user-level ε = 2. Hence why we adopt the
sample-once model.)

### B.2 — Allocate σ_c across channels (water-filling)

`phase1_gkd_distill_v2.py` internally:
1. Loads caps from `public_caps.csv` → `\widehat{c} ∈ R^{1024}`
2. Loads importance from `public_importance.csv` → `\widehat{s} ∈ R^{1024}`
3. Converts ε → ρ via Bun-Steinke
4. For each channel: `σ_c = κ · √Δ_c / s_c^{1/4}` (the WF formula —
   see §6)

### B.3 — Precompute the noisy release

For each of the 20 DRIVE training images:
1. Forward through teacher → `z_i`
2. Clip + normalize per channel
3. Add Gaussian noise with stddev `σ_c` per channel
4. Denormalize
5. Cache the result keyed by image path

**This is the only place private DRIVE data sees noise.** After this
step finishes, the cached noisy features are the only thing the
student ever sees of the private data.

---

## 5. PHASE C — Student training (no DP work, post-processing)

Same script (`phase1_gkd_distill_v2.py`), same job. Once the
precompute pass is done, the training loop runs as usual:

For 40,000 iterations:
1. Sample a batch of (image_id, label) from DRIVE
2. Student forward → student bottleneck
3. Look up the cached noisy teacher bottleneck by image_id
   (no fresh noise! no fresh teacher query!)
4. Feature loss = MSE(adapter(student_bn), cached_noisy_target)
5. Task loss = CE + 3·Dice on the ground-truth mask  *(see §0.5.1 — open privacy question)*
6. Total = task + λ · feature
7. Backprop → update student + adapter only

By DP post-processing immunity, the **cached noisy features** are
(ε, δ)-DP and so is anything computed purely from them. The
*student model* trained with step 5 above also consumes the
**private GT mask**, so its DP status with respect to the labels is
a separate question (§0.5.1) — the ε we report covers the feature
release only.

### One-shot command for Phase B + C

```bash
# args: lambda  noise_type  epsilon  seed  precompute(yes|no)
sbatch run_lambda_sweep.sh 20 channel_WF 8 0 yes
```

Output goes to `work_dirs/GKD_LAMSWP_public-proxy_sOnce_l20_channel_WF_eps8_seed0/`.

### Full WACV matrix in one go

```bash
bash submit_all_wacv.sh
# Submits 60 jobs covering diagnostics + lambda sweep + sample-once.
# Wall time ~3 days at 5-10 parallel.
```

---

## 6. ★ HOW NOISE IS ADDED — full detail ★

This is the answer to "if I add noise, how do I add it." It happens in
**exactly one place**: lines 617–628 of `phase1_gkd_distill_v2.py`
(the precompute path). The math below matches the code line-by-line.

### 6.1 — Setup

Given:
- Image `x_i ∈ R^(3×H×W)`
- Teacher encoder `φ_T`
- Per-channel caps `\widehat{c} ∈ R^{1024}` (from public HRF)
- Per-channel importance `\widehat{s} ∈ R^{1024}` (from public HRF)
- Target user-level zCDP budget `ρ_total` (derived from `--epsilon`)

### 6.2 — Step-by-step mechanism

```
[Step 1]  Compute clean bottleneck
          z_i = φ_T(x_i)                      # shape (1024, H_b, W_b)

[Step 2]  Per-channel L2 clip
          For each channel c:
              if ‖z_{i,c}‖_2 > \widehat{c}_c:
                  z_{i,c}  ←  z_{i,c} · \widehat{c}_c / ‖z_{i,c}‖_2
          After this: ‖z_{i,c}‖_2  ≤  \widehat{c}_c  for every c.

[Step 3]  Normalize to unit-norm space
          z'_{i,c} = z_{i,c} / \widehat{c}_c   # ‖z'_{i,c}‖_2 ≤ 1

[Step 4]  Compute per-channel noise scales (water-filling)
          ρ_rel = ρ_total                                  (public-proxy: nothing else to pay for)
          Δ_c   = 2 / K                                    (K = #teachers; K=1 in current setup)
          κ     = √( Σ_c (Δ_c · √s_c) / ρ_rel )
          σ_c   = κ · √Δ_c / s_c^{1/4}                     # Theorem 1

[Step 5]  ★ ADD NOISE in the unit-norm space ★
          For each channel c, sample once per image i:
              η_{i,c} ∈ R^{H_b × W_b} with  η_{i,c}^{(h,w)} ~ N(0, σ_c^2)
              \tilde{z}'_{i,c} = z'_{i,c} + η_{i,c}

[Step 6]  De-normalize (post-processing, no privacy cost)
          \tilde{z}_{i,c} = \tilde{z}'_{i,c} · \widehat{c}_c

[Step 7]  Cache
          noisy_cache[ img_path(x_i) ] = \tilde{z}_i        # store on disk
```

### 6.3 — Same thing as code (literal Python)

```python
# 1. Clean bottleneck
with torch.no_grad():
    enc = unet_encoder(teacher.backbone, x_i)
    z = enc[-1]                                     # (B, 1024, H_b, W_b)

# 2 + 3. Clip + normalize to unit-norm space
z_norm = clip_and_normalise(z, caps)                # each channel norm ≤ 1

# 4. σ_c was computed once at the start of the job:
#    sigma = waterfilling_sigma(deltas, importance, rho)   # shape (1024,)

# 5. Sample fresh Gaussian noise per channel
noise = torch.randn_like(z_norm) * sigma.view(1, 1024, 1, 1)

# 6. Denormalize
z_noisy = denormalise(z_norm + noise, caps)         # back to original scale

# 7. Cache (one entry per image)
noisy_cache[img_id(x_i)] = z_noisy.detach().cpu()
```

### 6.4 — Why this gives (ε, δ)-DP

- After Step 3, every channel lies in the unit ball, so the replace-one
  L2 sensitivity per channel is `Δ_c = 2/K` (data-independent).
- Step 5 is the Gaussian mechanism with per-channel scale `σ_c`. Its
  per-release zCDP cost is `Σ_c Δ_c² / (2 σ_c²) = ρ_rel` (by Step 4's
  budget allocation).
- Each image's release uses only that image's data → **disjoint
  mechanisms** → **parallel composition** → the union of all releases
  is still `ρ_rel`-zCDP **per image**.
- **DRIVE has k = 1 image per patient** (§0.5), so per-image release
  collapses to per-patient release: `ρ_user = ρ_rel`. This identity is
  dataset-specific — MIMIC-CXR (k ≤ 4) needs extra composition for
  those patients.
- Caps and importance came from public HRF → `ρ_caps = ρ_imp = 0`.
- Total: `ρ_user = 0 + 0 + ρ_rel = ρ_total`.
- Convert back via Bun-Steinke: `ε_user = ε` (the value you passed via
  `--epsilon`).

### 6.5 — What changes if `noise_type` is different

| `--noise-type` | What happens at Step 4 |
|----------------|-------------------------|
| `channel_WF` | σ_c = κ · √Δ_c / s_c^{1/4} (above) |
| `uniform` | σ_c = √(Σ_c Δ_c² / ρ_rel), same for every channel |
| `none` | σ_c = 0 for all c (zero noise — diagnostic upper bound) |

Steps 5, 6, 7 are identical in all three cases (only the σ vector
differs).

---

## 7. PHASE D — Privacy accounting / paper tables

After Phase B+C runs finish, you have a bunch of `summary.json` files
in `work_dirs/GKD_LAMSWP_*/`. The privacy accounting confirms what the
user-level ε actually is:

```bash
python3 phase1_distillation/scripts/privacy_accounting.py
```

This reads all `summary.json` files, finds the unique `epsilon` values
used, and prints a table comparing user-level ε under:
- per-iteration release (prior work's implicit, wrong, model)
- sample-once
- **public-proxy + sample-once** (what we report)

The output table is what goes into **paper Table 1**.

For the **§3.4 budget-split ablation**, run:

```bash
python3 phase1_distillation/scripts/budget_split_analysis.py --total-eps 8
```

This is purely analytical (no GPU, runs in seconds) and outputs **paper
Table 2**.

---

## 8. End-to-end submission recipe (do this today)

Assuming you're starting fresh on the cluster:

```bash
# --- one-time setup ---
mkdir -p ~/public_retinal/HRF
cd ~/public_retinal/HRF
wget https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/all.zip
unzip all.zip                                                  # 45 .jpg files

cd /data/user/home/ialam/slurm_scripts

# Step 1: Public proxy (~30 min)
sbatch run_compute_public_proxy.sh HRF

# Wait for the job to finish. Verify:
ls /data/user/home/ialam/mmsegmentation/work_dirs/PUBLIC_PROXY_HRF/
# Should see: public_caps.csv  public_importance.csv

# Step 2: Submit the full experiment matrix (~3 days wall time)
bash submit_all_wacv.sh

# Step 3 (after results land): privacy accounting + budget split
python3 /data/user/home/ialam/mmsegmentation/tools/analysis/privacy_accounting.py
python3 /data/user/home/ialam/mmsegmentation/tools/analysis/budget_split_analysis.py --total-eps 8
```

---

## 9. The mental cheat sheet

### Three architectural facts

1. **Noise is added exactly once per training image, in Step 5 of §6**
   — that is the only DP-relevant operation in the entire pipeline.

2. **Everything else is either public (Phase A) or post-processing
   (Phase C)**. Public ops cost 0 privacy. Post-processing ops cost 0
   privacy (DP immunity theorem).

3. **The reported ε IS the user-level ε** (no composition blow-up),
   because (a) caps + importance came from public data, (b) each
   training image is released exactly once, **and (c) DRIVE has
   k = 1 image per patient** so per-image = per-patient. (c) is
   dataset-specific (§0.5). Note: this ε is for the released
   *features* only — the student's task loss on private GT labels is
   a separate, currently-unaccounted data-dependent path (§0.5.1).

### Two conceptual confusions, resolved

4. **σ (scale) vs η (sample) — they are different things.**
   - `σ_c` is the standard deviation that defines the Gaussian noise
     distribution. It is **computed once** per job and is **the same
     for every image** within that job (same channel → same σ_c).
   - `η_{i,c}` is the actual noise draw applied to image `i`, channel
     `c`. It is **resampled i.i.d. for every image and every (h,w)
     pixel** within that channel. The scale stays fixed; the random
     draw changes.
   - Across jobs: σ changes when `--epsilon` changes (smaller ε →
     bigger σ). σ does NOT change when only `--seed` changes; only the
     `η` draws change.
   - Analogy: a scale with ±0.1 g precision (σ) gives different
     readings (+0.05, −0.07, +0.12 …) every time you weigh something
     (η). The instrument's precision is fixed; each reading differs.

5. **The final ε is computed, not measured.**
   - The number reported as "user-level ε" comes from **plugging your
     chosen σ into the Bun-Steinke formula**, not from any experiment
     on the data. It is a worst-case mathematical bound, guaranteed by
     the Gaussian-mechanism theorem + parallel composition over
     disjoint per-image releases.
   - `privacy_accounting.py` does NOT measure ε — it RE-DERIVES it
     analytically and prints comparison tables under different threat
     models. The numbers in paper Table 1 are theorems, not
     measurements.
   - What CAN be measured empirically: utility (mDice) and the success
     of a membership-inference attack (which gives a **lower bound**
     on actual privacy leakage — useful as a sanity audit but not the
     formal privacy claim).
   - Implication for the paper: never report ε and mDice in the same
     sentence as if they had the same epistemic status. ε is a
     theorem; mDice is a measurement.

---

## 10. When something breaks — debugging

| Symptom | Probable cause | Fix |
|---------|----------------|-----|
| `--public-caps-csv` required | Phase A.3 wasn't run | Run `run_compute_public_proxy.sh HRF` first |
| Smoke test fails on new code | You changed something | `python3 smoke_test_diagnostics.py` to see which assertion broke |
| All mDice ≈ baseline regardless of ε | distillation pipeline inert (see earlier diagnostics) | Run `run_diag_A_lambda0.sh` to confirm |
| Cluster job OOM | adapter or bottleneck too big | Reduce `--max-iters` to 10000 for a smoke run |
| Importance ratio < 5 | Public HRF importance is too flat | Try CHASE-DB1 instead; or paper falls back to uniform |
