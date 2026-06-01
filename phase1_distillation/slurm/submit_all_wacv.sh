#!/bin/bash
# ============================================================
# MASTER SUBMIT — all WACV-deadline critical-path jobs.
#
# Submits ~48 jobs in priority order. The queue will process
# 5-10 in parallel; expected wall-clock: ~3 days.
#
# Run from the slurm/ directory on the cluster.
# ============================================================

set -e

cd "$(dirname "$0")"

# ----------------------------
# PRIORITY 1 — Diagnostics (decides framing). 12 jobs.
# ----------------------------
echo "Submitting diagnostics (12 jobs)..."

for SEED in 0 1 2; do
    sbatch run_diag_A_lambda0.sh $SEED
done

for COND in huge none; do
    for SEED in 0 1 2; do
        sbatch run_diag_B.sh $COND $SEED
    done
done

for SEED in 0 1 2; do
    sbatch run_diag_C_scratch.sh $SEED
done

# ----------------------------
# PRIORITY 2 — eps=2 lambda sweep (the most important regime
# for Framing 1 — tight privacy is where channel-WF should win).
# 18 jobs: 3 lambdas × 2 noise × 3 seeds.
# ----------------------------
echo "Submitting eps=2 lambda sweep (18 jobs)..."

for LAM in 5 20 50; do
    for NT in uniform channel_WF; do
        for SEED in 0 1 2; do
            sbatch run_lambda_sweep.sh $LAM $NT 2 $SEED
        done
    done
done

# ----------------------------
# PRIORITY 3 — eps=16 sanity (gap should shrink at high eps).
# 6 jobs: 1 lambda × 2 noise × 3 seeds.
# ----------------------------
echo "Submitting eps=16 lambda=20 sanity (6 jobs)..."

for NT in uniform channel_WF; do
    for SEED in 0 1 2; do
        sbatch run_lambda_sweep.sh 20 $NT 16 $SEED
    done
done

# ----------------------------
# PRIORITY 4 — eps=8 lambda=20 (mid-eps, optional anchor point).
# 6 jobs. Per-iteration release (default).
# ----------------------------
echo "Submitting eps=8 lambda=20 anchor (6 jobs)..."

for NT in uniform channel_WF; do
    for SEED in 0 1 2; do
        sbatch run_lambda_sweep.sh 20 $NT 8 $SEED no
    done
done

# ----------------------------
# PRIORITY 5 — SAMPLE-ONCE-PER-IMAGE threat model (paper §3).
# This is the threat model we adopt in the paper. Reported epsilon
# now equals true user-level epsilon (no T-round composition blowup).
# Mirrors priority 2/3/4 setup but with --precompute-noise.
# 18 jobs: 3 eps × 2 noise × 3 seeds @ lambda=20.
# ----------------------------
echo "Submitting sample-once threat model (18 jobs)..."

for EPS in 2 8 16; do
    for NT in uniform channel_WF; do
        for SEED in 0 1 2; do
            sbatch run_lambda_sweep.sh 20 $NT $EPS $SEED yes
        done
    done
done

echo ""
echo "All submitted. Total: 60 jobs."
echo "  - 12 diagnostics (A/B/C)"
echo "  - 30 lambda sweep, per-iteration release"
echo "  - 18 sample-once threat model"
echo "Monitor with: squeue -u \$USER"
