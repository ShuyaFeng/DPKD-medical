#!/bin/bash
# Submit all 6 supervisor strategy experiments across 3 datasets (18 jobs total).
#
# Strategies covered:
#   1. canal_random_selection   — random vs importance-based channel selection (PRIORITY)
#   2. canal_lambda_sweep       — feature loss weight sweep (10%–91% of total loss)
#   3. canal_importance_scaling — power-transform importance + reverse CANAL sanity check
#   4. canal_equal_base         — teacher base=16, student base=16 (no adapter mismatch)
#   5. canal_more_teachers      — K=3, 5, 7 teachers
#   6. canal_layer_ablation     — distill at e1, e2, or e3

set -e
cd "$(dirname "$0")"

echo "==========================="
echo "Submitting supervisor strategy jobs"
echo "Date: $(date)"
echo "==========================="

# 1. Random selection (PRIORITY)
echo "--- Exp 1: Random vs importance selection ---"
J1A=$(sbatch --parsable run_rand_sel_isic.sh);   echo "  ISIC    JID=$J1A"
J1B=$(sbatch --parsable run_rand_sel_kvasir.sh); echo "  Kvasir  JID=$J1B"
J1C=$(sbatch --parsable run_rand_sel_busi.sh);   echo "  BUSI    JID=$J1C"

# 2. Lambda sweep
echo "--- Exp 2: Lambda (feature loss weight) sweep ---"
J2A=$(sbatch --parsable run_lambda_sweep_isic.sh);   echo "  ISIC    JID=$J2A"
J2B=$(sbatch --parsable run_lambda_sweep_kvasir.sh); echo "  Kvasir  JID=$J2B"
J2C=$(sbatch --parsable run_lambda_sweep_busi.sh);   echo "  BUSI    JID=$J2C"

# 3. Importance scaling + reverse CANAL
echo "--- Exp 3: Importance scaling + reverse CANAL ---"
J3A=$(sbatch --parsable run_imp_scale_isic.sh);   echo "  ISIC    JID=$J3A"
J3B=$(sbatch --parsable run_imp_scale_kvasir.sh); echo "  Kvasir  JID=$J3B"
J3C=$(sbatch --parsable run_imp_scale_busi.sh);   echo "  BUSI    JID=$J3C"

# 4. Equal base
echo "--- Exp 4: Equal base (teacher=16, student=16) ---"
J4A=$(sbatch --parsable run_equal_base_isic.sh);   echo "  ISIC    JID=$J4A"
J4B=$(sbatch --parsable run_equal_base_kvasir.sh); echo "  Kvasir  JID=$J4B"
J4C=$(sbatch --parsable run_equal_base_busi.sh);   echo "  BUSI    JID=$J4C"

# 5. More teachers
echo "--- Exp 5: More teachers (K=3,5,7) ---"
J5A=$(sbatch --parsable run_more_teachers_isic.sh);   echo "  ISIC    JID=$J5A"
J5B=$(sbatch --parsable run_more_teachers_kvasir.sh); echo "  Kvasir  JID=$J5B"
J5C=$(sbatch --parsable run_more_teachers_busi.sh);   echo "  BUSI    JID=$J5C"

# 6. Layer ablation
echo "--- Exp 6: Layer ablation (e1, e2, e3) ---"
J6A=$(sbatch --parsable run_layer_abl_isic.sh);   echo "  ISIC    JID=$J6A"
J6B=$(sbatch --parsable run_layer_abl_kvasir.sh); echo "  Kvasir  JID=$J6B"
J6C=$(sbatch --parsable run_layer_abl_busi.sh);   echo "  BUSI    JID=$J6C"

echo ""
echo "==========================="
echo "All 18 jobs submitted."
echo "Monitor: squeue -u \$USER"
echo ""
echo "Expected output files (in phase1_distillation/results/):"
echo "  {isic,kvasir,busi}_random_selection_results.json"
echo "  {isic,kvasir,busi}_lambda_sweep_results.json"
echo "  {isic,kvasir,busi}_importance_scaling_results.json"
echo "  {isic,kvasir,busi}_equal_base_results.json"
echo "  {isic,kvasir,busi}_more_teachers_results.json"
echo "  {isic,kvasir,busi}_layer_ablation_results.json"
echo "==========================="
