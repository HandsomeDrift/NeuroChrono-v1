#!/usr/bin/env bash
# Evaluate Tier 3 alpha channel ablation results with full 14-metric eval.
# Runs after all inference is complete.
# Usage: bash tools/launch_eval_ablate_alpha.sh

set -euo pipefail

PROJ_DIR="/public/home/maoyaoxin/zhangt/xxt/NeuroChrono-v1"
GT_JSON="/public/home/maoyaoxin/zhangt/xxt/datasets/full540_gt.json"
PYTHON="/public/home/maoyaoxin/anaconda3/envs/cinebrain/bin/python"

RESULT_DIR="${PROJ_DIR}/results/ablate_alpha"
OUTPUT="${PROJ_DIR}/results/ablate_alpha_metrics.json"

VARIANTS=("E6_nokey" "E7_notxt" "E8_nomot" "E9_nobrain" "E10_prior_only" "E11_all_off")

# Build --result-dir args
RESULT_ARGS=""
for NAME in "${VARIANTS[@]}"; do
    RESULT_ARGS="${RESULT_ARGS} --result-dir ${NAME}=${RESULT_DIR}/${NAME}"
done

echo "[$(date)] Running full 14-metric eval on Tier 3 results..."
echo "  Variants: ${VARIANTS[*]}"
echo "  Output: ${OUTPUT}"

ssh ts3 "ssh gpu2 'cd ${PROJ_DIR} && \
  CUDA_VISIBLE_DEVICES=0 \
  ${PYTHON} tools/eval_full14.py \
    --gt-jsonpath ${GT_JSON} \
    ${RESULT_ARGS} \
    --output ${OUTPUT}'"

echo "[$(date)] Eval complete. Results: ${OUTPUT}"
