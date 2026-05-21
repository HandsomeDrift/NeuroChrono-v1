#!/usr/bin/env bash
# Launch Tier 3 alpha channel ablation inference on gpu2 GPUs 5,6,7.
# Batch 1 (E6-E8): 3 variants in parallel, ~13.5h
# Batch 2 (E9-E11): run after batch 1 completes
# Usage:
#   bash tools/launch_ablate_alpha_gpu5-7.sh 1    # Batch 1: E6, E7, E8
#   bash tools/launch_ablate_alpha_gpu5-7.sh 2    # Batch 2: E9, E10, E11 (after batch 1)

set -euo pipefail

BATCH="${1:?Usage: $0 [1|2]}"

PROJ_DIR="/public/home/maoyaoxin/zhangt/xxt/NeuroChrono-v1"
INFER_BASE="configs/sf_v1/cinebrain_sf_v3_pathB_model.yaml"
INFER_ARGS="configs/sf_v1/infer_pathB_p1_iter500.yaml"
PYTHON="/public/home/maoyaoxin/anaconda3/envs/cinebrain/bin/python"

if [ "$BATCH" = "1" ]; then
    NAMES=("E6_nokey" "E7_notxt" "E8_nomot")
    OVERRIDES=(
        "configs/sf_v1/ablate_alpha_E6_nokey.yaml"
        "configs/sf_v1/ablate_alpha_E7_notxt.yaml"
        "configs/sf_v1/ablate_alpha_E8_nomot.yaml"
    )
elif [ "$BATCH" = "2" ]; then
    NAMES=("E9_nobrain" "E10_prior_only" "E11_all_off")
    OVERRIDES=(
        "configs/sf_v1/ablate_alpha_E9_nobrain.yaml"
        "configs/sf_v1/ablate_alpha_E10_prior_only.yaml"
        "configs/sf_v1/ablate_alpha_E11_all_off.yaml"
    )
else
    echo "Usage: $0 [1|2]"
    exit 1
fi

GPUS=(5 6 7)

echo "============================================"
echo " Tier 3 Batch ${BATCH}: ${NAMES[*]}"
echo " GPUs: ${GPUS[*]} (180 samples each)"
echo "============================================"

for i in 0 1 2; do
    GPU="${GPUS[$i]}"
    NAME="${NAMES[$i]}"
    OVERRIDE="${OVERRIDES[$i]}"
    OUTDIR="${PROJ_DIR}/results/ablate_alpha/${NAME}"
    LOG="${PROJ_DIR}/logs/ablate_${NAME}_gpu${GPU}.log"
    JSONPATH="/public/home/maoyaoxin/zhangt/xxt/datasets/full540_3split${i}.json"
    PORT=$((29840 + GPU))

    echo "[$(date)] ${NAME} → GPU ${GPU} (split ${i})"

    ssh ts3 "ssh gpu2 'mkdir -p ${OUTDIR} && rm -rf ${OUTDIR}/* && \
      cd ${PROJ_DIR} && \
      CUDA_VISIBLE_DEVICES=${GPU} \
      nohup ${PYTHON} -m torch.distributed.run \
          --standalone --nproc_per_node=1 --master_port=${PORT} \
          sample_brain_va.py \
          --base ${INFER_BASE} ${INFER_ARGS} ${OVERRIDE} \
          --seed 42 --jsonpath ${JSONPATH} --output_dir ${OUTDIR} \
          > ${LOG} 2>&1 &'"

    echo "  Log: ${LOG}"
done

echo ""
echo "[$(date)] Batch ${BATCH} launched."
echo ""
echo "Monitor:"
for i in 0 1 2; do
    NAME="${NAMES[$i]}"
    echo "  ssh ts3 \"ssh gpu2 'ls ${PROJ_DIR}/results/ablate_alpha/${NAME}/*.mp4 2>/dev/null | wc -l'\"  # ${NAME} (expect 180)"
done
echo ""
if [ "$BATCH" = "1" ]; then
    echo "After batch 1 completes (~13.5h), run: bash tools/launch_ablate_alpha_gpu5-7.sh 2"
fi
