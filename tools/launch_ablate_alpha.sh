#!/usr/bin/env bash
# Launch Tier 3 alpha channel ablation inference on gpu2.
# Runs 6 variants (E6-E11) as 8-way parallel inference each.
# Uses Path B P1 iter500 checkpoint.
# Usage: bash tools/launch_ablate_alpha.sh

set -euo pipefail

PROJ_DIR="/public/home/maoyaoxin/zhangt/xxt/NeuroChrono-v1"
INFER_BASE="configs/sf_v1/cinebrain_sf_v3_pathB_model.yaml"
INFER_ARGS="configs/sf_v1/infer_pathB_p1_iter500.yaml"
PYTHON="/public/home/maoyaoxin/anaconda3/envs/cinebrain/bin/python"

# Tier 3 variants
declare -A VARIANTS
VARIANTS[E6_nokey]="configs/sf_v1/ablate_alpha_E6_nokey.yaml"
VARIANTS[E7_notxt]="configs/sf_v1/ablate_alpha_E7_notxt.yaml"
VARIANTS[E8_nomot]="configs/sf_v1/ablate_alpha_E8_nomot.yaml"
VARIANTS[E9_nobrain]="configs/sf_v1/ablate_alpha_E9_nobrain.yaml"
VARIANTS[E10_prior_only]="configs/sf_v1/ablate_alpha_E10_prior_only.yaml"
VARIANTS[E11_all_off]="configs/sf_v1/ablate_alpha_E11_all_off.yaml"

echo "============================================"
echo " Tier 3: Alpha Channel Ablation Inference"
echo " Checkpoint: iter500"
echo " Samples: 540 per variant"
echo "============================================"

for NAME in "${!VARIANTS[@]}"; do
    OVERRIDE="${VARIANTS[$NAME]}"
    OUTDIR="${PROJ_DIR}/results/ablate_alpha/${NAME}"
    LOGDIR="${PROJ_DIR}/logs"

    echo ""
    echo "[$(date)] Launching ${NAME}..."
    echo "  Override: ${OVERRIDE}"
    echo "  Output:   ${OUTDIR}"

    ssh ts3 "ssh gpu2 'cd ${PROJ_DIR} && \
      rm -rf ${OUTDIR} && mkdir -p ${OUTDIR} ${LOGDIR} && \
      : > ${LOGDIR}/ablate_${NAME}.pids'"

    for gpu in 0 1 2 3 4 5 6 7; do
        split=${gpu}
        log="${LOGDIR}/ablate_${NAME}_gpu${gpu}.log"
        port=$((29840 + gpu))
        jsonpath="/public/home/maoyaoxin/zhangt/xxt/datasets/full540_8split${split}.json"

        ssh ts3 "ssh gpu2 'cd ${PROJ_DIR} && \
          CUDA_VISIBLE_DEVICES=${gpu} \
          nohup ${PYTHON} -m torch.distributed.run \
              --standalone --nproc_per_node=1 --master_port=${port} \
              sample_brain_va.py \
              --base ${INFER_BASE} ${INFER_ARGS} ${OVERRIDE} \
              --seed 42 --jsonpath ${jsonpath} --output_dir ${OUTDIR} \
              > ${log} 2>&1 &'"

        echo "  ${NAME} gpu${gpu} split${split} launched"
    done
    echo "[$(date)] ${NAME} 8-way inference launched"
    sleep 3
done

echo ""
echo "============================================"
echo " All Tier 3 jobs launched."
echo " Monitor progress:"
for NAME in "${!VARIANTS[@]}"; do
    echo "  ssh ts3 \"ssh gpu2 'ls ${PROJ_DIR}/results/ablate_alpha/${NAME}/*.mp4 2>/dev/null | wc -l'\"  # ${NAME}"
done
echo "============================================"
