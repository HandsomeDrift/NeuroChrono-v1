#!/usr/bin/env bash
# Launch all 6 Tier 3 alpha channel ablation variants in parallel.
# One variant per GPU (GPUs 0-5), each processes all 540 samples.
# Total: ~40-45h for all 6 variants.
# Usage: bash tools/launch_ablate_alpha_all.sh

set -euo pipefail

PROJ="/public/home/maoyaoxin/zhangt/xxt/NeuroChrono-v1"
PY="/public/home/maoyaoxin/anaconda3/envs/cinebrain/bin/python"
BASE="configs/sf_v1/cinebrain_sf_v3_pathB_model.yaml"
ARGS="configs/sf_v1/infer_pathB_p1_iter500.yaml"
JSON="/public/home/maoyaoxin/zhangt/xxt/datasets/full540.json"

declare -A NAMES
NAMES[0]="E6_nokey"
NAMES[1]="E7_notxt"
NAMES[2]="E8_nomot"
NAMES[3]="E9_nobrain"
NAMES[4]="E10_prior_only"
NAMES[5]="E11_all_off"

declare -A OVERRIDES
OVERRIDES[0]="configs/sf_v1/ablate_alpha_E6_nokey.yaml"
OVERRIDES[1]="configs/sf_v1/ablate_alpha_E7_notxt.yaml"
OVERRIDES[2]="configs/sf_v1/ablate_alpha_E8_nomot.yaml"
OVERRIDES[3]="configs/sf_v1/ablate_alpha_E9_nobrain.yaml"
OVERRIDES[4]="configs/sf_v1/ablate_alpha_E10_prior_only.yaml"
OVERRIDES[5]="configs/sf_v1/ablate_alpha_E11_all_off.yaml"

echo "============================================"
echo " Tier 3: All 6 variants in parallel"
echo " GPUs: 0-5, 540 samples each"
echo " ETA: ~42h"
echo "============================================"

for gpu in 0 1 2 3 4 5; do
    NAME="${NAMES[$gpu]}"
    OVERRIDE="${OVERRIDES[$gpu]}"
    OUTDIR="${PROJ}/results/ablate_alpha/${NAME}"
    LOG="${PROJ}/logs/ablate_${NAME}_gpu${gpu}.log"
    PORT=$((29850 + gpu))

    echo "[$(date)] ${NAME} → GPU ${gpu}"

    ssh ts3 "ssh gpu2 'mkdir -p ${OUTDIR} && rm -rf ${OUTDIR}/* && \
      cd ${PROJ} && \
      CUDA_VISIBLE_DEVICES=${gpu} \
      nohup ${PY} -m torch.distributed.run \
          --standalone --nproc_per_node=1 --master_port=${PORT} \
          sample_brain_va.py \
          --base ${BASE} ${ARGS} ${OVERRIDE} \
          --seed 42 --jsonpath ${JSON} --output_dir ${OUTDIR} \
          > ${LOG} 2>&1 &'"

    echo "  ${NAME}: log=${LOG}"
done

echo ""
echo "[$(date)] All 6 launched."
echo "Monitor:"
for gpu in 0 1 2 3 4 5; do
    NAME="${NAMES[$gpu]}"
    echo "  ssh ts3 \"ssh gpu2 'ls ${PROJ}/results/ablate_alpha/${NAME}/*.mp4 2>/dev/null | wc -l'\"  # ${NAME} (expect 540)"
done
