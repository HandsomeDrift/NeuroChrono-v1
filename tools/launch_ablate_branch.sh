#!/usr/bin/env bash
# Launch Tier 1 architecture ablation training on gpu2.
# Usage: bash tools/launch_ablate_branch.sh [slow|fast]
#
#   slow  → E1: w/o Slow Branch (fast-only, active_branches="fast_only")
#   fast  → E2: w/o Fast Branch (slow-only, active_branches="slow_only")
#
# Training: 2000 iter, 4 GPU, ~9h per experiment.
# Checkpoints saved every 500 iter.

set -euo pipefail

ABLATE="${1:?Usage: $0 [slow|fast]}"

if [ "$ABLATE" = "slow" ]; then
    NAME="ablate_slow"
    OVERRIDE="configs/sf_v1/ablate_slow_branch.yaml"
elif [ "$ABLATE" = "fast" ]; then
    NAME="ablate_fast"
    OVERRIDE="configs/sf_v1/ablate_fast_branch.yaml"
else
    echo "Usage: $0 [slow|fast]"
    exit 1
fi

PROJ_DIR="/public/home/maoyaoxin/zhangt/xxt/NeuroChrono-v1"
LOG="${PROJ_DIR}/logs/${NAME}.log"

echo "[$(date)] Launching ${NAME} training on gpu2..."
echo "  Override: ${OVERRIDE}"
echo "  Log: ${LOG}"

ssh ts3 "ssh gpu2 'cd ${PROJ_DIR} && \
  CUDA_HOME=/usr/local/cuda-12.4 CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_TIMEOUT=3600 \
  nohup /public/home/maoyaoxin/anaconda3/envs/cinebrain/bin/python \
  -m torch.distributed.run --standalone --nproc_per_node=4 \
  train_video_fmri.py \
  --base configs/sf_v1/cinebrain_sf_v3_pathB_model.yaml \
         configs/sf_v1/sf_v3_pathB_train.yaml \
         ${OVERRIDE} \
  --seed 42 \
  > ${LOG} 2>&1 &'"

echo "[$(date)] Launched. Monitor progress:"
echo "  ssh ts3 'ssh gpu2 \"tail -f ${LOG}\"'"
echo "  ssh ts3 'ssh gpu2 \"ls ${PROJ_DIR}/ckpts_5b/${NAME}*/checkpoints/\"'"
