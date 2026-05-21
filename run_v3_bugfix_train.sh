#!/bin/bash
cd /public/home/maoyaoxin/zhangt/xxt/SF-v3
export CUDA_HOME=/usr/local/cuda-12.4
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_TIMEOUT=3600

/public/home/maoyaoxin/anaconda3/envs/cinebrain/bin/python   -m torch.distributed.run --standalone --nproc_per_node=4   train_video_fmri.py   --base configs/sf_v1/cinebrain_sf_v1_model.yaml configs/sf_v1/sf_v3_stage3_bugfix_resume.yaml   --seed 42   > logs/sf_v3_bugfix_train.log 2>&1
