#!/bin/bash
cd /public/home/maoyaoxin/zhangt/xxt/SF-v3
export CUDA_HOME=/usr/local/cuda-12.4
export CUDA_VISIBLE_DEVICES=6

/public/home/maoyaoxin/anaconda3/envs/cinebrain/bin/python   sample_brain_va.py   --base configs/sf_v1/cinebrain_sf_v1_model.yaml configs/sf_v1/infer_stage3_v2.yaml   --seed 42   --jsonpath /public/home/maoyaoxin/zhangt/xxt/datasets/sub-0005_test_va_mini20.json   --output_dir results/v3_verify_v2_mini20   > logs/v3_verify_v2_mini20.log 2>&1
