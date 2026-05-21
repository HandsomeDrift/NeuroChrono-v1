#!/bin/bash
cd /public/home/maoyaoxin/zhangt/xxt/SF-v3
export CUDA_HOME=/usr/local/cuda-12.4
export CUDA_VISIBLE_DEVICES=0

/public/home/maoyaoxin/anaconda3/envs/cinebrain/bin/python   eval_mini20_compare.py   results/v3_bugfix_iter2500_mini20   /public/home/maoyaoxin/zhangt/xxt/SF-v1/CineBrain/results/stage3_v2_sub05   /public/home/maoyaoxin/zhangt/xxt/datasets/sub-0005_test_va_mini20.json   > logs/eval_mini20_compare.log 2>&1
