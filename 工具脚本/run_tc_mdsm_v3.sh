#!/usr/bin/env bash
# 在项目根目录按四阶段运行，参数传给每个阶段。
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${TC_MDSM_GPU:-0}"
for stage in generate train evaluate report; do
    python 3.实验评估/tc_mdsm_v3_runner.py "$stage" "$@"
done
