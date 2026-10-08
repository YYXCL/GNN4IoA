#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${TC_MDSM_GPU:-0}"
python 4.实验评估/tc_mdsm_v3_runner.py generate
python 4.实验评估/tc_mdsm_v3_runner.py train --device cuda
python 4.实验评估/tc_mdsm_v3_runner.py freeze
python 4.实验评估/tc_mdsm_v3_runner.py evaluate --device cuda
python 4.实验评估/tc_mdsm_v3_runner.py report
