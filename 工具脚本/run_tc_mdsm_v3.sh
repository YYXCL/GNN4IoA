#!/usr/bin/env bash
# TC-MDSM v3 全流程一键脚本：依次执行 数据生成 → 训练 → 冻结 → 评估 → 报告。
# 用法：在已安装依赖（PyTorch/OR-Tools 等）的环境中直接运行本脚本。

# 关键报错即退出：-e 出错停止，-u 使用未定义变量报错，-o pipefail 管道任一环节失败即失败
set -euo pipefail

# 切换到仓库根目录（本脚本位于 工具脚本/ 下，其上一级目录即仓库根）
cd "$(dirname "$0")/.."

# 指定使用的 GPU 编号；可用环境变量 TC_MDSM_GPU 覆盖，默认使用 0 号卡
export CUDA_VISIBLE_DEVICES="${TC_MDSM_GPU:-0}"

# 1) 生成数据集（写入运行目录下的 数据/）
python 4.实验评估/tc_mdsm_v3_runner.py generate
# 2) 在全部随机种子上训练各方法（使用 GPU）
python 4.实验评估/tc_mdsm_v3_runner.py train --device cuda
# 3) 冻结实验协议：记录数据/代码/检查点的哈希，后续评估只认这份协议
python 4.实验评估/tc_mdsm_v3_runner.py freeze
# 4) 在冻结协议下逐实例执行评估（使用 GPU）
python 4.实验评估/tc_mdsm_v3_runner.py evaluate --device cuda
# 5) 汇总结果并生成图表
python 4.实验评估/tc_mdsm_v3_runner.py report
