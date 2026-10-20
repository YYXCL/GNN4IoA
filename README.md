# TC-MDSM

面向异构智能体的任务分配与调度实验：状态编码 → 候选检索 → 图预测 → CP-SAT 调度 → 动态仿真 → 结果绘图。

## 安装与运行

需要 Python 3.9+。先安装适合运行平台的 PyTorch，再在项目根目录执行：

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
python 3.实验评估/tc_mdsm_paper_runner.py --describe
python 3.实验评估/tc_mdsm_paper_runner.py all --device auto
```

支持单独运行 `generate`、`train`、`evaluate`、`report`；续跑使用原参数并加 `--output 原输出目录 --resume`。完整默认实验耗时较长，先用 `--describe` 查看设置。

## 目录与结果

- `1.数据`：本地基础输入及数据处理脚本。
- `2.模型/tc_mdsm`：模型、训练、检索、调度和执行。
- `3.实验评估`：实验入口与真实结果绘图。
- `4.结果输出`：参数、样本、权重、逐请求结果和四幅图。

当前实现包含 TC-MDSM、GMATANN、HM-MATAS（图例显示 TARGETNET）、Rule-Latency、Full-State、StaticGraph；学习基线为本任务的结构适配。当前代码尚未完整对齐论文：可靠性硬约束、失败重试、候选扩展及部分实验设置仍待补齐。不能将排版参考图当作实测结果。

完成率包含拒绝、失败和迟到；当前平均时延只统计按期成功请求。不同方法使用相同场景和环境种子，运行参数与逐请求结果保存在输出目录。

GitHub 更新不包含本地环境、新增实验数据、权重、结果及理想趋势脚本；分支历史中已有的数据文件不会因忽略规则自动移除。基础输入需在本地准备，见 `1.数据/README.md`。
