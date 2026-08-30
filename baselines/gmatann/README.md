# GMATANN 单子任务分配 baseline

本目录复现论文《Heterogeneous Multiagent Task Allocation Based on Graph-Based Convolutional Assignment Neural Network》（IEEE Internet of Things Journal, 2025, DOI: `10.1109/JIOT.2025.3535641`）中强化学习之前的完整表征链路，并适配基于 LLM 的多智能体任务分配。

## 已实现范围

- 当前任务与候选智能体组成异构 `task-agent` 图；任务是独立节点。
- 事件窗口按时间重放，形成每个候选智能体的状态序列，不再只取最后一条记录。
- 时序条件变分自编码器：双向 LSTM 编码、条件化后验、Gaussian 重参数化、条件 LSTM 解码、掩码重构损失和 KL 损失。
- task→agent、agent→task 和 agent→agent 三类关系注意力；边状态显式参与 key、value 和 attention bias。
- 按论文式(2)–(5)分别实现局部注意力和全局注意力，保留可学习的约束系数 `tau`、原特征残差和深层融合。
- 智能体状态/能力文本与任务描述/Skill/QoS 使用类型专属嵌入；必需 Skill 作为最终硬约束。
- 返回图注意力、局部/全局注意力、融合权重、CVAE 均值/方差与重构序列，便于解释和消融。

强化学习决策部分未实现。复杂任务 DAG 也不进入该 baseline；每个当前子任务独立构图与调用，作为与仓库主版本任务节点方法进行对比的单子任务基线。

详细的论文对应关系、公式落地和复现假设见 [`IMPLEMENTATION.md`](IMPLEMENTATION.md)。

## 安装与测试

从仓库根目录执行：

```powershell
python -m pip install -r requirements.txt
python -m pytest "baselines/gmatann/tests" -q
```

## 训练与评估

```powershell
python "baselines/gmatann/train.py" --dataset "1.data/datasets/sixg_ma_state" --epochs 10 --output "outputs/gcan/model.pt"
python "baselines/gmatann/evaluate.py" --dataset "1.data/datasets/sixg_ma_state" --checkpoint "outputs/gcan/model.pt" --reliability "1.data/4.智能体可靠性/agent_reliability.csv" --output "outputs/gcan/scenario_metrics.json"
```

由于模型结构已经补全，旧版 `outputs/model.pt` 与新结构不兼容，需要重新训练。

统一训练基线和主模型并生成对比图的入口位于：

```powershell
python "experiments/compare_allocation_models.py" --epochs 10 --no-include-v1
```

如需同时包含归档的 V1，请按仓库根目录 README 中的说明创建 V1 worktree。
