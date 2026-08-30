# V3 三类节点协同分配模型与 Baseline

## 1. 事件、置信度和趋势

事件序列在训练和推理中承担相同角色：按时间顺序把最新节点/边状态送入 GRU。训练阶段学习 GRU 和注意力参数；部署阶段使用这些已训练参数更新隐藏状态，不需要为每条新事件重新训练模型。

旧版曾使用

`effective_confidence = observed_confidence * exp(-age / tau)`

其中基础置信度主要是生成器设置的 0.88–0.97，并非由真实标注估计，因此当前主模型不再使用它。当前也没有手工构造“短期趋势”数值；变化方向和幅度由 GRU 从连续状态中学习。旧实现已隔离到 `ma_tgn/v1/`，默认入口不会导入。V2 的共享 `MLP` 和分段 softmax 位于 `ma_tgn/v2/layers.py`。

## 2. 为什么选择 GRU

当前窗口只有 16 个源时间戳、训练样本为 54 个。GRU 比 LSTM 少一组门和一份细胞状态，参数更少、训练更稳定，也足以处理这种短到中等长度的状态序列。若以后窗口扩展到数百或数千步，再比较 LSTM、TCN 或时序 Transformer 更合适。

节点和边分别更新：

`h_node(t) = GRU_node(x_node(t), h_node(t-1))`

`h_edge(t) = GRU_edge(x_edge(t), h_edge(t-1))`

这里不输入置信度或人工趋势。

## 3. 三类节点和分配过程

节点类型为智能体、算力节点和瞬时任务节点。每个候选智能体先执行一层邻居聚合，形成五组特征：自身历史、智能体邻居、固定算力节点、通信边历史、未结束任务。

对于子任务 `s` 和候选智能体 `a`，先在五组特征上计算任务相关权重：

`beta(s,a,:) = softmax(q_task(s) · k_feature(a,:))`

再用加权特征计算任务—智能体注意力，并以精确 Skill id 做硬掩码。不同任务的 `q_task` 不同，因此会学习不同的特征权重。

每个复杂任务输出：

- 每个子任务到全部候选智能体的注意力矩阵；
- 每个子任务的最高分智能体；
- 按子任务顺序逐步累加的智能体注意力；
- 最终复杂任务的智能体总注意力和去重参与者集合。

## 4. Baseline

四个新增 baseline 已和 `gmatann/` 一样拆成独立目录，并各自提供名称、模型、训练/评估入口和 README：

1. `baselines/rule_based_latency/`（Rule-Latency）：先精确 Skill 匹配，再按上一状态估计的链路传输、GPU 执行和未完成任务等待时间选择最短者。
2. `baselines/mlp_state/`（State-MLP）：不建图，将最近智能体、算力、链路和任务状态拼接；共享 MLP 为每个候选生成 logit，Skill 掩码后用 softmax 分配，交叉熵训练。
3. `baselines/static_graph/`（StaticGraph）：只读取最近快照，做一层静态消息聚合，不使用 GRU 历史。
4. `baselines/minimal_state_ablation/`（GRU-Minimal）：保持主算法结构，但只开放 GPU 剩余算力和链路时延，并关闭智能体可靠性。

这些模型使用同一候选集合和 Skill 硬约束，当前四图实验只比较完成准确率与 DAG 完成时延。

共享的数据抽取、损失和 CLI 运行时集中在 `baselines/common.py` 与 `baselines/runtime.py`，避免四个目录复制相同实现。
