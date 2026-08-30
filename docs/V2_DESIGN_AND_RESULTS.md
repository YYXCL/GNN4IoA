# V2：任务节点条件化的异构时序图分配

V2 是一条自定义的任务条件异构时序 GNN 路线：

- 节点/边历史由 GRU 编码，接近 TGN 的“事件序列 + 记忆”思想；
- 智能体、算力、任务三类节点和两类系统关系，接近 HGT 的异构关系建模思想；
- 瞬时任务节点与所有候选智能体形成二部边，再做任务—智能体交叉注意力；
- DAG Skill 硬约束和逐子任务分配接近 dependency-aware heterogeneous task allocation。

它不是标准 HGT：邻居聚合尚未为每种节点/边类型分别设置完整 Q/K/V；也不是标准 TGN：当前使用节点 GRUCell、边 GRUCell 和一层自定义聚合器。

## 两阶段注意力

第一阶段在每个候选智能体的入边内归一化：

`alpha(u→a) = softmax_a(MLP([h_a, h_u, h_edge, r_type]))`

它聚合相邻智能体、固定算力节点和通信边。当前任务需求不直接参与这一层边注意力。

第二阶段由子任务节点决定五组聚合结果的重要性：

`beta(s,a,f) = softmax_f(q_task(z_s)^T k_feature(c_a,f) / sqrt(d))`

其中 `f` 是自身、智能体邻居、算力节点、通信边和未完成任务。不同 Skill、时延要求、准确率要求及 DAG 位置生成不同的任务节点向量，因此可以学习不同的属性权重。

任务—智能体边的 logit 为：

`l(s,a) = q_agent(z_s)^T k_agent(h_s,a)/sqrt(d) + MLP([z_s,h_s,a])`

精确 Skill ID 不匹配的边被硬掩码，然后在候选智能体维度 softmax。候选集由数据层和 Skill 约束给出，注意力负责在可执行候选中排序，并不会凭空生成候选。

V2 不使用 Task Query 对象、能力槽、人工状态置信度、显式趋势或快/慢更新尺度。代码位于 `2.模型/ma_tgn/v2/`；历史 V1 位于 `2.模型/ma_tgn/v1/`。
