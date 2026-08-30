# 模型包

当前入口是 `ma_tgn.v2.TaskConditionedHGTGRU`。V1 与 V2 已按版本隔离，二者不再共享实现模块。

| 目录/文件 | 作用 |
| --- | --- |
| `v2/model.py` | 节点/边 GRU 历史编码、一层邻居聚合、任务条件特征注意力与任务—智能体注意力 |
| `v2/task_node_encoder.py` | 把 Skill ID、时延、准确率和 DAG 依赖编码为瞬时任务节点 |
| `v2/normalization.py` | 仅用训练集拟合的分类型标准化 |
| `v2/text_encoder.py` / `v2/layers.py` | Skill ID 哈希编码、MLP 与分段 softmax |
| `v1/` | 旧双时间尺度、置信度和能力槽版本，仅用于历史复现 |
| `config.py` | 只保留 V2 实际使用的配置；V1 使用 `v1/config.py` |
| `state_schema.py` / `types.py` | 状态字段、事件、图、任务和训练样本契约 |

V2 没有 Task Query 对象。DAG 中每个子任务在一次分配中作为瞬时任务节点，与全部候选智能体形成边；任务节点不会写入跨时间 GRU 记忆。

智能体能力不使用能力槽。每个智能体保存精确 Skill ID 集合，Skill 匹配是硬约束；动态状态只保留上下文、环境、未完成任务和一条总体可靠性向量。模型也不使用人工置信度、显式趋势或快/慢状态更新尺度。
