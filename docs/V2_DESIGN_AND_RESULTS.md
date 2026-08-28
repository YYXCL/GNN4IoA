# GNN4IoA V2：DAG-query 条件化任务分配

## 1. 建模边界

系统异构图只允许两类节点：

```text
agent <-- communication/task_dependency --> agent
  ^
  | compute_access
  |
compute
```

复杂任务及其子任务不进入系统图。拆分器输出的 DAG 被表示为独立 query：

```text
subtask description + required Skill id + QoS
                         |
               bidirectional DAG encoder
                         |
        current dependency-conditioned task vector
                         |
agent/compute HGT state --+--> allocation decision
```

这样做避免把一次性任务与长期存在的智能体/算力实体混成同一节点空间，也使同一张系统图能被不同任务需求重复查询。

## 2. 条件化分配决策

每个子任务 query 包含：任务描述、必需 Skill id、8 维 QoS（时延、准确率、成本、可靠性、隐私、上下文、工具、多模态）以及 DAG 中的前驱/后继边。

1. 智能体能力描述、Skill、API、知识库、运行环境和传感器组成能力文本，和任务文本共用 subword hashing + 可训练投影编码器。
2. DAG 编码器分别聚合前驱与后继消息，保留依赖方向、入度和出度，得到当前子任务的依赖条件向量。
3. HGT-GRU 编码智能体/算力动态图，Cross-Attention 让每个候选智能体读取 DAG query token。
4. 状态头预测成功率、归一化完成时间、工具失败率、幻觉率和可执行性。
5. 最终 logit 由学习分数、能力语义匹配和 QoS 加权效用共同构成；缺少必需 Skill 的候选由结构化硬掩码排除。

主要文件：

- `2.模型/ma_tgn/dag_query.py`
- `2.模型/ma_tgn/v2_model.py`
- `2.模型/ma_tgn/text_encoder.py`
- `1.data/ma_tgn_data/foundation.py`

## 3. 数据中的 DAG

现有 72 条任务是单子任务记录，没有人工标注 DAG。数据构建器采用可审计规则生成局部复杂任务：

- 只从同一业务场景选择相关子任务；
- 优先同源智能体类型和相同 Skill 能力域；
- 按 `采集/查询/监测/检测 -> 分析/分类 -> 融合/评估 -> 优化/控制 -> 生成/问答` 建立有向无环依赖；
- 每个 query 保存子任务列表、边、当前子任务索引和复杂任务 id。

生成结果位于 `1.data/datasets/sixg_ma_state/task_dag_catalog.json`。这些边是研究用派生标签，不应被描述为真实生产工作流标注；获得人工 DAG 后可以直接替换，模型接口不变。

## 4. 论文对应关系

- G-Designer：使用 agent 能力/角色文本和 task-specific 条件共同学习多智能体结构。本项目保留能力文本编码与任务条件思想，但遵照当前研究假设，不把 task 放入系统图。
- HGRL-TA：复杂任务拆成有依赖、异质 Skill 需求的子任务，并以多关系图编码依赖状态。本项目据此加入方向敏感的 DAG query 编码。
- TARGETNET（NeurIPS 2025）：用残差异构图 Transformer 联合编码智能体能力与时空约束。本项目据此保留 agent/compute HGT，并把依赖条件放在 query 侧。
- GMATANN/GCAN：baseline 目录实现单子任务 task-agent 图、条件变分编码、图注意力和全局/局部融合；它有意忽略 DAG 边。

参考：

- <https://arxiv.org/abs/2410.11782>
- <https://arxiv.org/abs/2410.15449>
- <https://papers.nips.cc/paper_files/paper/2025/hash/82a227f51354b7ebafea51023d547d29-Abstract-Conference.html>
- <https://doi.org/10.1109/JIOT.2025.3535641>

## 5. 公平比较与输出

`experiments/compare_allocation_models.py` 使用同一 JSONL、同一时间划分、同一候选顺序、标签和随机种子训练：

- V1 MA-TGN；
- GMATANN/GCAN 单子任务基线；
- DAG-query HGT-GRU。

两张主图分别报告三种场景（智慧家居、智慧工厂、应急救援）的平均任务分配准确率和平均完成时延。完成时延严格按 `agent_reliability.csv` 中所选智能体的 `平均任务完成时间(秒)` 估算。

输出目录：`outputs/scenario_comparison/`。

注意：当前测试集仅 9 个样本（三场景分别为 3/4/2），图表适合验证端到端实验链路，不足以支持统计显著性结论。正式论文实验应增加样本和随机种子，并报告均值、标准差或置信区间。

## 6. 本次实测（10 epochs, seed=7）

| 模型 | 总体平均准确率 | 总体平均完成时延(s) |
|---|---:|---:|
| V1 MA-TGN | 0.2222 | 8.7094 |
| GMATANN/GCAN | 0.8889 | 8.4769 |
| DAG-query HGT-GRU | 1.0000 | 8.6852 |

按场景的准确率：智慧家居为 `0.3333 / 0.6667 / 1.0000`，智慧工厂为 `0 / 1 / 1`，应急救援为 `0.5 / 1 / 1`（顺序均为 V1 / GCAN / DAG-query）。DAG-query 比 GCAN 多正确分配 1 个智慧家居测试任务；其总体平均时延略高，是因为该任务改为选择正确但可靠性表平均完成时间更长的智能体。测试集过小且 DAG 是研究用派生流程，当前结果仍不能作为统计显著的优越性结论。后续应增加同一 Skill 下能力/可靠性不同的候选、人工 DAG 和更大的测试集。
