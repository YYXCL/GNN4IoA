# 当前模型与基线

`tc_mdsm/` 是唯一保留的模型实现。`types.py` 中的方法集合包含 TC-MDSM 和论文列出的五个基线。

| 方法 | 当前实现口径 |
|---|---|
| TC-MDSM | 历史状态、任务条件稀疏选择、异构图预测与 CP-SAT 调度 |
| GMATANN | 条件时序 VAE、局部与全局注意力，接入统一预测与调度接口 |
| HM-MATAS | 异构节点与边注意力，接入统一预测与调度接口 |
| Rule-Latency | 以观测时延筛选候选，使用共享 CP-SAT 后端 |
| Full-State | 与 TC-MDSM 相同的历史 GRU、任务查询、图预测、8候选预算与 CP-SAT，仅取消状态组门控及维度掩码 |
| StaticGraph | 与TC-MDSM相同的门控、检索和图预测，仅使用最新状态、不使用历史GRU |

GMATANN 与 HM-MATAS 是面向本任务的结构适配；图例使用论文给定的原名称，不加 adapted。旧 MA-TGN 的模型和独立基线目录已清理。
