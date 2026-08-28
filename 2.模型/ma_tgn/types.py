"""异构连续时间图的数据结构。"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Optional

import torch

Tensor = torch.Tensor


class EventType(IntEnum):
    """与 ``1.data/6.事件数据/event_type_codebook.csv`` 保持一致。"""

    CONTEXT_UPDATE = 0
    ENVIRONMENT_UPDATE = 1
    COMPUTE_UPDATE = 2
    LINK_UPDATE = 3
    TASK_RESULT = 4
    CAPABILITY_UPDATE = 5


class RelationType(IntEnum):
    """异构图中的稳定关系编码。"""

    COMPUTE_ACCESS = 0
    COMMUNICATION = 1
    TASK_DEPENDENCY = 2


@dataclass
class GraphEvent:
    src: int
    dst: int
    src_node_type: int
    dst_node_type: int
    timestamp: float
    event_type: int
    relation_type: int
    src_state: Dict[str, Tensor]
    dst_state: Dict[str, Tensor]
    edge_state: Tensor
    # 只给快变/观测状态提供置信度；能力及逐能力可靠性不使用。
    src_confidence: Optional[Dict[str, Tensor]] = None
    dst_confidence: Optional[Dict[str, Tensor]] = None
    src_observed_at: Optional[Dict[str, float]] = None
    dst_observed_at: Optional[Dict[str, float]] = None
    edge_confidence: float = 1.0
    edge_observed_at: Optional[float] = None


@dataclass
class TaskRequirement:
    values: Tensor
    capability_embedding: Tensor
    description: str = ""
    required_capability: str = ""
    # DAG 是条件化 query 序列，不属于系统图节点。
    dag_descriptions: tuple[str, ...] = ()
    dag_required_capabilities: tuple[str, ...] = ()
    dag_qos: Optional[Tensor] = None
    dag_edge_index: Optional[Tensor] = None
    dag_current_index: int = 0
    complex_task_id: str = ""


@dataclass
class QueryGraph:
    candidate_ids: Tensor
    edge_index: Tensor
    edge_state: Tensor
    relation_type: Tensor
    edge_timestamp: Tensor
    edge_confidence: Tensor
    query_time: float
    # 与 candidate_ids 等长。V2 用于任务文本和智能体能力文本的语义匹配。
    candidate_texts: tuple[str, ...] = ()
    # 必需 Skill 是可执行性硬约束，不依赖文本相似度猜测。
    candidate_skill_mask: Optional[Tensor] = None


@dataclass
class TrainingCase:
    events: list[GraphEvent]
    graph: QueryGraph
    task: TaskRequirement
    label: Tensor
    # [成功率, 执行时间(归一化), 工具失败率, 幻觉率, 可执行性]
    candidate_targets: Tensor
    teacher_scores: Tensor
    metadata: Dict[str, object]
