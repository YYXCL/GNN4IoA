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


@dataclass
class QueryGraph:
    candidate_ids: Tensor
    edge_index: Tensor
    edge_state: Tensor
    relation_type: Tensor
    edge_timestamp: Tensor
    edge_confidence: Tensor
    query_time: float


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
