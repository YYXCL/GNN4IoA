"""面向 6G 多智能体协同决策的异构连续时间图注意力模型（MA-TGN）。"""

from .config import ModelConfig
from .v2.model import TaskConditionedHGTGRU
TaskConditionedHeterogeneousTGN = TaskConditionedHGTGRU
from .types import EventType, GraphEvent, QueryGraph, RelationType, TaskRequirement, TrainingCase

__all__ = [
    "ModelConfig",
    "EventType",
    "GraphEvent",
    "QueryGraph",
    "RelationType",
    "TaskConditionedHeterogeneousTGN",
    "TaskConditionedHGTGRU",
    "TaskRequirement",
    "TrainingCase",
]
