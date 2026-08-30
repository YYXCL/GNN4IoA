"""V1：双时间尺度记忆与任务条件关系注意力的历史实现。"""

from .config import LegacyModelConfig
from .model import TaskConditionedHeterogeneousTGN

__all__ = ["LegacyModelConfig", "TaskConditionedHeterogeneousTGN"]
