"""TC-MDSM：任务条件化-多维状态建模"""

# 对外暴露的核心数据结构：配置、快照、任务、工作流、观测、样本、预测、调度方案
from .types import ModelConfig, Snapshot, Task, Workflow, Observation, Example, Prediction, Plan

__version__ = "1.0.0"
__all__ = ["ModelConfig", "Snapshot", "Task", "Workflow", "Observation", "Example", "Prediction", "Plan"]
