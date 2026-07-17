"""兼容入口：使用研究型 6G 状态仿真器生成单个训练样本。"""

from typing import List, Tuple

import torch

from .config import ModelConfig
from .simulator import SixGStateSimulator
from .types import GraphEvent, QueryGraph, TaskRequirement


def make_synthetic_case(
    config: ModelConfig, seed: int = 7
) -> Tuple[List[GraphEvent], QueryGraph, TaskRequirement, torch.Tensor]:
    """保留旧 API，但不再生成相互独立的均匀随机状态。"""
    case = SixGStateSimulator(config, seed=seed).make_case(max_candidates=min(8, config.num_nodes))
    return case.events, case.graph, case.task, case.label
