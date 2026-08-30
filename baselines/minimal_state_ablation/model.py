"""GRU-Minimal：主算法的最小状态消融。"""

from dataclasses import replace

from ma_tgn.config import ModelConfig
from ma_tgn.v2.model import TaskConditionedHGTGRU


class MinimalStateAblationAllocator(TaskConditionedHGTGRU):
    name = "GRU-Minimal"
    slug = "minimal_state_ablation"
    temporal = True
    trainable = True

    def __init__(self, config: ModelConfig):
        minimal = replace(
            config,
            compute_feature_mask=(True, False, False),
            edge_feature_mask=(True, False, False, False),
            use_agent_reliability=False,
        )
        super().__init__(minimal)
