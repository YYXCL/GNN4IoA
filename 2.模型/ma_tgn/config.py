"""模型与异构状态配置。"""

from dataclasses import dataclass

from .state_schema import EDGE_METRICS, TASK_QOS_METRICS


@dataclass
class ModelConfig:
    num_agents: int = 100
    num_compute_nodes: int = 12
    edge_feature_dim: int = len(EDGE_METRICS)
    task_qos_dim: int = len(TASK_QOS_METRICS)
    state_dim: int = 64
    attention_heads: int = 4
    text_hash_dim: int = 256
    text_dim: int = 64
    dag_layers: int = 2
    dropout: float = 0.1
    compute_feature_mask: tuple[bool, bool, bool] = (True, True, True)
    edge_feature_mask: tuple[bool, bool, bool, bool] = (True, True, True, True)
    use_agent_reliability: bool = True

    @property
    def num_nodes(self) -> int:
        return self.num_agents + self.num_compute_nodes

    def __post_init__(self) -> None:
        if self.state_dim % self.attention_heads:
            raise ValueError("state_dim 必须能被 attention_heads 整除")
