"""V1 历史模型的独立配置；仅用于复现实验，不被 V2 引用。"""

from dataclasses import dataclass


@dataclass
class LegacyModelConfig:
    num_agents: int = 100
    num_compute_nodes: int = 12
    max_capabilities: int = 8
    capability_embedding_dim: int = 8
    capability_reliability_dim: int = 4
    edge_feature_dim: int = 4
    task_qos_dim: int = 2
    task_feature_dim: int = 10
    event_types: tuple[str, ...] = (
        "context_update", "environment_update", "compute_update", "link_update",
        "task_result", "capability_update",
    )
    relation_types: tuple[str, ...] = ("compute_access", "communication", "task_assignment")
    node_types: tuple[str, ...] = ("agent", "compute", "task")
    modality_dim: int = 16
    state_dim: int = 64
    memory_dim: int = 64
    time_dim: int = 16
    event_dim: int = 64
    task_dim: int = 32
    attention_heads: int = 4
    hgt_layers: int = 1
    dropout: float = 0.1
    slow_update_rate: float = 0.1
    confidence_tau_seconds: float = 2.0
    use_queue_batch_in_decision: bool = False

    @property
    def num_nodes(self) -> int:
        return self.num_agents + self.num_compute_nodes


ModelConfig = LegacyModelConfig
