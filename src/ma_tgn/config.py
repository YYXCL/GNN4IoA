"""模型与异构状态配置。"""

from dataclasses import dataclass

from .state_schema import EDGE_METRICS, TASK_QOS_METRICS


@dataclass
class ModelConfig:
    num_agents: int = 100
    num_compute_nodes: int = 12
    max_capabilities: int = 8
    capability_embedding_dim: int = 8
    capability_reliability_dim: int = 4
    edge_feature_dim: int = len(EDGE_METRICS)
    task_qos_dim: int = len(TASK_QOS_METRICS)
    # 任务还携带细粒度能力语义向量。
    task_feature_dim: int = len(TASK_QOS_METRICS) + 8
    event_types: tuple[str, ...] = (
        "context_change", "environment_change", "compute_change", "link_change",
        "task_result", "capability_added",
    )
    relation_types: tuple[str, ...] = (
        "compute_access", "communication", "historical_collaboration",
    )
    modality_dim: int = 16
    state_dim: int = 64
    memory_dim: int = 64
    time_dim: int = 16
    event_dim: int = 64
    task_dim: int = 32
    attention_heads: int = 4
    dropout: float = 0.1
    slow_update_rate: float = 0.1
    confidence_tau_seconds: float = 2.0
    # 严格遵守当前要求；设为 True 可做“队列/batch 是否有用”的消融实验。
    use_queue_batch_in_decision: bool = False

    @property
    def num_nodes(self) -> int:
        return self.num_agents + self.num_compute_nodes

    def __post_init__(self) -> None:
        if self.memory_dim % self.attention_heads:
            raise ValueError("memory_dim 必须能被 attention_heads 整除")
        if self.capability_reliability_dim != 4:
            raise ValueError("每个能力必须保留成功率、执行时间、工具失败率、幻觉率四项可靠性")
        if self.task_feature_dim != self.task_qos_dim + self.capability_embedding_dim:
            raise ValueError("task_feature_dim 必须包含 QoS 和能力语义向量")
