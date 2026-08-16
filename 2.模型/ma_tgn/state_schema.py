"""异构图状态指标契约：只保留当前研究要求中明确提出的状态。"""

from __future__ import annotations

from typing import Dict, Tuple

NODE_TYPES = ("agent", "compute")
COMPUTE_KINDS = ("cloud", "edge")

# 智能体自身的负载只描述上下文；算力资源属于独立算力节点。
AGENT_CONTEXT_METRICS: Tuple[str, ...] = (
    "context_window_tokens",
    "context_utilization",
)

# 环境是快速变化的可执行条件。API 相关状态不再放入负载。
AGENT_ENVIRONMENT_METRICS: Tuple[str, ...] = (
    "tool_api_rate_limit_remaining",
    "tool_api_request_occupancy",
    "knowledge_api_rate_limit_remaining",
    "knowledge_api_request_occupancy",
    "user_context_available",
    "physical_sensor_available",
    "physical_environment_match",
)

# 每个细粒度能力独立保留四类可靠性，而不是先聚合成一个可靠性分数。
CAPABILITY_RELIABILITY_METRICS: Tuple[str, ...] = (
    "task_success_rate",
    "mean_execution_time_ms",
    "tool_call_failure_rate",
    "hallucination_rate",
)

# 原始物理量。队列与 batch 进入图数据，但默认不进入调度决策编码器。
COMPUTE_METRICS: Tuple[str, ...] = (
    "gpu_total_tflops",
    "gpu_remaining_tflops",
    "vram_total_gb",
    "vram_used_gb",
    "gpu_memory_bandwidth_total_mbps",
    "gpu_memory_bandwidth_used_mbps",
    "task_queue_length",
    "active_batch_size",
    "max_batch_size",
)
COMPUTE_DECISION_METRICS = COMPUTE_METRICS[:6]
COMPUTE_OBSERVABILITY_ONLY_METRICS = COMPUTE_METRICS[6:]

# 不编码路径性质；所有通信/历史协作边只使用可观测数值。
EDGE_METRICS: Tuple[str, ...] = (
    "latency_ms",
    "bandwidth_mbps",
    "packet_loss_rate",
    "jitter_ms",
)

TASK_QOS_METRICS: Tuple[str, ...] = (
    "latency",
    "accuracy",
    "cost",
    "reliability",
    "privacy",
    "context",
    "tool",
    "multimodal",
)

# 原始量到模型量的稳健截断上限；原始数据仍以真实单位保存。
NORMALIZATION_SCALES: Dict[str, Tuple[float, ...]] = {
    "agent_context": (262_144.0, 1.0),
    "agent_environment": (1.0,) * len(AGENT_ENVIRONMENT_METRICS),
    "compute": (10_000.0, 10_000.0, 512.0, 512.0, 10_000_000.0, 10_000_000.0, 256.0, 128.0, 128.0),
    "edge": (1_000.0, 100_000.0, 1.0, 200.0),
    "reliability": (1.0, 120_000.0, 1.0, 1.0),
}

CONFIDENCE_MODALITIES = ("agent_context", "agent_environment", "compute", "edge")
NO_CONFIDENCE_MODALITIES = ("capability", "capability_reliability")


def schema_dict() -> dict:
    return {
        "node_types": list(NODE_TYPES),
        "compute_kinds": list(COMPUTE_KINDS),
        "agent_context_metrics": list(AGENT_CONTEXT_METRICS),
        "agent_environment_metrics": list(AGENT_ENVIRONMENT_METRICS),
        "capability_reliability_metrics": list(CAPABILITY_RELIABILITY_METRICS),
        "compute_metrics": list(COMPUTE_METRICS),
        "compute_decision_metrics": list(COMPUTE_DECISION_METRICS),
        "compute_observability_only_metrics": list(COMPUTE_OBSERVABILITY_ONLY_METRICS),
        "edge_metrics": list(EDGE_METRICS),
        "task_qos_metrics": list(TASK_QOS_METRICS),
        "units": {
            "context_window_tokens": "token", "gpu_total_tflops": "TFLOPS",
            "gpu_remaining_tflops": "TFLOPS", "vram_total_gb": "GB", "vram_used_gb": "GB",
            "gpu_memory_bandwidth_total_mbps": "Mbps", "gpu_memory_bandwidth_used_mbps": "Mbps",
            "latency_ms": "ms", "bandwidth_mbps": "Mbps", "jitter_ms": "ms",
            "mean_execution_time_ms": "ms",
        },
        "confidence": {
            "applies_to": list(CONFIDENCE_MODALITIES),
            "excluded_from": list(NO_CONFIDENCE_MODALITIES),
            "decay": "effective_confidence = observed_confidence * exp(-age_seconds / tau_seconds)",
        },
        "normalization_scales": {key: list(value) for key, value in NORMALIZATION_SCALES.items()},
    }
