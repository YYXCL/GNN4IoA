"""异构图状态指标契约：只保留当前研究要求中明确提出的状态。"""

from __future__ import annotations

from typing import Dict, Tuple

NODE_TYPES = ("agent", "compute", "task")
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

# 智能体总体可靠性；Skill ID 是静态硬约束，不再维护逐能力槽。
AGENT_RELIABILITY_METRICS: Tuple[str, ...] = (
    "task_success_rate",
    "mean_execution_time_ms",
    "tool_call_failure_rate",
    "hallucination_rate",
)

# 任务运行直接需要的三类剩余资源；利用率、队列和 batch 不进入图编码。
COMPUTE_METRICS: Tuple[str, ...] = (
    "gpu_remaining_tflops",
    "memory_remaining_gb",
    "vram_remaining_gb",
)
AGENT_WORKLOAD_METRICS: Tuple[str, ...] = ("active_unfinished_tasks",)
COMPUTE_DECISION_METRICS = COMPUTE_METRICS
COMPUTE_OBSERVABILITY_ONLY_METRICS: Tuple[str, ...] = ()

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
)

# 原始量到模型量的稳健截断上限；原始数据仍以真实单位保存。
NORMALIZATION_SCALES: Dict[str, Tuple[float, ...]] = {
    "agent_context": (262_144.0, 1.0),
    "agent_environment": (1.0,) * len(AGENT_ENVIRONMENT_METRICS),
    "compute": (10_000.0, 2_048.0, 1_024.0),
    "edge": (1_000.0, 100_000.0, 1.0, 200.0),
    "reliability": (1.0, 120_000.0, 1.0, 1.0),
}

CONFIDENCE_MODALITIES = ("agent_context", "agent_environment", "compute", "edge")
NO_CONFIDENCE_MODALITIES = ("skill_ids", "agent_reliability")


def schema_dict() -> dict:
    return {
        "node_types": list(NODE_TYPES),
        "compute_kinds": list(COMPUTE_KINDS),
        "agent_context_metrics": list(AGENT_CONTEXT_METRICS),
        "agent_environment_metrics": list(AGENT_ENVIRONMENT_METRICS),
        "agent_workload_metrics": list(AGENT_WORKLOAD_METRICS),
        "agent_reliability_metrics": list(AGENT_RELIABILITY_METRICS),
        "compute_metrics": list(COMPUTE_METRICS),
        "compute_decision_metrics": list(COMPUTE_DECISION_METRICS),
        "compute_observability_only_metrics": list(COMPUTE_OBSERVABILITY_ONLY_METRICS),
        "edge_metrics": list(EDGE_METRICS),
        "task_qos_metrics": list(TASK_QOS_METRICS),
        "units": {
            "context_window_tokens": "token", "gpu_remaining_tflops": "TFLOPS",
            "memory_remaining_gb": "GB", "vram_remaining_gb": "GB",
            "latency_ms": "ms", "bandwidth_mbps": "Mbps", "jitter_ms": "ms",
            "mean_execution_time_ms": "ms",
        },
        "confidence": {"used_by_current_model": False, "note": "保留序列化兼容字段，但不进入节点、边或任务编码"},
        "normalization_scales": {key: list(value) for key, value in NORMALIZATION_SCALES.items()},
        "v2_normalization": "train-split-only typewise z-score; statistics stored in checkpoint",
        "relation_types": ["compute_access", "communication", "task_assignment"],
        "task_representation": "transient task nodes connected to every candidate agent",
    }
