"""智能体/算力异构节点、链路、时间和任务编码器。"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from torch import nn

from .config import ModelConfig
from .state_schema import (
    AGENT_CONTEXT_METRICS, AGENT_ENVIRONMENT_METRICS, COMPUTE_METRICS,
    NORMALIZATION_SCALES,
)

Tensor = torch.Tensor


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        hidden = max(in_dim, out_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def decayed_confidence(value: Optional[Tensor], observed_at: Optional[float], now: float, tau: float, device: torch.device) -> Tensor:
    """计算衰减置信度。"""
    base = torch.tensor(1.0, device=device) if value is None else value.float().to(device).mean().clamp(0, 1)
    age = 0.0 if observed_at is None else max(0.0, now - observed_at)
    return base * math.exp(-age / tau)


class HeterogeneousStateEncoder(nn.Module):
    """按节点类型编码；能力/可靠性不使用趋势或置信度。"""

    modality_names = ("context", "environment", "capability", "capability_reliability", "compute")

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        d = config.modality_dim
        self.context_encoder = MLP(len(AGENT_CONTEXT_METRICS) + 1, d, config.dropout)
        self.environment_encoder = MLP(len(AGENT_ENVIRONMENT_METRICS) + 1, d, config.dropout)
        slot_input = config.capability_embedding_dim + 1
        self.capability_slot_encoder = MLP(slot_input, d, config.dropout)
        self.reliability_slot_encoder = MLP(config.capability_reliability_dim, d, config.dropout)
        compute_decision_dim = len(COMPUTE_METRICS) if config.use_queue_batch_in_decision else 6
        self.compute_encoder = MLP(compute_decision_dim + 1, d, config.dropout)
        self.node_type_embedding = nn.Embedding(2, d)
        self.fusion = MLP(len(self.modality_names) * d + d, config.state_dim, config.dropout)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device   # 获取模型参数所在设备

    @staticmethod
    def _required(state: Dict[str, Tensor], key: str) -> Tensor:
        if key not in state:
            raise ValueError(f"节点状态缺少 {key}")  # 缺少必填状态
        return state[key]

    def _pool_slots(self, values: Tensor, mask: Tensor, encoder: nn.Module) -> Tensor:
        """对能力槽进行池化。"""
        tokens = encoder(values)
        weights = mask.float().to(tokens.device).reshape(-1, 1)
        return (tokens * weights).sum(0) / weights.sum().clamp_min(1.0)

    def _agent(self, state: Dict[str, Tensor], confidence: Dict[str, Tensor], observed_at: Dict[str, float], now: float) -> list[Tensor]:
        """编码智能体状态。"""
        context = self._required(state, "context").float().to(self.device).reshape(-1)
        if context.numel() != len(AGENT_CONTEXT_METRICS):
            raise ValueError("context 状态维度不正确")
        context = (context / torch.tensor(NORMALIZATION_SCALES["agent_context"], device=self.device)).clamp(0, 1)
        context_conf = decayed_confidence(confidence.get("context"), observed_at.get("context"), now, self.config.confidence_tau_seconds, self.device)
        context_token = self.context_encoder(torch.cat([context, context_conf.reshape(1)])) * context_conf

        environment = self._required(state, "environment").float().to(self.device).reshape(-1).clamp(0, 1)
        if environment.numel() != len(AGENT_ENVIRONMENT_METRICS):
            raise ValueError("environment 状态维度不正确")
        environment_conf = decayed_confidence(confidence.get("environment"), observed_at.get("environment"), now, self.config.confidence_tau_seconds, self.device)
        environment_token = self.environment_encoder(torch.cat([environment, environment_conf.reshape(1)])) * environment_conf

        embeddings = self._required(state, "capability_embeddings").float().to(self.device)
        mask = self._required(state, "capability_mask").float().to(self.device)
        usage = self._required(state, "capability_usage").float().to(self.device)
        reliability = self._required(state, "capability_reliability").float().to(self.device).clone()
        expected = (self.config.max_capabilities, self.config.capability_embedding_dim)
        if tuple(embeddings.shape) != expected or mask.numel() != self.config.max_capabilities:
            raise ValueError(f"能力槽必须为 {expected} 并带等长 mask")
        capability_token = self._pool_slots(torch.cat([embeddings, usage.reshape(-1, 1)], dim=1), mask, self.capability_slot_encoder)
        reliability[:, 1] = (reliability[:, 1] / NORMALIZATION_SCALES["reliability"][1]).clamp(0, 1)
        reliability_token = self._pool_slots(reliability.clamp(0, 1), mask, self.reliability_slot_encoder)
        return [context_token, environment_token, capability_token, reliability_token, torch.zeros_like(context_token)]

    def _compute(self, state: Dict[str, Tensor], confidence: Dict[str, Tensor], observed_at: Dict[str, float], now: float) -> list[Tensor]:
        """编码计算节点状态。"""
        raw = self._required(state, "compute").float().to(self.device).reshape(-1)
        if raw.numel() != len(COMPUTE_METRICS):
            raise ValueError("compute 状态维度不正确")
        normalized = (raw / torch.tensor(NORMALIZATION_SCALES["compute"], device=self.device)).clamp(0, 1)
        decision = normalized if self.config.use_queue_batch_in_decision else normalized[:6]
        confidence_value = decayed_confidence(confidence.get("compute"), observed_at.get("compute"), now, self.config.confidence_tau_seconds, self.device)
        token = self.compute_encoder(torch.cat([decision, confidence_value.reshape(1)])) * confidence_value
        zeros = torch.zeros_like(token)
        return [zeros, zeros, zeros, zeros, token]

    def forward(self, node_type: int, state: Dict[str, Tensor], confidence: Optional[Dict[str, Tensor]] = None, observed_at: Optional[Dict[str, float]] = None, now: float = 0.0) -> Tuple[Tensor, Tensor]:
        confidence = confidence or {}
        observed_at = observed_at or {}
        if node_type == 0:
            tokens = self._agent(state, confidence, observed_at, now)
        elif node_type == 1:
            tokens = self._compute(state, confidence, observed_at, now)
        else:
            raise ValueError(f"未知节点类型: {node_type}")
        token_tensor = torch.stack(tokens)
        type_token = self.node_type_embedding(torch.tensor(node_type, device=self.device))
        return self.fusion(torch.cat([token_tensor.reshape(-1), type_token])), token_tensor


class EdgeStateEncoder(nn.Module):
    """仅编码时延、带宽、丢包、抖动及时间衰减后的观测置信度。"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = MLP(config.edge_feature_dim + 1, config.state_dim, config.dropout)

    def forward(self, edge_state: Tensor, confidence: Optional[Tensor] = None) -> Tensor:
        values = edge_state.float()
        single = values.ndim == 1
        if single:
            values = values.unsqueeze(0)
        scales = torch.tensor(NORMALIZATION_SCALES["edge"], device=values.device)
        normalized = (values / scales).clamp(0, 1)
        if confidence is None:
            confidence = torch.ones(values.shape[0], device=values.device)
        confidence = confidence.float().to(values.device).reshape(-1, 1).clamp(0, 1)
        encoded = self.encoder(torch.cat([normalized, confidence], dim=-1)) * confidence
        return encoded.squeeze(0) if single else encoded


class TimeEncoder(nn.Module):
    def __init__(self, time_dim: int):
        super().__init__()
        self.frequency = nn.Parameter(torch.logspace(0, -3, time_dim))
        self.phase = nn.Parameter(torch.zeros(time_dim))

    def forward(self, delta_t: Tensor) -> Tensor:
        delta_t = torch.log1p(delta_t.float().clamp_min(0)).reshape(-1, 1)
        return torch.cos(delta_t * self.frequency.reshape(1, -1) + self.phase)


class TaskEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = MLP(config.task_feature_dim, config.task_dim, config.dropout)
        self.modality_attention = nn.Linear(config.task_dim, len(HeterogeneousStateEncoder.modality_names))
        self.latent_gate = nn.Linear(config.task_dim, config.memory_dim * 2)

    def forward(self, qos: Tensor, capability_embedding: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        raw = torch.cat([qos.float(), capability_embedding.float()])
        task = self.encoder(raw)
        return task, torch.softmax(self.modality_attention(task), -1), torch.sigmoid(self.latent_gate(task))
