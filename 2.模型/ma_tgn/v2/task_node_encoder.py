"""依赖感知的瞬时任务节点编码器。"""

from __future__ import annotations

import torch
from torch import nn

from ..config import ModelConfig
from .layers import MLP
from .normalization import TypewiseStandardizer
from .text_encoder import TextSemanticEncoder
from ..types import TaskRequirement


class BidirectionalDAGLayer(nn.Module):
    """分别聚合前驱和后继，让当前子任务看到上下游依赖。"""

    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.predecessor = nn.Linear(dim, dim, bias=False)
        self.successor = nn.Linear(dim, dim, bias=False)
        self.update = MLP(dim * 3 + 2, dim, dropout)
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            return x
        src, dst = edge_index
        pred_values = torch.zeros_like(x)
        pred_degree = torch.zeros(x.shape[0], 1, device=x.device)
        pred_values.index_add_(0, dst, self.predecessor(x[src]))
        pred_degree.index_add_(0, dst, torch.ones(dst.numel(), 1, device=x.device))
        predecessors = pred_values / pred_degree.clamp_min(1.0)

        succ_values = torch.zeros_like(x)
        succ_degree = torch.zeros(x.shape[0], 1, device=x.device)
        succ_values.index_add_(0, src, self.successor(x[dst]))
        succ_degree.index_add_(0, src, torch.ones(src.numel(), 1, device=x.device))
        successors = succ_values / succ_degree.clamp_min(1.0)

        structural = torch.cat([
            pred_degree.clamp_max(8.0) / 8.0,
            succ_degree.clamp_max(8.0) / 8.0,
        ], dim=-1)
        proposal = self.update(torch.cat([x, predecessors, successors, structural], dim=-1))
        gate = self.gate(torch.cat([x, proposal], dim=-1))
        return self.norm(x + gate * proposal)


class TaskNodeEncoder(nn.Module):
    """仅用 Skill id、时延、准确率和 DAG 方向生成任务节点向量。"""

    def __init__(
        self,
        config: ModelConfig,
        normalizer: TypewiseStandardizer,
        text_encoder: TextSemanticEncoder,
    ) -> None:
        super().__init__()
        self.config = config
        self.normalizer = normalizer
        self.text_encoder = text_encoder
        self.qos = MLP(config.task_qos_dim, config.state_dim, config.dropout)
        self.fusion = MLP(config.state_dim * 2, config.state_dim, config.dropout)
        self.layers = nn.ModuleList(
            BidirectionalDAGLayer(config.state_dim, config.dropout)
            for _ in range(config.dag_layers)
        )

    def _fields(self, task: TaskRequirement) -> tuple[list[str], list[str], torch.Tensor, int, torch.Tensor]:
        descriptions = list(task.dag_descriptions) or [task.description]
        skills = list(task.dag_required_capabilities) or [task.required_capability]
        if len(descriptions) != len(skills):
            raise ValueError("DAG 子任务描述与 Skill 数量不一致")
        qos = task.dag_qos
        if qos is None or qos.numel() == 0:
            qos = task.values.reshape(1, -1)
        if qos.shape != (len(descriptions), self.config.task_qos_dim):
            raise ValueError("dag_qos 形状必须为 [subtasks, task_qos_dim]")
        current = int(task.dag_current_index)
        if not 0 <= current < len(descriptions):
            raise ValueError("dag_current_index 越界")
        edges = task.dag_edge_index
        if edges is None:
            edges = torch.empty(2, 0, dtype=torch.long)
        return descriptions, skills, qos, current, edges

    def forward(self, task: TaskRequirement) -> torch.Tensor:
        descriptions, skills, qos, current, edges = self._fields(task)
        del descriptions
        skill_tokens = self.text_encoder(skills)
        qos_tokens = self.qos(self.normalizer.normalize("task_qos", qos.to(skill_tokens.device).float()))
        tokens = self.fusion(torch.cat([skill_tokens, qos_tokens], dim=-1))
        edges = edges.long().to(tokens.device)
        for layer in self.layers:
            tokens = layer(tokens, edges)
        return tokens
