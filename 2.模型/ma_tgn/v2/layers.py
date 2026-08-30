"""V2 模型共享的轻量神经网络层。"""

from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    """两层感知机；只保留当前 GRU-DAG 模型实际使用的实现。"""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        hidden = max(in_dim, out_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


def segmented_softmax(
    scores: torch.Tensor,
    groups: torch.Tensor,
    group_count: int,
) -> torch.Tensor:
    """分别在每个目标节点的入边集合上计算 softmax。"""

    output = torch.zeros_like(scores)
    for group in range(group_count):
        mask = groups == group
        if mask.any():
            output[mask] = torch.softmax(scores[mask], dim=0)
    return output
