"""旧版实验使用的纯 PyTorch Heterogeneous Graph Transformer。"""

from __future__ import annotations

import math

import torch
from torch import nn


def segmented_softmax(scores: torch.Tensor, groups: torch.Tensor, group_count: int) -> torch.Tensor:
    result = torch.zeros_like(scores)
    for group in range(group_count):
        mask = groups == group
        if mask.any():
            result[mask] = torch.softmax(scores[mask], dim=0)
    return result


class HGTLayer(nn.Module):
    """节点类型专属 Q/K/V + 关系专属注意力/消息变换。"""

    def __init__(self, dim: int, heads: int, node_types: int, relations: int, dropout: float):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.q = nn.ModuleList(nn.Linear(dim, dim) for _ in range(node_types))
        self.k = nn.ModuleList(nn.Linear(dim, dim) for _ in range(node_types))
        self.v = nn.ModuleList(nn.Linear(dim, dim) for _ in range(node_types))
        self.residual = nn.ModuleList(nn.Linear(dim, dim) for _ in range(node_types))
        self.norm = nn.ModuleList(nn.LayerNorm(dim) for _ in range(node_types))
        self.relation_attention = nn.Parameter(torch.empty(relations, heads, self.head_dim, self.head_dim))
        self.relation_message = nn.Parameter(torch.empty(relations, heads, self.head_dim, self.head_dim))
        self.relation_prior = nn.Parameter(torch.ones(relations, heads))
        self.edge_key = nn.Linear(dim, dim)
        self.edge_value = nn.Linear(dim, dim)
        self.edge_bias = nn.Linear(dim, heads)
        self.output = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.relation_attention)
        nn.init.xavier_uniform_(self.relation_message)

    def _typed_projection(self, layers: nn.ModuleList, x: torch.Tensor, node_type: torch.Tensor) -> torch.Tensor:
        output = torch.zeros_like(x)
        for kind, layer in enumerate(layers):
            mask = node_type == kind
            if mask.any():
                output[mask] = layer(x[mask])
        return output.view(-1, self.heads, self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        node_type: torch.Tensor,
        edge_index: torch.Tensor,
        relation_type: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index
        q = self._typed_projection(self.q, x, node_type)[dst]
        k = self._typed_projection(self.k, x, node_type)[src]
        v = self._typed_projection(self.v, x, node_type)[src]
        relation_attention = self.relation_attention[relation_type]
        relation_message = self.relation_message[relation_type]
        k = torch.einsum("ehd,ehdf->ehf", k, relation_attention)
        v = torch.einsum("ehd,ehdf->ehf", v, relation_message)
        edge_k = self.edge_key(edge_attr).view(-1, self.heads, self.head_dim)
        edge_v = self.edge_value(edge_attr).view(-1, self.heads, self.head_dim)
        score = ((q * (k + edge_k)).sum(-1) / math.sqrt(self.head_dim))
        score = score * self.relation_prior[relation_type] + self.edge_bias(edge_attr)
        alpha = segmented_softmax(score, dst, x.shape[0])
        messages = (alpha.unsqueeze(-1) * (v + edge_v)).reshape(-1, self.dim)
        aggregate = torch.zeros_like(x)
        aggregate.index_add_(0, dst, messages)
        transformed = self.output(aggregate)
        output = torch.zeros_like(x)
        for kind, (residual, norm) in enumerate(zip(self.residual, self.norm)):
            mask = node_type == kind
            if mask.any():
                output[mask] = norm(residual(x[mask]) + self.dropout(transformed[mask]))
        return output, alpha


class HGTEncoder(nn.Module):
    def __init__(self, dim: int, heads: int, node_types: int, relations: int, layers: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            HGTLayer(dim, heads, node_types, relations, dropout) for _ in range(layers)
        )

    def forward(self, *args: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x, node_type, edge_index, relation_type, edge_attr = args
        attention = []
        for layer in self.layers:
            x, alpha = layer(x, node_type, edge_index, relation_type, edge_attr)
            attention.append(alpha)
        return x, attention
