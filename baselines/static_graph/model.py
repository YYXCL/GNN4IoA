"""StaticGraph：只对查询前最近快照做一层图聚合。"""

from __future__ import annotations

import torch
from torch import nn

from baselines.common import latest_candidate_features, skill_mask
from ma_tgn.types import TrainingCase


class StaticGraphAllocator(nn.Module):
    name = "StaticGraph"
    slug = "static_graph"
    temporal = False
    trainable = True

    def __init__(self, hidden_dim: int = 64, dropout: float = 0.05):
        super().__init__()
        self.node = nn.Sequential(nn.Linear(24, hidden_dim), nn.GELU())
        self.edge = nn.Sequential(nn.Linear(4, hidden_dim), nn.GELU())
        self.message = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU())
        self.task = nn.Sequential(nn.Linear(2, hidden_dim), nn.GELU())
        self.scorer = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(self, case: TrainingCase) -> dict[str, torch.Tensor]:
        candidate_ids = case.graph.candidate_ids.tolist()
        position = {node: index for index, node in enumerate(candidate_ids)}
        own = self.node(latest_candidate_features(case))
        aggregate = torch.zeros_like(own)
        degree = torch.zeros(len(candidate_ids), 1)
        edge_values = case.graph.edge_state.float() / torch.tensor([1_000.0, 100_000.0, 1.0, 200.0])
        edge_tokens = self.edge(edge_values)
        for edge_index, (source, destination) in enumerate(case.graph.edge_index.t().tolist()):
            if destination not in position:
                continue
            target = position[destination]
            source_token = own[position[source]] if source in position else torch.zeros_like(own[target])
            aggregate[target] += self.message(torch.cat([source_token, edge_tokens[edge_index]]))
            degree[target] += 1
        aggregate = aggregate / degree.clamp_min(1.0)
        task = self.task(case.task.values.float()).expand_as(own)
        logits = self.scorer(torch.cat([own, aggregate, task], dim=-1)).squeeze(-1)
        mask = skill_mask(case)
        logits = logits.masked_fill(~mask, -1.0e4)
        return {
            "candidate_ids": case.graph.candidate_ids,
            "candidate_logits": logits,
            "candidate_scores": torch.softmax(logits, dim=0),
            "eligibility_mask": mask,
        }
