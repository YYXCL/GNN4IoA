"""State-MLP：拼接最近状态并直接输出分配概率。"""

from __future__ import annotations

import torch
from torch import nn

from baselines.common import latest_candidate_features, skill_mask
from ma_tgn.types import TrainingCase


class MLPStateAllocator(nn.Module):
    name = "State-MLP"
    slug = "mlp_state"
    temporal = False
    trainable = True

    def __init__(self, hidden_dim: int = 64, dropout: float = 0.05):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(24, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
        )

    def forward(self, case: TrainingCase) -> dict[str, torch.Tensor]:
        logits = self.scorer(latest_candidate_features(case)).squeeze(-1)
        mask = skill_mask(case).to(logits.device)
        logits = logits.masked_fill(~mask, -1.0e4)
        return {
            "candidate_ids": case.graph.candidate_ids,
            "candidate_logits": logits,
            "candidate_scores": torch.softmax(logits, dim=0),
            "eligibility_mask": mask,
        }
