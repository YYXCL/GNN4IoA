"""Rule-Latency：Skill 硬匹配后按上一状态估计最短时延。"""

from __future__ import annotations

import torch

from baselines.common import latest_agent_states, skill_mask
from ma_tgn.types import TrainingCase


class RuleBasedLatencyScheduler:
    name = "Rule-Latency"
    slug = "rule_based_latency"
    temporal = False
    trainable = False

    def __call__(self, case: TrainingCase) -> dict[str, torch.Tensor]:
        candidates = case.graph.candidate_ids
        mask = skill_mask(case)
        latest = latest_agent_states(case)
        estimates = []
        for agent_id in candidates.tolist():
            compute_index = case.metadata["agent_compute"][agent_id]
            gpu_remaining = max(1.0, float(case.metadata["compute_states"][compute_index][0]))
            incoming = case.graph.edge_index[1] == agent_id
            compute_edges = incoming & (case.graph.relation_type == 0)
            latency_ms = float(case.graph.edge_state[compute_edges, 0].min()) if compute_edges.any() else 1_000.0
            unfinished = float(latest[agent_id].get("workload", torch.zeros(1))[0])
            estimates.append(latency_ms / 1_000.0 + 240.0 / gpu_remaining + 0.25 * unfinished)
        predicted_latency = torch.tensor(estimates)
        logits = (-predicted_latency).masked_fill(~mask, -1.0e4)
        return {
            "candidate_ids": candidates,
            "candidate_logits": logits,
            "candidate_scores": torch.softmax(logits, dim=0),
            "predicted_completion_latency_s": predicted_latency,
            "eligibility_mask": mask,
        }
