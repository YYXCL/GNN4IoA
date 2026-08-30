"""各分配基线共享的数据抽取与损失函数。"""

from __future__ import annotations

import torch
from torch import nn

from ma_tgn.types import TrainingCase

Tensor = torch.Tensor


def latest_agent_states(case: TrainingCase) -> dict[int, dict[str, Tensor]]:
    latest: dict[int, dict[str, Tensor]] = {}
    candidates = set(case.graph.candidate_ids.tolist())
    for event in sorted(case.events, key=lambda item: item.timestamp):
        if event.src_node_type == 0 and event.src in candidates:
            latest[event.src] = event.src_state
        if event.dst_node_type == 0 and event.dst in candidates:
            latest[event.dst] = event.dst_state
    return latest


def skill_mask(case: TrainingCase) -> Tensor:
    if case.graph.candidate_skill_mask is not None:
        return case.graph.candidate_skill_mask.bool()
    return torch.ones(case.graph.candidate_ids.numel(), dtype=torch.bool)


def latest_candidate_features(case: TrainingCase, include_reliability: bool = True) -> Tensor:
    """抽取查询前最后一个快照，供无时序基线共用。"""

    latest = latest_agent_states(case)
    rows: list[Tensor] = []
    for position, agent_id in enumerate(case.graph.candidate_ids.tolist()):
        state = latest[agent_id]
        context = state["context"].float().clone()
        context[0] = torch.log1p(context[0]) / 13.0
        environment = state["environment"].float()
        workload = state.get("workload", torch.zeros(1)).float() / 16.0
        reliability = state["reliability"].float().clone()
        reliability[1] = reliability[1].clamp_min(0).log1p() / 12.0
        if not include_reliability:
            reliability.zero_()
        compute_index = case.metadata["agent_compute"][agent_id]
        compute = torch.tensor(case.metadata["compute_states"][compute_index], dtype=torch.float32)
        compute = compute / torch.tensor([10_000.0, 2_048.0, 1_024.0])
        incoming = case.graph.edge_index[1] == agent_id
        edge = case.graph.edge_state[incoming].float().mean(0)
        edge = edge / torch.tensor([1_000.0, 100_000.0, 1.0, 200.0])
        rows.append(torch.cat([
            context, environment, reliability, workload, compute, edge,
            case.task.values.float(), skill_mask(case)[position].float().reshape(1),
        ]))
    return torch.stack(rows)


def allocation_loss(output: dict[str, Tensor], label: Tensor) -> Tensor:
    return nn.functional.cross_entropy(output["candidate_logits"].unsqueeze(0), label.reshape(1))
