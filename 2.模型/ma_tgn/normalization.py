"""仅用训练集拟合的、按物理类型分组的数值标准化。"""

from __future__ import annotations

from collections import defaultdict

import torch
from torch import nn

from .types import TrainingCase


FEATURE_DIMS = {
    "agent_context": 2,
    "agent_environment": 7,
    "capability_usage": 1,
    "capability_reliability": 4,
    "compute": 9,
    "edge": 4,
    "task_qos": 8,
}


class TypewiseStandardizer(nn.Module):
    """不同量纲不共享统计量，并将统计量随 checkpoint 保存。"""

    def __init__(self) -> None:
        super().__init__()
        for name, dim in FEATURE_DIMS.items():
            self.register_buffer(f"{name}_mean", torch.zeros(dim))
            self.register_buffer(f"{name}_std", torch.ones(dim))
        self.register_buffer("is_fitted", torch.tensor(False))

    def normalize(self, name: str, values: torch.Tensor) -> torch.Tensor:
        mean = getattr(self, f"{name}_mean").to(values.device)
        std = getattr(self, f"{name}_std").to(values.device)
        return ((values.float() - mean) / std).clamp(-6.0, 6.0)

    @torch.no_grad()
    def fit(self, cases: list[TrainingCase]) -> None:
        values: dict[str, list[torch.Tensor]] = defaultdict(list)

        def collect(node_type: int, state: dict[str, torch.Tensor]) -> None:
            if node_type == 0:
                values["agent_context"].append(state["context"].reshape(-1, 2))
                values["agent_environment"].append(state["environment"].reshape(-1, 7))
                mask = state["capability_mask"].bool().reshape(-1)
                if mask.any():
                    values["capability_usage"].append(state["capability_usage"].reshape(-1, 1)[mask])
                    values["capability_reliability"].append(state["capability_reliability"].reshape(-1, 4)[mask])
            elif node_type == 1:
                values["compute"].append(state["compute"].reshape(-1, 9))

        for case in cases:
            values["task_qos"].append(case.task.values.reshape(-1, 8))
            if case.task.dag_qos is not None and case.task.dag_qos.numel():
                values["task_qos"].append(case.task.dag_qos.reshape(-1, 8))
            values["edge"].append(case.graph.edge_state.reshape(-1, 4))
            for event in case.events:
                collect(event.src_node_type, event.src_state)
                collect(event.dst_node_type, event.dst_state)
                values["edge"].append(event.edge_state.reshape(-1, 4))

        for name, dim in FEATURE_DIMS.items():
            if not values[name]:
                raise ValueError(f"训练集中缺少 {name}，无法拟合归一化统计量")
            stacked = torch.cat(values[name], dim=0).float().reshape(-1, dim)
            mean = stacked.mean(0)
            std = stacked.std(0, unbiased=False).clamp_min(1e-4)
            getattr(self, f"{name}_mean").copy_(mean)
            getattr(self, f"{name}_std").copy_(std)
        self.is_fitted.fill_(True)

    def summary(self) -> dict[str, dict[str, list[float]]]:
        return {
            name: {
                "mean": getattr(self, f"{name}_mean").detach().cpu().tolist(),
                "std": getattr(self, f"{name}_std").detach().cpu().tolist(),
            }
            for name in FEATURE_DIMS
        }
