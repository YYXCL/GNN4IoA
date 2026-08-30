"""三类节点协同任务分配模型：GRU 历史编码、一层聚合和任务条件注意力。"""

from __future__ import annotations

import math
from typing import Dict

import torch
from torch import nn

from ..config import ModelConfig
from .task_node_encoder import TaskNodeEncoder
from .layers import MLP, segmented_softmax
from .normalization import TypewiseStandardizer
from .text_encoder import TextSemanticEncoder
from ..types import GraphEvent, QueryGraph, TaskRequirement, TrainingCase

Tensor = torch.Tensor


class TypedNumericStateEncoder(nn.Module):
    """把智能体自身状态或三项算力剩余状态编码为单时刻输入。"""

    modality_names = ("self", "agent_neighbors", "compute_neighbor", "communication_edges", "unfinished_tasks")

    def __init__(self, config: ModelConfig, normalizer: TypewiseStandardizer):
        super().__init__()
        self.config = config
        self.normalizer = normalizer
        d = config.state_dim
        self.context = MLP(2, d, config.dropout)
        self.environment = MLP(7, d, config.dropout)
        self.reliability = MLP(4, d, config.dropout)
        self.workload = MLP(1, d, config.dropout)
        self.compute = MLP(3, d, config.dropout)
        self.agent_fusion = MLP(d * 4, d, config.dropout)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, node_type: int, state: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        if node_type == 0:
            context = self.context(self.normalizer.normalize("agent_context", state["context"].to(self.device)))
            environment = self.environment(
                self.normalizer.normalize("agent_environment", state["environment"].to(self.device))
            )
            reliability = self.reliability(self.normalizer.normalize(
                "agent_reliability", state["reliability"].to(self.device).float()
            ))
            if not self.config.use_agent_reliability:
                reliability = torch.zeros_like(reliability)
            workload = self.workload(self.normalizer.normalize(
                "agent_workload", state.get("workload", torch.zeros(1)).to(self.device).reshape(-1)
            ))
            fused = self.agent_fusion(torch.cat([context, environment, reliability, workload]))
            return fused, {
                "context": context, "environment": environment,
                "reliability": reliability, "workload": workload,
            }
        if node_type == 1:
            compute_values = self.normalizer.normalize("compute", state["compute"].to(self.device))
            compute_mask = torch.tensor(self.config.compute_feature_mask, dtype=torch.float32, device=self.device)
            compute = self.compute(compute_values * compute_mask)
            return compute, {"compute": compute}
        raise ValueError(f"历史状态编码只接受 agent/compute，收到 node_type={node_type}")


class EdgeStateEncoder(nn.Module):
    """将时延、固定带宽、丢包率、抖动编码为单时刻边输入。"""

    def __init__(self, config: ModelConfig, normalizer: TypewiseStandardizer):
        super().__init__()
        self.config = config
        self.normalizer = normalizer
        self.encoder = MLP(config.edge_feature_dim, config.state_dim, config.dropout)

    def forward(self, edge: Tensor) -> Tensor:
        values = self.normalizer.normalize("edge", edge.float())
        mask = torch.tensor(self.config.edge_feature_mask, dtype=torch.float32, device=values.device)
        return self.encoder(values * mask)


class NodeEdgeGRUHistory(nn.Module):
    """节点状态序列和链路状态序列分别使用 GRU 编码，不使用置信度或人工趋势。"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        d = config.state_dim
        self.node_gru = nn.GRUCell(d, d)
        self.edge_gru = nn.GRUCell(d, d)
        self.register_buffer("node_memory", torch.zeros(config.num_nodes, d))
        self.register_buffer("last_update", torch.zeros(config.num_nodes))
        self._node_graph: dict[int, Tensor] = {}
        self._edge_memory: dict[tuple[int, int, int], Tensor] = {}
        self._edge_graph: dict[tuple[int, int, int], Tensor] = {}

    @staticmethod
    def edge_key(src: int, dst: int, relation: int) -> tuple[int, int, int]:
        return min(src, dst), max(src, dst), relation

    def reset_state(self) -> None:
        self.node_memory.zero_()
        self.last_update.zero_()
        self._node_graph.clear()
        self._edge_memory.clear()
        self._edge_graph.clear()

    def update_node(self, node_id: int, value: Tensor, timestamp: float, track_grad: bool) -> Tensor:
        old = self._node_graph.get(node_id)
        if old is None:
            old = self.node_memory[node_id].clone()
        updated = self.node_gru(value.unsqueeze(0), old.unsqueeze(0)).squeeze(0)
        with torch.no_grad():
            self.node_memory[node_id].copy_(updated.detach())
            self.last_update[node_id] = timestamp
        if track_grad:
            self._node_graph[node_id] = updated
        return updated

    def update_edge(self, key: tuple[int, int, int], value: Tensor, track_grad: bool) -> Tensor:
        old = self._edge_graph.get(key, self._edge_memory.get(key, torch.zeros_like(value)))
        updated = self.edge_gru(value.unsqueeze(0), old.unsqueeze(0)).squeeze(0)
        self._edge_memory[key] = updated.detach()
        if track_grad:
            self._edge_graph[key] = updated
        return updated

    def read_nodes(self, node_ids: Tensor, query_time: float) -> tuple[Tensor, Tensor]:
        values = torch.stack([self._node_graph.get(int(node), self.node_memory[int(node)]) for node in node_ids])
        ages = torch.as_tensor(query_time, device=values.device) - self.last_update[node_ids]
        return values, ages.clamp_min(0)

    def read_edge(self, key: tuple[int, int, int], fallback: Tensor) -> Tensor:
        return self._edge_graph.get(key, self._edge_memory.get(key, fallback))


class TaskConditionedHGTGRU(nn.Module):
    """GRU 历史编码 + 一层邻居聚合 + 子任务到智能体的可训练注意力。"""

    task_dimension_names = ("latency", "accuracy")
    feature_names = TypedNumericStateEncoder.modality_names

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.normalizer = TypewiseStandardizer()
        self.text_encoder = TextSemanticEncoder(config.text_hash_dim, config.text_dim, config.dropout)
        self.agent_skill_projection = nn.Linear(config.text_dim, config.state_dim)
        self.node_encoder = TypedNumericStateEncoder(config, self.normalizer)
        self.edge_encoder = EdgeStateEncoder(config, self.normalizer)
        self.task_node_encoder = TaskNodeEncoder(config, self.normalizer, self.text_encoder)
        self.history = NodeEdgeGRUHistory(config)
        self.memory = self.history
        d = config.state_dim
        self.relation_embedding = nn.Embedding(2, d)
        self.neighbor_score = MLP(d * 4, 1, config.dropout)
        self.workload_memory = nn.GRUCell(d, d)
        self.register_buffer("latest_workload", torch.zeros(config.num_agents, d))
        self._workload_graph: dict[int, Tensor] = {}
        self.feature_task_query = nn.Linear(d, d, bias=False)
        self.feature_key = nn.Linear(d, d, bias=False)
        self.task_agent_query = nn.Linear(d, d, bias=False)
        self.task_agent_key = nn.Linear(d, d, bias=False)
        self.task_agent_bias = MLP(d * 2, 1, config.dropout)
        self.state_head = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 5))

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def fit_normalizers(self, cases: list[TrainingCase]) -> None:
        self.normalizer.fit(cases)

    def reset_state(self) -> None:
        self.history.reset_state()
        self.latest_workload.zero_()
        self._workload_graph.clear()

    def _update_workload(self, node_id: int, token: Tensor, track_grad: bool) -> Tensor:
        if node_id >= self.config.num_agents:
            return torch.zeros_like(token)
        old = self._workload_graph.get(node_id)
        if old is None:
            old = self.latest_workload[node_id].clone()
        updated = self.workload_memory(token.unsqueeze(0), old.unsqueeze(0)).squeeze(0)
        with torch.no_grad():
            self.latest_workload[node_id].copy_(updated.detach())
        if track_grad:
            self._workload_graph[node_id] = updated
        return updated

    def process_event(self, event: GraphEvent, track_grad: bool = False) -> Dict[str, Tensor]:
        src_input, src_parts = self.node_encoder(event.src_node_type, event.src_state)
        dst_input, dst_parts = self.node_encoder(event.dst_node_type, event.dst_state)
        edge_input = self.edge_encoder(event.edge_state.to(self.device))
        src_memory = self.history.update_node(event.src, src_input, event.timestamp, track_grad)
        dst_memory = self.history.update_node(event.dst, dst_input, event.timestamp, track_grad)
        edge_memory = self.history.update_edge(
            self.history.edge_key(event.src, event.dst, event.relation_type), edge_input, track_grad
        )
        if event.src_node_type == 0:
            self._update_workload(event.src, src_parts["workload"], track_grad)
        if event.dst_node_type == 0:
            self._update_workload(event.dst, dst_parts["workload"], track_grad)
        return {"src_memory": src_memory, "dst_memory": dst_memory, "edge_memory": edge_memory}

    def _skill_matrix(self, graph: QueryGraph, task: TaskRequirement, subtask_count: int) -> Tensor:
        skills = list(task.dag_required_capabilities) or [task.required_capability]
        if graph.candidate_skill_ids and len(graph.candidate_skill_ids) == graph.candidate_ids.numel():
            mask = torch.tensor([
                [skill in set(candidate_skills) for candidate_skills in graph.candidate_skill_ids]
                for skill in skills
            ], dtype=torch.bool, device=self.device)
        else:
            active = graph.candidate_skill_mask
            if active is None:
                active = torch.ones(graph.candidate_ids.numel(), dtype=torch.bool)
            mask = torch.ones(subtask_count, graph.candidate_ids.numel(), dtype=torch.bool, device=self.device)
            mask[task.dag_current_index] = active.to(self.device)
        for row in range(mask.shape[0]):
            if not mask[row].any():
                mask[row] = True
        return mask

    def _agent_skill_tokens(self, graph: QueryGraph) -> Tensor:
        """编码精确 Skill ID 集合；硬匹配仍由 ``_skill_matrix`` 决定。"""
        rows = []
        for skills in graph.candidate_skill_ids:
            ids = tuple(skills) or ("<no-skill>",)
            encoded = torch.cat([self.text_encoder((skill,)) for skill in ids], dim=0).mean(0)
            rows.append(self.agent_skill_projection(encoded))
        if not rows:
            return torch.zeros(graph.candidate_ids.numel(), self.config.state_dim, device=self.device)
        return torch.stack(rows)

    def assign(self, graph: QueryGraph, task: TaskRequirement) -> Dict[str, Tensor]:
        candidate_ids = graph.candidate_ids.long().to(self.device)
        edge_index = graph.edge_index.long().to(self.device)
        relation = graph.relation_type.long().to(self.device)
        all_ids = torch.unique(torch.cat([candidate_ids, edge_index.reshape(-1)]), sorted=True)
        id_to_local = {int(node): index for index, node in enumerate(all_ids.tolist())}
        local_src = torch.tensor([id_to_local[int(node)] for node in edge_index[0].tolist()], device=self.device)
        local_dst = torch.tensor([id_to_local[int(node)] for node in edge_index[1].tolist()], device=self.device)
        candidate_local = torch.tensor([id_to_local[int(node)] for node in candidate_ids.tolist()], device=self.device)
        node_memory, node_age = self.history.read_nodes(all_ids, graph.query_time)

        raw_edge = self.edge_encoder(graph.edge_state.to(self.device))
        edge_memory = torch.stack([
            self.history.read_edge(
                self.history.edge_key(int(src), int(dst), int(rel)), raw_edge[index]
            )
            for index, (src, dst, rel) in enumerate(zip(edge_index[0], edge_index[1], relation))
        ])
        relation_token = self.relation_embedding(relation.clamp_max(1))
        edge_scores = self.neighbor_score(torch.cat([
            node_memory[local_dst], node_memory[local_src], edge_memory, relation_token
        ], dim=-1)).squeeze(-1)
        edge_alpha = segmented_softmax(edge_scores.unsqueeze(-1), local_dst, len(all_ids)).squeeze(-1)

        d = self.config.state_dim
        components = torch.zeros(candidate_ids.numel(), len(self.feature_names), d, device=self.device)
        components[:, 0] = node_memory[candidate_local] + self._agent_skill_tokens(graph)
        candidate_position = {int(node): index for index, node in enumerate(candidate_ids.tolist())}
        for edge_pos, (source, destination) in enumerate(zip(edge_index[0].tolist(), edge_index[1].tolist())):
            if destination not in candidate_position:
                continue
            target = candidate_position[destination]
            message = edge_alpha[edge_pos] * (node_memory[local_src[edge_pos]] + edge_memory[edge_pos])
            if source < self.config.num_agents:
                components[target, 1] += message
            else:
                components[target, 2] += message
            components[target, 3] += edge_alpha[edge_pos] * edge_memory[edge_pos]
        components[:, 4] = torch.stack([
            self._workload_graph.get(int(node), self.latest_workload[int(node)]) for node in candidate_ids
        ])

        task_tokens = self.task_node_encoder(task)
        feature_scores = torch.einsum(
            "sd,cfd->scf", self.feature_task_query(task_tokens), self.feature_key(components)
        ) / math.sqrt(d)
        feature_attention = torch.softmax(feature_scores, dim=-1)
        fused = torch.einsum("scf,cfd->scd", feature_attention, components)
        task_query = self.task_agent_query(task_tokens)
        agent_key = self.task_agent_key(fused)
        pair_bias = self.task_agent_bias(torch.cat([
            task_tokens[:, None, :].expand_as(fused), fused
        ], dim=-1)).squeeze(-1)
        task_agent_logits = torch.einsum("sd,scd->sc", task_query, agent_key) / math.sqrt(d) + pair_bias
        eligibility = self._skill_matrix(graph, task, task_tokens.shape[0])
        task_agent_logits = task_agent_logits.masked_fill(~eligibility, -1.0e4)
        task_agent_attention = torch.softmax(task_agent_logits, dim=-1)
        current = int(task.dag_current_index)
        logits = task_agent_logits[current]
        state_predictions = torch.sigmoid(self.state_head(fused[current]))
        complex_attention = task_agent_attention.sum(dim=0)
        assigned_positions = task_agent_attention.argmax(dim=-1)
        selected_positions = torch.unique(assigned_positions)

        task_count = task_tokens.shape[0]
        task_global_ids = torch.arange(self.config.num_nodes, self.config.num_nodes + task_count, device=self.device)
        task_sources = task_global_ids.repeat_interleave(candidate_ids.numel())
        task_destinations = candidate_ids.repeat(task_count)
        return {
            "candidate_ids": candidate_ids, "candidate_embeddings": fused[current],
            "candidate_logits": logits, "candidate_scores": task_agent_attention[current],
            "state_predictions": state_predictions,
            "attention_weights": edge_alpha.unsqueeze(-1).repeat(1, self.config.attention_heads),
            "feature_attention_weights": feature_attention, "feature_names": self.feature_names,
            "task_agent_logits": task_agent_logits, "task_agent_attention": task_agent_attention,
            "complex_agent_attention": complex_attention,
            "cumulative_complex_attention": task_agent_attention.cumsum(dim=0),
            "subtask_assignments": candidate_ids[assigned_positions],
            "selected_agent_ids": candidate_ids[selected_positions],
            "task_agent_edge_index": torch.stack([task_sources, task_destinations]),
            "dag_attention_weights": task_agent_attention,
            "modality_weights": feature_attention.mean(dim=(0, 1)),
            "eligibility_mask": eligibility[current], "subtask_eligibility_mask": eligibility,
            "candidate_state_age": node_age[candidate_local], "source_state_age": node_age[local_src],
            "id_to_local": id_to_local,
            "system_node_types": torch.cat([
                torch.where(all_ids < self.config.num_agents, 0, 1),
                torch.full((task_count,), 2, dtype=torch.long, device=self.device),
            ]),
            "current_task_node_embedding": task_tokens[current],
        }

    @staticmethod
    def topk(output: Dict[str, Tensor], k: int = 3) -> Dict[str, Tensor]:
        k = min(k, output["candidate_scores"].numel())
        scores, indices = torch.topk(output["candidate_scores"], k)
        return {"node_ids": output["candidate_ids"][indices], "scores": scores}
