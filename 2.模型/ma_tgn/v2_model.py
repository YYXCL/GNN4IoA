"""仅含智能体/算力节点的系统图 + 依赖感知 DAG query 条件化模型。"""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from .config import ModelConfig
from .dag_query import DAGQueryEncoder
from .encoders import MLP, decayed_confidence
from .hgt import HGTEncoder
from .memory import DualTimescaleMemory
from .normalization import TypewiseStandardizer
from .text_encoder import TextSemanticEncoder
from .types import GraphEvent, QueryGraph, TaskRequirement, TrainingCase

Tensor = torch.Tensor


class TypedNumericStateEncoder(nn.Module):
    """智能体与算力节点使用独立字段、独立统计量和独立 MLP。"""

    modality_names = ("context", "environment", "capability", "capability_reliability", "compute")

    def __init__(self, config: ModelConfig, normalizer: TypewiseStandardizer):
        super().__init__()
        self.config = config
        self.normalizer = normalizer
        d = config.modality_dim
        self.context = MLP(3, d, config.dropout)
        self.environment = MLP(8, d, config.dropout)
        self.capability = MLP(config.capability_embedding_dim + 1, d, config.dropout)
        self.reliability = MLP(config.capability_reliability_dim, d, config.dropout)
        compute_dim = 9 if config.use_queue_batch_in_decision else 6
        self.compute = MLP(compute_dim + 1, d, config.dropout)
        self.type_embedding = nn.Embedding(2, d)
        self.fusion = MLP(d * 6, config.state_dim, config.dropout)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @staticmethod
    def _pool(tokens: Tensor, mask: Tensor) -> Tensor:
        weight = mask.float().reshape(-1, 1)
        return (tokens * weight).sum(0) / weight.sum().clamp_min(1.0)

    def forward(
        self,
        node_type: int,
        state: dict[str, Tensor],
        confidence: dict[str, Tensor] | None = None,
        observed_at: dict[str, float] | None = None,
        now: float = 0.0,
    ) -> tuple[Tensor, Tensor]:
        confidence = confidence or {}
        observed_at = observed_at or {}
        d = self.config.modality_dim
        zeros = torch.zeros(d, device=self.device)
        if node_type == 0:
            context_conf = decayed_confidence(
                confidence.get("context"), observed_at.get("context"), now,
                self.config.confidence_tau_seconds, self.device,
            )
            environment_conf = decayed_confidence(
                confidence.get("environment"), observed_at.get("environment"), now,
                self.config.confidence_tau_seconds, self.device,
            )
            context = self.normalizer.normalize("agent_context", state["context"].to(self.device).reshape(-1))
            environment = self.normalizer.normalize("agent_environment", state["environment"].to(self.device).reshape(-1))
            mask = state["capability_mask"].to(self.device).reshape(-1)
            usage = self.normalizer.normalize(
                "capability_usage", state["capability_usage"].to(self.device).reshape(-1, 1),
            )
            capability = state["capability_embeddings"].to(self.device).float()
            reliability = self.normalizer.normalize(
                "capability_reliability", state["capability_reliability"].to(self.device).float(),
            )
            tokens = [
                self.context(torch.cat([context, context_conf.reshape(1)])) * context_conf,
                self.environment(torch.cat([environment, environment_conf.reshape(1)])) * environment_conf,
                self._pool(self.capability(torch.cat([capability, usage], dim=1)), mask),
                self._pool(self.reliability(reliability), mask),
                zeros,
            ]
        elif node_type == 1:
            compute_conf = decayed_confidence(
                confidence.get("compute"), observed_at.get("compute"), now,
                self.config.confidence_tau_seconds, self.device,
            )
            compute = self.normalizer.normalize("compute", state["compute"].to(self.device).reshape(-1))
            if not self.config.use_queue_batch_in_decision:
                compute = compute[:6]
            token = self.compute(torch.cat([compute, compute_conf.reshape(1)])) * compute_conf
            tokens = [zeros, zeros, zeros, zeros, token]
        else:
            raise ValueError(f"系统图只允许 agent/compute 节点，收到 node_type={node_type}")
        stacked = torch.stack(tokens)
        type_token = self.type_embedding(torch.tensor(node_type, device=self.device))
        return self.fusion(torch.cat([stacked.reshape(-1), type_token])), stacked


class TypedEdgeEncoder(nn.Module):
    def __init__(self, config: ModelConfig, normalizer: TypewiseStandardizer):
        super().__init__()
        self.normalizer = normalizer
        self.encoder = MLP(config.edge_feature_dim + 2, config.state_dim, config.dropout)

    def forward(self, edge: Tensor, confidence: Tensor, age: Tensor) -> Tensor:
        single = edge.ndim == 1
        if single:
            edge, confidence, age = edge.unsqueeze(0), confidence.reshape(1), age.reshape(1)
        normalized = self.normalizer.normalize("edge", edge)
        freshness = torch.log1p(age.float()).reshape(-1, 1) / 10.0
        encoded = self.encoder(torch.cat([normalized, confidence.reshape(-1, 1), freshness], dim=-1))
        encoded = encoded * confidence.reshape(-1, 1)
        return encoded.squeeze(0) if single else encoded


class TaskConditionedHGTGRU(nn.Module):
    """任务不是节点：DAG query 与系统图表征在决策头处条件融合。"""

    task_dimension_names = (
        "latency", "accuracy", "cost", "reliability", "privacy", "context", "tool", "multimodal",
    )

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.normalizer = TypewiseStandardizer()
        self.text_encoder = TextSemanticEncoder(config.text_hash_dim, config.text_dim, config.dropout)
        if config.text_dim != config.state_dim:
            raise ValueError("当前实现要求 text_dim == state_dim")
        self.node_encoder = TypedNumericStateEncoder(config, self.normalizer)
        self.edge_encoder = TypedEdgeEncoder(config, self.normalizer)
        self.dag_query_encoder = DAGQueryEncoder(config, self.normalizer, self.text_encoder)
        self.memory = DualTimescaleMemory(config)
        self.memory_projection = MLP(config.memory_dim * 2, config.state_dim, config.dropout)
        self.node_type_embedding = nn.Embedding(len(config.node_types), config.state_dim)
        self.agent_text_fusion = nn.Linear(config.text_dim, config.state_dim, bias=False)
        self.hgt = HGTEncoder(
            config.state_dim, config.attention_heads, len(config.node_types),
            len(config.relation_types), config.hgt_layers, config.dropout,
        )
        self.cross_attention = nn.MultiheadAttention(
            config.state_dim, config.attention_heads, dropout=config.dropout, batch_first=True,
        )
        self.condition_gate = nn.Sequential(nn.Linear(config.state_dim * 2, config.state_dim), nn.Sigmoid())
        self.condition_norm = nn.LayerNorm(config.state_dim)
        self.scorer = nn.Sequential(
            nn.Linear(config.state_dim * 2 + 1, config.state_dim),
            nn.GELU(), nn.Dropout(config.dropout), nn.Linear(config.state_dim, 1),
        )
        self.state_head = nn.Sequential(
            nn.Linear(config.state_dim, config.state_dim // 2), nn.GELU(),
            nn.Linear(config.state_dim // 2, 5),
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def fit_normalizers(self, cases: list[TrainingCase]) -> None:
        self.normalizer.fit(cases)

    def reset_state(self) -> None:
        self.memory.reset_state()

    def process_event(self, event: GraphEvent, track_grad: bool = False) -> Dict[str, Tensor]:
        src_state, src_tokens = self.node_encoder(
            event.src_node_type, event.src_state, event.src_confidence, event.src_observed_at, event.timestamp,
        )
        dst_state, dst_tokens = self.node_encoder(
            event.dst_node_type, event.dst_state, event.dst_confidence, event.dst_observed_at, event.timestamp,
        )
        edge_age_value = max(0.0, event.timestamp - (event.edge_observed_at or event.timestamp))
        edge_age = torch.tensor(edge_age_value, device=self.device)
        confidence = torch.tensor(event.edge_confidence, device=self.device) * torch.exp(
            -edge_age / self.config.confidence_tau_seconds
        )
        edge_state = self.edge_encoder(event.edge_state.to(self.device), confidence, edge_age)
        src_delta = torch.tensor(
            max(0.0, event.timestamp - float(self.memory.last_update[event.src])), device=self.device,
        )
        dst_delta = torch.tensor(
            max(0.0, event.timestamp - float(self.memory.last_update[event.dst])), device=self.device,
        )
        src_message, src_gates = self.memory.build_message(
            src_state, dst_state, edge_state, event.event_type, event.relation_type, src_delta,
        )
        dst_message, dst_gates = self.memory.build_message(
            dst_state, src_state, edge_state, event.event_type, event.relation_type, dst_delta,
        )
        src_fast, src_slow = self.memory.update_one(
            event.src, src_message, src_gates, event.timestamp, track_grad,
        )
        dst_fast, dst_slow = self.memory.update_one(
            event.dst, dst_message, dst_gates, event.timestamp, track_grad,
        )
        return {
            "src_memory": torch.cat([src_fast, src_slow]),
            "dst_memory": torch.cat([dst_fast, dst_slow]),
            "src_modality_tokens": src_tokens,
            "dst_modality_tokens": dst_tokens,
            "src_update_gates": src_gates,
            "dst_update_gates": dst_gates,
        }

    def _eligibility(self, graph: QueryGraph, task: TaskRequirement, texts: list[str]) -> Tensor:
        if graph.candidate_skill_mask is not None:
            mask = graph.candidate_skill_mask.to(self.device).bool().reshape(-1)
        else:
            skill = task.required_capability.strip().lower()
            mask = torch.tensor([bool(skill) and skill in text.lower() for text in texts], device=self.device)
        if mask.numel() != graph.candidate_ids.numel():
            raise ValueError("candidate_skill_mask 与候选智能体数量不一致")
        if not mask.any():
            mask = torch.ones_like(mask)
        return mask

    def _qos_utility(self, predictions: Tensor, task: TaskRequirement) -> Tensor:
        qos = task.values.to(predictions.device).float()
        success, execution, tool_failure, hallucination, executable = predictions.unbind(-1)
        weights = torch.stack([qos[1], qos[0], qos[6], qos[1], qos[3]]).clamp_min(0.05)
        components = torch.stack([
            success, 1.0 - execution, 1.0 - tool_failure, 1.0 - hallucination, executable,
        ], dim=-1)
        return (components * weights).sum(-1) / weights.sum()

    def query(self, graph: QueryGraph, task: TaskRequirement) -> Dict[str, Tensor]:
        candidate_ids = graph.candidate_ids.long().to(self.device)
        original_edge_index = graph.edge_index.long().to(self.device)
        all_global_ids = torch.unique(torch.cat([candidate_ids, original_edge_index.reshape(-1)]), sorted=True)
        id_to_local = {int(node_id): index for index, node_id in enumerate(all_global_ids.tolist())}
        local_edges = torch.tensor(
            [[id_to_local[int(node)] for node in original_edge_index[0].tolist()],
             [id_to_local[int(node)] for node in original_edge_index[1].tolist()]],
            dtype=torch.long, device=self.device,
        )
        candidate_local = torch.tensor(
            [id_to_local[int(node)] for node in candidate_ids.tolist()], device=self.device,
        )

        memories, _ = self.memory.read(all_global_ids, graph.query_time)
        node_type = torch.where(all_global_ids < self.config.num_agents, 0, 1).long()
        x = self.memory_projection(memories) + self.node_type_embedding(node_type)
        texts = list(graph.candidate_texts)
        if len(texts) != candidate_ids.numel():
            texts = [""] * candidate_ids.numel()
        agent_text = self.text_encoder(texts)
        x = x.clone()
        x[candidate_local] = x[candidate_local] + self.agent_text_fusion(agent_text)

        edge_age = (graph.query_time - graph.edge_timestamp.to(self.device).float()).clamp_min(0)
        edge_confidence = graph.edge_confidence.to(self.device).float() * torch.exp(
            -edge_age / self.config.confidence_tau_seconds
        )
        edge_attr = self.edge_encoder(graph.edge_state.to(self.device).float(), edge_confidence, edge_age)
        hgt_nodes, hgt_attention = self.hgt(
            x, node_type, local_edges, graph.relation_type.to(self.device).long(), edge_attr,
        )
        candidates = hgt_nodes[candidate_local]

        task_vector, dag_tokens, current_skill_token = self.dag_query_encoder(task)
        task_context, dag_attention = self.cross_attention(
            candidates.unsqueeze(0), dag_tokens.unsqueeze(0), dag_tokens.unsqueeze(0),
            need_weights=True, average_attn_weights=False,
        )
        task_context = task_context.squeeze(0)
        gate = self.condition_gate(torch.cat([candidates, task_context], dim=-1))
        conditioned = self.condition_norm(candidates + gate * task_context)

        learned_match = torch.nn.functional.cosine_similarity(
            current_skill_token.unsqueeze(0).expand_as(agent_text), agent_text, dim=-1,
        )
        raw_task_text = self.text_encoder.hash_features([task.required_capability]).to(self.device)
        raw_agent_text = self.text_encoder.hash_features(texts).to(self.device)
        raw_match = torch.nn.functional.cosine_similarity(
            raw_task_text.expand_as(raw_agent_text), raw_agent_text, dim=-1,
        )
        semantic_match = 0.75 * raw_match + 0.25 * learned_match
        state_predictions = torch.sigmoid(self.state_head(conditioned))
        qos_utility = self._qos_utility(state_predictions, task)
        logits = self.scorer(torch.cat([
            conditioned,
            task_vector.unsqueeze(0).expand_as(conditioned),
            semantic_match.unsqueeze(-1),
        ], dim=-1)).squeeze(-1)
        logits = (
            logits
            + self.config.semantic_prior_weight * semantic_match
            + self.config.qos_utility_weight * qos_utility
        )
        eligibility = self._eligibility(graph, task, texts)
        logits = logits.masked_fill(~eligibility, -1.0e4)

        _, source_age = self.memory.read(original_edge_index[0], graph.query_time)
        _, candidate_age = self.memory.read(candidate_ids, graph.query_time)
        return {
            "candidate_ids": candidate_ids,
            "candidate_embeddings": conditioned,
            "candidate_logits": logits,
            "candidate_scores": torch.softmax(logits, dim=0),
            "state_predictions": state_predictions,
            "attention_weights": hgt_attention[-1],
            "dag_attention_weights": dag_attention.squeeze(0),
            "modality_weights": dag_attention.mean(dim=(0, 1, 2)),
            "semantic_match": semantic_match,
            "qos_utility": qos_utility,
            "eligibility_mask": eligibility,
            "candidate_state_age": candidate_age,
            "source_state_age": source_age,
            "id_to_local": id_to_local,
            "system_node_types": node_type,
            "dag_query_vector": task_vector,
        }

    @staticmethod
    def topk(output: Dict[str, Tensor], k: int = 3) -> Dict[str, Tensor]:
        k = min(k, output["candidate_scores"].numel())
        scores, indices = torch.topk(output["candidate_scores"], k)
        return {"node_ids": output["candidate_ids"][indices], "scores": scores}
