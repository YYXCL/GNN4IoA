from typing import Dict

import torch
from torch import nn 

from .attention import TaskConditionedRelationalAttention
from .config import ModelConfig
from .encoders import EdgeStateEncoder, HeterogeneousStateEncoder, TaskEncoder
from .memory import DualTimescaleMemory
from .types import GraphEvent, QueryGraph, TaskRequirement

Tensor = torch.Tensor


class TaskConditionedHeterogeneousTGN(nn.Module):
    """异构编码、CTDG/TGN 和记忆式任务条件注意力的统一模型。"""

    task_dimension_names = ("latency", "accuracy", "cost", "reliability", "privacy", "context", "tool", "multimodal")

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.node_encoder = HeterogeneousStateEncoder(config)
        self.edge_encoder = EdgeStateEncoder(config)
        self.task_encoder = TaskEncoder(config)
        self.memory = DualTimescaleMemory(config)
        self.attention = TaskConditionedRelationalAttention(config)
        self.scorer = nn.Sequential(
            nn.Linear(config.memory_dim + config.task_dim, config.memory_dim),  # 输入层，形状 (memory_dim + task_dim, memory_dim)。
            nn.GELU(),  # 激活函数。
            nn.Dropout(config.dropout),  # 随机失活层。
            nn.Linear(config.memory_dim, 1),  # 输出层，形状 (memory_dim, 1)。
        )
        # 不聚合可靠性：[成功率, 执行时间, 工具失败率, 幻觉率, 可执行性]。
        self.state_head = nn.Sequential(
            nn.Linear(config.memory_dim, config.memory_dim // 2),
            nn.GELU(),
            nn.Linear(config.memory_dim // 2, 5),
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device  # 返回模型参数所在设备。

    def reset_state(self) -> None:
        self.memory.reset_state()  # 重置记忆状态。

    def process_event(self, event: GraphEvent, track_grad: bool = False) -> Dict[str, Tensor]:
        """状态发生变化时调用；无需按固定时间间隔重放全图。"""
        src_state, src_tokens = self.node_encoder(
            event.src_node_type, event.src_state, event.src_confidence,
            event.src_observed_at, event.timestamp,
        )
        dst_state, dst_tokens = self.node_encoder(
            event.dst_node_type, event.dst_state, event.dst_confidence,
            event.dst_observed_at, event.timestamp,
        )
        edge_age = max(0.0, event.timestamp - (event.edge_observed_at or event.timestamp))
        edge_confidence = event.edge_confidence * torch.exp(torch.tensor(-edge_age / self.config.confidence_tau_seconds))
        edge_state = self.edge_encoder(event.edge_state.to(self.device), edge_confidence.to(self.device))
        src_delta = torch.tensor( # 源节点时间戳与上次更新时间戳的差值。
            max(0.0, event.timestamp - float(self.memory.last_update[event.src])),
            device=self.device,
        )
        dst_delta = torch.tensor( # 目标节点时间戳与上次更新时间戳的差值。
            max(0.0, event.timestamp - float(self.memory.last_update[event.dst])),
            device=self.device,
        )
        src_message, src_gates = self.memory.build_message( # 源节点消息和更新门限。
            src_state, dst_state, edge_state, event.event_type, event.relation_type, src_delta
        )
        dst_message, dst_gates = self.memory.build_message( # 目标节点消息和更新门限。
            dst_state, src_state, edge_state, event.event_type, event.relation_type, dst_delta
        )
        src_fast, src_slow = self.memory.update_one( # 源节点记忆更新。
            event.src, src_message, src_gates, event.timestamp, track_grad
        )
        dst_fast, dst_slow = self.memory.update_one( # 目标节点记忆更新。
            event.dst, dst_message, dst_gates, event.timestamp, track_grad
        )
        return {
            "src_memory": torch.cat([src_fast, src_slow]),
            "dst_memory": torch.cat([dst_fast, dst_slow]),
            "src_modality_tokens": src_tokens,
            "dst_modality_tokens": dst_tokens,
            "src_update_gates": src_gates,
            "dst_update_gates": dst_gates,
        }

    def query(self, graph: QueryGraph, task: TaskRequirement) -> Dict[str, Tensor]:
        """仅在需要编排或调度决策时，按需形成任务相关状态嵌入。"""
        candidate_ids = graph.candidate_ids.long().to(self.device)  # 候选节点ID。
        edge_index = graph.edge_index.long().to(self.device)  # 边索引。
        edge_state_raw = graph.edge_state.float().to(self.device)  # 边状态。
        relation_type = graph.relation_type.long().to(self.device)  # 关系类型。
        edge_timestamp = graph.edge_timestamp.float().to(self.device)  # 边时间戳。
        task_values = task.values.float().to(self.device)
        task_capability = task.capability_embedding.float().to(self.device)

        candidate_memory, candidate_age = self.memory.read(candidate_ids, graph.query_time)  # 候选节点记忆与年龄。
        src_memory, src_age = self.memory.read(edge_index[0], graph.query_time)  # 源节点记忆与年龄。
        edge_age = (graph.query_time - edge_timestamp).clamp_min(0)  # 边年龄。
        edge_confidence = graph.edge_confidence.float().to(self.device) * torch.exp(-edge_age / self.config.confidence_tau_seconds)
        encoded_edges = self.edge_encoder(edge_state_raw, edge_confidence)
        task_embedding, modality_weights, latent_gate = self.task_encoder(task_values, task_capability)
        embeddings, attention, id_to_local = self.attention(  # 关系注意力。
            candidate_ids,
            candidate_memory,
            src_memory,
            edge_index[1],
            encoded_edges,
            relation_type,
            edge_age,
            task_embedding,
            latent_gate,
        )
        repeated_task = task_embedding.unsqueeze(0).expand(candidate_ids.numel(), -1)  # 任务嵌入重复。
        logits = self.scorer(torch.cat([embeddings, repeated_task], dim=-1)).squeeze(-1)  # 候选节点日志。
        state_predictions = torch.sigmoid(self.state_head(embeddings))
        return {
            "candidate_ids": candidate_ids,
            "candidate_embeddings": embeddings,
            "candidate_logits": logits,
            "candidate_scores": torch.sigmoid(logits),
            "state_predictions": state_predictions,
            "attention_weights": attention,
            "modality_weights": modality_weights,
            "candidate_state_age": candidate_age,
            "source_state_age": src_age,
            "id_to_local": id_to_local,
        }

    def topk(self, output: Dict[str, Tensor], k: int = 3) -> Dict[str, Tensor]:
        k = min(k, output["candidate_scores"].numel())  # 取前 k 个候选节点。
        scores, indices = torch.topk(output["candidate_scores"], k)  # 候选节点日志排序。
        return {"node_ids": output["candidate_ids"][indices], "scores": scores}  # 返回前 k 个候选节点ID和日志。
        
