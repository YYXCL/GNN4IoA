"""GMATANN 可训练复现：时序 CVAE、task-agent 图与全局-局部注意力融合。"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn

from .data import GCANCase


@dataclass
class GCANConfig:
    numeric_dim: int = 24
    text_dim: int = 256
    hidden_dim: int = 64
    heads: int = 4
    layers: int = 2
    edge_dim: int = 8
    cvae_lstm_layers: int = 1
    dropout: float = 0.05

    def __post_init__(self) -> None:
        if self.hidden_dim % self.heads:
            raise ValueError("hidden_dim 必须能被 heads 整除")
        if self.hidden_dim % 2:
            raise ValueError("双向 LSTM 要求 hidden_dim 为偶数")

    def to_dict(self) -> dict:
        return asdict(self)


def segmented_softmax(scores: torch.Tensor, destinations: torch.Tensor, node_count: int) -> torch.Tensor:
    output = torch.zeros_like(scores)
    for destination in range(node_count):
        mask = destinations == destination
        if mask.any():
            output[mask] = torch.softmax(scores[mask], dim=0)
    return output


class TimeSeriesConditionalVariationalAutoencoder(nn.Module):
    """论文 A 节的条件 VAE，并以双向 LSTM 显式处理有序状态序列。"""

    def __init__(self, input_dim: int, dim: int, layers: int, dropout: float):
        super().__init__()
        recurrent_dropout = dropout if layers > 1 else 0.0
        self.observation = nn.Sequential(nn.Linear(input_dim + 1, dim), nn.LayerNorm(dim), nn.GELU())
        self.encoder = nn.LSTM(
            dim * 2,
            dim // 2,
            num_layers=layers,
            dropout=recurrent_dropout,
            bidirectional=True,
            batch_first=True,
        )
        self.mean = nn.Linear(dim, dim)
        self.log_variance = nn.Linear(dim, dim)
        self.decoder_seed = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU())
        self.decoder = nn.LSTM(
            dim,
            dim,
            num_layers=layers,
            dropout=recurrent_dropout,
            batch_first=True,
        )
        self.reconstruction = nn.Linear(dim, input_dim)

    def forward(
        self,
        sequence: torch.Tensor,
        mask: torch.Tensor,
        delta_time: torch.Tensor,
        condition: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        lengths = mask.long().sum(dim=1).clamp_min(1)
        observation = self.observation(torch.cat([sequence, delta_time], dim=-1))
        conditioned = torch.cat([
            observation,
            condition.unsqueeze(1).expand(-1, sequence.shape[1], -1),
        ], dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(
            conditioned, lengths.cpu(), batch_first=True, enforce_sorted=False,
        )
        _, (hidden, _) = self.encoder(packed)
        encoded = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        mean = self.mean(encoded)
        log_variance = self.log_variance(encoded).clamp(-8.0, 5.0)
        if self.training:
            latent = mean + torch.exp(0.5 * log_variance) * torch.randn_like(mean)
        else:
            latent = mean

        seed = self.decoder_seed(torch.cat([latent, condition], dim=-1))
        decoder_input = seed.unsqueeze(1).expand(-1, sequence.shape[1], -1)
        decoded, _ = self.decoder(decoder_input)
        reconstruction = self.reconstruction(decoded)
        valid = mask.unsqueeze(-1).expand_as(sequence)
        reconstruction_loss = (reconstruction[valid] - sequence[valid]).square().mean()
        kl_loss = -0.5 * (1.0 + log_variance - mean.square() - log_variance.exp()).mean()
        return {
            "latent": latent,
            "mean": mean,
            "log_variance": log_variance,
            "reconstruction": reconstruction,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }


class RelationalGraphAttention(nn.Module):
    """关系与边特征共同参与 Q/K/V 的 task-agent 图注意力层。"""

    def __init__(self, dim: int, heads: int, relation_count: int, edge_dim: int, dropout: float):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.relation_key = nn.Embedding(relation_count, dim)
        self.relation_value = nn.Embedding(relation_count, dim)
        self.edge_key = nn.Linear(edge_dim, dim)
        self.edge_value = nn.Linear(edge_dim, dim)
        self.edge_bias = nn.Linear(edge_dim, heads)
        self.output = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        relation: torch.Tensor,
        edge_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index
        query = self.query(x)[dst].view(-1, self.heads, self.head_dim)
        key = (
            self.key(x)[src] + self.relation_key(relation) + self.edge_key(edge_state)
        ).view(-1, self.heads, self.head_dim)
        value = (
            self.value(x)[src] + self.relation_value(relation) + self.edge_value(edge_state)
        ).view(-1, self.heads, self.head_dim)
        scores = (query * key).sum(-1) / math.sqrt(self.head_dim) + self.edge_bias(edge_state)
        attention = segmented_softmax(scores, dst, x.shape[0])
        messages = (attention.unsqueeze(-1) * value).reshape(-1, self.dim)
        aggregate = torch.zeros_like(x)
        aggregate.index_add_(0, dst, messages)
        return self.norm(x + self.dropout(self.output(aggregate))), attention


class FeatureAttention(nn.Module):
    """对应论文式(2)-(5)：Q/K/V、softmax、加权求和及 tau 残差。"""

    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, dim)
        self.tau = nn.Parameter(torch.tensor(1.0))
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, delta: torch.Tensor, allowed: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        node_count = delta.shape[0]
        query = self.query(delta).view(node_count, self.heads, self.head_dim).transpose(0, 1)
        key = self.key(delta).view(node_count, self.heads, self.head_dim).transpose(0, 1)
        value = self.value(delta).view(node_count, self.heads, self.head_dim).transpose(0, 1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if allowed is not None:
            scores = scores.masked_fill(~allowed.unsqueeze(0), -1.0e4)
        attention = torch.softmax(scores, dim=-1)
        attended = torch.matmul(self.dropout(attention), value)
        attended = attended.transpose(0, 1).contiguous().view(node_count, -1)
        output = self.norm(delta + self.tau * self.output(attended))
        return output, attention


class GlobalLocalFeatureFusion(nn.Module):
    """局部邻域与全图两条注意力支路的深层特征融合。"""

    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.local_attention = FeatureAttention(dim, heads, dropout)
        self.global_attention = FeatureAttention(dim, heads, dropout)
        self.branch_gate = nn.Linear(dim * 3, 2)
        self.deep_fusion = nn.Sequential(
            nn.Linear(dim * 3, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, delta: torch.Tensor, local_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        local, local_weights = self.local_attention(delta, local_mask)
        global_, global_weights = self.global_attention(delta, None)
        joined = torch.cat([delta, local, global_], dim=-1)
        weights = torch.softmax(self.branch_gate(joined), dim=-1)
        mixed = weights[:, :1] * local + weights[:, 1:] * global_
        fused = self.norm(delta + mixed + self.deep_fusion(joined))
        return {
            "fused": fused,
            "local_features": local,
            "global_features": global_,
            "local_attention": local_weights,
            "global_attention": global_weights,
            "fusion_weights": weights,
        }


class GraphConvolutionalAssignmentNetwork(nn.Module):
    """论文的单当前任务 task-agent 图；不读取复杂任务 DAG 字段。"""

    def __init__(self, config: GCANConfig):
        super().__init__()
        self.config = config
        d = config.hidden_dim
        self.latest_numeric = nn.Sequential(nn.Linear(config.numeric_dim, d), nn.LayerNorm(d), nn.GELU())
        self.text = nn.Sequential(nn.Linear(config.text_dim, d), nn.LayerNorm(d), nn.GELU())
        self.task_qos = nn.Sequential(nn.Linear(2, d), nn.LayerNorm(d), nn.GELU())
        self.task_fusion = nn.Sequential(nn.Linear(d * 3, d), nn.GELU())
        self.cvae_condition = nn.Sequential(nn.Linear(d * 2, d), nn.GELU())
        self.cvae = TimeSeriesConditionalVariationalAutoencoder(
            config.numeric_dim, d, config.cvae_lstm_layers, config.dropout,
        )
        self.agent_fusion = nn.Sequential(nn.Linear(d * 3, d), nn.GELU())
        self.node_type = nn.Embedding(2, d)
        self.layers = nn.ModuleList(
            RelationalGraphAttention(d, config.heads, 3, config.edge_dim, config.dropout)
            for _ in range(config.layers)
        )
        self.global_local_fusion = GlobalLocalFeatureFusion(d, config.heads, config.dropout)
        self.assignment = nn.Sequential(nn.Linear(d * 3, d), nn.GELU(), nn.Linear(d, 1))
        self.state_head = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 5))

    @staticmethod
    def _graph(case: GCANCase, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        candidate_count = case.candidate_ids.numel()
        task = candidate_count
        task_edges: list[tuple[int, int]] = []
        task_relations: list[int] = []
        task_features: list[torch.Tensor] = []
        edge_features = case.task_agent_edge_state.to(device)
        for agent in range(candidate_count):
            task_edges.extend([(task, agent), (agent, task)])
            task_relations.extend([0, 1])
            task_features.extend([edge_features[agent], edge_features[agent]])

        edge_index = torch.tensor(task_edges, dtype=torch.long, device=device).t().contiguous()
        relations = torch.tensor(task_relations, dtype=torch.long, device=device)
        states = torch.stack(task_features)
        if case.collaboration_edge_index.numel():
            collaboration = case.collaboration_edge_index.to(device)
            edge_index = torch.cat([edge_index, collaboration], dim=1)
            relations = torch.cat([
                relations,
                torch.full((collaboration.shape[1],), 2, dtype=torch.long, device=device),
            ])
            states = torch.cat([states, case.collaboration_edge_state.to(device)], dim=0)
        return edge_index, relations, states

    @staticmethod
    def _local_mask(edge_index: torch.Tensor, node_count: int) -> torch.Tensor:
        mask = torch.eye(node_count, dtype=torch.bool, device=edge_index.device)
        if edge_index.numel():
            mask[edge_index[1], edge_index[0]] = True
        return mask

    def forward(self, case: GCANCase) -> dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        agent_text = self.text(case.agent_text.to(device))
        task_text = case.task_text.to(device)
        task = self.task_fusion(torch.cat([
            self.text(task_text[0]),
            self.text(task_text[1]),
            self.task_qos(case.task_qos.to(device)),
        ], dim=-1))
        condition = self.cvae_condition(torch.cat([
            agent_text,
            task.unsqueeze(0).expand_as(agent_text),
        ], dim=-1))
        cvae = self.cvae(
            case.agent_history.to(device),
            case.history_mask.to(device),
            case.history_delta_time.to(device),
            condition,
        )
        latest = self.latest_numeric(case.agent_numeric.to(device))
        agents = self.agent_fusion(torch.cat([cvae["latent"], latest, agent_text], dim=-1))
        nodes = torch.cat([
            agents + self.node_type(torch.tensor(0, device=device)),
            (task + self.node_type(torch.tensor(1, device=device))).unsqueeze(0),
        ])

        edge_index, relation, edge_state = self._graph(case, device)
        graph_attentions = []
        for layer in self.layers:
            nodes, attention = layer(nodes, edge_index, relation, edge_state)
            graph_attentions.append(attention)

        fusion = self.global_local_fusion(nodes, self._local_mask(edge_index, nodes.shape[0]))
        fused = fusion["fused"]
        agent_nodes = fused[:-1]
        task_node = fused[-1]
        logits = self.assignment(torch.cat([
            agent_nodes,
            task_node.unsqueeze(0).expand_as(agent_nodes),
            agent_nodes * task_node.unsqueeze(0),
        ], dim=-1)).squeeze(-1)
        skill_mask = case.skill_mask.to(device)
        logits = logits.masked_fill(~skill_mask, -1.0e4)
        return {
            "candidate_logits": logits,
            "candidate_scores": torch.softmax(logits, dim=0),
            "state_predictions": torch.sigmoid(self.state_head(agent_nodes)),
            "kl_loss": cvae["kl_loss"],
            "reconstruction_loss": cvae["reconstruction_loss"],
            "reconstructed_history": cvae["reconstruction"],
            "cvae_mean": cvae["mean"],
            "cvae_log_variance": cvae["log_variance"],
            "attention_weights": graph_attentions[-1],
            "graph_attention_weights": graph_attentions,
            "local_attention_weights": fusion["local_attention"],
            "global_attention_weights": fusion["global_attention"],
            "fusion_weights": fusion["fusion_weights"],
            "node_embeddings": fused,
            "edge_index": edge_index,
            "edge_state": edge_state,
            "relation_type": relation,
        }
