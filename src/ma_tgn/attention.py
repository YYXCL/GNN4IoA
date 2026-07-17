"""任务条件化多头图注意力层。"""

import math
from typing import Dict, Tuple

import torch
from torch import nn

from .config import ModelConfig
from .encoders import TimeEncoder

Tensor = torch.Tensor


def segmented_softmax(scores: Tensor, groups: Tensor, num_groups: int) -> Tensor:
    """按目标节点分组的 softmax，每个节点对其入边独立归一化。

    Args:
        scores: 注意力分数，形状 (E, H)，E=边数，H=注意力头数。
        groups: 每条边所属的目标节点局部索引，形状 (E,)。
        num_groups: 目标节点总数 C。

    Returns:
        归一化权重，形状 (E, H)。
    """
    output = torch.zeros_like(scores)
    for group_id in range(num_groups):
        mask = groups == group_id
        if torch.any(mask):
            output[mask] = torch.softmax(scores[mask], dim=0)
    return output


class TaskConditionedRelationalAttention(nn.Module):
    """融合任务条件、关系类型、边特征和时间新鲜度的多头图注意力。

    核心设计:
        1. Query = 候选节点记忆 + 任务嵌入。
        2. Key = 源节点记忆 + 关系嵌入 + 任务嵌入。
        3. Value = 源节点记忆 + 边特征。
        4. 注意力分数叠加边偏差和时间新鲜度偏差。
        5. 按目标节点分段 softmax → 加权聚合。
        6. 残差连接 + MLP 输出。
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.heads = config.attention_heads
        self.head_dim = config.memory_dim // config.attention_heads
        # 记忆输入 = 快记忆 + 慢记忆 拼接。
        memory_in = config.memory_dim * 2

        # ---- 核心注意力投影 ----
        self.query = nn.Linear(memory_in, config.memory_dim)
        self.key = nn.Linear(memory_in, config.memory_dim)
        self.value = nn.Linear(memory_in, config.memory_dim)
        self.residual = nn.Linear(memory_in, config.memory_dim)

        # ---- 任务条件注入 ----
        self.task_query = nn.Linear(config.task_dim, config.memory_dim)
        self.task_key = nn.Linear(config.task_dim, config.memory_dim)

        # ---- 边特征增强 ----
        self.edge_value = nn.Linear(config.state_dim, config.memory_dim)
        self.edge_bias = nn.Linear(config.state_dim, self.heads)

        # ---- 关系类型编码 ----
        self.relation = nn.Embedding(len(config.relation_types), config.memory_dim)

        # ---- 时间新鲜度 ----
        self.time_encoder = TimeEncoder(config.time_dim)
        self.time_bias = nn.Linear(config.time_dim, self.heads)

        # ---- 输出变换 ----
        self.output = nn.Sequential(
            nn.Linear(config.memory_dim * 2, config.memory_dim),
            nn.LayerNorm(config.memory_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        candidate_ids: Tensor,
        candidate_memory: Tensor,
        src_memory: Tensor,
        edge_dst_ids: Tensor,
        edge_state: Tensor,
        relation_type: Tensor,
        edge_age: Tensor,
        task: Tensor,
        latent_gate: Tensor,
    ) -> Tuple[Tensor, Tensor, Dict[int, int]]:
        """前向传播：计算任务条件化的图注意力。

        Args:
            candidate_ids: 候选节点 ID，形状 (C,)。
            candidate_memory: 候选节点快慢记忆拼接，形状 (C, memory_dim*2)。
            src_memory: 源节点快慢记忆拼接，形状 (E, memory_dim*2)。
            edge_dst_ids: 边目的节点全局 ID，形状 (E,)。
            edge_state: 边特征，形状 (E, state_dim)。
            relation_type: 关系类型索引，形状 (E,)。
            edge_age: 边年龄（Δt），形状 (E,)。
            task: 任务嵌入，形状 (task_dim,)。
            latent_gate: 门控向量，形状 (memory_dim*2,)，值域 (0,1)。

        Returns:
            output: 候选节点嵌入，形状 (C, memory_dim)。
            alpha: 注意力权重，形状 (E, H)。
            id_to_local: 全局 ID → 局部索引映射。

        Raises:
            ValueError: edge_dst_ids 中存在不在 candidate_ids 中的节点。
        """
        # 用任务相关的潜在门控调控记忆读出量。
        candidate_memory = candidate_memory * latent_gate
        src_memory = src_memory * latent_gate

        # 构建全局节点 ID → 局部候选索引映射。
        id_to_local = {int(node_id): k for k, node_id in enumerate(candidate_ids.tolist())}
        try:
            dst_local = torch.tensor(
                [id_to_local[int(i)] for i in edge_dst_ids.tolist()],
                dtype=torch.long,
                device=candidate_ids.device,
            )
        except KeyError as exc:
            raise ValueError("每条查询边的目标节点必须包含在 candidate_ids 中") from exc

        # Query = 候选节点记忆 + 任务嵌入，选目的节点对应行 → (E, H, D_head)。
        q_all = self.query(candidate_memory) + self.task_query(task).unsqueeze(0)
        q = q_all[dst_local].view(-1, self.heads, self.head_dim)

        # Key = 源节点记忆 + 关系嵌入 + 任务嵌入 → (E, H, D_head)。
        relation = self.relation(relation_type.long())
        k = (self.key(src_memory) + relation + self.task_key(task)).view(
            -1, self.heads, self.head_dim
        )

        # Value = 源节点记忆 + 边特征 → (E, H, D_head)。
        v = (self.value(src_memory) + self.edge_value(edge_state)).view(
            -1, self.heads, self.head_dim
        )

        # 缩放点积注意力分数 (E, H)。
        score = (q * k).sum(-1) / math.sqrt(self.head_dim)
        # 边特征偏差：不同链路质量获得不同注意力偏好。
        score = score + self.edge_bias(edge_state)
        # 时间新鲜度偏差：较新的边获得更高分数。
        score = score + self.time_bias(self.time_encoder(edge_age))

        # 按目的节点分段 softmax 归一化。
        alpha = segmented_softmax(score, dst_local, candidate_ids.numel())

        # 加权聚合：注意力权重 × Value → 展平合并所有头 → 按目的节点求和。
        messages = (alpha.unsqueeze(-1) * v).reshape(-1, self.heads * self.head_dim)
        aggregated = torch.zeros(
            candidate_ids.numel(), self.heads * self.head_dim, device=messages.device
        )
        aggregated.index_add_(0, dst_local, messages)

        # 残差连接 + 输出 MLP。
        base = self.residual(candidate_memory)
        return self.output(torch.cat([base, aggregated], dim=-1)), alpha, id_to_local
