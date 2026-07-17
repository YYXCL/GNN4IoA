"""事件驱动的快慢双时间尺度节点记忆。"""

from typing import Tuple

import torch
from torch import nn

from .config import ModelConfig
from .encoders import MLP, TimeEncoder

Tensor = torch.Tensor


class DualTimescaleMemory(nn.Module):
    """快慢双时间尺度记忆模块。

    快记忆（GRU 标准更新）: 快速响应短期变化。
    慢记忆（GRU 低速率更新）: 保留长期稳定表征。
    两套记忆拼接后作为注意力输入。

    更新公式:
        fast = old_fast + gate[0] * (proposed_fast - old_fast)
        slow = old_slow + gate[1] * slow_update_rate * (proposed_slow - old_slow)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        # 时间编码器，将 Δt 转为高维特征。
        self.time_encoder = TimeEncoder(config.time_dim)
        # 事件/关系类型嵌入表。
        self.event_embedding = nn.Embedding(len(config.event_types), config.event_dim)          #（5，64）
        self.relation_embedding = nn.Embedding(len(config.relation_types), config.event_dim)    #（4，64）
        # 消息输入: 自身状态 + 对方状态 + 边状态 + 时间编码 + 事件嵌入 + 关系嵌入。
        message_in = config.state_dim * 3 + config.time_dim + config.event_dim * 2  # 消息输入维度之和。
        self.message_encoder = MLP(message_in, config.event_dim, config.dropout)  # 消息编码器。
        # 快记忆 GRU。
        self.fast_cell = nn.GRUCell(config.event_dim, config.memory_dim)
        # 慢记忆 GRU。
        self.slow_cell = nn.GRUCell(config.event_dim, config.memory_dim)
        # 门控网络: [消息, 事件嵌入] → 2 维 (快门, 慢门)，sigmoid 后取值 (0,1)。
        self.update_gate = nn.Sequential(
            nn.Linear(config.event_dim * 2, 2), nn.Sigmoid()
        )

        # ---- 持久记忆缓存（推理用，不参与梯度） ----
        self.register_buffer("fast_memory", torch.zeros(config.num_nodes, config.memory_dim))
        self.register_buffer("slow_memory", torch.zeros(config.num_nodes, config.memory_dim))
        self.register_buffer("last_update", torch.zeros(config.num_nodes))
        # 训练时保留计算图的缓存字典。
        self._fast_graph = {}
        self._slow_graph = {}

    def reset_state(self) -> None:
        """重置所有记忆和缓存为零。"""
        self.fast_memory.zero_()    # 推理时的持久快记忆缓存。
        self.slow_memory.zero_()    # 推理时的持久慢记忆缓存。
        self.last_update.zero_()    # 推理时的最后更新时间戳缓存。
        self._fast_graph.clear()    # 训练时保留的快记忆缓存。
        self._slow_graph.clear()    # 训练时保留的慢记忆缓存。

    def build_message(self, own_state: Tensor, other_state: Tensor, edge_state: Tensor, event_type: int, relation_type: int, delta_t: Tensor,) -> Tuple[Tensor, Tensor]:
        """构建记忆更新消息和门控。
        Returns:
            message: 编码消息，形状 (event_dim,)。
            gates: [快门, 慢门]，形状 (2,)，值域 (0, 1)。
        """
        device = own_state.device  # 获取模型参数所在设备。
        event_id = torch.tensor(event_type, dtype=torch.long, device=device)    # 是事件类型的索引，范围 [0, 5)。
        relation_id = torch.tensor(relation_type, dtype=torch.long, device=device)  # 是关系类型的索引，范围 [0, 4)。
        event_token = self.event_embedding(event_id)  # 事件类型的嵌入，形状 (64,)。
        relation_token = self.relation_embedding(relation_id)  # 关系类型的嵌入，形状 (64,)。
        time_token = self.time_encoder(delta_t.reshape(1)).squeeze(0)  # 时间编码。
        # 拼接所有信息源 → 消息编码。
        raw = torch.cat(
            [own_state, other_state, edge_state, time_token, event_token, relation_token]
        )
        message = self.message_encoder(raw)  # MLP消息编码。
        # 门控值由消息和事件嵌入共同决定。 作用：根据事件类型和消息，动态调整快记忆和慢记忆的更新比例。
        gates = self.update_gate(torch.cat([message, event_token]))
        return message, gates

    def update_one(self, node_id: int, message: Tensor, gates: Tensor, timestamp: float, track_grad: bool = False,) -> Tuple[Tensor, Tensor]:
        """对单个节点执行一次记忆更新。
        Args:
            node_id: 节点 ID，范围 [0, num_nodes)。
            message: 事件消息，形状 (event_dim,)。
            gates: [快门, 慢门]，形状 (2,)。
            timestamp: 当前时间戳。
            track_grad: 训练时 True 保留计算图，推理时 False 节省内存。

        Returns:
            fast: 更新后快记忆，形状 (memory_dim,)。
            slow: 更新后慢记忆，形状 (memory_dim,)。
        """
        # 优先从计算图缓存读取旧状态，否则从持久 buffer 克隆。
        old_fast = self._fast_graph.get(node_id)
        old_slow = self._slow_graph.get(node_id)
        if old_fast is None:
            old_fast = self.fast_memory[node_id].clone()
        if old_slow is None:
            old_slow = self.slow_memory[node_id].clone()

        # GRUCell 计算候选新状态。作用：根据消息和旧状态，计算出新的快记忆和慢记忆。
        proposed_fast = self.fast_cell(message.unsqueeze(0), old_fast.unsqueeze(0)).squeeze(0)
        proposed_slow = self.slow_cell(message.unsqueeze(0), old_slow.unsqueeze(0)).squeeze(0)

        # 增量门控更新。作用：根据门控值和候选新状态，更新快记忆和慢记忆。
        fast = old_fast + gates[0] * (proposed_fast - old_fast)  # 快记忆更新。
        slow_rate = gates[1] * self.config.slow_update_rate     # 慢记忆额外衰减。
        slow = old_slow + slow_rate * (proposed_slow - old_slow)  # 慢记忆更新。

        # 无梯度写回持久 buffer。
        with torch.no_grad():
            self.fast_memory[node_id].copy_(fast.detach())  # 将更新了的快记忆写回持久 buffer。detach函数的作用：将张量从计算图中分离，避免计算梯度。
            self.slow_memory[node_id].copy_(slow.detach())  # 将更新了的慢记忆写回持久 buffer。
            self.last_update[node_id] = timestamp    # 添加上更新的时间戳。

        # 训练时保留计算图。
        if track_grad:
            self._fast_graph[node_id] = fast
            self._slow_graph[node_id] = slow

        return fast, slow

    def read(self, node_ids: Tensor, query_time: float) -> Tuple[Tensor, Tensor]:
        """批量读取节点记忆和状态年龄。

        训练时优先从图缓存读取（保留梯度），推理时从持久 buffer 读取。

        Returns:
            memory: 快慢记忆拼接，形状 (N, memory_dim * 2)。
            age: 状态年龄 = query_time - last_update，形状 (N,)。
        """
        # 训练时读图缓存，推理时读持久 buffer。
        fast = torch.stack(
            [self._fast_graph.get(int(i), self.fast_memory[int(i)]) for i in node_ids]  # 从图缓存读取快记忆，形状 (N, memory_dim)。
        )
        slow = torch.stack(
            [self._slow_graph.get(int(i), self.slow_memory[int(i)]) for i in node_ids]  # 从图缓存读取慢记忆，形状 (N, memory_dim)。
        )
        age = torch.as_tensor(query_time, device=fast.device) - self.last_update[node_ids]  # 计算状态年龄，形状 (N,)。
        return torch.cat([fast, slow], dim=-1), age.clamp_min(0)
        
