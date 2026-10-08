"""第2步的第二个前置：纯 PyTorch 实现的异构注意力层，不依赖可选的 scatter 扩展库。"""
import math
import torch
from torch import nn


def segment_softmax(values, index, size):
    """分段 softmax：对属于同一索引分组的元素做归一化。

    values: [边数, 头数]；index: 每条边对应的目标节点；size: 节点总数。
    """
    expanded = index[:, None].expand_as(values)
    maximum = values.new_full((size, values.shape[1]), -torch.inf)
    maximum.scatter_reduce_(0, expanded, values, reduce="amax", include_self=True)
    weight = (values - maximum[index]).exp()
    total = values.new_zeros((size, values.shape[1])).scatter_add_(0, expanded, weight)
    return weight / total[index].clamp_min(1e-10)


class HeterogeneousLayer(nn.Module):
    """异构图注意力层：按节点类型与关系类型分别使用独立的投影参数。"""

    def __init__(self, d, heads, node_types=6, relations=10, edge_updates=False):
        super().__init__()
        self.d, self.heads, self.edge_updates = d, heads, edge_updates
        # 每种节点类型各有独立的 Q/K/V 投影
        self.q = nn.ModuleList(nn.Linear(d, d) for _ in range(node_types))
        self.k = nn.ModuleList(nn.Linear(d, d) for _ in range(node_types))
        self.v = nn.ModuleList(nn.Linear(d, d) for _ in range(node_types))
        # 每种关系类型各有独立的 K/V 投影（关系相关注意力）
        self.rel_k = nn.ModuleList(nn.Linear(d, d, bias=False) for _ in range(relations))
        self.rel_v = nn.ModuleList(nn.Linear(d, d, bias=False) for _ in range(relations))
        self.ek, self.ev, self.eq = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, d * 2), nn.GELU(), nn.Linear(d * 2, d))
        self.norm = nn.LayerNorm(d)
        self.raw_node = nn.Linear(d * 2, d)
        self.raw_edge = nn.Linear(d * 2, d)

    @staticmethod
    def typed(modules, x, kinds):
        """按类型给每个元素选择对应的投影层并计算。"""
        out = torch.zeros_like(x)
        for kind, layer in enumerate(modules):
            mask = kinds == kind
            if mask.any():
                out[mask] = layer(x[mask])
        return out

    def forward(self, x, kinds, source, target, relations, edge, initial_x=None, initial_edge=None):
        # 计算每条边的 Q/K/V（分别来自目标节点、源节点，并叠加关系投影）
        q = self.typed(self.q, x, kinds)[target]
        k = self.typed(self.k, x, kinds)[source]
        v = self.typed(self.v, x, kinds)[source]
        k = self.typed(self.rel_k, k, relations)
        v = self.typed(self.rel_v, v, relations)
        reshape = lambda a: a.reshape(-1, self.heads, self.d // self.heads)
        if self.edge_updates:
            # TARGETNET：节点注意力与边注意力分离；边注意力不做 softmax 归一化。
            alpha = segment_softmax((reshape(q) * reshape(k)).sum(-1) / math.sqrt(self.d // self.heads), target, len(x))
            beta = (reshape(self.eq(x[target])) * reshape(self.ek(edge))).sum(-1) / math.sqrt(self.d // self.heads)
            em = beta.unsqueeze(-1) * reshape(self.ev(edge))
            message = alpha.unsqueeze(-1) * reshape(v) + em
            new_edge = self.raw_edge(torch.cat([em.flatten(1), initial_edge], -1))
        else:
            # 标准做法：把边特征直接加到 K/V 上再算注意力
            alpha = segment_softmax((reshape(q) * reshape(k + self.ek(edge))).sum(-1) / math.sqrt(self.d // self.heads), target, len(x))
            message = alpha.unsqueeze(-1) * reshape(v + self.ev(edge))
            new_edge = edge
        # 按目标节点聚合消息，再经前馈与残差归一化
        aggregate = torch.zeros_like(x).index_add_(0, target, message.flatten(1))
        result = self.norm(x + self.ff(aggregate))
        if self.edge_updates:
            result = self.raw_node(torch.cat([result, initial_x], -1))
        return result, new_edge
