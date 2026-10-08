"""第2步的第三个前置：精确 MIPS 检索与固定种子的多表 LSH 检索；可行性判断始终先于排序。"""
from __future__ import annotations

import torch


class CandidateIndex:
    """候选检索索引：支持精确最大内积搜索（MIPS）或 LSH 近似检索。"""

    def __init__(self, keys, mode="exact", seed=2718, tables=4, bits=8):
        self.keys = keys.detach()
        self.mode = mode
        self.fallbacks = 0
        if mode == "lsh":
            # 非对称的 MIPS→余弦 变换；未归一化的内积顺序保持不变。
            radius = self.keys.square().sum(-1).max().sqrt().clamp_min(1e-6)
            x = self.keys / radius
            # 增广一维，把内积检索转成余弦/汉明空间上的近似检索
            x = torch.cat([x, (1 - x.square().sum(-1, keepdim=True)).clamp_min(0).sqrt()], -1)
            generator = torch.Generator(device="cpu").manual_seed(seed)
            # 每张表的随机投影，得到每张表的二进制签名
            self.projection = torch.randn(tables, keys.shape[-1] + 1, bits, generator=generator).to(keys)
            self.signatures = torch.einsum("nd,tdb->tnb", x, self.projection) >= 0

    def search(self, query, feasible, k):
        """在可行集合 feasible 内检索与 query 内积最大的 k 个候选。"""
        if not feasible:
            return []
        ids = torch.tensor(feasible, device=self.keys.device, dtype=torch.long)
        k = min(k, len(feasible))
        pool = ids
        if self.mode == "lsh" and len(ids) > max(4 * k, 32):
            # 查询向量同样做增广，再与各表签名比较汉明距离
            q = torch.cat([query.detach(), query.new_zeros(1)])
            sig = torch.einsum("d,tdb->tb", q, self.projection) >= 0
            distance = (self.signatures[:, ids] != sig[:, None]).sum(-1).min(0).values
            pool = ids[distance <= 1]
            if len(pool) < k:
                # 先用最近的桶试探，再做精确回退；回退次数会被记录并披露。
                pool = ids[torch.argsort(distance, stable=True)[:max(4 * k, 32)]]
            if len(pool) < k:
                pool = ids
                self.fallbacks += 1
        scores = self.keys[pool] @ query
        return pool[torch.argsort(scores, descending=True, stable=True)[:k]].tolist()


def recall_at_k(index, query, feasible, k):
    """以精确检索为基准，计算近似检索的 Recall@k。"""
    exact = CandidateIndex(index.keys).search(query, feasible, k)
    approx = index.search(query, feasible, k)
    return len(set(exact) & set(approx)) / max(1, len(exact))
