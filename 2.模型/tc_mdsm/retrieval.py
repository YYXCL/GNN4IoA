"""Exact MIPS and fixed-seed multi-table LSH; feasibility always precedes ranking."""
from __future__ import annotations

import torch


class CandidateIndex:
    def __init__(self, keys, mode="exact", seed=2718, tables=4, bits=8):
        self.keys = keys.detach()
        self.mode = mode
        self.fallbacks = 0
        if mode == "lsh":
            # Asymmetric MIPS-to-cosine transform; unnormalised inner products remain ordered.
            radius = self.keys.square().sum(-1).max().sqrt().clamp_min(1e-6)
            x = self.keys / radius
            x = torch.cat([x, (1 - x.square().sum(-1, keepdim=True)).clamp_min(0).sqrt()], -1)
            generator = torch.Generator(device="cpu").manual_seed(seed)
            self.projection = torch.randn(tables, keys.shape[-1] + 1, bits, generator=generator).to(keys)
            self.signatures = torch.einsum("nd,tdb->tnb", x, self.projection) >= 0

    def search(self, query, feasible, k):
        if not feasible:
            return []
        ids = torch.tensor(feasible, device=self.keys.device, dtype=torch.long)
        k = min(k, len(feasible))
        pool = ids
        if self.mode == "lsh" and len(ids) > max(4 * k, 32):
            q = torch.cat([query.detach(), query.new_zeros(1)])
            sig = torch.einsum("d,tdb->tb", q, self.projection) >= 0
            distance = (self.signatures[:, ids] != sig[:, None]).sum(-1).min(0).values
            pool = ids[distance <= 1]
            if len(pool) < k:
                # Probe closest buckets before exact fallback. Fallback cost is disclosed.
                pool = ids[torch.argsort(distance, stable=True)[:max(4 * k, 32)]]
            if len(pool) < k:
                pool = ids
                self.fallbacks += 1
        scores = self.keys[pool] @ query
        return pool[torch.argsort(scores, descending=True, stable=True)[:k]].tolist()


def recall_at_k(index, query, feasible, k):
    exact = CandidateIndex(index.keys).search(query, feasible, k)
    approx = index.search(query, feasible, k)
    return len(set(exact) & set(approx)) / max(1, len(exact))
