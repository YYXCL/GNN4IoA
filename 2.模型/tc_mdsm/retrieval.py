"""在满足权限的智能体中进行精确最大内积检索。"""
import torch


class CandidateIndex:
    def __init__(self, keys):
        self.keys = keys.detach()

    def scores(self, query, feasible, dimensions=None):
        ids = torch.tensor(feasible, device=self.keys.device, dtype=torch.long)
        keys = self.keys[ids]
        if dimensions is not None:
            keys = keys.index_select(1, dimensions)
            query = query.index_select(0, dimensions)
        return keys @ query

    def search(self, query, feasible, k, dimensions=None):
        if not feasible:
            return []
        ids = torch.tensor(feasible, device=self.keys.device, dtype=torch.long)
        scores = self.scores(query, feasible, dimensions)
        order = torch.argsort(scores, descending=True, stable=True)[:k]
        return ids[order].tolist()
