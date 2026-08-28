"""可复现、无需联网的中英文与 Skill-id 文本特征。"""

from __future__ import annotations

import hashlib
import re

import torch


def hash_text(text: str, dimension: int = 256) -> torch.Tensor:
    normalized = re.sub(r"\s+", " ", str(text).strip().lower())
    compact = normalized.replace(" ", "")
    parts = [part for part in re.split(r"[^\w\u4e00-\u9fff]+|[._:/-]+", normalized) if part]
    grams = parts + [
        compact[index:index + width]
        for width in (1, 2, 3)
        for index in range(max(0, len(compact) - width + 1))
    ]
    vector = torch.zeros(dimension)
    for gram in grams:
        raw = int.from_bytes(hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest(), "little")
        vector[raw % dimension] += 1.0 if ((raw >> 8) & 1) else -1.0
    return torch.nn.functional.normalize(vector, dim=0) if vector.abs().sum() else vector
