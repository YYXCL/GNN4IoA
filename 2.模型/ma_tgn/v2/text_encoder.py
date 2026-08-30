"""V2 使用的、无需联网的 Skill id 字符 n-gram 编码器。"""

from __future__ import annotations

import hashlib
import re

import torch
from torch import nn


class TextSemanticEncoder(nn.Module):
    """智能体能力/任务描述共享的 signed subword hashing + 可训练投影。

    Hashing 层无词表、能覆盖中文和 skill id；投影层与卸载目标端到端训练。
    后续若接入本地预训练编码器，只需保持 forward 的输出维度不变。
    """

    def __init__(self, hash_dim: int, output_dim: int, dropout: float = 0.0):
        super().__init__()
        self.hash_dim = hash_dim
        self._cache: dict[str, torch.Tensor] = {}
        self.projection = nn.Sequential(
            nn.Linear(hash_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

    def _hash(self, text: str) -> torch.Tensor:
        normalized = re.sub(r"\s+", " ", text.strip().lower())
        cached = self._cache.get(normalized)
        if cached is not None:
            return cached
        compact = normalized.replace(" ", "")
        tokens = normalized.split()
        # Skill-id 的层级片段与英文词显式进入共享语义空间；中文继续使用字符 n-gram。
        identifier_parts = [part for part in re.split(r"[^\w\u4e00-\u9fff]+|[._:/-]+", normalized) if part]
        english_words = re.findall(r"[a-z][a-z0-9_]+", normalized)
        grams = tokens + identifier_parts + english_words + [
            compact[i : i + n]
            for n in (1, 2, 3)
            for i in range(max(0, len(compact) - n + 1))
        ]
        vector = torch.zeros(self.hash_dim)
        for gram in grams:
            digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
            raw = int.from_bytes(digest, "little")
            vector[raw % self.hash_dim] += 1.0 if ((raw >> 8) & 1) else -1.0
        vector = torch.nn.functional.normalize(vector, dim=0) if vector.abs().sum() else vector
        self._cache[normalized] = vector
        return vector

    def forward(self, texts: list[str] | tuple[str, ...]) -> torch.Tensor:
        device = next(self.parameters()).device
        return self.projection(self.hash_features(texts).to(device))

    def hash_features(self, texts: list[str] | tuple[str, ...]) -> torch.Tensor:
        """返回未经过可训练投影的可审计语义特征。"""
        return torch.stack([self._hash(str(text)) for text in texts])
