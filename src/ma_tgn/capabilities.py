"""能力 profile 细粒度拆分、动态槽管理与逐能力可靠性更新。"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Protocol

import torch


class CapabilityDecomposer(Protocol):
    def decompose(self, profile: dict) -> list[str]: ...


class TextEmbedder(Protocol):
    def embed(self, text: str, dimension: int) -> torch.Tensor: ...


class HashTextEmbedder:
    """离线可复现后备；生产环境应换成语义嵌入模型。"""

    def embed(self, text: str, dimension: int) -> torch.Tensor:
        vector = torch.zeros(dimension)
        tokens = re.findall(r"[\w\u4e00-\u9fff]+", text.lower()) or [text]
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            vector[int.from_bytes(digest[:4], "little") % dimension] += 1.0
        return vector / vector.norm().clamp_min(1e-6)


class HeuristicCapabilityDecomposer:
    """无 LLM 时从 profile 的能力列表安全降级。"""

    def decompose(self, profile: dict) -> list[str]:
        values = profile.get("capabilities", [])
        if isinstance(values, str):
            values = re.split(r"[，,；;\n]", values)
        return [str(value).strip() for value in values if str(value).strip()]


class LLMCapabilityDecomposer:
    """模型无关的 LLM 适配器。

    complete 接收提示词并返回 JSON 数组，例如
    `["判断视频帧是否有火灾", "判断视频帧是否有烟雾"]`。
    这样项目不绑定特定厂商或密钥，调用方可注入任意大模型客户端。
    """

    def __init__(self, complete: Callable[[str], str]):
        self.complete = complete

    def decompose(self, profile: dict) -> list[str]:
        prompt = (
            "把智能体能力拆成不可再分、可由任务验证的细粒度能力。"
            "只返回 JSON 字符串数组，不要解释。profile=" + json.dumps(profile, ensure_ascii=False)
        )
        raw = json.loads(self.complete(prompt))
        if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
            raise ValueError("LLM 必须返回字符串数组")
        return [item.strip() for item in raw if item.strip()]


@dataclass
class CapabilitySlot:
    name: str
    description: str
    usage_count: int = 1
    task_success_rate: float = 0.8
    mean_execution_time_ms: float = 2_000.0
    tool_call_failure_rate: float = 0.1
    hallucination_rate: float = 0.1


class CapabilityRegistry:
    def __init__(self, max_slots: int = 8, embedding_dim: int = 8, embedder: TextEmbedder | None = None):
        self.max_slots = max_slots
        self.embedding_dim = embedding_dim
        self.embedder = embedder or HashTextEmbedder()
        self.slots: list[CapabilitySlot] = []

    @classmethod
    def from_profile(cls, profile: dict, decomposer: CapabilityDecomposer, max_slots: int = 8, embedding_dim: int = 8) -> "CapabilityRegistry":
        registry = cls(max_slots, embedding_dim)
        for capability in decomposer.decompose(profile):
            registry.add_or_touch(capability)
        return registry

    def _similarity(self, left: str, right: str) -> float:
        a = self.embedder.embed(left, self.embedding_dim)
        b = self.embedder.embed(right, self.embedding_dim)
        return float(torch.dot(a, b))

    def match(self, capability: str, threshold: float = 0.82) -> int | None:
        if not self.slots:
            return None
        scores = [self._similarity(capability, slot.description) for slot in self.slots]
        index = max(range(len(scores)), key=scores.__getitem__)
        return index if scores[index] >= threshold else None

    def add_or_touch(self, capability: str) -> int:
        matched = self.match(capability)
        if matched is not None:
            self.slots[matched].usage_count += 1
            return matched
        slot = CapabilitySlot(capability, capability)
        if len(self.slots) < self.max_slots:
            self.slots.append(slot)
            return len(self.slots) - 1
        # 固定输入维度下，淘汰最少使用的槽；同频时淘汰更靠前的旧槽。
        index = min(range(len(self.slots)), key=lambda i: (self.slots[i].usage_count, i))
        self.slots[index] = slot
        return index

    def record_result(self, capability: str, success: bool, execution_time_ms: float, tool_failed: bool, hallucinated: bool, rate: float = 0.1) -> int:
        index = self.add_or_touch(capability)
        slot = self.slots[index]
        slot.task_success_rate = (1-rate)*slot.task_success_rate + rate*float(success)
        slot.mean_execution_time_ms = (1-rate)*slot.mean_execution_time_ms + rate*execution_time_ms
        slot.tool_call_failure_rate = (1-rate)*slot.tool_call_failure_rate + rate*float(tool_failed)
        slot.hallucination_rate = (1-rate)*slot.hallucination_rate + rate*float(hallucinated)
        return index

    def tensors(self) -> dict[str, torch.Tensor]:
        embeddings = torch.zeros(self.max_slots, self.embedding_dim)
        mask = torch.zeros(self.max_slots)
        usage = torch.zeros(self.max_slots)
        reliability = torch.zeros(self.max_slots, 4)
        max_usage = max((slot.usage_count for slot in self.slots), default=1)
        for index, slot in enumerate(self.slots):
            embeddings[index] = self.embedder.embed(slot.description, self.embedding_dim)
            mask[index] = 1.0
            usage[index] = slot.usage_count / max_usage
            reliability[index] = torch.tensor([
                slot.task_success_rate, slot.mean_execution_time_ms,
                slot.tool_call_failure_rate, slot.hallucination_rate,
            ])
        return {
            "capability_embeddings": embeddings,
            "capability_mask": mask,
            "capability_usage": usage,
            "capability_reliability": reliability,
        }

    def to_profile(self, agent_id: int, metadata: dict | None = None) -> dict:
        return {"agent_id": agent_id, "capabilities": [asdict(slot) for slot in self.slots], "metadata": metadata or {}}

    def save_profile(self, path: Path, agent_id: int, metadata: dict | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_profile(agent_id, metadata), ensure_ascii=False, indent=2), encoding="utf-8")
