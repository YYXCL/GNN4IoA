"""把 GNN4IoA JSONL 转换为论文式单子任务 task-agent 图输入。"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch

from .text import hash_text


@dataclass
class GCANCase:
    """一个当前子任务及其候选智能体的完整时序图样本。"""

    candidate_ids: torch.Tensor
    agent_numeric: torch.Tensor
    agent_history: torch.Tensor
    history_mask: torch.Tensor
    history_delta_time: torch.Tensor
    agent_text: torch.Tensor
    task_text: torch.Tensor
    task_qos: torch.Tensor
    task_agent_edge_state: torch.Tensor
    collaboration_edge_index: torch.Tensor
    collaboration_edge_state: torch.Tensor
    skill_mask: torch.Tensor
    label: torch.Tensor
    teacher_scores: torch.Tensor
    candidate_targets: torch.Tensor
    scenario: str
    task_id: str


def _compute_features(values: list[float]) -> list[float]:
    remaining_gpu, remaining_memory, remaining_vram = values
    return [
        min(1.0, remaining_gpu / 10_000.0),
        min(1.0, remaining_memory / 2_048.0),
        min(1.0, remaining_vram / 1_024.0),
        0.0, 0.0,
        math.log1p(max(0.0, remaining_gpu)) / 10.0,
    ]


def _link_features(values: list[float]) -> list[float]:
    latency, bandwidth, loss, jitter = values
    return [
        min(1.0, latency / 1000.0),
        min(1.0, bandwidth / 100000.0),
        float(loss),
        min(1.0, jitter / 200.0),
    ]


def _numeric_features(state: dict, compute: list[float], link: list[float]) -> list[float]:
    context = state["context"]
    environment = state["environment"]
    if "reliability" in state:
        pooled_reliability = [float(value) for value in state["reliability"]]
    else:
        # 仅用于直接读取旧 JSONL；当前数据加载器不会把能力槽送入模型。
        mask = [bool(value) for value in state["capability_mask"]]
        rows = [row for row, keep in zip(state["capability_reliability"], mask) if keep]
        pooled_reliability = [
            sum(row[index] for row in rows) / len(rows) for index in range(4)
        ] if rows else [0.0] * 4
    pooled_reliability[1] = math.log1p(max(0.0, pooled_reliability[1])) / 10.0
    return [
        math.log1p(float(context[0])) / 13.0,
        float(context[1]),
        *[float(value) for value in environment],
        *pooled_reliability,
        math.log1p(float(state.get("workload", [0.0])[0])) / 10.0,
        *compute,
        *link,
    ]


def _agent_histories(
    raw: dict, candidate_ids: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """按事件时间重放状态，保留每个候选的观测序列而非只取最后一帧。"""

    candidate_set = set(candidate_ids)
    latest_state: dict[int, dict] = {}
    latest_compute = {candidate: [0.0] * 6 for candidate in candidate_ids}
    latest_link = {candidate: [0.0] * 4 for candidate in candidate_ids}
    histories: dict[int, list[list[float]]] = {candidate: [] for candidate in candidate_ids}
    timestamps: dict[int, list[float]] = {candidate: [] for candidate in candidate_ids}

    for event in sorted(raw["events"], key=lambda item: float(item["timestamp"])):
        touched: set[int] = set()
        for side in ("src", "dst"):
            node_id = int(event[side])
            if node_id in candidate_set and int(event[f"{side}_node_type"]) == 0:
                latest_state[node_id] = event[f"{side}_state"]
                latest_link[node_id] = _link_features(event["edge_state"])
                touched.add(node_id)

        if int(event["src_node_type"]) == 1 and int(event["dst"]) in candidate_set:
            values = event["src_state"].get("compute")
            if values is not None:
                latest_compute[int(event["dst"])] = _compute_features(values)
        if int(event["dst_node_type"]) == 1 and int(event["src"]) in candidate_set:
            values = event["dst_state"].get("compute")
            if values is not None:
                latest_compute[int(event["src"])] = _compute_features(values)

        for candidate in touched:
            histories[candidate].append(_numeric_features(
                latest_state[candidate], latest_compute[candidate], latest_link[candidate],
            ))
            timestamps[candidate].append(float(event["timestamp"]))

    missing = [candidate for candidate in candidate_ids if not histories[candidate]]
    if missing:
        raise ValueError(f"候选智能体在事件窗口中没有状态: {missing}")

    max_steps = max(len(histories[candidate]) for candidate in candidate_ids)
    feature_dim = len(histories[candidate_ids[0]][0])
    values = torch.zeros(len(candidate_ids), max_steps, feature_dim, dtype=torch.float32)
    mask = torch.zeros(len(candidate_ids), max_steps, dtype=torch.bool)
    delta = torch.zeros(len(candidate_ids), max_steps, 1, dtype=torch.float32)
    for row, candidate in enumerate(candidate_ids):
        sequence = histories[candidate]
        length = len(sequence)
        values[row, :length] = torch.tensor(sequence, dtype=torch.float32)
        mask[row, :length] = True
        previous = timestamps[candidate][0]
        for step, timestamp in enumerate(timestamps[candidate]):
            elapsed = 0.0 if step == 0 else max(0.0, timestamp - previous)
            delta[row, step, 0] = math.log1p(elapsed) / 10.0
            previous = timestamp
    return values, mask, delta


def _collaboration_graph(raw: dict, candidate_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """抽取候选之间的观测关系；task-agent 边由模型显式补齐。"""

    graph = raw["graph"]
    local = {node_id: index for index, node_id in enumerate(candidate_ids)}
    edges: list[tuple[int, int]] = []
    features: list[list[float]] = []
    query_time = float(graph["query_time"])
    for index, (source, destination) in enumerate(zip(*graph["edge_index"])):
        source, destination = int(source), int(destination)
        if source not in local or destination not in local:
            continue
        latency, bandwidth, loss, jitter = _link_features(graph["edge_state"][index])
        confidence = float(graph["edge_confidence"][index])
        age = max(0.0, query_time - float(graph["edge_timestamp"][index]))
        edges.append((local[source], local[destination]))
        features.append([latency, bandwidth, loss, jitter, confidence, math.log1p(age) / 10.0, 0.0, 0.0])
    if not edges:
        return torch.empty(2, 0, dtype=torch.long), torch.empty(0, 8, dtype=torch.float32)
    return torch.tensor(edges, dtype=torch.long).t().contiguous(), torch.tensor(features, dtype=torch.float32)


def case_from_dict(raw: dict, hash_dimension: int = 256) -> GCANCase:
    graph = raw["graph"]
    candidate_ids = [int(value) for value in graph["candidate_ids"]]
    history, history_mask, history_delta = _agent_histories(raw, candidate_ids)
    last_indices = history_mask.long().sum(dim=1) - 1
    agent_numeric = history[torch.arange(len(candidate_ids)), last_indices]

    task = raw["task"]
    candidate_texts = list(graph.get("candidate_texts", raw.get("metadata", {}).get("candidate_roles", [])))
    if len(candidate_texts) < len(candidate_ids):
        candidate_texts.extend([""] * (len(candidate_ids) - len(candidate_texts)))
    candidate_texts = candidate_texts[:len(candidate_ids)]
    skill_mask = graph.get("candidate_skill_mask")
    if not skill_mask:
        required = str(task.get("required_capability", "")).lower()
        skill_mask = [required in str(text).lower() for text in candidate_texts]
    if not any(skill_mask):
        skill_mask = [True] * len(candidate_ids)

    agent_text = torch.stack([hash_text(text, hash_dimension) for text in candidate_texts])
    required_text = hash_text(task.get("required_capability", ""), hash_dimension)
    semantic = torch.nn.functional.cosine_similarity(
        agent_text, required_text.unsqueeze(0).expand_as(agent_text), dim=-1,
    )
    qos = torch.tensor(task["qos"], dtype=torch.float32)
    task_agent_edge_state = torch.stack([
        torch.tensor([
            float(skill_mask[index]),
            0.0,
            float(qos[0]), float(qos[1]), 0.0, 0.0, 0.0, 0.0,
        ])
        for index in range(len(candidate_ids))
    ])
    collaboration_edge_index, collaboration_edge_state = _collaboration_graph(raw, candidate_ids)

    return GCANCase(
        candidate_ids=torch.tensor(candidate_ids, dtype=torch.long),
        agent_numeric=agent_numeric,
        agent_history=history,
        history_mask=history_mask,
        history_delta_time=history_delta,
        agent_text=agent_text,
        task_text=torch.stack([
            hash_text(task.get("description", raw.get("metadata", {}).get("task_prompt", "")), hash_dimension),
            required_text,
        ]),
        task_qos=qos,
        task_agent_edge_state=task_agent_edge_state,
        collaboration_edge_index=collaboration_edge_index,
        collaboration_edge_state=collaboration_edge_state,
        skill_mask=torch.tensor(skill_mask, dtype=torch.bool),
        label=torch.tensor(raw["label"], dtype=torch.long),
        teacher_scores=torch.tensor(raw["teacher_scores"], dtype=torch.float32),
        candidate_targets=torch.tensor(raw["candidate_targets"], dtype=torch.float32),
        scenario=str(raw["metadata"]["task_type"]),
        task_id=str(raw["metadata"]["task_id"]),
    )


def load_cases(path: Path, hash_dimension: int = 256) -> list[GCANCase]:
    return [
        case_from_dict(json.loads(line), hash_dimension)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
