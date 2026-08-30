"""新版异构图 JSONL 数据集持久化。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

import torch

from ma_tgn.config import ModelConfig
from ma_tgn.state_schema import schema_dict
from ma_tgn.types import GraphEvent, QueryGraph, TaskRequirement, TrainingCase

from .simulator import SixGStateSimulator


def _list(tensor: torch.Tensor) -> list:
    return tensor.detach().cpu().tolist()


def case_to_dict(case: TrainingCase) -> dict:
    def event_dict(event: GraphEvent) -> dict:
        return {
            "src": event.src, "dst": event.dst, "src_node_type": event.src_node_type,
            "dst_node_type": event.dst_node_type, "timestamp": event.timestamp,
            "event_type": event.event_type, "relation_type": event.relation_type,
            "src_state": {key: _list(value) for key, value in event.src_state.items()},
            "dst_state": {key: _list(value) for key, value in event.dst_state.items()},
            "src_confidence": {key: _list(value) for key, value in (event.src_confidence or {}).items()},
            "dst_confidence": {key: _list(value) for key, value in (event.dst_confidence or {}).items()},
            "src_observed_at": event.src_observed_at or {}, "dst_observed_at": event.dst_observed_at or {},
            "edge_state": _list(event.edge_state), "edge_confidence": event.edge_confidence,
            "edge_observed_at": event.edge_observed_at,
        }
    graph = case.graph
    return {
        "events": [event_dict(event) for event in case.events],
        "graph": {
            "candidate_ids": _list(graph.candidate_ids), "edge_index": _list(graph.edge_index),
            "edge_state": _list(graph.edge_state), "relation_type": _list(graph.relation_type),
            "edge_timestamp": _list(graph.edge_timestamp), "edge_confidence": _list(graph.edge_confidence),
            "query_time": graph.query_time, "candidate_texts": list(graph.candidate_texts),
            "candidate_skill_mask": (
                _list(graph.candidate_skill_mask) if graph.candidate_skill_mask is not None else []
            ),
            "candidate_skill_ids": [list(items) for items in graph.candidate_skill_ids],
        },
        "task": {
            "qos": _list(case.task.values),
            "capability_embedding": _list(case.task.capability_embedding),
            "description": case.task.description,
            "required_capability": case.task.required_capability,
            "dag_descriptions": list(case.task.dag_descriptions),
            "dag_required_capabilities": list(case.task.dag_required_capabilities),
            "dag_qos": _list(case.task.dag_qos) if case.task.dag_qos is not None else [],
            "dag_edge_index": _list(case.task.dag_edge_index) if case.task.dag_edge_index is not None else [[], []],
            "dag_current_index": case.task.dag_current_index,
            "complex_task_id": case.task.complex_task_id,
        },
        "label": int(case.label), "candidate_targets": _list(case.candidate_targets),
        "teacher_scores": _list(case.teacher_scores), "metadata": case.metadata,
    }


def case_from_dict(data: dict) -> TrainingCase:
    def tensors(values: dict) -> dict[str, torch.Tensor]:
        converted = {key: torch.tensor(value, dtype=torch.float32) for key, value in values.items()}
        # 读取 6.0 以前的能力槽数据时，只迁移为智能体总体可靠性；V2 不暴露槽结构。
        if "reliability" not in converted and "capability_reliability" in converted:
            reliability = converted["capability_reliability"]
            mask = converted.get("capability_mask", torch.ones(reliability.shape[0])).bool()
            converted["reliability"] = reliability[mask].mean(0) if mask.any() else torch.zeros(4)
        for legacy in ("capability_embeddings", "capability_mask", "capability_usage", "capability_reliability"):
            converted.pop(legacy, None)
        return converted
    events = [GraphEvent(
        src=item["src"], dst=item["dst"], src_node_type=item["src_node_type"], dst_node_type=item["dst_node_type"],
        timestamp=item["timestamp"], event_type=item["event_type"], relation_type=item["relation_type"],
        src_state=tensors(item["src_state"]), dst_state=tensors(item["dst_state"]),
        edge_state=torch.tensor(item["edge_state"], dtype=torch.float32),
        src_confidence=tensors(item["src_confidence"]), dst_confidence=tensors(item["dst_confidence"]),
        src_observed_at=item["src_observed_at"], dst_observed_at=item["dst_observed_at"],
        edge_confidence=float(item["edge_confidence"]), edge_observed_at=item["edge_observed_at"],
    ) for item in data["events"]]
    raw = data["graph"]
    graph = QueryGraph(
        candidate_ids=torch.tensor(raw["candidate_ids"], dtype=torch.long),
        edge_index=torch.tensor(raw["edge_index"], dtype=torch.long),
        edge_state=torch.tensor(raw["edge_state"], dtype=torch.float32),
        relation_type=torch.tensor(raw["relation_type"], dtype=torch.long),
        edge_timestamp=torch.tensor(raw["edge_timestamp"], dtype=torch.float32),
        edge_confidence=torch.tensor(raw["edge_confidence"], dtype=torch.float32), query_time=float(raw["query_time"]),
        candidate_texts=tuple(raw.get("candidate_texts", data.get("metadata", {}).get("candidate_roles", []))),
        candidate_skill_mask=(
            torch.tensor(raw["candidate_skill_mask"], dtype=torch.bool)
            if raw.get("candidate_skill_mask") else None
        ),
        candidate_skill_ids=tuple(tuple(items) for items in raw.get("candidate_skill_ids", ())),
    )
    task = data["task"]
    return TrainingCase(
        events, graph,
        TaskRequirement(
            torch.tensor(task["qos"], dtype=torch.float32),
            torch.tensor(task["capability_embedding"], dtype=torch.float32),
            str(task.get("description", data.get("metadata", {}).get("task_prompt", ""))),
            str(task.get("required_capability", data.get("metadata", {}).get("required_capability", ""))),
            tuple(task.get("dag_descriptions", ())),
            tuple(task.get("dag_required_capabilities", ())),
            torch.tensor(task["dag_qos"], dtype=torch.float32) if task.get("dag_qos") else None,
            torch.tensor(task["dag_edge_index"], dtype=torch.long) if task.get("dag_edge_index") else None,
            int(task.get("dag_current_index", 0)),
            str(task.get("complex_task_id", "")),
        ),
        torch.tensor(data["label"], dtype=torch.long), torch.tensor(data["candidate_targets"], dtype=torch.float32),
        torch.tensor(data["teacher_scores"], dtype=torch.float32), data["metadata"],
    )


def iter_cases(path: Path) -> Iterator[TrainingCase]:
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield case_from_dict(json.loads(line))


def write_cases(path: Path, cases: Iterable[TrainingCase]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as stream:
        for case in cases:
            stream.write(json.dumps(case_to_dict(case), ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def generate_dataset(output_dir: Path, num_agents: int = 100, num_compute_nodes: int = 12, samples: int = 48, seed: int = 7, update_interval_ms: int = 100, scenario_duration_minutes: int = 30) -> dict:
    if num_agents < 100:
        raise ValueError("研究指标要求智能体规模不少于 100")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = ModelConfig(num_agents=num_agents, num_compute_nodes=num_compute_nodes)
    train_count = max(1, int(samples*.75)); val_count = max(1, int(samples*.125))
    split_counts = {"train": train_count, "val": val_count, "test": samples-train_count-val_count}
    task_counts: dict[str, int] = {}; compute_counts: dict[str, int] = {}
    for split_index, (split, count) in enumerate(split_counts.items()):
        simulator = SixGStateSimulator(config, seed+split_index*10_000, update_interval_ms)
        cases = []
        for _ in range(count):
            case = simulator.make_case(); cases.append(case)
            task = str(case.metadata["task_type"]); task_counts[task] = task_counts.get(task, 0)+1
        write_cases(output_dir/f"{split}.jsonl", cases)
        if split == "train":
            profile_dir = output_dir/"profiles"
            profile_dir.mkdir(parents=True, exist_ok=True)
            for agent_id, skills in enumerate(simulator.agent_skills):
                profile = {
                    "agent_id": agent_id, "role": simulator.roles[agent_id],
                    "skill_ids": list(skills),
                    "reliability": simulator.agent_reliability[agent_id].tolist(),
                }
                (profile_dir/f"agent_{agent_id:03d}.json").write_text(
                    json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            compute_counts = {kind: simulator.compute_kinds.count(kind) for kind in set(simulator.compute_kinds)}
    metadata = {
        "dataset": "6G-Agent-Compute-Heterogeneous-State", "version": "2.0",
        "num_agents": num_agents, "num_compute_nodes": num_compute_nodes, "compute_node_distribution": compute_counts,
        "samples": samples, "update_interval_ms": update_interval_ms,
        "scenario_duration_minutes": scenario_duration_minutes,
        "virtual_probe_ticks": scenario_duration_minutes*60*1000//update_interval_ms,
        "splits": split_counts, "task_distribution": task_counts, "profile_format": "profiles/agent_NNN.json",
        "design_constraints": {
            "queue_batch_in_graph": True,
            "queue_batch_used_for_decision": False,
            "link_path_type_encoded": False,
            "capability_representation": "exact Skill ID set; no capability slots",
        },
        "schema": schema_dict(),
    }
    (output_dir/"metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata
