"""训练六种分配方法并生成场景、难度、图规模与并发负载论文图。"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT), str(ROOT / "2.模型"), str(ROOT / "1.data"),
    str(ROOT / "baselines" / "gmatann"),
]

from baselines import (  # noqa: E402
    MLPStateAllocator, MinimalStateAblationAllocator,
    RuleBasedLatencyScheduler, StaticGraphAllocator, allocation_loss,
)
from baselines.gmatann.gcan_baseline.data import GCANCase, load_cases  # noqa: E402
from baselines.gmatann.gcan_baseline.model import (  # noqa: E402
    GCANConfig, GraphConvolutionalAssignmentNetwork,
)
from baselines.gmatann.gcan_baseline.text import hash_text  # noqa: E402
from baselines.common import latest_agent_states  # noqa: E402
from ma_tgn import ModelConfig, TaskConditionedHeterogeneousTGN  # noqa: E402
from ma_tgn.types import GraphEvent, QueryGraph, TrainingCase  # noqa: E402
from ma_tgn_data.dataset import iter_cases  # noqa: E402


METHODS = (
    "Ours (DAG-GRU)", "GMATANN", "Rule-Latency",
    "State-MLP", "StaticGraph", "GRU-Minimal",
)
SCENARIOS = ("智慧工厂", "智慧家居", "应急救援")
SCENARIO_EN = {"智慧工厂": "Smart factory", "智慧家居": "Smart home", "应急救援": "Emergency response"}
DIFFICULTIES = ("简单", "中等", "困难")
DIFFICULTY_EN = {"简单": "Simple", "中等": "Medium", "困难": "Hard"}
GRAPH_SIZES = (50, 100, 200, 400, 800, 1600)
CONCURRENCY = (1, 2, 4, 8, 16, 32, 64)


def native_temporal_output(model, case: TrainingCase, train: bool = False):
    model.reset_state()
    warmup = max(0, len(case.events) - 24) if train else len(case.events)
    with torch.no_grad():
        for event in case.events[:warmup]:
            model.process_event(event)
    for event in case.events[warmup:]:
        model.process_event(event, track_grad=train)
    return model.assign(case.graph, case.task)


def dag_supervision(case: TrainingCase) -> torch.Tensor:
    """用同一物理规则为每个子任务构造监督分数，包含历史趋势和邻居外部性。"""

    candidates = [int(value) for value in case.graph.candidate_ids.tolist()]
    position = {agent: index for index, agent in enumerate(candidates)}
    latest = latest_agent_states(case)
    workload_series = {agent: [] for agent in candidates}
    context_series = {agent: [] for agent in candidates}
    link_series = {agent: [] for agent in candidates}
    compute_series: dict[int, list[torch.Tensor]] = {
        index: [] for index in range(len(case.metadata["compute_states"]))
    }
    for event in sorted(case.events, key=lambda item: item.timestamp):
        for node, node_type, state in (
            (event.src, event.src_node_type, event.src_state),
            (event.dst, event.dst_node_type, event.dst_state),
        ):
            if node_type == 0 and node in position:
                workload_series[node].append(float(state.get("workload", torch.zeros(1))[0]))
                context_series[node].append(float(state["context"][1]))
                link_series[node].append(event.edge_state.float())
            elif node_type == 1 and "compute" in state:
                compute_index = int(node) - len(case.metadata["agent_compute"])
                if compute_index in compute_series:
                    compute_series[compute_index].append(state["compute"].float())

    peer_sources: dict[int, list[int]] = {agent: [] for agent in candidates}
    for edge_pos, (source, destination) in enumerate(case.graph.edge_index.t().tolist()):
        if int(case.graph.relation_type[edge_pos]) == 1 and source in position and destination in position:
            peer_sources[destination].append(source)

    base_scores = []
    for agent in candidates:
        row = position[agent]
        target = case.candidate_targets[row].float()
        state = latest[agent]
        compute_index = int(case.metadata["agent_compute"][agent])
        compute_now = torch.tensor(case.metadata["compute_states"][compute_index]).float()
        compute_scale = torch.tensor([4_000.0, 2_048.0, 1_024.0])
        compute_quality = float((compute_now / compute_scale).clamp(0, 1).mean())
        incoming = case.graph.edge_index[1] == agent
        link_now = case.graph.edge_state[incoming].float().mean(0)
        link_quality = float(
            (1.0 - (link_now[0] / 1_000.0).clamp(0, 1))
            * (link_now[1] / 3_000.0).clamp(0, 1)
            * (1.0 - link_now[2].clamp(0, 1))
            * (1.0 - (link_now[3] / 200.0).clamp(0, 1))
        )
        workload = float(state.get("workload", torch.zeros(1))[0]) / 16.0
        environment = float(state["environment"].mean())
        peers = peer_sources[agent]
        neighbor_pressure = sum(
            float(latest[peer].get("workload", torch.zeros(1))[0]) / 16.0 for peer in peers
        ) / max(1, len(peers))

        def delta(series, scale=1.0):
            return 0.0 if len(series) < 2 else (float(series[-1]) - float(series[0])) / scale

        workload_rise = max(0.0, delta(workload_series[agent], 16.0))
        context_rise = max(0.0, delta(context_series[agent], 1.0))
        links = link_series[agent]
        latency_rise = 0.0 if len(links) < 2 else max(0.0, float(links[-1][0] - links[0][0]) / 1_000.0)
        loss_rise = 0.0 if len(links) < 2 else max(0.0, float(links[-1][2] - links[0][2]))
        resources = compute_series.get(compute_index, [])
        resource_decline = 0.0
        if len(resources) >= 2:
            resource_decline = float(((resources[0] - resources[-1]) / compute_scale).clamp_min(0).mean())
        trend_risk = (
            0.25 * workload_rise + 0.18 * context_rise + 0.25 * resource_decline
            + 0.18 * latency_rise + 0.14 * loss_rise
        )
        accuracy_utility = (
            0.58 * float(target[0]) + 0.18 * (1.0 - float(target[3]))
            + 0.12 * (1.0 - float(target[2])) + 0.12 * environment
        )
        latency_utility = (
            0.34 * (1.0 - float(target[1])) + 0.22 * compute_quality + 0.14 * link_quality
            + 0.12 * (1.0 - workload) + 0.10 * (1.0 - neighbor_pressure)
            + 0.08 * (1.0 - trend_risk)
        )
        base_scores.append((accuracy_utility, latency_utility))

    rows = []
    for subtask_index, skill in enumerate(case.task.dag_required_capabilities):
        latency_priority = float(case.task.dag_qos[subtask_index, 0])
        accuracy_priority = float(case.task.dag_qos[subtask_index, 1])
        scores = torch.tensor([
            accuracy_priority * accuracy + (0.55 + latency_priority) * latency
            for accuracy, latency in base_scores
        ])
        eligible = torch.tensor([
            skill in set(skill_ids) for skill_ids in case.graph.candidate_skill_ids
        ], dtype=torch.bool)
        if not eligible.any():
            raise ValueError(f"{case.metadata['task_id']} 的子任务 Skill 无可执行候选: {skill}")
        rows.append(scores.masked_fill(~eligible, -1.0e4))
    return torch.stack(rows)


def validation_score(model, cases, kind: str, gcan_cases=None) -> float:
    correct = 0
    ndcg = 0.0
    source = gcan_cases if kind == "gmatann" else cases
    model.eval()
    with torch.no_grad():
        item_count = 0
        for case in source:
            output = model(case) if kind in {"gmatann", "flat"} else native_temporal_output(model, case)
            logits_rows = output["task_agent_logits"] if kind == "temporal" else output["candidate_logits"].unsqueeze(0)
            relevance_rows = dag_supervision(case) if kind == "temporal" else case.teacher_scores.unsqueeze(0)
            labels = relevance_rows.argmax(dim=-1)
            for logits, relevance, label in zip(logits_rows, relevance_rows, labels):
                ranking = torch.argsort(logits, descending=True).cpu()
                correct += int(int(ranking[0]) == int(label))
                k = min(5, ranking.numel())
                discount = torch.log2(torch.arange(k, dtype=torch.float32) + 2.0)
                dcg = ((torch.pow(2.0, relevance[ranking[:k]]) - 1.0) / discount).sum()
                ideal = torch.sort(relevance, descending=True).values[:k]
                ideal_dcg = ((torch.pow(2.0, ideal) - 1.0) / discount).sum().clamp_min(1e-8)
                ndcg += float(dcg / ideal_dcg)
                item_count += 1
    count = max(1, item_count)
    return correct / count + 0.25 * ndcg / count


def train_model(model, train_cases, val_cases, kind: str, epochs: int, seed: int, gcan_val=None):
    random.seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_score = -math.inf
    best_state = copy.deepcopy(model.state_dict())
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = list(range(len(train_cases)))
        random.shuffle(order)
        total = 0.0
        for index in order:
            case = train_cases[index]
            if kind == "gmatann":
                output = model(case)
                loss = allocation_loss(output, case.label)
                loss = loss + 0.20 * nn.functional.binary_cross_entropy(
                    output["state_predictions"], case.candidate_targets,
                )
                teacher = torch.softmax(case.teacher_scores / 0.15, dim=0)
                loss = loss + 0.10 * nn.functional.kl_div(
                    torch.log_softmax(output["candidate_logits"], dim=0), teacher, reduction="batchmean",
                )
                loss = loss + 0.03 * output["kl_loss"] + 0.05 * output["reconstruction_loss"]
            elif kind == "flat":
                output = model(case)
                loss = allocation_loss(output, case.label)
                teacher = torch.softmax(case.teacher_scores / 0.15, dim=0)
                loss = loss + 0.10 * nn.functional.kl_div(
                    torch.log_softmax(output["candidate_logits"], dim=0), teacher, reduction="batchmean",
                )
            else:
                output = native_temporal_output(model, case, train=True)
                teacher_scores = dag_supervision(case).to(output["task_agent_logits"].device)
                labels = teacher_scores.argmax(dim=-1)
                loss = nn.functional.cross_entropy(output["task_agent_logits"], labels)
                loss = loss + 0.40 * nn.functional.binary_cross_entropy(
                    output["state_predictions"], case.candidate_targets,
                )
                teacher = torch.softmax(teacher_scores / 0.15, dim=-1)
                loss = loss + 0.10 * nn.functional.kl_div(
                    torch.log_softmax(output["task_agent_logits"], dim=-1), teacher, reduction="batchmean",
                )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach())
        score = validation_score(model, val_cases, kind, gcan_val)
        history.append({"epoch": epoch, "train_loss": total / len(train_cases), "validation_score": score})
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    return history


def subtask_native_case(case: TrainingCase, index: int) -> TrainingCase:
    item = copy.deepcopy(case)
    skill = item.task.dag_required_capabilities[index]
    item.task.required_capability = skill
    item.task.values = item.task.dag_qos[index].clone()
    item.task.dag_current_index = index
    item.graph.candidate_skill_mask = torch.tensor(
        [skill in set(skills) for skills in item.graph.candidate_skill_ids], dtype=torch.bool,
    )
    scores = dag_supervision(case)[index]
    item.teacher_scores = scores
    item.label = scores.argmax()
    return item


def subtask_gcan_case(case: GCANCase, native: TrainingCase, index: int) -> GCANCase:
    item = copy.deepcopy(case)
    skill = native.task.dag_required_capabilities[index]
    qos = native.task.dag_qos[index].clone()
    mask = torch.tensor([skill in set(skills) for skills in native.graph.candidate_skill_ids], dtype=torch.bool)
    if not mask.any():
        mask[:] = True
    item.skill_mask = mask
    item.task_qos = qos
    item.task_text = item.task_text.clone()
    item.task_text[1] = hash_text(skill, item.task_text.shape[-1])
    item.task_agent_edge_state = item.task_agent_edge_state.clone()
    item.task_agent_edge_state[:, 0] = mask.float()
    item.task_agent_edge_state[:, 2:4] = qos
    scores = dag_supervision(native)[index]
    item.teacher_scores = scores
    item.label = scores.argmax()
    return item


def dag_aware_decode(output: dict[str, torch.Tensor], case: TrainingCase) -> list[int]:
    """联合考虑并行层资源冲突；这是主方法相对单子任务基线的 DAG 解码器。"""

    logits = output["task_agent_logits"].detach().cpu()
    candidate_ids = [int(value) for value in output["candidate_ids"].cpu().tolist()]
    predecessors: dict[int, list[int]] = {index: [] for index in range(logits.shape[0])}
    if case.task.dag_edge_index is not None:
        for source, destination in case.task.dag_edge_index.t().tolist():
            predecessors[int(destination)].append(int(source))
    levels = [0] * logits.shape[0]
    for index in range(logits.shape[0]):
        if predecessors[index]:
            levels[index] = 1 + max(levels[parent] for parent in predecessors[index])
    selected: list[int] = []
    level_agents: Counter = Counter()
    level_compute: Counter = Counter()
    current_level = -1
    for index in range(logits.shape[0]):
        if levels[index] != current_level:
            current_level = levels[index]
            level_agents.clear(); level_compute.clear()
        attention_score = torch.softmax(logits[index], dim=0)
        predictions = output["state_predictions"].detach().cpu()
        latency_priority = float(case.task.dag_qos[index, 0])
        accuracy_priority = float(case.task.dag_qos[index, 1])
        predicted_quality = predictions[:, 0] * (1.0 - predictions[:, 3])
        predicted_speed = 1.0 - predictions[:, 1]
        objective = (
            1.10 * attention_score
            + 0.55 * accuracy_priority * predicted_quality
            + 0.55 * latency_priority * predicted_speed
        )
        for position, agent in enumerate(candidate_ids):
            compute = int(case.metadata["agent_compute"][agent])
            objective[position] -= 0.80 * level_agents[agent]
            objective[position] -= 0.45 * level_compute[compute]
            if any(selected[parent] == agent for parent in predecessors[index]):
                objective[position] += 0.10
        chosen_position = int(objective.argmax())
        chosen_agent = candidate_ids[chosen_position]
        selected.append(chosen_agent)
        level_agents[chosen_agent] += 1
        level_compute[int(case.metadata["agent_compute"][chosen_agent])] += 1
    return selected


def assignments(model, native: TrainingCase, method: str, gcan: GCANCase | None = None) -> list[int]:
    with torch.no_grad():
        if method in {"Ours (DAG-GRU)", "GRU-Minimal"}:
            output = native_temporal_output(model, native)
            if method == "Ours (DAG-GRU)":
                return dag_aware_decode(output, native)
            return [int(value) for value in output["subtask_assignments"].cpu().tolist()]
        selected = []
        for index in range(len(native.task.dag_required_capabilities)):
            if method == "GMATANN":
                output = model(subtask_gcan_case(gcan, native, index))
                selected.append(int(gcan.candidate_ids[int(output["candidate_logits"].argmax())]))
            else:
                case = subtask_native_case(native, index)
                output = model(case)
                selected.append(int(case.graph.candidate_ids[int(output["candidate_logits"].argmax())]))
        return selected


def select_candidate_positions(case: TrainingCase, size: int) -> list[int]:
    count = case.graph.candidate_ids.numel()
    if size >= count:
        return list(range(count))
    label = int(case.label)
    required = set(case.task.dag_required_capabilities)
    skill_sets = [set(skills) & required for skills in case.graph.candidate_skill_ids]
    for combination in itertools.combinations(range(count), size):
        if label not in combination:
            continue
        covered = set().union(*(skill_sets[index] for index in combination))
        if covered == required:
            return list(combination)
    raise ValueError(f"{case.metadata['task_id']} 无法在 {size} 个候选中覆盖全部 DAG Skill")


def subset_native_case(case: TrainingCase, positions: list[int]) -> TrainingCase:
    item = copy.deepcopy(case)
    keep_agents = {int(case.graph.candidate_ids[index]) for index in positions}
    keep_edges = torch.tensor([
        int(destination) in keep_agents and (
            int(source) in keep_agents or int(source) >= max(case.metadata["agent_compute"].__len__(), 1)
        )
        for source, destination in case.graph.edge_index.t().tolist()
    ], dtype=torch.bool)
    item.graph.candidate_ids = case.graph.candidate_ids[positions].clone()
    item.graph.candidate_skill_mask = case.graph.candidate_skill_mask[positions].clone()
    item.graph.candidate_skill_ids = tuple(case.graph.candidate_skill_ids[index] for index in positions)
    item.graph.edge_index = case.graph.edge_index[:, keep_edges].clone()
    item.graph.edge_state = case.graph.edge_state[keep_edges].clone()
    item.graph.relation_type = case.graph.relation_type[keep_edges].clone()
    item.graph.edge_timestamp = case.graph.edge_timestamp[keep_edges].clone()
    item.graph.edge_confidence = case.graph.edge_confidence[keep_edges].clone()
    item.candidate_targets = case.candidate_targets[positions].clone()
    item.teacher_scores = case.teacher_scores[positions].clone()
    item.label = torch.tensor(positions.index(int(case.label)), dtype=torch.long)
    item.metadata["candidate_roles"] = [case.metadata["candidate_roles"][index] for index in positions]
    return item


def subset_gcan_case(case: GCANCase, positions: list[int]) -> GCANCase:
    item = copy.deepcopy(case)
    old_to_new = {old: new for new, old in enumerate(positions)}
    edge_keep = [
        source in old_to_new and destination in old_to_new
        for source, destination in case.collaboration_edge_index.t().tolist()
    ]
    if edge_keep:
        keep = torch.tensor(edge_keep, dtype=torch.bool)
        edges = case.collaboration_edge_index[:, keep]
        item.collaboration_edge_index = torch.tensor([
            [old_to_new[int(value)] for value in edges[0]],
            [old_to_new[int(value)] for value in edges[1]],
        ], dtype=torch.long)
        item.collaboration_edge_state = case.collaboration_edge_state[keep].clone()
    else:
        item.collaboration_edge_index = torch.empty(2, 0, dtype=torch.long)
        item.collaboration_edge_state = torch.empty(0, case.collaboration_edge_state.shape[-1])
    for field in (
        "candidate_ids", "agent_numeric", "agent_history", "history_mask", "history_delta_time",
        "agent_text", "task_agent_edge_state", "skill_mask", "teacher_scores", "candidate_targets",
    ):
        setattr(item, field, getattr(case, field)[positions].clone())
    item.label = torch.tensor(positions.index(int(case.label)), dtype=torch.long)
    return item


def expand_native_case(case: TrainingCase, agent_count: int, fanout: int = 4) -> TrainingCase:
    """以实测候选为模板扩展系统图，并保持 O(N) 的固定邻居采样。"""

    base_ids = [int(value) for value in case.graph.candidate_ids.tolist()]
    base_count = len(base_ids)
    latest: dict[int, dict[str, torch.Tensor]] = {}
    for event in sorted(case.events, key=lambda item: item.timestamp):
        if event.src_node_type == 0 and event.src in base_ids:
            latest[event.src] = event.src_state
        if event.dst_node_type == 0 and event.dst in base_ids:
            latest[event.dst] = event.dst_state
    compute_edges: dict[int, torch.Tensor] = {}
    peer_states = case.graph.edge_state[case.graph.relation_type == 1]
    peer_template = peer_states if peer_states.numel() else torch.tensor([[20.0, 1000.0, 0.01, 2.0]])
    for edge_pos, (source, destination) in enumerate(case.graph.edge_index.t().tolist()):
        if int(case.graph.relation_type[edge_pos]) == 0 and destination in base_ids:
            compute_edges[destination] = case.graph.edge_state[edge_pos]

    agent_compute = []
    expanded_targets: list[torch.Tensor] = []
    expanded_scores: list[torch.Tensor] = []
    sources, destinations, edge_states, relations = [], [], [], []
    events: list[GraphEvent] = []
    observed = case.graph.query_time - 0.2
    for virtual_id in range(agent_count):
        template_pos = virtual_id % base_count
        template_id = base_ids[template_pos]
        compute_index = int(case.metadata["agent_compute"][template_id])
        agent_compute.append(compute_index)
        compute_id = agent_count + compute_index
        compute_edge = compute_edges.get(template_id, torch.tensor([50.0, 1000.0, 0.01, 3.0])).clone()
        sources.append(compute_id); destinations.append(virtual_id)
        edge_states.append(compute_edge); relations.append(0)
        state = {key: value.clone() for key, value in latest[template_id].items()}
        base_target = case.candidate_targets[template_pos].clone()
        reliability_delta = 0.055 * math.sin((virtual_id + 1) * 1.61803398875)
        execution_factor = 0.72 + 0.56 * (0.5 + 0.5 * math.sin((virtual_id + 3) * 0.731))
        base_target[0] = (base_target[0] + reliability_delta).clamp(0.65, 0.995)
        base_target[1] = (base_target[1] * execution_factor).clamp(0.01, 1.0)
        base_target[2] = (base_target[2] - 0.35 * reliability_delta).clamp(0.001, 0.25)
        base_target[3] = (base_target[3] - 0.25 * reliability_delta).clamp(0.001, 0.20)
        base_target[4] = (base_target[4] + 0.50 * reliability_delta).clamp(0.05, 1.0)
        state["reliability"] = torch.tensor([
            base_target[0], base_target[1] * 120_000.0, base_target[2], base_target[3]
        ])
        state["workload"] = (
            state.get("workload", torch.zeros(1)) + (virtual_id % 7) * 0.35
        ).clamp(0, 16)
        expanded_targets.append(base_target)
        expanded_scores.append(case.teacher_scores[template_pos] + 0.8 * reliability_delta - 0.15 * (execution_factor - 1.0))
        events.append(GraphEvent(
            src=compute_id, dst=virtual_id, src_node_type=1, dst_node_type=0,
            timestamp=observed, event_type=2, relation_type=0,
            src_state={"compute": torch.tensor(case.metadata["compute_states"][compute_index])},
            dst_state=state,
            edge_state=compute_edge, edge_confidence=1.0, edge_observed_at=observed,
        ))
        for offset in range(1, min(fanout, agent_count - 1) + 1):
            sources.append((virtual_id - offset) % agent_count); destinations.append(virtual_id)
            edge_states.append(peer_template[(virtual_id + offset) % len(peer_template)].clone())
            relations.append(1)

    template_positions = torch.arange(agent_count) % base_count
    skills = tuple(case.graph.candidate_skill_ids[int(index)] for index in template_positions)
    active_skill = case.task.required_capability
    graph = QueryGraph(
        candidate_ids=torch.arange(agent_count),
        edge_index=torch.tensor([sources, destinations], dtype=torch.long),
        edge_state=torch.stack(edge_states), relation_type=torch.tensor(relations),
        edge_timestamp=torch.full((len(sources),), observed),
        edge_confidence=torch.ones(len(sources)), query_time=case.graph.query_time,
        candidate_skill_mask=torch.tensor([active_skill in set(items) for items in skills]),
        candidate_skill_ids=skills,
    )
    item = TrainingCase(
        events=events, graph=graph, task=copy.deepcopy(case.task),
        label=torch.tensor(int(case.label)),
        candidate_targets=torch.stack(expanded_targets),
        teacher_scores=torch.stack(expanded_scores),
        metadata=copy.deepcopy(case.metadata),
    )
    item.metadata["agent_compute"] = agent_compute
    item.metadata["candidate_roles"] = [case.metadata["candidate_roles"][int(index)] for index in template_positions]
    item.metadata["virtual_source_agents"] = [base_ids[int(index)] for index in template_positions]
    return item


def expand_gcan_case(case: GCANCase, native: TrainingCase, agent_count: int, fanout: int = 4) -> GCANCase:
    base_count = case.candidate_ids.numel()
    positions = torch.arange(agent_count) % base_count
    item = copy.deepcopy(case)
    for field in (
        "agent_numeric", "agent_history", "history_mask", "history_delta_time", "agent_text",
        "task_agent_edge_state", "teacher_scores", "candidate_targets",
    ):
        setattr(item, field, getattr(case, field)[positions].clone())
    item.candidate_ids = torch.arange(agent_count)
    active = native.task.required_capability
    item.skill_mask = torch.tensor([
        active in set(native.graph.candidate_skill_ids[int(index)]) for index in positions
    ], dtype=torch.bool)
    item.task_agent_edge_state[:, 0] = item.skill_mask.float()
    sources, destinations = [], []
    for destination in range(agent_count):
        for offset in range(1, min(fanout, agent_count - 1) + 1):
            sources.append((destination - offset) % agent_count); destinations.append(destination)
    item.collaboration_edge_index = torch.tensor([sources, destinations], dtype=torch.long)
    if case.collaboration_edge_state.numel():
        edge_positions = torch.arange(len(sources)) % case.collaboration_edge_state.shape[0]
        item.collaboration_edge_state = case.collaboration_edge_state[edge_positions].clone()
    else:
        item.collaboration_edge_state = torch.zeros(len(sources), 8)
    item.label = torch.tensor(int(case.label))
    return item


def resized_temporal_model(model, config: ModelConfig, agent_count: int, method: str):
    resized_config = copy.deepcopy(config)
    resized_config.num_agents = agent_count
    resized = (
        TaskConditionedHeterogeneousTGN(resized_config)
        if method == "Ours (DAG-GRU)" else MinimalStateAblationAllocator(resized_config)
    )
    source = model.state_dict()
    target = resized.state_dict()
    compatible = {key: value for key, value in source.items() if key in target and value.shape == target[key].shape}
    resized.load_state_dict(compatible, strict=False)
    resized.eval()
    return resized


def task_metrics(
    case: TrainingCase,
    assigned_agents: list[int],
    duration_multiplier: dict[int, float] | None = None,
) -> dict[str, float]:
    duration_multiplier = duration_multiplier or {}
    candidate_position = {int(agent): index for index, agent in enumerate(case.graph.candidate_ids.tolist())}
    compute_latency: dict[int, float] = {}
    peer_latency: dict[tuple[int, int], float] = {}
    for edge_index, (source, destination) in enumerate(case.graph.edge_index.t().tolist()):
        latency_s = float(case.graph.edge_state[edge_index, 0]) / 1_000.0
        if int(case.graph.relation_type[edge_index]) == 0:
            compute_latency[destination] = min(compute_latency.get(destination, math.inf), latency_s)
        else:
            peer_latency[(source, destination)] = latency_s
    finish = [0.0] * len(assigned_agents)
    agent_available: dict[int, float] = {}
    compute_available: dict[int, float] = {}
    predecessors: dict[int, list[int]] = {index: [] for index in range(len(assigned_agents))}
    if case.task.dag_edge_index is not None:
        for source, destination in case.task.dag_edge_index.t().tolist():
            predecessors[int(destination)].append(int(source))
    qualities = []
    for index, agent in enumerate(assigned_agents):
        position = candidate_position[agent]
        target = case.candidate_targets[position]
        # 并发过载下把超时/中断的尝试按 0 分计入提交任务平均准确率；单任务时系数为 1。
        contention_quality = 1.0 / math.sqrt(duration_multiplier.get(agent, 1.0))
        qualities.append(float(target[0]) * (1.0 - float(target[3])) * contention_quality)
        ready = 0.0
        for predecessor in predecessors[index]:
            previous_agent = assigned_agents[predecessor]
            communication = 0.0 if previous_agent == agent else peer_latency.get((previous_agent, agent), 1.0)
            ready = max(ready, finish[predecessor] + communication)
        duration = float(target[1]) * 120.0 * duration_multiplier.get(agent, 1.0)
        compute_index = int(case.metadata["agent_compute"][agent])
        start = max(ready, agent_available.get(agent, 0.0), compute_available.get(compute_index, 0.0))
        finish[index] = start + duration + compute_latency.get(agent, 1.0)
        agent_available[agent] = finish[index]
        compute_available[compute_index] = finish[index]
    latency = max(finish, default=0.0)
    return {
        "completion_accuracy": sum(qualities) / max(1, len(qualities)),
        "completion_latency_s": latency,
    }


def method_models(config: ModelConfig, seed: int):
    torch.manual_seed(seed)
    return {
        "Ours (DAG-GRU)": TaskConditionedHeterogeneousTGN(copy.deepcopy(config)),
        "GMATANN": GraphConvolutionalAssignmentNetwork(GCANConfig()),
        "Rule-Latency": RuleBasedLatencyScheduler(),
        "State-MLP": MLPStateAllocator(),
        "StaticGraph": StaticGraphAllocator(),
        "GRU-Minimal": MinimalStateAblationAllocator(copy.deepcopy(config)),
    }


def fit_seed(seed, epochs, dataset, output, reuse_checkpoints: bool = False):
    native = {split: list(iter_cases(dataset / f"{split}.jsonl")) for split in ("train", "val", "test")}
    gcan = {split: load_cases(dataset / f"{split}.jsonl") for split in ("train", "val", "test")}
    native_subtasks = {split: [] for split in native}
    gcan_subtasks = {split: [] for split in native}
    for split in native:
        for case_index, case in enumerate(native[split]):
            for subtask_index in range(len(case.task.dag_required_capabilities)):
                native_subtasks[split].append(subtask_native_case(case, subtask_index))
                gcan_subtasks[split].append(subtask_gcan_case(
                    gcan[split][case_index], case, subtask_index,
                ))
    metadata = json.loads((dataset / "metadata.json").read_text(encoding="utf-8"))
    config = ModelConfig(
        num_agents=int(metadata["num_agents"]), num_compute_nodes=int(metadata["num_compute_nodes"]), dropout=0.05,
    )
    models = method_models(config, seed)
    histories = {}
    for method in METHODS:
        model = models[method]
        if method == "Rule-Latency":
            histories[method] = []
            continue
        if hasattr(model, "fit_normalizers"):
            model.fit_normalizers(native["train"])
        checkpoint_dir = output / "checkpoints" / f"seed_{seed}"
        checkpoint = checkpoint_dir / f"{method.replace(' ', '_')}.pt"
        if reuse_checkpoints and checkpoint.exists():
            checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            model.load_state_dict(checkpoint_payload["model"])
            model.eval()
            histories[method] = checkpoint_payload.get("history", [])
            print(json.dumps({"seed": seed, "reused": method}, ensure_ascii=False), flush=True)
            continue
        kind = "gmatann" if method == "GMATANN" else "flat" if method in {"State-MLP", "StaticGraph"} else "temporal"
        train_source = (
            gcan_subtasks["train"] if kind == "gmatann"
            else native_subtasks["train"] if kind == "flat"
            else native["train"]
        )
        val_source = native_subtasks["val"] if kind == "flat" else native["val"]
        histories[method] = train_model(
            model, train_source, val_source, kind, epochs, seed, gcan_subtasks["val"],
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"name": method, "model": model.state_dict(), "history": histories[method]}, checkpoint)
        print(json.dumps({"seed": seed, "trained": method, "best_validation": max(row["validation_score"] for row in histories[method])}, ensure_ascii=False), flush=True)

    case_rows = []
    graph_rows = []
    assignment_cache: dict[str, list[list[int]]] = {method: [] for method in METHODS}
    for method in METHODS:
        model = models[method]
        for index, case in enumerate(native["test"]):
            chosen = assignments(model, case, method, gcan["test"][index])
            assignment_cache[method].append(chosen)
            metrics = task_metrics(case, chosen)
            case_rows.append({
                "seed": seed, "method": method, "task_id": case.metadata["task_id"],
                "scenario": case.metadata["task_type"], "difficulty": case.metadata["difficulty_level"],
                **metrics,
            })
    # 规模实验使用固定 fanout=4 的稀疏计算图；候选智能体数就是横坐标节点数。
    scale_indices = [
        index for index, case in enumerate(native["test"])
        if case.metadata["difficulty_level"] == "中等"
    ]
    for size in GRAPH_SIZES:
        expanded_cases = {
            index: expand_native_case(native["test"][index], size) for index in scale_indices
        }
        expanded_gcan = {
            index: expand_gcan_case(gcan["test"][index], expanded_cases[index], size)
            for index in scale_indices
        }
        for method in METHODS:
            model = models[method]
            if method in {"Ours (DAG-GRU)", "GRU-Minimal"}:
                model = resized_temporal_model(model, config, size, method)
            for index in scale_indices:
                scaled_case = expanded_cases[index]
                scaled_gcan = expanded_gcan[index]
                chosen = assignments(model, scaled_case, method, scaled_gcan)
                graph_rows.append({
                    "seed": seed, "method": method, "graph_size": size,
                    "task_id": scaled_case.metadata["task_id"],
                    **task_metrics(scaled_case, chosen),
                })

    concurrency_rows = []
    medium_indices = [
        index for index, case in enumerate(native["test"])
        if case.metadata["difficulty_level"] == "中等"
    ]
    if not medium_indices:
        raise ValueError("测试集没有中等难度任务，无法运行并发任务曲线")
    test_count = len(medium_indices)
    for method in METHODS:
        for concurrent in CONCURRENCY:
            for rotation in range(test_count):
                indices = [medium_indices[(rotation + offset) % test_count] for offset in range(concurrent)]
                agent_load = Counter()
                compute_load = Counter()
                for index in indices:
                    case = native["test"][index]
                    for agent in assignment_cache[method][index]:
                        agent_load[agent] += 1
                        compute_load[int(case.metadata["agent_compute"][agent])] += 1
                for index in indices:
                    case = native["test"][index]
                    multipliers = {
                        agent: 1.0 + 0.15 * (agent_load[agent] - 1)
                        + 0.06 * (compute_load[int(case.metadata["agent_compute"][agent])] - 1)
                        for agent in set(assignment_cache[method][index])
                    }
                    metrics = task_metrics(case, assignment_cache[method][index], multipliers)
                    concurrency_rows.append({
                        "seed": seed, "method": method, "concurrent_tasks": concurrent,
                        "rotation": rotation, "task_id": case.metadata["task_id"], **metrics,
                    })
    return case_rows, graph_rows, concurrency_rows, histories


def plot_results(case_frame: pd.DataFrame, graph_frame: pd.DataFrame, concurrency_frame: pd.DataFrame, output: Path):
    config_dir = output / ".matplotlib"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(config_dir)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
        "axes.labelsize": 9, "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    colors = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00")
    markers = ("o", "s", "^", "D", "P", "X")
    hatches = ("", "//", "..", "xx", "\\\\", "++")

    def save(figure, stem):
        figure.savefig(output / f"{stem}.png", dpi=360, bbox_inches="tight")
        figure.savefig(output / f"{stem}.pdf", bbox_inches="tight")
        plt.close(figure)

    specs = (
        ("completion_accuracy", "Average completion accuracy (%)", (70, 100), "01_accuracy_scenario_difficulty"),
        ("completion_latency_s", "Average DAG completion latency (s)", None, "02_latency_scenario_difficulty"),
    )
    for metric, ylabel, ylim, stem in specs:
        figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.0), sharey=True, constrained_layout=True)
        x = np.arange(len(DIFFICULTIES))
        width = 0.13
        for scenario_index, scenario in enumerate(SCENARIOS):
            axis = axes[scenario_index]
            for method_index, method in enumerate(METHODS):
                means, errors = [], []
                for difficulty in DIFFICULTIES:
                    values = case_frame[
                        (case_frame.scenario == scenario) & (case_frame.difficulty == difficulty)
                        & (case_frame.method == method)
                    ][metric]
                    if metric != "completion_latency_s":
                        values = values * 100.0
                    means.append(values.mean())
                    error = values.std(ddof=0)
                    errors.append(error)
                axis.bar(
                    x + (method_index - 2.5) * width, means, width,
                    yerr=errors, capsize=1.8, color=colors[method_index], edgecolor="#333333",
                    linewidth=0.35, hatch=hatches[method_index], label=method,
                )
            axis.set_title(SCENARIO_EN[scenario], fontweight="bold")
            axis.set_xticks(x, [DIFFICULTY_EN[item] for item in DIFFICULTIES])
            axis.grid(axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.75)
            if ylim:
                axis.set_ylim(*ylim)
        axes[0].set_ylabel(ylabel)
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="outside upper center", ncol=6, frameon=False)
        save(figure, stem)

    curve_specs = (
        ("completion_accuracy", "Completion accuracy (%)"),
        ("completion_latency_s", "DAG completion latency (s)"),
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=True)
    for axis, (metric, ylabel) in zip(axes, curve_specs):
        for method_index, method in enumerate(METHODS):
            means, errors = [], []
            for size in GRAPH_SIZES:
                by_seed = graph_frame[(graph_frame.method == method) & (graph_frame.graph_size == size)].groupby("seed")[metric].mean()
                if metric != "completion_latency_s":
                    by_seed = by_seed * 100.0
                means.append(by_seed.mean())
                errors.append(by_seed.std(ddof=0))
            axis.errorbar(
                GRAPH_SIZES, means, yerr=errors, color=colors[method_index], marker=markers[method_index],
                linewidth=1.7, markersize=4.5, capsize=2, label=method,
            )
        axis.set(xlabel="Agent nodes in modeled graph", ylabel=ylabel, xticks=GRAPH_SIZES, xscale="log")
        if metric == "completion_accuracy":
            axis.set_ylim(70, 100)
        axis.grid(color="#d9d9d9", linewidth=0.6, alpha=0.75)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
    save(figure, "03_graph_scale_curves")

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=True)
    for axis, metric, ylabel in (
        (axes[0], "completion_accuracy", "Completion accuracy (%)"),
        (axes[1], "completion_latency_s", "DAG completion latency (s)"),
    ):
        for method_index, method in enumerate(METHODS):
            means, errors = [], []
            for concurrent in CONCURRENCY:
                by_seed = concurrency_frame[
                    (concurrency_frame.method == method) & (concurrency_frame.concurrent_tasks == concurrent)
                ].groupby("seed")[metric].mean()
                if metric == "completion_accuracy":
                    by_seed = by_seed * 100.0
                means.append(by_seed.mean())
                errors.append(by_seed.std(ddof=0))
            axis.errorbar(
                CONCURRENCY, means, yerr=errors, color=colors[method_index], marker=markers[method_index],
                linewidth=1.7, markersize=4.5, capsize=2, label=method,
            )
        axis.set(xlabel="Concurrent medium-difficulty tasks", ylabel=ylabel, xticks=CONCURRENCY, xscale="log")
        axis.grid(color="#d9d9d9", linewidth=0.6, alpha=0.75)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
    save(figure, "04_concurrency_curves")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "1.data" / "datasets" / "sixg_ma_state")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seeds", default="7,17,29")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "paper_simulation_v4")
    parser.add_argument("--reuse-checkpoints", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    all_cases, all_graph, all_concurrency = [], [], []
    all_histories = {}
    for seed in [int(value) for value in args.seeds.split(",")]:
        case_rows, graph_rows, concurrency_rows, histories = fit_seed(
            seed, args.epochs, args.dataset, args.output, args.reuse_checkpoints,
        )
        all_cases.extend(case_rows)
        all_graph.extend(graph_rows)
        all_concurrency.extend(concurrency_rows)
        all_histories[str(seed)] = histories
    case_frame = pd.DataFrame(all_cases)
    graph_frame = pd.DataFrame(all_graph)
    concurrency_frame = pd.DataFrame(all_concurrency)
    case_frame.to_csv(args.output / "scenario_difficulty_metrics.csv", index=False, encoding="utf-8-sig")
    graph_frame.to_csv(args.output / "graph_scale_metrics.csv", index=False, encoding="utf-8-sig")
    concurrency_frame.to_csv(args.output / "concurrency_metrics.csv", index=False, encoding="utf-8-sig")
    (args.output / "training_history.json").write_text(
        json.dumps(all_histories, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    plot_results(case_frame, graph_frame, concurrency_frame, args.output)
    summary = case_frame.groupby("method")[["completion_accuracy", "completion_latency_s"]].mean().reindex(METHODS)
    protocol = {
        "methods": list(METHODS), "epochs": args.epochs,
        "seeds": [int(value) for value in args.seeds.split(",")],
        "split": "54/9/9 chronological, non-overlapping, scenario-difficulty-stratified",
        "candidate_protocol": "48 candidates per case; every subtask Skill has 4-5 eligible agents; skills shared by instances of the same agent type",
        "supervision_protocol": "every DAG subtask is supervised with the same QoS-weighted physical teacher; not only dag_current_index",
        "main_decoder": "DAG-aware joint decoder using learned attention/state predictions and same-level agent/compute contention penalties",
        "metric_definitions": {
            "completion_accuracy": "mean over subtasks of success_rate * (1 - hallucination_rate); overloaded interrupted attempts count as zero in concurrency simulation",
            "completion_latency_s": "DAG critical-path latency with agent execution and communication/compute-access latency",
        },
        "latency_model": "DAG critical path plus serialization on reused agent/compute resources",
        "concurrency_model": "processor-sharing approximation: 15% per extra task on an agent + 6% per extra task on its compute node",
        "graph_sizes": list(GRAPH_SIZES), "concurrent_tasks": list(CONCURRENCY),
        "summary": json.loads(summary.to_json(orient="index")),
        "graph_scaling_protocol": "fixed-fanout sparse neighbor sampling; x-axis counts all modeled agent nodes",
    }
    (args.output / "simulation_summary.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(summary.to_string(), flush=True)
    print(f"output={args.output}", flush=True)


if __name__ == "__main__":
    main()
