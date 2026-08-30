"""把基础 CSV/JSON 表转换为 MA-TGN 可直接读取的滑动窗口 JSONL。"""

from __future__ import annotations

import hashlib
import json
import math
import random
from bisect import bisect_right
from pathlib import Path

import pandas as pd
import torch

from ma_tgn.config import ModelConfig
from ma_tgn.state_schema import schema_dict
from ma_tgn.types import EventType, GraphEvent, QueryGraph, RelationType, TaskRequirement, TrainingCase

from .dataset import write_cases


TASK_SKILL_FIELD = "必需的Skill id"
TASK_OUTPUT_FIELD = "输出结果的要求"
TASK_COMPLETENESS_FIELD = "字段完整率要求"
TASK_LATENCY_FIELD = "端到端完成时延要求(s)"

# 基于现有 72 条任务整理的研究用复杂任务流程。顺序表示依赖的主干方向；
# 同一子任务可能出现在多个合理流程中，数据构建时采用列表中第一个匹配流程。
WORKFLOW_TEMPLATES: tuple[tuple[str, ...], ...] = (
    ("T0001", "T0007", "T0019", "T0031", "T0020", "T0021"),  # 居家跌倒响应
    ("T0002", "T0012", "T0016", "T0017", "T0018", "T0034"),  # 居家入侵处置
    ("T0004", "T0013", "T0026", "T0025", "T0027", "T0033"),  # 居家环境优化
    ("T0010", "T0022", "T0023", "T0024", "T0035", "T0034"),  # 居家能源报告
    ("T0010", "T0028", "T0029", "T0032", "T0030"),  # 居家自动化控制
    ("T0005", "T0014", "T0021", "T0027", "T0034"),  # 居家状态问答/汇总
    ("T0008", "T0009", "T0011", "T0043", "T0060", "T0045", "T0044"),  # 工厂故障维护
    ("T0003", "T0006", "T0015", "T0042", "T0063"),  # 工厂多源问答/摘要
    ("T0037", "T0038", "T0058", "T0062", "T0061", "T0039"),  # 工厂质量分析
    ("T0049", "T0041", "T0040", "T0050", "T0059", "T0051"),  # 生产与机器人协同
    ("T0046", "T0047", "T0048", "T0063"),  # 工厂安全闭环
    ("T0052", "T0053", "T0054", "T0063"),  # 工厂能源诊断
    ("T0055", "T0056", "T0057", "T0040"),  # 厂内物流补货调度
    ("T0064", "T0066", "T0065", "T0036"),  # 楼宇安防事件处置
    ("T0068", "T0069", "T0067", "T0036"),  # 楼宇环境联动
    ("T0071", "T0070", "T0072", "T0036"),  # 园区车辆通行
)


def _stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def _clip(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return min(maximum, max(minimum, float(value)))


def _edge_tensor(row: pd.Series | dict) -> torch.Tensor:
    return torch.tensor([
        float(row["时延(ms)"]),
        float(row["带宽(Mbps)"]),
        float(row["网络丢包率"]),
        float(row["网络抖动(ms)"]),
    ], dtype=torch.float32)


def _build_agent_reliability(agents: list[dict], reliability: dict[int, dict]) -> list[torch.Tensor]:
    rows: list[torch.Tensor] = []
    for agent in agents:
        agent_id = int(agent["智能体编号"])
        metrics = reliability[agent_id]
        hallucination_rate = _clip((1.0 - float(metrics["任务成功率"])) * 0.35, 0.005, 0.10)
        rows.append(torch.tensor([
            float(metrics["任务成功率"]),
            float(metrics["平均任务完成时间(秒)"]) * 1_000.0,
            float(metrics["工具调用失败率"]),
            hallucination_rate,
        ], dtype=torch.float32))
    return rows


def _agent_state(agent: dict, reliability: torch.Tensor, tick: int) -> dict[str, torch.Tensor]:
    total_tokens = float(agent["上下文窗口总量(tokens)"])
    base_usage = float(agent["上下文窗口当前占用率"])
    utilization = _clip(base_usage + 0.035 * math.sin((tick + int(agent["智能体编号"])) / 7.0))
    api_limit_fraction = _clip(float(agent["工具API速率限制(次/分钟)"]) / 240.0)
    api_occupied = 1.0 if agent["工具API是否被占用"] == "是" else 0.0
    has_knowledge = agent["接入知识库"] != "无"
    has_sensor = agent["物理传感器可用情况"] != "无"
    digital_only = agent["运行环境"] == "数字环境"
    environment = torch.tensor([
        api_limit_fraction * (0.35 if api_occupied else 1.0),
        api_occupied,
        1.0 if has_knowledge else 0.0,
        0.12 if has_knowledge else 0.0,
        1.0 - utilization,
        1.0 if has_sensor or digital_only else 0.5,
        1.0,
    ], dtype=torch.float32)
    return {
        "context": torch.tensor([total_tokens, utilization], dtype=torch.float32),
        "workload": torch.tensor([
            max(0.0, round(6.0 * utilization + (2.0 if api_occupied else 0.0) + 1.5 * math.sin((tick + int(agent["智能体编号"])) / 5.0)))
        ], dtype=torch.float32),
        "environment": environment,
        "reliability": reliability.clone(),
    }


def _compute_state(load_row: pd.Series, inventory_row: dict) -> torch.Tensor:
    del inventory_row
    return torch.tensor([
        float(load_row["GPU算力剩余量(TFLOPS)"]),
        float(load_row["内存剩余量(GB)"]),
        float(load_row["显存剩余量(GB)"]),
    ], dtype=torch.float32)


def _task_qos(task: dict) -> torch.Tensor:
    latency_seconds = float(task[TASK_LATENCY_FIELD])
    return torch.tensor([
        _clip(1.0 - latency_seconds / 120.0),
        float(task["准确率要求"]),
    ], dtype=torch.float32)


def _dependency_stage(skill: str) -> int:
    action = skill.lower().split(".")[-1]
    if action in {"collect", "query", "monitor", "detect", "recognize", "status"}:
        return 0
    if action in {"analyze", "classify", "calculate", "extract"}:
        return 1
    if action in {"fuse", "assess", "verify", "evaluate"}:
        return 2
    if action in {"optimize", "recommend", "control", "prepare", "decide"}:
        return 3
    return 4


def _task_similarity(left: dict, right: dict) -> float:
    left_skill = str(left[TASK_SKILL_FIELD]).lower().split(".")
    right_skill = str(right[TASK_SKILL_FIELD]).lower().split(".")
    left_domain = set(left_skill[2:-1])
    right_domain = set(right_skill[2:-1])
    union = left_domain | right_domain
    overlap = len(left_domain & right_domain) / max(1, len(union))
    same_source = left.get("source_agent_type_id") == right.get("source_agent_type_id")
    return 4.0 * float(same_source) + 2.0 * overlap - 0.15 * abs(
        _dependency_stage(str(left[TASK_SKILL_FIELD])) - _dependency_stage(str(right[TASK_SKILL_FIELD]))
    )


def _build_task_dag(current: dict, tasks: list[dict], max_subtasks: int = 5) -> tuple[list[dict], list[list[int]], int, str]:
    """从同场景、同能力域任务构造可审计的局部复杂任务 DAG。"""
    if current.get("任务类型") == "复杂任务" and current.get("子任务"):
        selected = list(current["子任务"])
        id_to_index = {item["任务id"]: index for index, item in enumerate(selected)}
        edges = [(id_to_index[source], id_to_index[target]) for source, target in current["DAG"]["边"]]
        current_index = _stable_int(str(current["任务id"])) % len(selected)
        return (
            selected,
            [[source for source, _ in edges], [target for _, target in edges]],
            current_index,
            str(current["任务id"]),
        )
    task_by_id = {str(task["任务id"]): task for task in tasks}
    template_index = next(
        (index for index, template in enumerate(WORKFLOW_TEMPLATES) if current["任务id"] in template),
        None,
    )
    if template_index is not None:
        selected = [task_by_id[task_id] for task_id in WORKFLOW_TEMPLATES[template_index]]
    else:
        pool = [task for task in tasks if task["场景"] == current["场景"] and task["任务id"] != current["任务id"]]
        pool.sort(key=lambda task: (-_task_similarity(current, task), str(task["任务id"])))
        selected = [current, *pool[: max(1, max_subtasks - 1)]]
        selected.sort(key=lambda task: (_dependency_stage(str(task[TASK_SKILL_FIELD])), str(task["任务id"])))
    current_index = next(index for index, task in enumerate(selected) if task["任务id"] == current["任务id"])
    edges: list[tuple[int, int]] = []
    for index in range(len(selected) - 1):
        edges.append((index, index + 1))
        source_stage = _dependency_stage(str(selected[index][TASK_SKILL_FIELD]))
        if index + 2 < len(selected):
            target_stage = _dependency_stage(str(selected[index + 2][TASK_SKILL_FIELD]))
            if source_stage < target_stage:
                edges.append((index, index + 2))
    edge_index = [
        [source for source, _ in edges],
        [target for _, target in edges],
    ]
    workflow = f"W{template_index + 1:02d}" if template_index is not None else str(current["任务id"])
    complex_task_id = f"DAG-{current['场景']}-{workflow}"
    return selected, edge_index, current_index, complex_task_id


def _task_requirement(task: dict, tasks: list[dict]) -> TaskRequirement:
    selected, edge_index, current_index, complex_task_id = _build_task_dag(task, tasks)
    active = selected[current_index]
    required_skill = str(active[TASK_SKILL_FIELD])
    qos = _task_qos(active)
    return TaskRequirement(
        qos,
        torch.empty(0),
        description=str(active["任务具体提示词"]),
        required_capability=required_skill,
        dag_descriptions=tuple(str(item["任务具体提示词"]) for item in selected),
        dag_required_capabilities=tuple(str(item[TASK_SKILL_FIELD]) for item in selected),
        dag_qos=torch.stack([_task_qos(item) for item in selected]),
        dag_edge_index=torch.tensor(edge_index, dtype=torch.long),
        dag_current_index=current_index,
        complex_task_id=complex_task_id,
    )


def _agent_ability_text(agent: dict) -> str:
    """保留可审计的原始能力语义；不把标签或 teacher score 写入文本。"""
    skill_text = "；".join(
        f"{skill['id']} {skill.get('name', '')} {skill.get('description', '')}"
        for skill in agent.get("skills", [])
    )
    tools = "、".join(str(item) for item in agent.get("tool_apis", []))
    return (
        f"{agent.get('智能体能力描述', '')}。技能：{skill_text}。"
        f"工具API：{tools}。知识库：{agent.get('接入知识库', '无')}。"
        f"运行环境：{agent.get('运行环境', '')}。传感器：{agent.get('物理传感器可用情况', '')}"
    )


def _candidate_agents(task: dict, agents: list[dict], candidate_count: int, seed: int) -> list[int]:
    required_skill = str(task[TASK_SKILL_FIELD])
    source_type = str(task["source_agent_type_id"])
    scenario = str(task["场景"])
    exact = [index for index, agent in enumerate(agents) if required_skill in {skill["id"] for skill in agent["skills"]}]
    same_source = [index for index, agent in enumerate(agents) if agent["源智能体类型id"] == source_type and index not in exact]
    same_scenario = [index for index, agent in enumerate(agents) if agent["场景"] == scenario and index not in exact and index not in same_source]
    others = [index for index in range(len(agents)) if index not in exact and index not in same_source and index not in same_scenario]
    rng = random.Random(seed + _stable_int(str(task["任务id"])))
    rng.shuffle(exact)
    rng.shuffle(same_source)
    rng.shuffle(same_scenario)
    rng.shuffle(others)
    candidates = (exact + same_source + same_scenario + others)[:candidate_count]
    rng.shuffle(candidates)
    if not any(required_skill in {skill["id"] for skill in agents[index]["skills"]} for index in candidates):
        raise AssertionError(f"任务 {task['任务id']} 没有具备必需 skill 的候选智能体")
    return candidates


def _candidate_agents_for_dag(
    subtasks: list[dict], active_task: dict, agents: list[dict], candidate_count: int, seed: int,
    min_eligible_per_skill: int = 4,
) -> list[int]:
    """候选集中为每个子任务保留多个精确 Skill 候选。"""
    skill_sets = [{skill["id"] for skill in agent["skills"]} for agent in agents]
    mandatory: list[int] = []
    for subtask in subtasks:
        skill = str(subtask[TASK_SKILL_FIELD])
        exact = [index for index, available in enumerate(skill_sets) if skill in available]
        if not exact:
            raise AssertionError(f"子任务 {subtask['任务id']} 的 Skill {skill} 没有可用智能体")
        exact.sort(key=lambda index: _stable_int(f"{subtask['任务id']}:{index}:{seed}"))
        for chosen in exact[:min(min_eligible_per_skill, len(exact))]:
            if chosen not in mandatory:
                mandatory.append(chosen)
    if len(mandatory) > candidate_count:
        raise ValueError("candidate_count 小于复杂任务所需的不同 Skill 智能体数量")
    preferred = _candidate_agents(active_task, agents, candidate_count, seed)
    scenario = str(active_task["场景"])
    remainder = [index for index, agent in enumerate(agents) if agent["场景"] == scenario]
    remainder += list(range(len(agents)))
    candidates = mandatory.copy()
    for index in [*preferred, *remainder]:
        if index not in candidates:
            candidates.append(index)
        if len(candidates) == candidate_count:
            break
    random.Random(seed + _stable_int(str(active_task["父任务id"]))).shuffle(candidates)
    return candidates


def _share_skills_within_agent_type(agents: list[dict]) -> None:
    """同一智能体类型的实例共享能力卡，避免 Skill 筛选后退化为唯一候选。"""

    by_type: dict[str, dict[str, dict]] = {}
    for agent in agents:
        role = str(agent["源智能体类型id"])
        union = by_type.setdefault(role, {})
        for skill in agent["skills"]:
            union[str(skill["id"])] = skill
    for agent in agents:
        role = str(agent["源智能体类型id"])
        agent["skills"] = [by_type[role][skill_id] for skill_id in sorted(by_type[role])]


def _split_counts(total: int) -> dict[str, int]:
    train = max(1, int(total * 0.75))
    val = max(1, int(total * 0.125))
    if train + val >= total:
        train = total - 2
    return {"train": train, "val": val, "test": total - train - val}


def _stratified_task_order(tasks: list[dict], seed: int) -> tuple[list[dict], bool]:
    """让时间顺序切分后的 val/test 均覆盖每个场景×难度组合。"""

    groups: dict[tuple[str, str], list[dict]] = {}
    for task in tasks:
        key = (str(task.get("场景", "")), str(task.get("难度级别", "")))
        groups.setdefault(key, []).append(task)
    if len(groups) != 9 or any(len(items) < 3 for items in groups.values()):
        shuffled = tasks.copy()
        random.Random(seed).shuffle(shuffled)
        return shuffled, False
    partitions: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for key in sorted(groups):
        items = groups[key].copy()
        random.Random(seed + _stable_int("|".join(key))).shuffle(items)
        counts = _split_counts(len(items))
        cursor = 0
        for split, count in counts.items():
            partitions[split].extend(items[cursor:cursor + count])
            cursor += count
    for offset, split in enumerate(("train", "val", "test")):
        random.Random(seed + 10_000 + offset).shuffle(partitions[split])
    ordered = partitions["train"] + partitions["val"] + partitions["test"]
    return ordered, True


def generate_foundation_dataset(
    data_root: Path,
    output_dir: Path,
    *,
    seed: int = 20260814,
    window_size_timestamps: int = 16,
    candidate_count: int = 12,
    event_count: int = 24,
    neighbor_fanout: int = 8,
    share_role_skills: bool = True,
    samples: int | None = None,
) -> dict:
    """从基础表生成按时间滑动的 train/val/test JSONL。

    时间窗口按 Alibaba 匿名时间戳顺序选取；默认 72 个窗口与 72 个任务一一对应。
    窗口起点均匀分布，默认相邻窗口无重叠，并按时间先后切分数据集。
    """
    if window_size_timestamps < 2:
        raise ValueError("window_size_timestamps 至少为 2")
    if candidate_count < 2:
        raise ValueError("candidate_count 至少为 2")
    if event_count < candidate_count:
        raise ValueError("event_count 不能小于 candidate_count，否则无法观测全部候选")
    agents_path = data_root / "3.智能体能力数据(包括智能体环境和知识库、API权限)" / "agent_capabilities.json"
    reliability_path = data_root / "4.智能体可靠性" / "agent_reliability.csv"
    inventory_path = data_root / "1.负载数据" / "compute_node_inventory.csv"
    load_path = data_root / "1.负载数据" / "compute_node_load_timeseries.csv"
    links_path = data_root / "2.链路数据" / "communication_links.csv"
    tasks_path = data_root / "5.任务数据" / "tasks.json"
    agents = json.loads(agents_path.read_text(encoding="utf-8"))
    if share_role_skills:
        _share_skills_within_agent_type(agents)
    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    reliability_frame = pd.read_csv(reliability_path)
    inventory_frame = pd.read_csv(inventory_path)
    load_frame = pd.read_csv(load_path)
    links = pd.read_csv(links_path)
    required_task_fields = {
        "任务id", "任务具体提示词", TASK_SKILL_FIELD, TASK_OUTPUT_FIELD,
        TASK_COMPLETENESS_FIELD, TASK_LATENCY_FIELD, "准确率要求", "场景", "任务类型", "子任务", "DAG",
    }
    for task in tasks:
        missing = required_task_fields - set(task)
        if missing:
            raise ValueError(f"任务 {task.get('任务id')} 缺少字段: {sorted(missing)}")
        if "required_skill_id" in task or "实验要求" in task:
            raise ValueError("任务 JSON 仍含旧字段 required_skill_id 或 实验要求")
        if not task["子任务"]:
            raise ValueError(f"复杂任务 {task['任务id']} 没有子任务")
    if samples is None:
        samples = len(tasks)
    if samples < 3:
        raise ValueError("样本数至少为 3，才能生成 train/val/test")
    if samples > len(tasks):
        raise ValueError("样本数不能超过任务数；每个窗口默认绑定一个唯一任务")
    num_agents = len(agents)
    num_compute_nodes = len(inventory_frame)
    candidate_count = min(candidate_count, num_agents)
    config = ModelConfig(num_agents=num_agents, num_compute_nodes=num_compute_nodes)
    reliability = {int(row["智能体编号"]): row for row in reliability_frame.to_dict("records")}
    inventory = {str(row["算力节点编号"]): row for row in inventory_frame.to_dict("records")}
    agent_reliability = _build_agent_reliability(agents, reliability)
    roles = [str(agent["源智能体类型名称"]) for agent in agents]
    compute_ids = inventory_frame["算力节点编号"].astype(str).tolist()
    compute_kinds = ["cloud" if inventory[node]["节点类型"] == "云侧" else "edge" for node in compute_ids]

    links["时间戳"] = links["时间戳"].astype("int64")
    link_timestamps = sorted(links["时间戳"].unique().tolist())
    first_links = links[links["时间戳"] == link_timestamps[0]]
    link_lookup = {
        (int(row["时间戳"]), *sorted((str(row["数据源"]), str(row["数据目的地"])))): row
        for row in links.to_dict("records")
    }

    def link_at(raw_timestamp: int, source: str, target: str) -> dict:
        position = max(0, bisect_right(link_timestamps, int(raw_timestamp)) - 1)
        return link_lookup[(link_timestamps[position], *sorted((source, target)))]

    agent_compute: list[int] = []
    for index, agent in enumerate(agents):
        agent_key = str(agent["agent_key"])
        attached = first_links[(first_links["数据源"] == agent_key) & (first_links["链路类型"] == "智能体-算力节点")]
        if attached.empty:
            raise ValueError(f"{agent_key} 没有算力节点链路")
        if len(attached) != 1:
            raise ValueError(f"{agent_key} 必须有且只有一个固定算力节点链路")
        row = attached.iloc[0]
        compute_index = compute_ids.index(str(row["数据目的地"]))
        agent_compute.append(compute_index)

    load_frame["时间戳"] = load_frame["时间戳"].astype("int64")
    timestamps = sorted(load_frame["时间戳"].unique().tolist())
    if len(timestamps) < window_size_timestamps * samples:
        raise ValueError("时间戳数量不足以构造默认无重叠窗口")
    load_lookup = {
        (int(row["时间戳"]), str(row["算力节点编号"])): row
        for row in load_frame.to_dict("records")
    }
    max_start = len(timestamps) - window_size_timestamps
    starts = [round(index * max_start / max(1, samples - 1)) for index in range(samples)]
    if any(starts[index + 1] < starts[index] + window_size_timestamps for index in range(len(starts) - 1)):
        raise AssertionError("默认滑动窗口发生重叠，请减少样本数或窗口长度")
    shuffled_tasks, stratified_split = _stratified_task_order(tasks, seed)
    shuffled_tasks = shuffled_tasks[:samples]
    origin_timestamp = timestamps[0]
    cases: list[TrainingCase] = []
    case_ranges: list[tuple[int, int]] = []
    task_counts: dict[str, int] = {}

    for case_index, (start, task) in enumerate(zip(starts, shuffled_tasks)):
        window_timestamps = timestamps[start : start + window_size_timestamps]
        window_start, window_end = window_timestamps[0], window_timestamps[-1]
        case_ranges.append((window_start, window_end))
        task_requirement = _task_requirement(task, tasks)
        selected_subtasks, _, current_index, _ = _build_task_dag(task, tasks)
        active_task = selected_subtasks[current_index]
        required_skill = str(active_task[TASK_SKILL_FIELD])
        candidates = _candidate_agents_for_dag(selected_subtasks, active_task, agents, candidate_count, seed)
        query_time = float(window_end - origin_timestamp) + 0.1
        query_compute_states = [
            _compute_state(pd.Series(load_lookup[(window_end, node)]), inventory[node])
            for node in compute_ids
        ]
        events: list[GraphEvent] = []
        for position, agent_index in enumerate(candidates):
            raw_timestamp = window_timestamps[position % len(window_timestamps)]
            timestamp = float(raw_timestamp - origin_timestamp)
            compute_index = agent_compute[agent_index]
            compute_node = compute_ids[compute_index]
            load_row = pd.Series(load_lookup[(raw_timestamp, compute_node)])
            compute_state = _compute_state(load_row, inventory[compute_node])
            agent_state = _agent_state(agents[agent_index], agent_reliability[agent_index], start + position)
            edge_row = link_at(raw_timestamp, agents[agent_index]["agent_key"], compute_node)
            confidence = 1.0
            events.append(GraphEvent(
                src=num_agents + compute_index,
                dst=agent_index,
                src_node_type=1,
                dst_node_type=0,
                timestamp=timestamp,
                event_type=int(EventType.COMPUTE_UPDATE),
                relation_type=int(RelationType.COMPUTE_ACCESS),
                src_state={"compute": compute_state},
                dst_state=agent_state,
                edge_state=_edge_tensor(edge_row),
                src_confidence={"compute": torch.tensor([confidence])},
                dst_confidence={"context": torch.tensor([0.96]), "environment": torch.tensor([0.94])},
                src_observed_at={"compute": timestamp},
                dst_observed_at={"context": timestamp, "environment": timestamp},
                edge_confidence=confidence,
                edge_observed_at=timestamp,
            ))
        peer_cursor = 0
        while len(events) < event_count:
            dst = candidates[peer_cursor % len(candidates)]
            src = candidates[(peer_cursor + 1) % len(candidates)]
            raw_timestamp = window_timestamps[(len(events) - len(candidates)) % len(window_timestamps)]
            edge_row = link_at(raw_timestamp, agents[src]["agent_key"], agents[dst]["agent_key"])
            timestamp = float(raw_timestamp - origin_timestamp) + 0.01 * (peer_cursor + 1)
            events.append(GraphEvent(
                src=src,
                dst=dst,
                src_node_type=0,
                dst_node_type=0,
                timestamp=timestamp,
                event_type=int(EventType.LINK_UPDATE),
                relation_type=int(RelationType.COMMUNICATION),
                src_state=_agent_state(agents[src], agent_reliability[src], start + peer_cursor),
                dst_state=_agent_state(agents[dst], agent_reliability[dst], start + peer_cursor),
                edge_state=_edge_tensor(edge_row),
                src_confidence={"context": torch.tensor([0.95]), "environment": torch.tensor([0.93])},
                dst_confidence={"context": torch.tensor([0.95]), "environment": torch.tensor([0.93])},
                src_observed_at={"context": timestamp, "environment": timestamp},
                dst_observed_at={"context": timestamp, "environment": timestamp},
                edge_confidence=0.94,
                edge_observed_at=timestamp,
            ))
            peer_cursor += 1
        events.sort(key=lambda event: event.timestamp)

        edge_sources: list[int] = []
        edge_destinations: list[int] = []
        edge_states: list[torch.Tensor] = []
        relation_types: list[int] = []
        edge_timestamps: list[float] = []
        edge_confidences: list[float] = []
        candidate_link: dict[int, torch.Tensor] = {}
        for agent_index in candidates:
            compute_index = agent_compute[agent_index]
            edge_row = link_at(window_end, agents[agent_index]["agent_key"], compute_ids[compute_index])
            edge_state = _edge_tensor(edge_row)
            candidate_link[agent_index] = edge_state
            edge_sources.append(num_agents + compute_index)
            edge_destinations.append(agent_index)
            edge_states.append(edge_state)
            relation_types.append(int(RelationType.COMPUTE_ACCESS))
            edge_timestamps.append(query_time - 0.05)
            edge_confidences.append(0.96)
            peer_candidates = [peer for peer in candidates if peer != agent_index]
            peer_candidates.sort(key=lambda peer: _stable_int(
                f"{task['任务id']}:{agent_index}:{peer}:{seed}"
            ))
            peer_candidates = peer_candidates[:neighbor_fanout]
            for peer in peer_candidates:
                peer_row = link_at(window_end, agents[peer]["agent_key"], agents[agent_index]["agent_key"])
                edge_sources.append(peer)
                edge_destinations.append(agent_index)
                edge_states.append(_edge_tensor(peer_row))
                relation_types.append(int(RelationType.COMMUNICATION))
                edge_timestamps.append(query_time - 0.08)
                edge_confidences.append(0.93)
        graph = QueryGraph(
            candidate_ids=torch.tensor(candidates, dtype=torch.long),
            edge_index=torch.tensor([edge_sources, edge_destinations], dtype=torch.long),
            edge_state=torch.stack(edge_states),
            relation_type=torch.tensor(relation_types, dtype=torch.long),
            edge_timestamp=torch.tensor(edge_timestamps, dtype=torch.float32),
            edge_confidence=torch.tensor(edge_confidences, dtype=torch.float32),
            query_time=query_time,
            candidate_texts=(),
            candidate_skill_mask=torch.tensor([
                required_skill in {skill["id"] for skill in agents[index]["skills"]}
                for index in candidates
            ], dtype=torch.bool),
            candidate_skill_ids=tuple(
                tuple(skill["id"] for skill in agents[index]["skills"])
                for index in candidates
            ),
        )

        targets: list[torch.Tensor] = []
        teacher_scores: list[float] = []
        for agent_index in candidates:
            agent = agents[agent_index]
            metrics = reliability[int(agent["智能体编号"])]
            skill_ids = [skill["id"] for skill in agent["skills"]]
            exact = required_skill in skill_ids
            capability_score = 1.0 if exact else 0.0
            agent_state = _agent_state(agent, agent_reliability[agent_index], start + window_size_timestamps)
            context_score = float(1.0 - agent_state["context"][1]) * min(1.0, float(agent_state["context"][0]) / 32_768.0)
            environment_score = float(agent_state["environment"].mean())
            compute_state = query_compute_states[agent_compute[agent_index]]
            compute_node = compute_ids[agent_compute[agent_index]]
            gpu_total = 4_000.0 if inventory[compute_node]["节点类型"] == "云侧" else 400.0
            compute_score = float((
                compute_state[0] / gpu_total
                + compute_state[1] / float(inventory[compute_node]["估计内存总量(GB)"])
                + compute_state[2] / float(inventory[compute_node]["估计显存总量(GB)"])
            ) / 3.0)
            link_state = candidate_link[agent_index]
            link_score = float((1.0 - min(1.0, link_state[0] / 1_000.0)) * min(1.0, link_state[1] / 10_000.0) * (1.0 - link_state[2]) * (1.0 - min(1.0, link_state[3] / 200.0)))
            success = float(metrics["任务成功率"])
            execution_norm = min(1.0, float(metrics["平均任务完成时间(秒)"]) / 120.0)
            tool_failure = float(metrics["工具调用失败率"])
            hallucination = _clip((1.0 - success) * 0.35, 0.005, 0.10)
            executability = _clip(environment_score if exact else 0.0)
            targets.append(torch.tensor([success, execution_norm, tool_failure, hallucination, executability], dtype=torch.float32))
            latency_priority = float(task_requirement.values[0])
            accuracy_priority = float(task_requirement.values[1])
            teacher_scores.append(
                (1.0 if exact else -1.0)
                + 0.18 * capability_score + 0.10 * context_score + 0.14 * environment_score
                + latency_priority * (
                    0.12 * compute_score + 0.08 * link_score + 0.18 * (1.0 - execution_norm)
                )
                + accuracy_priority * (0.20 * success + 0.08 * (1.0 - hallucination))
                + 0.06 * (1.0 - tool_failure) + 0.06 * executability
            )
        scores = torch.tensor(teacher_scores, dtype=torch.float32)
        label = int(scores.argmax())
        if required_skill not in {skill["id"] for skill in agents[candidates[label]]["skills"]}:
            raise AssertionError(f"任务 {task['任务id']} 的最优标签不具备必需 skill")
        task_counts[str(task["场景"])] = task_counts.get(str(task["场景"]), 0) + 1
        cases.append(TrainingCase(
            events=events,
            graph=graph,
            task=task_requirement,
            label=torch.tensor(label, dtype=torch.long),
            candidate_targets=torch.stack(targets),
            teacher_scores=scores,
            metadata={
                "dataset_origin": "foundation_tables",
                "task_id": active_task["任务id"],
                "task_type": task["场景"],
                "difficulty_level": task["难度级别"],
                "task_prompt": active_task["任务具体提示词"],
                "required_capability": required_skill,
                "output_requirement": active_task[TASK_OUTPUT_FIELD],
                "field_completeness_min": float(active_task[TASK_COMPLETENESS_FIELD]),
                "end_to_end_latency_s_max": float(active_task[TASK_LATENCY_FIELD]),
                "complex_task_id": task_requirement.complex_task_id,
                "dag_task_ids": [item["任务id"] for item in selected_subtasks],
                "dag_edge_index": task_requirement.dag_edge_index.tolist(),
                "dag_current_index": task_requirement.dag_current_index,
                "source_timestamp_start": int(window_start),
                "source_timestamp_end": int(window_end),
                "candidate_roles": [roles[index] for index in candidates],
                "agent_roles": roles,
                "agent_compute": agent_compute,
                "compute_kinds": compute_kinds,
                "compute_states": [state.tolist() for state in query_compute_states],
            },
        ))

    output_dir.mkdir(parents=True, exist_ok=True)
    split_counts = _split_counts(len(cases))
    cursor = 0
    split_ranges: dict[str, dict[str, int]] = {}
    for split, count in split_counts.items():
        subset = cases[cursor : cursor + count]
        write_cases(output_dir / f"{split}.jsonl", subset)
        source_ranges = case_ranges[cursor : cursor + count]
        split_ranges[split] = {
            "source_timestamp_start": int(source_ranges[0][0]),
            "source_timestamp_end": int(source_ranges[-1][1]),
        }
        cursor += count
    profile_dir = output_dir / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    for agent_index, agent in enumerate(agents):
        profile = {
            "agent_id": agent_index,
            "agent_key": agent["agent_key"],
            "scenario": agent["场景"],
            "source_agent_type_id": agent["源智能体类型id"],
            "source_agent_type_name": agent["源智能体类型名称"],
            "skill_ids": [skill["id"] for skill in agent["skills"]],
            "reliability": agent_reliability[agent_index].tolist(),
        }
        (profile_dir / f"agent_{agent_index:03d}.json").write_text(
            json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    metadata = {
        "dataset": "6G-Agent-Compute-Foundation-Sliding-Window",
        "version": "6.0-task-agent-gru",
        "dataset_origin": "1.data numbered foundation tables",
        "num_agents": num_agents,
        "num_compute_nodes": num_compute_nodes,
        "compute_node_distribution": {kind: compute_kinds.count(kind) for kind in sorted(set(compute_kinds))},
        "samples": len(cases),
        "splits": split_counts,
        "split_source_timestamp_ranges": split_ranges,
        "window_size_timestamps": window_size_timestamps,
        "candidate_count": candidate_count,
        "events_per_window": event_count,
        "neighbor_fanout": neighbor_fanout,
        "window_overlap": False,
        "source_timestamp_origin": int(origin_timestamp),
        "source_timestamp_min": int(timestamps[0]),
        "source_timestamp_max": int(timestamps[-1]),
        "task_distribution": task_counts,
        "profile_format": "profiles/agent_NNN.json",
        "derived_field_notes": {
            "compute_features": "只使用 GPU 剩余算力、内存剩余量和显存剩余量",
            "gpu_total_tflops": "剩余 GPU 算力基于云侧/边侧 4000/400 TFLOPS 容量假设计算",
            "hallucination_rate": "基础可靠性表没有该字段，按 (1-task_success_rate)*0.35 截断到 0.5%-10%",
        },
        "design_constraints": {
            "chronological_split": True,
            "scenario_difficulty_stratified_split": stratified_split,
            "non_overlapping_windows": True,
            "required_skill_present_in_candidates": True,
            "label_agent_has_required_skill": True,
            "queue_batch_in_graph": False,
            "queue_batch_used_for_decision": False,
            "link_path_type_encoded": False,
            "explicit_task_node": True,
            "task_representation": "每个子任务是瞬时任务节点，并连接到所有候选智能体",
            "node_history_encoder": "GRU",
            "edge_history_encoder": "GRU",
            "one_hop_agent_aggregation": True,
            "confidence_used_by_model": False,
            "explicit_trend_feature_used_by_model": False,
            "required_skill_hard_mask": True,
            "skills_shared_within_agent_type": share_role_skills,
            "task_and_agent_text_in_jsonl": False,
            "task_encoding_fields": ["required_skill_id", "latency", "accuracy"],
            "capability_representation": "精确 Skill ID 集合；不使用能力槽",
        },
        "schema": schema_dict(),
    }
    dag_catalog = {
        case.metadata["task_id"]: {
            "complex_task_id": case.metadata["complex_task_id"],
            "task_ids": case.metadata["dag_task_ids"],
            "edge_index": case.metadata["dag_edge_index"],
            "current_index": case.metadata["dag_current_index"],
        }
        for case in cases
    }
    (output_dir / "task_dag_catalog.json").write_text(
        json.dumps(dag_catalog, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata
