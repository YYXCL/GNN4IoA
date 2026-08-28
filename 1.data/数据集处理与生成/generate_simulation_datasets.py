"""从能力卡与 Alibaba GenAI trace 生成仿真所需的基础数据集。

生成过程完全确定：同一输入文件、节点数和随机种子会得到相同结果。原始 Alibaba
文件不会被改写；由于原始数据没有物理服务器标识，本脚本使用容器 IP 的稳定哈希
构造虚拟算力节点，并在元数据中记录所有估计规则。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "2.模型"))
sys.path.insert(0, str(PROJECT_ROOT / "1.data"))

from ma_tgn_data.foundation import generate_foundation_dataset  # noqa: E402


SEED = 20260814
VARIANTS_PER_TYPE = 5
DEFAULT_COMPUTE_NODES = 12
EMERGENCY_WORDS = (
    "emergency", "rescue", "fire", "smoke", "fall", "intrusion", "alarm",
    "danger", "accident", "collision", "leak", "safety", "urgent", "evacuation",
    "应急", "救援", "火灾", "烟雾", "跌倒", "入侵", "告警", "危险", "事故", "泄漏",
)
FOCUS_RULES = (
    ("感知监测", ("detect", "monitor", "recognize", "track", "识别", "检测", "监测", "跟踪")),
    ("诊断分析", ("analy", "diagnos", "assess", "predict", "分析", "诊断", "评估", "预测")),
    ("控制协调", ("control", "manage", "schedule", "coordinate", "optimiz", "控制", "管理", "调度", "协调", "优化")),
    ("告警响应", EMERGENCY_WORDS + ("warning", "response", "risk", "风险", "响应", "预警")),
    ("汇总问答", ("report", "summary", "question", "answer", "log", "报告", "摘要", "问答", "日志")),
)


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def scenario_for(source_index: int, variant_index: int) -> str:
    if 6 <= source_index <= 12:
        return "智慧家居"
    if 13 <= source_index <= 21:
        return "智慧工厂"
    if 22 <= source_index <= 24:
        return "应急救援"
    return ("智慧家居", "智慧家居", "智慧工厂", "智慧工厂", "应急救援")[variant_index]


def skill_text(skill: dict) -> str:
    return " ".join((skill.get("id", ""), skill.get("name", ""), skill.get("description", ""))).lower()


def choose_skills(source: dict, source_index: int, variant_index: int) -> tuple[str, list[dict]]:
    focus, keywords = FOCUS_RULES[variant_index]
    scenario = scenario_for(source_index, variant_index)
    skills = source["skills"]
    target_count = 5 + (variant_index % 3)
    # 前五类是跨场景通用感知/设备智能体，先按原始标签限定场景，避免把工厂
    # skill 选入智慧家居派生智能体。应急救援没有原始标签，只按应急语义筛选。
    if source_index <= 5 and scenario in {"智慧家居", "智慧工厂"}:
        required_tag = "smart-home" if scenario == "智慧家居" else "smart-factory"
        tagged = [skill for skill in skills if required_tag in set(skill.get("tags", []))]
        if len(tagged) >= target_count:
            skills = tagged
    if scenario == "应急救援":
        emergency = [skill for skill in skills if any(word in skill_text(skill) for word in EMERGENCY_WORDS)]
        if len(emergency) >= target_count:
            skills = emergency
    ranked: list[tuple[int, int, dict]] = []
    for position, skill in enumerate(skills):
        text = skill_text(skill)
        tags = set(skill.get("tags", []))
        score = 4 * sum(word in text for word in keywords)
        if scenario == "智慧家居" and "smart-home" in tags:
            score += 5
        elif scenario == "智慧工厂" and "smart-factory" in tags:
            score += 5
        elif scenario == "应急救援":
            score += 7 * sum(word in text for word in EMERGENCY_WORDS)
            score += 2 * bool(tags & {"smart-building", "smart-campus"})
        score += stable_int(f"{source['id']}:{variant_index}:{skill['id']}") % 3
        ranked.append((score, -position, skill))
    selected = [item[2] for item in sorted(ranked, reverse=True)[:target_count]]
    return focus, selected


def infer_apis(selected: list[dict]) -> list[str]:
    text = " ".join(skill_text(item) for item in selected)
    rules = (
        (("video", "camera", "视频", "摄像"), "视频流访问 API"),
        (("image", "图像"), "图像对象存储 API"),
        (("audio", "sound", "语音", "音频", "声音"), "音频流访问 API"),
        (("sensor", "telemetry", "传感器", "遥测"), "传感器遥测 API"),
        (("iot", "device", "设备"), "IoT 网关 API"),
        (("control", "actuator", "控制", "执行"), "设备控制 API"),
        (("access", "security", "intrusion", "门禁", "安防", "入侵"), "门禁与安防告警 API"),
        (("health", "elderly", "patient", "健康", "老人", "患者"), "健康监测 API"),
        (("energy", "power", "能耗", "能源", "电力"), "能源管理 API"),
        (("production", "quality", "mes", "生产", "质量"), "MES 生产执行 API"),
        (("maintenance", "fault", "维修", "故障"), "CMMS 维修工单 API"),
        (("robot", "机器人"), "机器人编队控制 API"),
        (("logistics", "warehouse", "物流", "仓储"), "WMS 仓储管理 API"),
        (("traffic", "vehicle", "parking", "交通", "车辆", "停车"), "园区交通管理 API"),
        (("report", "summary", "报告", "摘要"), "报告导出 API"),
    )
    result = [api for words, api in rules if any(word in text for word in words)]
    result.append("智能体消息总线 API")
    return list(dict.fromkeys(result))[:4]


def infer_knowledge_base(scenario: str, focus: str, agent_id: int) -> str:
    # 约 15% 的智能体有知识库，符合“只有少部分接入”的要求。
    if agent_id % 7:
        return "无"
    if scenario == "智慧工厂":
        return "设备维护手册与工厂安全规程库" if focus != "汇总问答" else "生产运营指标口径库"
    if scenario == "智慧家居":
        return "家庭设备说明书与照护安全知识库"
    return "应急预案、疏散路线与处置规程库"


def infer_environment(selected: list[dict], apis: list[str]) -> tuple[str, str]:
    text = " ".join(skill_text(item) for item in selected)
    sensors = []
    for words, name in (
        (("video", "camera", "图像", "视频", "摄像"), "摄像头"),
        (("audio", "sound", "音频", "声音", "语音"), "麦克风阵列"),
        (("temperature", "humidity", "温度", "湿度"), "温湿度传感器"),
        (("smoke", "fire", "gas", "烟雾", "火灾", "气体"), "烟感/气体传感器"),
        (("vibration", "振动"), "振动传感器"),
        (("energy", "power", "能耗", "电力"), "智能电表"),
        (("location", "position", "track", "位置", "定位", "轨迹"), "定位传感器"),
        (("health", "vital", "健康", "生命体征"), "可穿戴生命体征传感器"),
    ):
        if any(word in text for word in words):
            sensors.append(name)
    has_control = any("控制" in api or "网关" in api for api in apis)
    if sensors and has_control:
        environment = "混合环境"
    elif sensors:
        environment = "物理环境"
    elif has_control:
        environment = "混合环境"
    else:
        environment = "数字环境"
    return environment, "|".join(dict.fromkeys(sensors)) if sensors else "无"


def build_agents(source_path: Path, seed: int) -> tuple[pd.DataFrame, list[dict], dict]:
    sources = json.loads(source_path.read_text(encoding="utf-8"))
    rng = random.Random(seed)
    rows: list[dict] = []
    structured: list[dict] = []
    for source_index, source in enumerate(sources, start=1):
        for variant_index in range(VARIANTS_PER_TYPE):
            agent_id = len(rows) + 1
            scenario = scenario_for(source_index, variant_index)
            focus, selected = choose_skills(source, source_index, variant_index)
            apis = infer_apis(selected)
            environment, sensors = infer_environment(selected, apis)
            knowledge = infer_knowledge_base(scenario, focus, agent_id)
            window = rng.choice((32768, 65536, 131072, 262144))
            usage = round(rng.uniform(0.08, 0.78), 4)
            rate_limit = rng.choice((30, 60, 120, 240))
            if any("控制" in api for api in apis):
                rate_limit = min(rate_limit, 60)
            skill_ids = [item["id"] for item in selected]
            names = "、".join(item["name"] for item in selected[:3])
            description = (
                f"【{scenario}】【{source['name']}】{focus}型智能体，重点承担{names}等任务；"
                f"仅配置源能力卡中的 {len(selected)} 项技能，不具备该类型的全部能力。"
            )
            row = {
                "智能体编号": agent_id,
                "智能体能力描述": description,
                "包含的skill": "|".join(skill_ids),
                "工具API权限": "|".join(apis),
                "接入知识库": knowledge,
                "运行环境": environment,
                "上下文窗口总量(tokens)": window,
                "上下文窗口当前占用率": usage,
                "工具API速率限制(次/分钟)": rate_limit,
                "工具API是否被占用": "是" if rng.random() < 0.28 else "否",
                "物理传感器可用情况": sensors if environment != "数字环境" else "无",
            }
            rows.append(row)
            structured.append({
                **row,
                "agent_key": f"A{agent_id:03d}",
                "场景": scenario,
                "派生重点": focus,
                "源智能体类型id": source["id"],
                "源智能体类型名称": source["name"],
                "skills": selected,
                "tool_apis": apis,
            })
    metadata = {
        "source": str(source_path),
        "source_type_count": len(sources),
        "variants_per_source_type": VARIANTS_PER_TYPE,
        "agent_count": len(rows),
        "seed": seed,
        "scenario_mapping_note": (
            "源文件没有 emergency/rescue 标签；应急救援智能体仅从源文件中的火灾、跌倒、入侵、"
            "告警、泄漏和交通事故等技能派生，未使用 part2 或外部技能。"
        ),
        "column_6_interpretation": "用户列说明跳过第6列；本数据集将其定义为运行环境。",
        "knowledge_base_access_rate": sum(item["接入知识库"] != "无" for item in rows) / len(rows),
    }
    return pd.DataFrame(rows), structured, metadata


def build_reliability(agents: list[dict], seed: int) -> pd.DataFrame:
    rng = random.Random(seed + 1)
    rows = []
    for agent in agents:
        calls = rng.randint(80, 6500)
        physical = agent["运行环境"] != "数字环境"
        tool_count = len(agent["tool_apis"])
        failure = min(0.12, max(0.002, rng.betavariate(1.6, 42) + 0.003 * max(0, tool_count - 2)))
        success = min(0.995, max(0.72, rng.uniform(0.84, 0.985) - 0.22 * failure - (0.012 if physical else 0)))
        mean_time = rng.uniform(0.7, 5.0) + 0.65 * len(agent["skills"]) + (1.8 if physical else 0)
        rows.append({
            "智能体编号": agent["智能体编号"],
            "被调用次数": calls,
            "任务成功率": round(success, 4),
            "平均任务完成时间(秒)": round(mean_time, 3),
            "工具调用失败率": round(failure, 4),
        })
    return pd.DataFrame(rows)


def container_node(container_ip: object, num_nodes: int) -> str | None:
    if pd.isna(container_ip) or not str(container_ip).strip():
        return None
    return f"C{stable_int(str(container_ip)) % num_nodes + 1:03d}"


def aggregate_metric(path: Path, value_name: str, num_nodes: int, aggregation: str) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["timestamp_anon", "value", "container_ip"])
    frame["compute_node_id"] = frame["container_ip"].map(lambda value: container_node(value, num_nodes))
    frame = frame.dropna(subset=["compute_node_id"])
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    grouped = frame.groupby(["timestamp_anon", "compute_node_id"], as_index=False)["value"]
    result = getattr(grouped, aggregation)().rename(columns={"value": value_name})
    return result


def build_compute_load(source_dir: Path, num_nodes: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    duty_raw = pd.read_csv(source_dir / "pod_gpu_duty_cycle_anon.csv", usecols=["timestamp_anon", "value", "container_ip"])
    duty_raw["compute_node_id"] = duty_raw["container_ip"].map(lambda value: container_node(value, num_nodes))
    duty_raw = duty_raw.dropna(subset=["compute_node_id"])
    duty_raw["value"] = pd.to_numeric(duty_raw["value"], errors="coerce")
    duty = duty_raw.groupby(["timestamp_anon", "compute_node_id"], as_index=False).agg(
        gpu_duty_cycle_percent=("value", "mean"), active_gpu_container_count=("container_ip", "nunique")
    )
    gpu_mem = aggregate_metric(source_dir / "pod_gpu_memory_used_bytes_anon.csv", "gpu_memory_used_bytes", num_nodes, "sum")
    memory = aggregate_metric(source_dir / "pod_memory_util_anon.csv", "memory_utilization", num_nodes, "mean")
    queue_rt = aggregate_metric(source_dir / "queue_rt_raw_anon.csv", "queue_rt_ms", num_nodes, "mean")
    model_predict = aggregate_metric(source_dir / "model_predict_data_anon.csv", "model_predict_value", num_nodes, "sum")
    pipeline = aggregate_metric(source_dir / "pipeline_inference_data_anon.csv", "pipeline_inference_value", num_nodes, "sum")
    load = duty
    for metric in (gpu_mem, memory, queue_rt, model_predict, pipeline):
        load = load.merge(metric, on=["timestamp_anon", "compute_node_id"], how="left", validate="one_to_one")
    load = load.sort_values(["compute_node_id", "timestamp_anon"]).reset_index(drop=True)
    raw_missing = load[["gpu_memory_used_bytes", "memory_utilization"]].isna().any(axis=1)
    for column in ("gpu_memory_used_bytes", "memory_utilization"):
        load[column] = load.groupby("compute_node_id")[column].transform(
            lambda values: values.interpolate(limit_direction="both")
        )
    load["memory_utilization"] = load["memory_utilization"].clip(0, 1)
    for column in ("queue_rt_ms", "model_predict_value", "pipeline_inference_value"):
        load[column] = load[column].fillna(0)
    node_order = [f"C{index:03d}" for index in range(1, num_nodes + 1)]
    memory_capacity = {node: (512, 768, 1024)[(index - 1) % 3] for index, node in enumerate(node_order, start=1)}
    max_gpu = load.groupby("compute_node_id")["gpu_memory_used_bytes"].max() / (1024 ** 3)
    gpu_capacity = {node: max(64, math.ceil(float(max_gpu.get(node, 0)) / 64) * 64) for node in node_order}
    unique_containers = duty_raw.groupby("compute_node_id")["container_ip"].nunique()
    inventory = pd.DataFrame([{
        "算力节点编号": node,
        "节点类型": "云侧" if index <= max(1, num_nodes // 3) else "边侧",
        "估计内存总量(GB)": memory_capacity[node],
        "估计显存总量(GB)": gpu_capacity[node],
        "映射容器数量": int(unique_containers.get(node, 0)),
        "节点构造方式": "container_ip稳定哈希映射",
    } for index, node in enumerate(node_order, start=1)])
    load["timestamp_anon"] = load["timestamp_anon"].astype("int64")
    load["gpu_memory_used_gb"] = load["gpu_memory_used_bytes"] / (1024 ** 3)
    load["memory_total_gb_estimated"] = load["compute_node_id"].map(memory_capacity)
    load["memory_used_gb_estimated"] = load["memory_utilization"] * load["memory_total_gb_estimated"]
    load["gpu_memory_total_gb_estimated"] = load["compute_node_id"].map(gpu_capacity)
    load["gpu_memory_utilization_estimated"] = (
        load["gpu_memory_used_gb"] / load["gpu_memory_total_gb_estimated"]
    ).clip(0, 1)
    load["server_load_rate_proxy"] = load[["memory_utilization", "gpu_memory_utilization_estimated"]].max(axis=1)
    load["data_quality_flag"] = raw_missing.map({True: "含插值", False: "原始聚合"})
    output_columns = [
        "timestamp_anon", "compute_node_id", "active_gpu_container_count", "server_load_rate_proxy",
        "memory_utilization", "memory_total_gb_estimated", "memory_used_gb_estimated",
        "gpu_duty_cycle_percent", "gpu_memory_total_gb_estimated", "gpu_memory_used_gb",
        "gpu_memory_utilization_estimated", "queue_rt_ms", "model_predict_value",
        "pipeline_inference_value", "data_quality_flag",
    ]
    load = load[output_columns]
    metadata = {
        "virtual_compute_nodes": num_nodes,
        "node_mapping": "sha256(container_ip) mod N; this is not a recovered physical server mapping",
        "server_load_rate_proxy": "max(memory_utilization, estimated_gpu_memory_utilization)",
        "cpu_metric": "source data does not contain CPU utilization; no CPU value was fabricated",
        "memory_capacity": "synthetic fixed capacities 512/768/1024 GB, used only to estimate used memory",
        "gpu_memory_capacity": "per-node maximum observed aggregate rounded up to a multiple of 64 GiB",
        "interpolation": "only missing memory utilization and GPU memory usage are interpolated within each node",
    }
    return load, inventory, metadata


def build_links(agents: list[dict], inventory: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = random.Random(seed + 2)
    node_ids = inventory["算力节点编号"].tolist()
    rows: list[dict] = []

    def link(source: str, target: str, dependency: bool, cross_type: bool) -> dict:
        latency = rng.uniform(7, 85) if cross_type else rng.uniform(1, 35)
        loss = min(0.12, rng.betavariate(1.4, 45))
        return {
            "数据源": source,
            "数据目的地": target,
            "时延(ms)": round(latency, 3),
            "带宽(Mbps)": round(rng.uniform(150, 20000) / (1.6 if cross_type else 1), 3),
            "网络丢包率": round(loss, 5),
            "网络抖动(ms)": round(max(0.05, latency * rng.uniform(0.03, 0.25)), 3),
            "是否有任务依赖关系": "是" if dependency else "否",
        }

    by_source: dict[str, list[dict]] = {}
    by_scenario: dict[str, list[dict]] = {}
    for agent in agents:
        by_source.setdefault(agent["源智能体类型id"], []).append(agent)
        by_scenario.setdefault(agent["场景"], []).append(agent)
        agent_key = agent["agent_key"]
        primary = node_ids[(agent["智能体编号"] - 1) % len(node_ids)]
        secondary = node_ids[stable_int(agent_key + agent["包含的skill"]) % len(node_ids)]
        if secondary == primary:
            secondary = node_ids[(node_ids.index(primary) + 1) % len(node_ids)]
        rows.append(link(agent_key, primary, True, True))
        rows.append(link(agent_key, secondary, False, True))
    for members in by_source.values():
        for index, agent in enumerate(members):
            peer = members[(index + 1) % len(members)]
            rows.append(link(agent["agent_key"], peer["agent_key"], True, False))
    for scenario_members in by_scenario.values():
        for agent in scenario_members:
            candidates = [item for item in scenario_members if item["源智能体类型id"] != agent["源智能体类型id"]]
            if not candidates:
                continue
            peer = min(candidates, key=lambda item: stable_int(agent["agent_key"] + item["agent_key"]))
            rows.append(link(agent["agent_key"], peer["agent_key"], False, False))
    return pd.DataFrame(rows)


def build_tasks(source_path: Path, agents: list[dict]) -> tuple[pd.DataFrame, list[dict]]:
    sources = json.loads(source_path.read_text(encoding="utf-8"))
    selected_by_source: dict[str, set[str]] = {}
    for agent in agents:
        selected_by_source.setdefault(agent["源智能体类型id"], set()).update(skill["id"] for skill in agent["skills"])
    rows = []
    structured = []
    for source_index, source in enumerate(sources, start=1):
        available = [skill for skill in source["skills"] if skill["id"] in selected_by_source[source["id"]]]
        picks = [available[0], available[len(available) // 2], available[-1]]
        for skill in picks:
            task_id = len(rows) + 1
            text = skill_text(skill)
            critical = any(word in text for word in EMERGENCY_WORDS)
            accuracy = 0.94 if critical else (0.90 if "control" in text or "控制" in text else 0.86)
            tags = set(skill.get("tags", []))
            if "smart-home" in tags:
                scenario = "智慧家居"
            elif "smart-factory" in tags:
                scenario = "智慧工厂"
            else:
                # 源数据中的智慧楼宇/园区任务映射到三场景实验中的应急救援侧。
                scenario = "应急救援"
            prompt = f"在{scenario}场景中，{skill['description']}请给出结构化结果、关键证据和置信度。"
            row = {
                "任务id": f"T{task_id:04d}",
                "任务具体提示词": prompt,
                "必需的Skill id": skill["id"],
                "输出结果的要求": "JSON",
                "字段完整率要求": 0.95,
                "端到端完成时延要求(s)": 5 if critical else 15,
                "准确率要求": accuracy,
            }
            rows.append(row)
            structured.append({**row, "场景": scenario, "source_agent_type_id": source["id"]})
    return pd.DataFrame(rows), structured


def build_event_tables(agents: list[dict], links: pd.DataFrame, tasks: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    event_types = pd.DataFrame([
        (0, "context_update", "智能体上下文窗口占用率或会话状态变化", "agent"),
        (1, "environment_update", "工具、知识库或物理传感器可用状态变化", "agent"),
        (2, "compute_update", "算力节点负载、内存或显存状态变化", "compute"),
        (3, "link_update", "链路时延、带宽、丢包或抖动变化", "link"),
        (4, "task_result", "任务完成并产生成功率、耗时和工具失败反馈", "task"),
        (5, "capability_update", "智能体新增、移除或更新 skill", "agent"),
    ], columns=["事件类型编码", "事件类型名称", "含义", "主要作用对象"])
    relations = pd.DataFrame([
        (0, "compute_access", "智能体访问算力节点"),
        (1, "communication", "智能体之间通信"),
        (2, "task_dependency", "具有任务依赖的历史协作"),
    ], columns=["关系类型编码", "关系类型名称", "含义"])
    events = []
    timestamp = 0.0
    for index, agent in enumerate(agents):
        timestamp += 0.1
        events.append((f"E{len(events)+1:06d}", round(timestamp, 1), 0, agent["agent_key"], agent["agent_key"], -1, "agent_capabilities.csv"))
        link_row = links.iloc[(index * 2) % len(links)]
        timestamp += 0.1
        relation = 0 if str(link_row["数据目的地"]).startswith("C") else (2 if link_row["是否有任务依赖关系"] == "是" else 1)
        events.append((f"E{len(events)+1:06d}", round(timestamp, 1), 3, link_row["数据源"], link_row["数据目的地"], relation, "communication_links.csv"))
        task = tasks[index % len(tasks)]
        timestamp += 0.1
        events.append((f"E{len(events)+1:06d}", round(timestamp, 1), 4, task["任务id"], agent["agent_key"], 2, "tasks.csv"))
    event_stream = pd.DataFrame(events, columns=[
        "事件id", "相对时间戳(s)", "事件类型编码", "数据源", "数据目的地", "关系类型编码", "载荷引用文件"
    ])
    return event_types, relations, event_stream


def main() -> None:
    data_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="生成模型训练与仿真基础数据集")
    parser.add_argument("--data-root", type=Path, default=data_root)
    parser.add_argument("--compute-nodes", type=int, default=DEFAULT_COMPUTE_NODES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--training-output", type=Path, default=None)
    parser.add_argument("--training-samples", type=int, default=None, help="默认每个任务生成一个时间窗口")
    parser.add_argument("--window-size", type=int, default=16, help="每个训练样本包含的源时间戳数量")
    parser.add_argument("--build-training-jsonl", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.compute_nodes < 2:
        raise ValueError("算力节点数至少为 2")
    source_path = args.data_root / "实验室的智能体能力列表" / "part1_24_revised_descriptions_skills.json"
    alibaba_source = args.data_root / "raw" / "Alibaba-cluster-trace-v2026-GenAI" / "源数据"
    agents_frame, agents, agent_meta = build_agents(source_path, args.seed)
    reliability = build_reliability(agents, args.seed)
    compute_load, inventory, load_meta = build_compute_load(alibaba_source, args.compute_nodes)
    links = build_links(agents, inventory, args.seed)
    tasks_frame, tasks = build_tasks(source_path, agents)
    event_types, relations, event_stream = build_event_tables(agents, links, tasks)

    paths = {
        "agents_csv": args.data_root / "3.智能体能力数据(包括智能体环境和知识库、API权限)" / "agent_capabilities.csv",
        "agents_json": args.data_root / "3.智能体能力数据(包括智能体环境和知识库、API权限)" / "agent_capabilities.json",
        "reliability_csv": args.data_root / "4.智能体可靠性" / "agent_reliability.csv",
        "compute_load_csv": args.data_root / "1.负载数据" / "compute_node_load_timeseries.csv",
        "compute_inventory_csv": args.data_root / "1.负载数据" / "compute_node_inventory.csv",
        "links_csv": args.data_root / "2.链路数据" / "communication_links.csv",
        "tasks_csv": args.data_root / "5.任务数据" / "tasks.csv",
        "tasks_json": args.data_root / "5.任务数据" / "tasks.json",
        "event_types_csv": args.data_root / "6.事件数据" / "event_type_codebook.csv",
        "relations_csv": args.data_root / "6.事件数据" / "relation_type_codebook.csv",
        "events_csv": args.data_root / "6.事件数据" / "event_stream.csv",
    }
    for frame, key in (
        (agents_frame, "agents_csv"), (reliability, "reliability_csv"),
        (compute_load, "compute_load_csv"), (inventory, "compute_inventory_csv"),
        (links, "links_csv"), (tasks_frame, "tasks_csv"),
        (event_types, "event_types_csv"), (relations, "relations_csv"), (event_stream, "events_csv"),
    ):
        write_csv(frame, paths[key])
    paths["agents_json"].write_text(json.dumps(agents, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["tasks_json"].write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    training_metadata = None
    if args.build_training_jsonl:
        training_output = args.training_output or args.data_root / "datasets" / "sixg_ma_state"
        training_metadata = generate_foundation_dataset(
            args.data_root,
            training_output,
            seed=args.seed,
            window_size_timestamps=args.window_size,
            samples=args.training_samples,
        )
    manifest = {
        "dataset_version": "1.1.0",
        "generation_seed": args.seed,
        "counts": {
            "agents": len(agents_frame), "source_agent_types": agent_meta["source_type_count"],
            "compute_nodes": len(inventory), "compute_load_rows": len(compute_load),
            "links": len(links), "tasks": len(tasks_frame), "events": len(event_stream),
        },
        "agent_generation": agent_meta,
        "compute_load_processing": load_meta,
        "training_jsonl": None if training_metadata is None else {
            "output_dir": str((args.training_output or args.data_root / "datasets" / "sixg_ma_state").relative_to(args.data_root)),
            "samples": training_metadata["samples"],
            "splits": training_metadata["splits"],
            "window_size_timestamps": training_metadata["window_size_timestamps"],
            "window_overlap": training_metadata["window_overlap"],
        },
        "link_column_6_interpretation": "用户重复写了网络丢包；按模型现有 EDGE_METRICS 将第6列定义为网络抖动(ms)。",
        "files": {key: str(path.relative_to(args.data_root)) for key, path in paths.items()},
    }
    manifest_path = args.data_root / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    assert agents_frame["智能体编号"].tolist() == list(range(1, len(agents_frame) + 1))
    assert reliability["智能体编号"].tolist() == agents_frame["智能体编号"].tolist()
    assert set(links["数据源"]).issubset({item["agent_key"] for item in agents})
    assert set(links.loc[links["数据目的地"].str.startswith("C"), "数据目的地"]).issubset(set(inventory["算力节点编号"]))
    assert agents_frame["接入知识库"].ne("无").mean() <= 0.20
    assert "实验要求" not in tasks_frame.columns
    assert "required_skill_id" not in json.dumps(tasks, ensure_ascii=False)
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
