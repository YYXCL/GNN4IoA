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

SEED = 20260814
VARIANTS_PER_TYPE = 5
DEFAULT_COMPUTE_NODES = 12
DEFAULT_LINK_TIMESTAMPS = 48
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
                f"可使用已登记的技能和工具完成对应场景工作。"
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
    node_kind = {row["算力节点编号"]: row["节点类型"] for row in inventory.to_dict("records")}
    load["gpu_total_tflops_estimated"] = load["compute_node_id"].map(
        lambda node: 4_000.0 if node_kind[node] == "云侧" else 400.0
    )
    load["gpu_remaining_tflops_estimated"] = (
        load["gpu_total_tflops_estimated"] * (1.0 - load["gpu_duty_cycle_percent"].clip(0, 100) / 100.0)
    )
    load["memory_remaining_gb_estimated"] = (
        load["memory_total_gb_estimated"] - load["memory_used_gb_estimated"]
    ).clip(lower=0)
    load["gpu_memory_remaining_gb_estimated"] = (
        load["gpu_memory_total_gb_estimated"] - load["gpu_memory_used_gb"]
    ).clip(lower=0)
    load["data_quality_flag"] = raw_missing.map({True: "含插值", False: "原始聚合"})
    output_columns = [
        "timestamp_anon", "compute_node_id", "active_gpu_container_count", "server_load_rate_proxy",
        "memory_utilization", "memory_total_gb_estimated", "memory_used_gb_estimated",
        "gpu_duty_cycle_percent", "gpu_memory_total_gb_estimated", "gpu_memory_used_gb",
        "gpu_memory_utilization_estimated", "queue_rt_ms", "model_predict_value",
        "pipeline_inference_value", "data_quality_flag",
        "gpu_remaining_tflops_estimated", "memory_remaining_gb_estimated",
        "gpu_memory_remaining_gb_estimated",
    ]
    load = load[output_columns].rename(columns={
        "timestamp_anon": "时间戳", "compute_node_id": "算力节点编号",
        "active_gpu_container_count": "活跃GPU容器数量", "server_load_rate_proxy": "服务器负载率代理",
        "memory_utilization": "内存使用率", "memory_total_gb_estimated": "估计内存总量(GB)",
        "memory_used_gb_estimated": "估计内存已用量(GB)", "gpu_duty_cycle_percent": "GPU利用率(%)",
        "gpu_memory_total_gb_estimated": "估计显存总量(GB)", "gpu_memory_used_gb": "显存已用量(GB)",
        "gpu_memory_utilization_estimated": "估计显存使用率", "queue_rt_ms": "队列响应时间(ms)",
        "model_predict_value": "模型预测请求量", "pipeline_inference_value": "流水线推理请求量",
        "data_quality_flag": "数据质量标记", "gpu_remaining_tflops_estimated": "GPU算力剩余量(TFLOPS)",
        "memory_remaining_gb_estimated": "内存剩余量(GB)",
        "gpu_memory_remaining_gb_estimated": "显存剩余量(GB)",
    })
    metadata = {
        "virtual_compute_nodes": num_nodes,
        "node_mapping": "sha256(container_ip) mod N; this is not a recovered physical server mapping",
        "server_load_rate_proxy": "max(memory_utilization, estimated_gpu_memory_utilization)",
        "cpu_metric": "source data does not contain CPU utilization; no CPU value was fabricated",
        "memory_capacity": "synthetic fixed capacities 512/768/1024 GB, used only to estimate used memory",
        "gpu_memory_capacity": "per-node maximum observed aggregate rounded up to a multiple of 64 GiB",
        "model_compute_features": (
            "仅编码 GPU算力剩余量(TFLOPS)、内存剩余量(GB)、显存剩余量(GB)；"
            "利用率、队列和 batch 仅保留作数据溯源，不进入模型"
        ),
        "interpolation": "only missing memory utilization and GPU memory usage are interpolated within each node",
    }
    return load, inventory, metadata


def build_links(
    agents: list[dict], inventory: pd.DataFrame, timestamps: list[int], seed: int,
    observation_count: int = DEFAULT_LINK_TIMESTAMPS,
) -> pd.DataFrame:
    """生成固定全连接拓扑上的动态链路状态。

    智能体—算力节点是一对一固定部署关系；智能体之间使用一条无向逻辑边表示
    双向可达。任务依赖只属于任务 DAG，不是通信链路属性。
    """
    if observation_count < 2:
        raise ValueError("链路观测时间戳数量至少为 2")
    node_ids = inventory["算力节点编号"].tolist()
    positions = [round(index * (len(timestamps) - 1) / (observation_count - 1)) for index in range(observation_count)]
    observed_timestamps = [int(timestamps[position]) for position in dict.fromkeys(positions)]
    topology: list[tuple[str, str, str]] = []
    for agent in agents:
        agent_key = agent["agent_key"]
        compute = node_ids[(agent["智能体编号"] - 1) % len(node_ids)]
        topology.append((agent_key, compute, "智能体-算力节点"))
    for left in range(len(agents)):
        for right in range(left + 1, len(agents)):
            topology.append((agents[left]["agent_key"], agents[right]["agent_key"], "智能体-智能体"))

    rows: list[dict] = []
    previous: dict[str, tuple[float, float, float, float]] = {}
    for tick, timestamp in enumerate(observed_timestamps):
        global_congestion = 0.5 + 0.28 * math.sin(2 * math.pi * tick / 24.0) + 0.12 * math.sin(2 * math.pi * tick / 7.0)
        for source, target, link_type in topology:
            link_id = f"L-{source}-{target}"
            local = random.Random(seed + stable_int(f"{link_id}:{tick}"))
            cross_type = link_type == "智能体-算力节点"
            if link_id not in previous:
                latency = local.uniform(7, 85) if cross_type else local.uniform(1, 35)
                bandwidth = float(local.choice((500, 1000, 2000, 3000)))
                loss = min(0.08, local.betavariate(1.4, 55))
                jitter = max(0.05, latency * local.uniform(0.03, 0.18))
            else:
                old_latency, old_bandwidth, old_loss, old_jitter = previous[link_id]
                pressure = max(0.0, min(1.0, global_congestion + local.uniform(-0.12, 0.12)))
                latency = 0.86 * old_latency + 0.14 * old_latency * (0.72 + 0.75 * pressure) + local.gauss(0, 0.35)
                bandwidth = old_bandwidth
                loss = 0.90 * old_loss + 0.10 * (0.001 + 0.035 * pressure) + local.gauss(0, 0.0008)
                jitter = 0.84 * old_jitter + 0.16 * latency * (0.025 + 0.13 * pressure)
            latency = max(0.2, latency)
            bandwidth = max(25.0, bandwidth)
            loss = max(0.0, min(0.12, loss))
            jitter = max(0.02, jitter)
            previous[link_id] = (latency, bandwidth, loss, jitter)
            rows.append({
                "时间戳": timestamp, "链路id": link_id, "链路类型": link_type,
                "数据源": source, "数据目的地": target, "是否双向可达": "是",
                "时延(ms)": round(latency, 3), "带宽(Mbps)": int(bandwidth),
                "网络丢包率": round(loss, 5), "网络抖动(ms)": round(jitter, 3),
            })
    return pd.DataFrame(rows)


SCENARIO_GOALS = {
    "智慧工厂": (
        "产线设备异常诊断与停机风险处置", "产品质量缺陷追踪与工艺校正", "厂内物流拥堵疏解与补料调度",
        "机器人协同作业安全检查", "能源异常定位与节能控制", "危险区域入侵联动处置",
        "预测性维护计划编排", "生产事件复盘与班组交接",
    ),
    "智慧家居": (
        "独居老人异常活动发现与照护响应", "家庭火灾隐患识别与设备联动", "门口可疑人员与遗留物处置",
        "室内舒适度分析与环境调节", "家庭能耗诊断与用电优化", "婴幼儿活动安全看护",
        "多房间设备故障排查", "家庭日程与设备自动化协同",
    ),
    "应急救援": (
        "火灾现场态势研判与疏散路线制定", "地震受困人员搜索与救援资源调度", "危化品泄漏范围评估与隔离",
        "交通事故伤员分级与通道保障", "洪涝区域风险监测与转移组织", "园区大面积停电应急恢复",
        "高层建筑人员滞留排查", "多源告警核验与指挥简报生成",
    ),
}
SCENARIO_GOAL_KEYWORDS = {
    "智慧工厂": (
        ("fault", "maintenance", "equipment", "diagnos", "predict", "故障", "维护", "设备"),
        ("quality", "defect", "inspect", "image", "label", "质量", "缺陷", "工艺"),
        ("logistics", "agv", "warehouse", "inventory", "route", "物流", "仓储", "补料"),
        ("robot", "collision", "safety", "control", "机器人", "碰撞", "安全"),
        ("energy", "power", "battery", "consumption", "能源", "电力", "能耗"),
        ("intrusion", "danger", "fire", "alarm", "safety", "入侵", "危险", "火灾"),
        ("maintenance", "fault", "predict", "schedule", "维修", "预测", "计划"),
        ("report", "timeline", "log", "summary", "event", "报告", "时间线", "日志"),
    ),
    "智慧家居": (
        ("elderly", "fall", "activity", "health", "老人", "跌倒", "照护"),
        ("fire", "smoke", "gas", "alarm", "火灾", "烟雾", "燃气"),
        ("intrusion", "person", "object", "door", "security", "可疑", "门口", "遗留"),
        ("temperature", "humidity", "comfort", "air", "hvac", "温度", "湿度", "舒适"),
        ("energy", "power", "electric", "consumption", "能源", "用电", "能耗"),
        ("baby", "child", "person", "activity", "cry", "婴幼儿", "儿童"),
        ("device", "fault", "status", "diagnos", "设备", "故障", "状态"),
        ("automation", "schedule", "control", "scene", "自动化", "日程", "控制"),
    ),
    "应急救援": (
        ("fire", "smoke", "evacuat", "person", "火灾", "烟雾", "疏散"),
        ("fall", "person", "location", "rescue", "跌倒", "人员", "定位", "救援"),
        ("leak", "gas", "danger", "risk", "泄漏", "气体", "危险"),
        ("traffic", "accident", "collision", "health", "交通", "事故", "伤员"),
        ("water", "flood", "risk", "weather", "洪涝", "积水", "风险"),
        ("power", "energy", "device", "alarm", "停电", "能源", "设备"),
        ("person", "building", "location", "detect", "人员", "建筑", "位置"),
        ("alarm", "report", "summary", "verify", "告警", "报告", "核验"),
    ),
}
DIFFICULTY_COUNTS = {"简单": (1, 2, 3), "中等": (4, 5, 6), "困难": (7, 8, 9)}
OUTPUT_STYLES = (
    "形成按优先级排序的处置清单，并标明触发条件", "给出处置结论、责任对象和可执行动作",
    "生成事件时间线与风险分级表", "输出资源调度方案及无法满足的约束",
    "汇总关键状态、异常位置和建议操作", "形成便于值守人员直接执行的操作卡",
    "给出诊断结论、候选原因与验证顺序", "生成阶段性结果和最终决策摘要",
)
SUBTASK_ENDINGS = (
    "记录发现的异常与位置。", "将结果交给后续步骤并标明时间范围。", "列出无法确认的信息。",
    "优先返回会改变处置决策的信号。", "保留原始设备或传感器标识。", "明确成功条件与失败原因。",
)


def _dag_for_subtasks(subtask_ids: list[str]) -> tuple[list[int], list[tuple[str, str]]]:
    """生成含串并行结构的 DAG；相同 step 的子任务可并行。"""
    count = len(subtask_ids)
    if count == 1:
        return [1], []
    if count == 2:
        steps = [1, 2]
    elif count == 3:
        steps = [1, 1, 2]
    else:
        middle = count - 2
        steps = [1] + [2 + (index // 2) for index in range(middle)] + [2 + ((middle - 1) // 2) + 1]
    edges: list[tuple[str, str]] = []
    for target_index, target_step in enumerate(steps):
        if target_step == 1:
            continue
        predecessors = [index for index, step in enumerate(steps) if step == target_step - 1]
        for source_index in predecessors:
            edges.append((subtask_ids[source_index], subtask_ids[target_index]))
    return steps, edges


def build_tasks(source_path: Path, agents: list[dict]) -> tuple[pd.DataFrame, list[dict]]:
    del source_path  # Skill 只取最终派生智能体实际拥有的集合，保证能力硬匹配可执行。
    scenario_skills: dict[str, dict[str, dict]] = {scenario: {} for scenario in SCENARIO_GOALS}
    skill_source: dict[str, str] = {}
    for agent in agents:
        for skill in agent["skills"]:
            scenario_skills[agent["场景"]].setdefault(skill["id"], skill)
            skill_source.setdefault(skill["id"], agent["源智能体类型id"])

    flat_rows: list[dict] = []
    complex_tasks: list[dict] = []
    complex_index = 0
    for scenario, goals in SCENARIO_GOALS.items():
        pool = sorted(scenario_skills[scenario].values(), key=lambda item: item["id"])
        if len(pool) < 9:
            raise ValueError(f"{scenario} 可用 Skill 少于 9 项，无法构造困难任务")
        for difficulty, counts in DIFFICULTY_COUNTS.items():
            for goal_index, goal in enumerate(goals):
                complex_index += 1
                count = counts[goal_index % len(counts)]
                keywords = SCENARIO_GOAL_KEYWORDS[scenario][goal_index]
                selected = sorted(
                    pool,
                    key=lambda skill: (
                        -sum(keyword in skill_text(skill) for keyword in keywords),
                        stable_int(f"{scenario}:{difficulty}:{goal}:{skill['id']}") % 10_000,
                    ),
                )[:count]
                complex_id = f"CT{complex_index:04d}"
                subtask_ids = [f"{complex_id}-S{index + 1:02d}" for index in range(count)]
                steps, edges = _dag_for_subtasks(subtask_ids)
                critical = scenario == "应急救援" or any(
                    word in skill_text(skill) for skill in selected for word in EMERGENCY_WORDS
                )
                latency = {"简单": 20, "中等": 45, "困难": 90}[difficulty]
                if critical:
                    latency = max(8, round(latency * 0.65))
                accuracy = {"简单": 0.88, "中等": 0.91, "困难": 0.94}[difficulty]
                output = OUTPUT_STYLES[goal_index]
                completeness = {"简单": 0.90, "中等": 0.93, "困难": 0.96}[difficulty]
                prompt_openers = (
                    f"请协调多个智能体完成“{goal}”。",
                    f"当前需要在{scenario}环境中处理“{goal}”，请组织协同执行。",
                    f"围绕“{goal}”发起一次{difficulty}级联合任务。",
                    f"值守系统收到“{goal}”请求，请按依赖关系推进各项工作。",
                )
                complex_prompt = prompt_openers[(goal_index + complex_index) % len(prompt_openers)]
                complex_row = {
                    "任务id": complex_id, "父任务id": "", "任务类型": "复杂任务", "执行步骤": "",
                    "前置子任务id": "", "场景": scenario, "难度级别": difficulty,
                    "任务具体提示词": complex_prompt, "必需的Skill id": "|".join(skill["id"] for skill in selected),
                    "输出结果的要求": output, "字段完整率要求": completeness,
                    "端到端完成时延要求(s)": latency, "准确率要求": accuracy,
                    "DAG边列表": "|".join(f"{source}->{target}" for source, target in edges),
                }
                flat_rows.append(complex_row)
                subtasks: list[dict] = []
                for sub_index, (subtask_id, skill, step) in enumerate(zip(subtask_ids, selected, steps)):
                    predecessors = [source for source, target in edges if target == subtask_id]
                    sub_prompt = (
                        f"{skill.get('description', skill.get('name', skill['id']))}"
                        f"{SUBTASK_ENDINGS[(sub_index + goal_index) % len(SUBTASK_ENDINGS)]}"
                    )
                    subtask = {
                        "任务id": subtask_id, "父任务id": complex_id, "任务类型": "子任务", "执行步骤": step,
                        "前置子任务id": "|".join(predecessors), "场景": scenario, "难度级别": difficulty,
                        "任务具体提示词": sub_prompt, "必需的Skill id": skill["id"],
                        "输出结果的要求": output if sub_index == count - 1 else "返回本步骤可供下游消费的结果",
                        "字段完整率要求": completeness,
                        "端到端完成时延要求(s)": round(latency / max(1, max(steps)), 2),
                        "准确率要求": accuracy, "DAG边列表": "",
                        "source_agent_type_id": skill_source[skill["id"]],
                    }
                    flat_rows.append(subtask)
                    subtasks.append(subtask)
                complex_tasks.append({
                    **complex_row, "子任务数量": count,
                    "DAG": {"节点": subtask_ids, "边": [[source, target] for source, target in edges]},
                    "子任务": subtasks,
                })
    return pd.DataFrame(flat_rows), complex_tasks


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
    ], columns=["关系类型编码", "关系类型名称", "含义"])
    events = []
    timestamp = 0.0
    for index, agent in enumerate(agents):
        timestamp += 0.1
        events.append((f"E{len(events)+1:06d}", round(timestamp, 1), 0, agent["agent_key"], agent["agent_key"], -1, "agent_capabilities.csv"))
        link_row = links.iloc[(index * 2) % len(links)]
        timestamp += 0.1
        relation = 0 if str(link_row["数据目的地"]).startswith("C") else 1
        events.append((f"E{len(events)+1:06d}", round(timestamp, 1), 3, link_row["数据源"], link_row["数据目的地"], relation, "communication_links.csv"))
        task = tasks[index % len(tasks)]
        timestamp += 0.1
        events.append((f"E{len(events)+1:06d}", round(timestamp, 1), 4, task["任务id"], agent["agent_key"], 1, "tasks.csv"))
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
    parser.add_argument("--link-timestamps", type=int, default=DEFAULT_LINK_TIMESTAMPS, help="覆盖全时间范围的链路观测点数量")
    args = parser.parse_args()
    if args.compute_nodes < 2:
        raise ValueError("算力节点数至少为 2")
    source_path = args.data_root / "智能体能力源数据" / "part1_24_revised_descriptions_skills.json"
    alibaba_source = args.data_root / "原始数据" / "Alibaba-cluster-trace-v2026-GenAI" / "源数据"
    agents_frame, agents, agent_meta = build_agents(source_path, args.seed)
    reliability = build_reliability(agents, args.seed)
    compute_load, inventory, load_meta = build_compute_load(alibaba_source, args.compute_nodes)
    compute_timestamps = sorted(compute_load["时间戳"].astype(int).unique().tolist())
    links = build_links(agents, inventory, compute_timestamps, args.seed, args.link_timestamps)
    tasks_frame, tasks = build_tasks(source_path, agents)
    event_types, relations, event_stream = build_event_tables(agents, links, tasks)

    paths = {
        "agents_csv": args.data_root / "3.智能体能力数据" / "agent_capabilities.csv",
        "agents_json": args.data_root / "3.智能体能力数据" / "agent_capabilities.json",
        "reliability_csv": args.data_root / "4.智能体可靠性数据" / "agent_reliability.csv",
        "compute_load_csv": args.data_root / "1.算力负载数据" / "compute_node_load_timeseries.csv",
        "compute_inventory_csv": args.data_root / "1.算力负载数据" / "compute_node_inventory.csv",
        "links_csv": args.data_root / "2.通信链路数据" / "communication_links.csv",
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
    manifest = {
        "dataset_version": "3.0.0",
        "generation_seed": args.seed,
        "counts": {
            "agents": len(agents_frame), "source_agent_types": agent_meta["source_type_count"],
            "compute_nodes": len(inventory), "compute_load_rows": len(compute_load),
            "link_state_rows": len(links), "link_topology_edges": int(links["链路id"].nunique()),
            "link_timestamps": int(links["时间戳"].nunique()), "complex_tasks": len(tasks),
            "task_table_rows": len(tasks_frame), "events": len(event_stream),
        },
        "agent_generation": agent_meta,
        "compute_load_processing": load_meta,
        "link_design": (
            "智能体间采用无向全连接逻辑拓扑；每个智能体固定且仅连接一个算力节点；"
            "通信边不携带任务依赖，任务依赖只保存在复杂任务 DAG 中"
        ),
        "files": {key: str(path.relative_to(args.data_root)) for key, path in paths.items()},
    }
    manifest_path = args.data_root / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    assert agents_frame["智能体编号"].tolist() == list(range(1, len(agents_frame) + 1))
    assert reliability["智能体编号"].tolist() == agents_frame["智能体编号"].tolist()
    assert set(links["数据源"]).issubset({item["agent_key"] for item in agents})
    assert set(links.loc[links["数据目的地"].str.startswith("C"), "数据目的地"]).issubset(set(inventory["算力节点编号"]))
    assert agents_frame["接入知识库"].ne("无").mean() <= 0.20
    first_snapshot = links[links["时间戳"] == links["时间戳"].min()]
    compute_degree = first_snapshot[first_snapshot["链路类型"] == "智能体-算力节点"].groupby("数据源").size()
    assert compute_degree.eq(1).all() and len(compute_degree) == len(agents)
    assert len(first_snapshot[first_snapshot["链路类型"] == "智能体-智能体"]) == len(agents) * (len(agents) - 1) // 2
    assert "是否有任务依赖关系" not in links.columns
    assert len(tasks) == 72 and set(tasks_frame["任务类型"]) == {"复杂任务", "子任务"}
    assert "实验要求" not in tasks_frame.columns
    assert "required_skill_id" not in json.dumps(tasks, ensure_ascii=False)
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
