"""异构智能体—算力图、任务节点、精确 Skill 和历史状态测试。"""

import json
import sys
import csv
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "2.模型"))
sys.path.insert(0, str(ROOT / "1.data"))

from ma_tgn import ModelConfig, TaskConditionedHeterogeneousTGN  # noqa: E402
from ma_tgn.state_schema import COMPUTE_METRICS, EDGE_METRICS, schema_dict  # noqa: E402
from ma_tgn_data.synthetic import make_synthetic_case  # noqa: E402
from ma_tgn_data.dataset import iter_cases  # noqa: E402


def small_config(**kwargs):
    return ModelConfig(num_agents=8, num_compute_nodes=2, dropout=0.0, **kwargs)


def test_end_to_end_shapes_and_normalized_attention():
    config = small_config()
    model = TaskConditionedHeterogeneousTGN(config).eval()
    events, graph, task, _ = make_synthetic_case(config, seed=11)
    with torch.no_grad():
        for event in events:
            model.process_event(event)
        output = model.assign(graph, task)
    assert output["candidate_scores"].shape == graph.candidate_ids.shape
    assert output["state_predictions"].shape == (graph.candidate_ids.numel(), 5)
    assert output["task_agent_attention"].shape == (1, graph.candidate_ids.numel())
    assert set(output["system_node_types"].tolist()) == {0, 1, 2}
    assert torch.allclose(output["feature_attention_weights"].sum(-1), torch.ones(1, graph.candidate_ids.numel()))
    for candidate in graph.candidate_ids:
        mask = graph.edge_index[1] == candidate
        incoming = output["attention_weights"][mask].sum(0)
        assert torch.allclose(incoming, torch.ones(config.attention_heads), atol=1e-5)


def test_node_and_edge_gru_histories_are_updated():
    config = small_config()
    model = TaskConditionedHeterogeneousTGN(config).eval()
    events, _, _, _ = make_synthetic_case(config, seed=3)
    with torch.no_grad():
        model.process_event(events[0])
    node = events[0].src
    assert model.history.node_memory[node].norm() > 0
    key = model.history.edge_key(events[0].src, events[0].dst, events[0].relation_type)
    assert model.history._edge_memory[key].norm() > 0


def test_gradient_flows_through_heterogeneous_encoder_and_memory():
    config = small_config()
    model = TaskConditionedHeterogeneousTGN(config).train()
    events, graph, task, label = make_synthetic_case(config, seed=5)
    for event in events[:8]:
        model.process_event(event, track_grad=True)
    output = model.assign(graph, task)
    torch.nn.functional.cross_entropy(output["candidate_logits"].unsqueeze(0), label.reshape(1)).backward()
    assert model.node_encoder.agent_fusion.net[0].weight.grad is not None
    assert model.history.node_gru.weight_ih.grad is not None
    assert model.history.edge_gru.weight_ih.grad is not None


def test_only_requested_state_contract_remains():
    schema = schema_dict()
    assert schema["node_types"] == ["agent", "compute", "task"]
    assert schema["relation_types"] == ["compute_access", "communication", "task_assignment"]
    assert "task nodes" in schema["task_representation"]
    assert EDGE_METRICS == ("latency_ms", "bandwidth_mbps", "packet_loss_rate", "jitter_ms")
    assert COMPUTE_METRICS == ("gpu_remaining_tflops", "memory_remaining_gb", "vram_remaining_gb")
    serialized = str(schema)
    for removed in ("path_gateway", "task_dependency_strength", "historical_collaboration_frequency", "trend"):
        assert removed not in serialized


def test_agent_skill_ids_are_static_and_slotless():
    config = small_config()
    events, graph, task, _ = make_synthetic_case(config, seed=17)
    assert graph.candidate_skill_ids
    assert task.required_capability
    for event in events:
        for state in (event.src_state, event.dst_state):
            assert not any(key.startswith("capability_") for key in state)
            if "context" in state:
                assert state["reliability"].shape == (4,)


def test_confidence_is_not_used_by_current_model():
    assert schema_dict()["confidence"]["used_by_current_model"] is False


def test_only_three_remaining_compute_resources_are_encoded():
    config = small_config()
    encoder = TaskConditionedHeterogeneousTGN(config).node_encoder.eval()
    encoder.normalizer.compute_std.copy_(torch.tensor([100.0, 100.0, 100.0]))
    base = torch.tensor([50., 128., 24.])
    changed = torch.tensor([25., 64., 12.])
    with torch.no_grad():
        first, _ = encoder(1, {"compute": base})
        second, _ = encoder(1, {"compute": changed})
    assert not torch.allclose(first, second, atol=1e-6)


def test_task_requirements_are_split_into_independent_fields():
    tasks_path = ROOT / "1.data" / "5.任务数据" / "tasks.json"
    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    required = {
        "必需的Skill id",
        "输出结果的要求",
        "字段完整率要求",
        "端到端完成时延要求(s)",
    }
    assert len(tasks) == 72
    assert all(required <= task.keys() for task in tasks)
    assert all(task["任务类型"] == "复杂任务" and task["子任务"] for task in tasks)
    assert all("实验要求" not in task and "required_skill_id" not in task for task in tasks)


def test_declared_complex_task_dags_cover_three_scenarios_and_difficulties():
    tasks = json.loads((ROOT / "1.data" / "5.任务数据" / "tasks.json").read_text(encoding="utf-8"))
    assert {(task["场景"], task["难度级别"]) for task in tasks} == {
        (scenario, difficulty)
        for scenario in ("智慧工厂", "智慧家居", "应急救援")
        for difficulty in ("简单", "中等", "困难")
    }
    limits = {"简单": (1, 3), "中等": (4, 6), "困难": (7, 9)}
    for task in tasks:
        subtasks = task["子任务"]
        minimum, maximum = limits[task["难度级别"]]
        assert minimum <= len(subtasks) <= maximum
        steps = {item["任务id"]: int(item["执行步骤"]) for item in subtasks}
        assert all(steps[source] < steps[target] for source, target in task["DAG"]["边"])
        assert all("结构化结果、关键证据和置信度" not in item["任务具体提示词"] for item in subtasks)


def test_links_are_dynamic_full_mesh_with_one_fixed_compute_node():
    path = ROOT / "1.data" / "2.链路数据" / "communication_links.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert "是否有任务依赖关系" not in rows[0]
    timestamps = sorted({int(row["时间戳"]) for row in rows})
    assert len(timestamps) == 48
    first = [row for row in rows if int(row["时间戳"]) == timestamps[0]]
    agent_compute = [row for row in first if row["链路类型"] == "智能体-算力节点"]
    agent_peer = [row for row in first if row["链路类型"] == "智能体-智能体"]
    assert len(agent_compute) == 120
    assert len({row["数据源"] for row in agent_compute}) == 120
    assert len(agent_peer) == 120 * 119 // 2
    first_link = rows[0]["链路id"]
    assert len({row["时延(ms)"] for row in rows if row["链路id"] == first_link}) > 1
    assert {int(float(row["带宽(Mbps)"])) for row in rows} <= {500, 1000, 2000, 3000}
    bandwidth_by_link: dict[str, set[str]] = {}
    for row in rows:
        bandwidth_by_link.setdefault(row["链路id"], set()).add(row["带宽(Mbps)"])
    assert all(len(values) == 1 for values in bandwidth_by_link.values())


def test_compute_load_headers_are_chinese_and_agent_boilerplate_is_removed():
    load_path = ROOT / "1.data" / "1.负载数据" / "compute_node_load_timeseries.csv"
    with load_path.open(encoding="utf-8-sig", newline="") as stream:
        headers = next(csv.reader(stream))
    assert "timestamp_anon" not in headers and "compute_node_id" not in headers
    assert {"时间戳", "算力节点编号", "GPU算力剩余量(TFLOPS)", "内存剩余量(GB)", "显存剩余量(GB)"} <= set(headers)
    agents_path = ROOT / "1.data" / "3.智能体能力数据(包括智能体环境和知识库、API权限)" / "agent_capabilities.csv"
    with agents_path.open(encoding="utf-8-sig", newline="") as stream:
        agents = list(csv.DictReader(stream))
    assert all("仅配置源能力卡" not in row["智能体能力描述"] for row in agents)


def test_foundation_jsonl_has_chronological_windows_and_skill_valid_label():
    dataset_dir = ROOT / "1.data" / "datasets" / "sixg_ma_state"
    metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["dataset"] == "6G-Agent-Compute-Foundation-Sliding-Window"
    assert metadata["samples"] == 72
    assert metadata["splits"] == {"train": 54, "val": 9, "test": 9}
    assert metadata["window_overlap"] is False
    assert metadata["design_constraints"]["scenario_difficulty_stratified_split"] is True

    agents_path = ROOT / "1.data" / "3.智能体能力数据(包括智能体环境和知识库、API权限)" / "agent_capabilities.json"
    agents = json.loads(agents_path.read_text(encoding="utf-8"))
    skill_by_agent = {int(agent["智能体编号"]) - 1: {skill["id"] for skill in agent["skills"]} for agent in agents}

    previous_end = None
    for split in ("train", "val", "test"):
        lines = (dataset_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        for line in lines:
            case = json.loads(line)
            meta = case["metadata"]
            start, end = meta["source_timestamp_start"], meta["source_timestamp_end"]
            assert previous_end is None or start > previous_end
            previous_end = end
            label_agent = case["graph"]["candidate_ids"][case["label"]]
            assert meta["required_capability"] in skill_by_agent[label_agent]
            assert meta["difficulty_level"] in {"简单", "中等", "困难"}
            assert case["graph"]["candidate_skill_mask"][case["label"]]
            assert 1 <= len(case["task"]["dag_descriptions"]) <= 9
            assert len(case["task"]["dag_descriptions"]) == len(case["task"]["dag_required_capabilities"])
            candidate_skills = [set(items) for items in case["graph"]["candidate_skill_ids"]]
            assert all(any(skill in skills for skills in candidate_skills) for skill in case["task"]["dag_required_capabilities"])
            edge_index = case["task"]["dag_edge_index"]
            assert all(src < dst for src, dst in zip(edge_index[0], edge_index[1]))


def test_subtasks_are_transient_task_nodes_with_agent_edges():
    config = small_config()
    model = TaskConditionedHeterogeneousTGN(config).eval()
    events, graph, task, _ = make_synthetic_case(config, seed=19)
    task.dag_descriptions = ("采集上游证据", task.description, "生成最终报告")
    task.dag_required_capabilities = (
        "sensor.collect", task.required_capability, "report.generate",
    )
    task.dag_qos = task.values.repeat(3, 1)
    task.dag_edge_index = torch.tensor([[0, 1], [1, 2]])
    task.dag_current_index = 1
    graph.candidate_skill_mask = torch.ones_like(graph.candidate_ids, dtype=torch.bool)
    with torch.no_grad():
        for event in events:
            model.process_event(event)
        output = model.assign(graph, task)
    assert output["task_agent_attention"].shape == (3, graph.candidate_ids.numel())
    assert output["task_agent_edge_index"].shape[1] == 3 * graph.candidate_ids.numel()
    assert (output["system_node_types"] == 2).sum() == 3
    assert output["subtask_assignments"].numel() == 3


def test_complex_task_attention_assigns_every_subtask_and_accumulates_agents():
    case = next(iter_cases(ROOT / "1.data" / "datasets" / "sixg_ma_state" / "train.jsonl"))
    config = ModelConfig(num_agents=120, num_compute_nodes=12, dropout=0.0)
    model = TaskConditionedHeterogeneousTGN(config).eval()
    with torch.no_grad():
        for event in case.events:
            model.process_event(event)
        output = model.assign(case.graph, case.task)
    subtask_count = len(case.task.dag_required_capabilities)
    assert output["task_agent_attention"].shape == (subtask_count, case.graph.candidate_ids.numel())
    assert torch.allclose(output["task_agent_attention"].sum(-1), torch.ones(subtask_count), atol=1e-6)
    assert torch.allclose(output["cumulative_complex_attention"][-1], output["complex_agent_attention"])
    assert output["subtask_eligibility_mask"].gather(1, output["task_agent_attention"].argmax(-1, keepdim=True)).all()
