"""异构智能体—算力图、基础表窗口、动态能力和置信度测试。"""

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "2.模型"))
sys.path.insert(0, str(ROOT / "1.data"))

from ma_tgn import ModelConfig, TaskConditionedHeterogeneousTGN  # noqa: E402
from ma_tgn.capabilities import CapabilityRegistry  # noqa: E402
from ma_tgn.encoders import HeterogeneousStateEncoder, decayed_confidence  # noqa: E402
from ma_tgn.state_schema import COMPUTE_METRICS, EDGE_METRICS, schema_dict  # noqa: E402
from ma_tgn_data.synthetic import make_synthetic_case  # noqa: E402


def small_config(**kwargs):
    return ModelConfig(num_agents=8, num_compute_nodes=2, dropout=0.0, **kwargs)


def test_end_to_end_shapes_and_normalized_attention():
    config = small_config()
    model = TaskConditionedHeterogeneousTGN(config).eval()
    events, graph, task, _ = make_synthetic_case(config, seed=11)
    with torch.no_grad():
        for event in events:
            model.process_event(event)
        output = model.query(graph, task)
    assert output["candidate_scores"].shape == graph.candidate_ids.shape
    assert output["state_predictions"].shape == (graph.candidate_ids.numel(), 5)
    assert output["dag_attention_weights"].shape[-1] == 1
    assert set(output["system_node_types"].tolist()) <= {0, 1}
    for candidate in graph.candidate_ids:
        mask = graph.edge_index[1] == candidate
        incoming = output["attention_weights"][mask].sum(0)
        assert torch.allclose(incoming, torch.ones(config.attention_heads), atol=1e-5)


def test_fast_memory_changes_more_than_slow_memory():
    config = small_config(slow_update_rate=0.05)
    model = TaskConditionedHeterogeneousTGN(config).eval()
    events, _, _, _ = make_synthetic_case(config, seed=3)
    with torch.no_grad():
        model.process_event(events[0])
    node = events[0].src
    assert model.memory.fast_memory[node].norm() > model.memory.slow_memory[node].norm()


def test_gradient_flows_through_heterogeneous_encoder_and_memory():
    config = small_config()
    model = TaskConditionedHeterogeneousTGN(config).train()
    events, graph, task, label = make_synthetic_case(config, seed=5)
    for event in events[:8]:
        model.process_event(event, track_grad=True)
    output = model.query(graph, task)
    torch.nn.functional.cross_entropy(output["candidate_logits"].unsqueeze(0), label.reshape(1)).backward()
    assert model.node_encoder.fusion.net[0].weight.grad is not None
    assert model.memory.fast_cell.weight_ih.grad is not None


def test_only_requested_state_contract_remains():
    schema = schema_dict()
    assert schema["node_types"] == ["agent", "compute"]
    assert schema["relation_types"] == ["compute_access", "communication", "task_dependency"]
    assert "DAG query" in schema["task_representation"]
    assert EDGE_METRICS == ("latency_ms", "bandwidth_mbps", "packet_loss_rate", "jitter_ms")
    assert COMPUTE_METRICS[-3:] == ("task_queue_length", "active_batch_size", "max_batch_size")
    serialized = str(schema)
    for removed in ("path_gateway", "task_dependency_strength", "historical_collaboration_frequency", "trend"):
        assert removed not in serialized


def test_dynamic_capability_slots_add_and_evict_least_used():
    registry = CapabilityRegistry(max_slots=2, embedding_dim=8)
    registry.add_or_touch("判断视频帧是否有火灾")
    registry.add_or_touch("判断视频帧是否有火灾")
    registry.add_or_touch("判断视频帧是否有烟雾")
    registry.record_result("识别视频帧中的人员", True, 800, False, False)
    names = [slot.name for slot in registry.slots]
    assert "判断视频帧是否有火灾" in names
    assert "识别视频帧中的人员" in names
    tensors = registry.tensors()
    assert tensors["capability_reliability"].shape == (2, 4)


def test_confidence_decays_but_reliability_has_no_confidence_field():
    now = decayed_confidence(torch.tensor([0.9]), 10.0, 10.0, 2.0, torch.device("cpu"))
    later = decayed_confidence(torch.tensor([0.9]), 10.0, 14.0, 2.0, torch.device("cpu"))
    assert later < now
    assert "capability_reliability" in schema_dict()["confidence"]["excluded_from"]


def test_queue_and_batch_are_in_graph_but_masked_from_decision():
    config = small_config(use_queue_batch_in_decision=False)
    encoder = HeterogeneousStateEncoder(config).eval()
    base = torch.tensor([100., 50., 48., 24., 400000., 200000., 0., 1., 8.])
    busy_queue = base.clone(); busy_queue[-3:] = torch.tensor([200., 64., 128.])
    kwargs = {"confidence": {"compute": torch.tensor([1.0])}, "observed_at": {"compute": 0.0}, "now": 0.0}
    with torch.no_grad():
        first, _ = encoder(1, {"compute": base}, **kwargs)
        second, _ = encoder(1, {"compute": busy_queue}, **kwargs)
    assert torch.allclose(first, second, atol=1e-6)


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
    assert all("实验要求" not in task and "required_skill_id" not in task for task in tasks)


def test_foundation_jsonl_has_chronological_windows_and_skill_valid_label():
    dataset_dir = ROOT / "1.data" / "datasets" / "sixg_ma_state"
    metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["dataset"] == "6G-Agent-Compute-Foundation-Sliding-Window"
    assert metadata["samples"] == 72
    assert metadata["splits"] == {"train": 54, "val": 9, "test": 9}
    assert metadata["window_overlap"] is False

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
            assert case["graph"]["candidate_skill_mask"][case["label"]]
            assert len(case["task"]["dag_descriptions"]) >= 2
            assert len(case["task"]["dag_descriptions"]) == len(case["task"]["dag_required_capabilities"])
            edge_index = case["task"]["dag_edge_index"]
            assert all(src < dst for src, dst in zip(edge_index[0], edge_index[1]))


def test_dag_is_query_condition_not_system_node():
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
        output = model.query(graph, task)
    assert output["dag_attention_weights"].shape[-1] == 3
    assert output["system_node_types"].numel() == len(output["id_to_local"])
    assert set(output["system_node_types"].tolist()) == {0, 1}
