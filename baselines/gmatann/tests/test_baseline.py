import json
import os
import sys
from pathlib import Path

import torch

BASELINE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASELINE_ROOT))

from gcan_baseline.data import case_from_dict
from gcan_baseline.model import GCANConfig, GraphConvolutionalAssignmentNetwork


def _dataset() -> Path:
    override = os.environ.get("GNN4IOA_DATASET") or os.environ.get("GNN4IOA_V2_DATASET")
    if override:
        return Path(override)
    return REPOSITORY_ROOT / "1.data" / "datasets" / "sixg_ma_state"


def _fixture() -> dict:
    return json.loads((_dataset() / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])


def _model_and_case():
    case = case_from_dict(_fixture())
    model = GraphConvolutionalAssignmentNetwork(GCANConfig(numeric_dim=case.agent_numeric.shape[1]))
    return model, case


def test_data_keeps_ordered_agent_histories():
    case = case_from_dict(_fixture())
    assert case.agent_history.ndim == 3
    assert case.history_mask.shape == case.agent_history.shape[:2]
    assert case.history_delta_time.shape == (*case.agent_history.shape[:2], 1)
    assert case.history_mask.sum(dim=1).min() >= 2
    last = case.history_mask.long().sum(dim=1) - 1
    assert torch.allclose(case.agent_numeric, case.agent_history[torch.arange(last.numel()), last])


def test_forward_builds_task_agent_heterogeneous_graph():
    model, case = _model_and_case()
    model.eval()
    with torch.no_grad():
        output = model(case)
    candidate_count = case.candidate_ids.numel()
    assert output["candidate_logits"].shape == case.candidate_ids.shape
    assert output["state_predictions"].shape == (candidate_count, 5)
    assert output["edge_index"].max() == candidate_count  # 最后一个局部节点是当前 task
    assert int((output["relation_type"] == 0).sum()) == candidate_count
    assert int((output["relation_type"] == 1).sum()) == candidate_count
    assert output["edge_state"].shape == (output["edge_index"].shape[1], 8)


def test_time_series_cvae_reconstructs_only_observed_steps():
    model, case = _model_and_case()
    model.train()
    output = model(case)
    assert output["reconstructed_history"].shape == case.agent_history.shape
    assert output["cvae_mean"].shape == (case.candidate_ids.numel(), model.config.hidden_dim)
    assert torch.isfinite(output["reconstruction_loss"])
    assert torch.isfinite(output["kl_loss"])


def test_global_local_attention_and_fusion_are_normalized():
    model, case = _model_and_case()
    model.eval()
    with torch.no_grad():
        output = model(case)
    node_count = case.candidate_ids.numel() + 1
    for name in ("local_attention_weights", "global_attention_weights"):
        weights = output[name]
        assert weights.shape == (model.config.heads, node_count, node_count)
        assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)), atol=1e-5)
    assert torch.allclose(
        output["fusion_weights"].sum(dim=-1),
        torch.ones(node_count),
        atol=1e-5,
    )


def test_dag_fields_are_intentionally_ignored():
    raw = _fixture()
    first = case_from_dict(raw)
    raw["task"]["dag_descriptions"] = ["完全不同的上游", "完全不同的下游"]
    raw["task"]["dag_edge_index"] = [[0], [1]]
    second = case_from_dict(raw)
    assert torch.equal(first.task_text, second.task_text)
    assert torch.equal(first.task_qos, second.task_qos)
    assert torch.equal(first.agent_history, second.agent_history)


def test_all_requested_components_receive_gradients():
    model, case = _model_and_case()
    output = model(case)
    loss = (
        torch.nn.functional.cross_entropy(output["candidate_logits"].unsqueeze(0), case.label.reshape(1))
        + 0.03 * output["kl_loss"]
        + 0.05 * output["reconstruction_loss"]
    )
    loss.backward()
    assert model.cvae.mean.weight.grad is not None
    assert model.global_local_fusion.local_attention.query.weight.grad is not None
    assert model.global_local_fusion.global_attention.query.weight.grad is not None
    assert model.assignment[0].weight.grad is not None
