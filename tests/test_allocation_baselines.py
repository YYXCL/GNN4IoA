"""规则、MLP、静态图和主模型特征消融 baseline 测试。"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "2.模型"))
sys.path.insert(0, str(ROOT / "1.data"))

from baselines import (  # noqa: E402
    MLPStateAllocator, MinimalStateAblationAllocator,
    RuleBasedLatencyScheduler, StaticGraphAllocator, allocation_loss,
)
from ma_tgn import ModelConfig  # noqa: E402
from ma_tgn_data.dataset import iter_cases  # noqa: E402


def fixture():
    return next(iter_cases(ROOT / "1.data" / "datasets" / "sixg_ma_state" / "train.jsonl"))


def test_rule_scheduler_respects_skill_before_latency():
    case = fixture()
    output = RuleBasedLatencyScheduler()(case)
    chosen = int(output["candidate_logits"].argmax())
    assert output["eligibility_mask"][chosen]
    eligible_latency = output["predicted_completion_latency_s"][output["eligibility_mask"]]
    assert torch.isclose(output["predicted_completion_latency_s"][chosen], eligible_latency.min())


def test_mlp_converts_concatenated_state_to_trainable_allocation_logits():
    case = fixture()
    model = MLPStateAllocator()
    output = model(case)
    assert output["candidate_logits"].shape == case.graph.candidate_ids.shape
    allocation_loss(output, case.label).backward()
    assert model.scorer[0].weight.grad is not None


def test_static_graph_uses_latest_snapshot_without_recurrent_layers():
    case = fixture()
    model = StaticGraphAllocator()
    output = model(case)
    assert output["candidate_scores"].shape == case.graph.candidate_ids.shape
    assert not any(isinstance(module, (torch.nn.GRU, torch.nn.GRUCell, torch.nn.LSTM)) for module in model.modules())


def test_minimal_ablation_keeps_main_algorithm_and_masks_requested_features():
    config = ModelConfig(num_agents=120, num_compute_nodes=12, dropout=0.0)
    model = MinimalStateAblationAllocator(config)
    assert model.config.compute_feature_mask == (True, False, False)
    assert model.config.edge_feature_mask == (True, False, False, False)
    assert model.config.use_agent_reliability is False
    assert isinstance(model.history.node_gru, torch.nn.GRUCell)
    assert isinstance(model.history.edge_gru, torch.nn.GRUCell)
