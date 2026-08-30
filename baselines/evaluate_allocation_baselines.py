"""统一训练/评估四类新增分配 baseline，并输出可供后续绘图的 JSON。"""

from __future__ import annotations

import argparse
import json
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


def run_temporal(model, case, train: bool):
    model.reset_state()
    for event in case.events:
        model.process_event(event, track_grad=train)
    return model.assign(case.graph, case.task)


def accuracy(model, cases, temporal: bool = False) -> float:
    correct = 0
    with torch.no_grad():
        for case in cases:
            output = run_temporal(model, case, False) if temporal else model(case)
            correct += int(output["candidate_logits"].argmax() == case.label)
    return correct / max(1, len(cases))


def train_model(model, cases, epochs: int, temporal: bool = False) -> list[float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    history = []
    for _ in range(epochs):
        total = 0.0
        for case in cases:
            output = run_temporal(model, case, True) if temporal else model(case)
            loss = allocation_loss(output, case.label)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += float(loss.detach())
        history.append(total / max(1, len(cases)))
    return history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "1.data" / "datasets" / "sixg_ma_state")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-train-cases", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "baseline_results_v3.json")
    args = parser.parse_args()
    train_cases = list(iter_cases(args.dataset / "train.jsonl"))
    if args.max_train_cases > 0:
        train_cases = train_cases[:args.max_train_cases]
    test_cases = list(iter_cases(args.dataset / "test.jsonl"))
    metadata = json.loads((args.dataset / "metadata.json").read_text(encoding="utf-8"))
    config = ModelConfig(num_agents=metadata["num_agents"], num_compute_nodes=metadata["num_compute_nodes"])

    rule = RuleBasedLatencyScheduler()
    mlp = MLPStateAllocator()
    static = StaticGraphAllocator()
    minimal = MinimalStateAblationAllocator(config)
    minimal.fit_normalizers(train_cases)
    results = {
        rule.name: {"top1_accuracy": accuracy(rule, test_cases), "train_loss": []},
        mlp.name: {"train_loss": train_model(mlp, train_cases, args.epochs)},
        static.name: {"train_loss": train_model(static, train_cases, args.epochs)},
        minimal.name: {"train_loss": train_model(minimal, train_cases, args.epochs, temporal=True)},
    }
    results[mlp.name]["top1_accuracy"] = accuracy(mlp, test_cases)
    results[static.name]["top1_accuracy"] = accuracy(static, test_cases)
    results[minimal.name]["top1_accuracy"] = accuracy(minimal, test_cases, temporal=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
