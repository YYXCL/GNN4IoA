"""独立 baseline 目录共用的训练与评估运行时。"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from baselines import (
    MLPStateAllocator,
    MinimalStateAblationAllocator,
    RuleBasedLatencyScheduler,
    StaticGraphAllocator,
    allocation_loss,
)
from ma_tgn import ModelConfig
from ma_tgn_data.dataset import iter_cases


def build_baseline(key: str, config: ModelConfig):
    builders = {
        "rule_based_latency": lambda: RuleBasedLatencyScheduler(),
        "mlp_state": lambda: MLPStateAllocator(),
        "static_graph": lambda: StaticGraphAllocator(),
        "minimal_state_ablation": lambda: MinimalStateAblationAllocator(config),
    }
    return builders[key]()


def run_output(model, case, train: bool = False):
    if not getattr(model, "temporal", False):
        return model(case)
    model.reset_state()
    warmup = max(0, len(case.events) - 24) if train else len(case.events)
    with torch.no_grad():
        for event in case.events[:warmup]:
            model.process_event(event)
    for event in case.events[warmup:]:
        model.process_event(event, track_grad=train)
    return model.assign(case.graph, case.task)


def train_cli(key: str) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(1)
    metadata = json.loads((args.dataset / "metadata.json").read_text(encoding="utf-8"))
    config = ModelConfig(num_agents=metadata["num_agents"], num_compute_nodes=metadata["num_compute_nodes"])
    model = build_baseline(key, config)
    train_cases = list(iter_cases(args.dataset / "train.jsonl"))
    if hasattr(model, "fit_normalizers"):
        model.fit_normalizers(train_cases)
    history = []
    if getattr(model, "trainable", False):
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        for epoch in range(1, args.epochs + 1):
            model.train()
            random.shuffle(train_cases)
            total = 0.0
            for case in train_cases:
                output = run_output(model, case, train=True)
                loss = allocation_loss(output, case.label)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total += float(loss.detach())
            history.append({"epoch": epoch, "train_loss": total / max(1, len(train_cases))})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "baseline": key,
        "name": model.name,
        "config": config.__dict__,
        "model": model.state_dict() if getattr(model, "trainable", False) else None,
        "history": history,
    }
    torch.save(payload, args.output)
    print(json.dumps({"checkpoint": str(args.output), "name": model.name, "history": history}, ensure_ascii=False))


def evaluate_cli(key: str) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    metadata = json.loads((args.dataset / "metadata.json").read_text(encoding="utf-8"))
    config = ModelConfig(num_agents=metadata["num_agents"], num_compute_nodes=metadata["num_compute_nodes"])
    model = build_baseline(key, config)
    train_cases = list(iter_cases(args.dataset / "train.jsonl"))
    if hasattr(model, "fit_normalizers"):
        model.fit_normalizers(train_cases)
    if getattr(model, "trainable", False):
        if args.checkpoint is None:
            parser.error("可训练 baseline 必须提供 --checkpoint")
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"])
        model.eval()
    cases = list(iter_cases(args.dataset / "test.jsonl"))
    predictions = []
    with torch.no_grad():
        for case in cases:
            output = run_output(model, case)
            chosen = int(output["candidate_logits"].argmax())
            predictions.append({
                "task_id": case.metadata["task_id"],
                "chosen": chosen,
                "label": int(case.label),
                "correct": chosen == int(case.label),
                "task_success_probability": float(case.candidate_targets[chosen, 0]),
                "normalized_completion_time": float(case.candidate_targets[chosen, 1]),
            })
    result = {
        "name": model.name,
        "samples": len(predictions),
        "top1_accuracy": sum(row["correct"] for row in predictions) / max(1, len(predictions)),
        "task_success_rate": sum(row["task_success_probability"] for row in predictions) / max(1, len(predictions)),
        "predictions": predictions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
