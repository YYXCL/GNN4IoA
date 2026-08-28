"""在隔离进程中评估旧版或 V2，避免同名 Python 包相互污染。"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch


def perturb_case(case, level: float, seed: int):
    if level <= 0:
        return case
    case = copy.deepcopy(case)
    torch.manual_seed(seed)

    def perturb_state(state: dict[str, torch.Tensor]) -> None:
        for key in ("context", "environment", "compute"):
            if key not in state:
                continue
            value = state[key]
            scale = value.abs().clamp_min(0.01) * level
            state[key] = (value + torch.randn_like(value) * scale).clamp_min(0)
            if key == "environment":
                state[key].clamp_(0, 1)
            elif key == "context" and value.numel() > 1:
                state[key][1].clamp_(0, 1)

    for event in case.events:
        perturb_state(event.src_state)
        perturb_state(event.dst_state)
        event.edge_state = (event.edge_state + torch.randn_like(event.edge_state) * event.edge_state.abs().clamp_min(0.01) * level).clamp_min(0)
        event.edge_state[2].clamp_(0, 1)
    graph = case.graph
    graph.edge_state = (graph.edge_state + torch.randn_like(graph.edge_state) * graph.edge_state.abs().clamp_min(0.01) * level).clamp_min(0)
    graph.edge_state[:, 2].clamp_(0, 1)
    return case


def run_case(model, case):
    model.reset_state()
    with torch.no_grad():
        for event in case.events:
            model.process_event(event)
        return model.query(case.graph, case.task)


def metrics(model, cases) -> dict[str, float]:
    correct = top3 = 0
    reciprocal_rank = ndcg = regret = 0.0
    errors = torch.zeros(5)
    candidates = 0
    model.eval()
    for case in cases:
        output = run_case(model, case)
        ranking = torch.argsort(output["candidate_logits"], descending=True).cpu()
        label = int(case.label)
        correct += int(int(ranking[0]) == label)
        top3 += int(label in ranking[:3].tolist())
        reciprocal_rank += 1.0 / (ranking.tolist().index(label) + 1)
        k = min(5, ranking.numel())
        relevance = case.teacher_scores.float()
        discount = torch.log2(torch.arange(k, dtype=torch.float32) + 2.0)
        dcg = ((torch.pow(2.0, relevance[ranking[:k]]) - 1.0) / discount).sum()
        ideal = torch.sort(relevance, descending=True).values[:k]
        ideal_dcg = ((torch.pow(2.0, ideal) - 1.0) / discount).sum().clamp_min(1e-8)
        ndcg += float(dcg / ideal_dcg)
        regret += float(relevance.max() - relevance[int(ranking[0])])
        errors += (output["state_predictions"].cpu() - case.candidate_targets).abs().sum(0)
        candidates += case.candidate_targets.shape[0]
    count = max(1, len(cases))
    names = ("success_mae", "execution_time_mae", "tool_failure_mae", "hallucination_mae", "executability_mae")
    result = {
        "top1_accuracy": correct / count,
        "top3_hit_rate": top3 / count,
        "mrr": reciprocal_rank / count,
        "ndcg_at_5": ndcg / count,
        "selection_regret": regret / count,
        "mean_state_mae": float(errors.sum()) / max(1, candidates * 5),
    }
    result.update({name: float(errors[index]) / max(1, candidates) for index, name in enumerate(names)})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--noise-levels", default="0,0.05,0.10,0.20,0.30")
    parser.add_argument("--latency-repeats", type=int, default=3)
    args = parser.parse_args()
    sys.path.insert(0, str(args.root / "2.模型"))
    sys.path.insert(0, str(args.root / "1.data"))
    from ma_tgn import ModelConfig, TaskConditionedHeterogeneousTGN
    from ma_tgn_data.dataset import iter_cases

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = ModelConfig(**checkpoint["config"])
    model = TaskConditionedHeterogeneousTGN(config)
    model.load_state_dict(checkpoint["model"])
    test_cases = list(iter_cases(args.dataset / "test.jsonl"))
    clean = metrics(model, test_cases)
    robustness = []
    for index, level in enumerate(float(value) for value in args.noise_levels.split(",")):
        noisy = [perturb_case(case, level, 10_000 * index + case_index) for case_index, case in enumerate(test_cases)]
        noisy_metrics = metrics(model, noisy)
        robustness.append({"noise": level, "top1_accuracy": noisy_metrics["top1_accuracy"], "ndcg_at_5": noisy_metrics["ndcg_at_5"]})

    run_case(model, test_cases[0])
    timings = []
    for _ in range(args.latency_repeats):
        for case in test_cases:
            start = time.perf_counter()
            run_case(model, case)
            timings.append((time.perf_counter() - start) * 1000.0)
    result = {
        "name": args.name,
        "clean_test": clean,
        "robustness": robustness,
        "latency_ms_per_case": sum(timings) / len(timings),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "test_cases": len(test_cases),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
