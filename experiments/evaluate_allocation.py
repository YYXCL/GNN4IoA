"""按三种业务场景评估分配准确率与所选智能体平均完成时延。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch


def run_case(model, case):
    model.reset_state()
    with torch.no_grad():
        for event in case.events:
            model.process_event(event)
        return model.query(case.graph, case.task)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reliability", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.root / "2.模型"))
    sys.path.insert(0, str(args.root / "1.data"))
    from ma_tgn import ModelConfig, TaskConditionedHeterogeneousTGN
    from ma_tgn_data.dataset import iter_cases

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = ModelConfig(**checkpoint["config"])
    model = TaskConditionedHeterogeneousTGN(config).eval()
    model.load_state_dict(checkpoint["model"])
    cases = list(iter_cases(args.dataset / "test.jsonl"))
    reliability = pd.read_csv(args.reliability)
    completion_time = {
        int(row["智能体编号"]) - 1: float(row["平均任务完成时间(秒)"])
        for row in reliability.to_dict("records")
    }

    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    predictions = []
    for case in cases:
        output = run_case(model, case)
        predicted_index = int(output["candidate_logits"].argmax().cpu())
        selected_agent = int(case.graph.candidate_ids[predicted_index])
        scenario = str(case.metadata["task_type"])
        correct = float(predicted_index == int(case.label))
        latency = completion_time[selected_agent]
        grouped[scenario]["correct"].append(correct)
        grouped[scenario]["latency"].append(latency)
        predictions.append({
            "task_id": case.metadata["task_id"],
            "scenario": scenario,
            "predicted_candidate_index": predicted_index,
            "selected_agent_id": selected_agent,
            "label_candidate_index": int(case.label),
            "correct": bool(correct),
            "estimated_completion_time_s": latency,
        })

    scenario_metrics = {
        scenario: {
            "samples": len(values["correct"]),
            "average_allocation_accuracy": sum(values["correct"]) / len(values["correct"]),
            "average_completion_latency_s": sum(values["latency"]) / len(values["latency"]),
        }
        for scenario, values in grouped.items()
    }
    all_correct = [item["correct"] for item in predictions]
    all_latency = [item["estimated_completion_time_s"] for item in predictions]
    result = {
        "name": args.name,
        "scenario_metrics": scenario_metrics,
        "overall": {
            "samples": len(predictions),
            "average_allocation_accuracy": sum(all_correct) / len(all_correct),
            "average_completion_latency_s": sum(all_latency) / len(all_latency),
        },
        "latency_definition": "mean task completion time of the selected agent from agent_reliability.csv",
        "predictions": predictions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
