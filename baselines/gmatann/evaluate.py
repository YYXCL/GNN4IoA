"""按三种场景评估 GMATANN/GCAN 的准确率与可靠性表时延。"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch

from gcan_baseline import GCANConfig, GraphConvolutionalAssignmentNetwork, load_cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reliability", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", default="GMATANN/GCAN")
    args = parser.parse_args()
    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = GraphConvolutionalAssignmentNetwork(GCANConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    cases = load_cases(args.dataset / "test.jsonl")
    with args.reliability.open(encoding="utf-8-sig", newline="") as stream:
        completion_time = {
            int(row["智能体编号"]) - 1: float(row["平均任务完成时间(秒)"])
            for row in csv.DictReader(stream)
        }
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    predictions = []
    with torch.no_grad():
        for case in cases:
            predicted_index = int(model(case)["candidate_logits"].argmax())
            selected_agent = int(case.candidate_ids[predicted_index])
            correct = float(predicted_index == int(case.label))
            latency = completion_time[selected_agent]
            grouped[case.scenario]["correct"].append(correct)
            grouped[case.scenario]["latency"].append(latency)
            predictions.append({
                "task_id": case.task_id,
                "scenario": case.scenario,
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
    result = {
        "name": args.name,
        "scenario_metrics": scenario_metrics,
        "overall": {
            "samples": len(predictions),
            "average_allocation_accuracy": sum(item["correct"] for item in predictions) / len(predictions),
            "average_completion_latency_s": sum(item["estimated_completion_time_s"] for item in predictions) / len(predictions),
        },
        "latency_definition": "mean task completion time of the selected agent from agent_reliability.csv",
        "dag_dependency_used": False,
        "predictions": predictions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
