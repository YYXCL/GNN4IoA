"""加载训练检查点和测试数据，输出 Top-K 与状态预测。"""

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "2.模型"))
sys.path.insert(0, str(PROJECT_ROOT / "1.data"))

from ma_tgn import ModelConfig, TaskConditionedHeterogeneousTGN  # noqa: E402
from ma_tgn.encoders import HeterogeneousStateEncoder  # noqa: E402
from ma_tgn_data.dataset import iter_cases  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="运行训练后模型的推理演示")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "checkpoints" / "model.pt",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "1.data" / "datasets" / "sixg_ma_state" / "test.jsonl",
    )
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = ModelConfig(**checkpoint["config"])
    model = TaskConditionedHeterogeneousTGN(config).eval()
    model.load_state_dict(checkpoint["model"])
    case = next(iter_cases(args.dataset))
    with torch.no_grad():
        model.reset_state()
        for event in case.events:
            model.process_event(event)
        output = model.query(case.graph, case.task)
        topk = model.topk(output, args.top_k)
    predictions = output["state_predictions"]
    id_to_row = {int(node): i for i, node in enumerate(output["candidate_ids"])}
    result = {
        "task_type": case.metadata["task_type"],
        "top_k_agents": [{
            "node_id": int(node), "score": round(float(score), 4),
            "task_success_rate": round(float(predictions[id_to_row[int(node)], 0]), 4),
            "execution_time_normalized": round(float(predictions[id_to_row[int(node)], 1]), 4),
            "tool_failure_rate": round(float(predictions[id_to_row[int(node)], 2]), 4),
            "hallucination_rate": round(float(predictions[id_to_row[int(node)], 3]), 4),
            "executability": round(float(predictions[id_to_row[int(node)], 4]), 4),
        } for node, score in zip(topk["node_ids"], topk["scores"])],
        "task_conditioned_modality_weights": {
            name: round(float(weight), 4)
            for name, weight in zip(HeterogeneousStateEncoder.modality_names, output["modality_weights"])
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
