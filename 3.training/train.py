"""从持久化 6G 多智能体状态数据集训练并输出可视化工件。"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "2.模型"))
sys.path.insert(0, str(PROJECT_ROOT / "1.data"))

from ma_tgn import ModelConfig, TaskConditionedHeterogeneousTGN, TrainingCase  # noqa: E402
from ma_tgn.state_schema import COMPUTE_METRICS, TASK_QOS_METRICS  # noqa: E402
from ma_tgn_data.dataset import generate_dataset, iter_cases  # noqa: E402


def run_case(model: TaskConditionedHeterogeneousTGN, case: TrainingCase, train: bool) -> dict[str, torch.Tensor]:
    model.reset_state()
    warmup = max(0, len(case.events) - 24) if train else len(case.events)
    with torch.no_grad():
        for event in case.events[:warmup]:
            model.process_event(event)
    for event in case.events[warmup:]:
        model.process_event(event, track_grad=train)
    return model.assign(case.graph, case.task)


def evaluate(model: TaskConditionedHeterogeneousTGN, cases: list[TrainingCase]) -> dict[str, float]:
    model.eval()
    correct = 0
    top3_correct = 0
    reciprocal_rank = 0.0
    ndcg5 = 0.0
    regret = 0.0
    total_candidates = 0
    absolute_errors = torch.zeros(5)
    with torch.no_grad():
        for case in cases:
            output = run_case(model, case, False)
            correct += int(output["candidate_logits"].argmax() == case.label)
            ranking = torch.argsort(output["candidate_logits"], descending=True).cpu()
            label = int(case.label)
            top3_correct += int(label in ranking[:3].tolist())
            rank = ranking.tolist().index(label) + 1
            reciprocal_rank += 1.0 / rank
            k = min(5, ranking.numel())
            relevance = case.teacher_scores.float()
            gains = torch.pow(2.0, relevance[ranking[:k]]) - 1.0
            discount = torch.log2(torch.arange(k, dtype=torch.float32) + 2.0)
            ideal = torch.sort(relevance, descending=True).values[:k]
            ideal_dcg = ((torch.pow(2.0, ideal) - 1.0) / discount).sum().clamp_min(1e-8)
            ndcg5 += float((gains / discount).sum() / ideal_dcg)
            chosen = int(ranking[0])
            regret += float(relevance.max() - relevance[chosen])
            predicted = output["state_predictions"].cpu()
            target = case.candidate_targets
            absolute_errors += (predicted-target).abs().sum(0)
            total_candidates += target.shape[0]
    return {
        "top1_accuracy": correct / max(1, len(cases)),
        "top3_hit_rate": top3_correct / max(1, len(cases)),
        "mrr": reciprocal_rank / max(1, len(cases)),
        "ndcg_at_5": ndcg5 / max(1, len(cases)),
        "selection_regret": regret / max(1, len(cases)),
        "task_success_rate_mae": float(absolute_errors[0])/max(1,total_candidates),
        "execution_time_mae": float(absolute_errors[1])/max(1,total_candidates),
        "tool_failure_rate_mae": float(absolute_errors[2])/max(1,total_candidates),
        "hallucination_rate_mae": float(absolute_errors[3])/max(1,total_candidates),
        "executability_mae": float(absolute_errors[4])/max(1,total_candidates),
        "mean_state_mae": float(absolute_errors.sum())/max(1,total_candidates*5),
    }


def dashboard_payload(config: ModelConfig, case: TrainingCase, output: dict[str, torch.Tensor], metrics: dict, history: list[dict]) -> dict:
    candidate_ids = case.graph.candidate_ids.tolist()
    roles = case.metadata["candidate_roles"]
    scores = output["candidate_scores"].detach().cpu().tolist()
    predictions = output["state_predictions"].detach().cpu().tolist()
    nodes = [{
        "id": int(node_id), "node_type": "agent", "role": roles[index], "score": scores[index],
        "task_success_rate": predictions[index][0], "execution_time_normalized": predictions[index][1],
        "tool_failure_rate": predictions[index][2], "hallucination_rate": predictions[index][3],
        "executability": predictions[index][4], "selected": index == int(output["candidate_logits"].argmax()),
    } for index, node_id in enumerate(candidate_ids)]
    compute_states = case.metadata["compute_states"]
    compute_kinds = case.metadata["compute_kinds"]
    attached = sorted({case.metadata["agent_compute"][int(agent)] for agent in candidate_ids})
    for compute_index in attached:
        raw = compute_states[compute_index]
        nodes.append({
            "id": config.num_agents+compute_index, "node_type": "compute", "kind": compute_kinds[compute_index],
            "gpu_remaining_tflops": raw[0], "memory_remaining_gb": raw[1],
            "vram_remaining_gb": raw[2],
        })
    edges = []
    edge_attention = output["attention_weights"].detach().cpu().mean(dim=1).tolist()
    for index in range(case.graph.edge_index.shape[1]):
        src = int(case.graph.edge_index[0, index]); dst = int(case.graph.edge_index[1, index])
        raw = case.graph.edge_state[index].tolist()
        relation = ("compute_access", "communication")[int(case.graph.relation_type[index])]
        src_type = "agent" if src < config.num_agents else "compute"
        edges.append({
            "source": src, "target": dst, "source_type": src_type, "relation": relation,
            "attention": edge_attention[index],
            "latency_ms": raw[0], "bandwidth_mbps": raw[1], "packet_loss_rate": raw[2], "jitter_ms": raw[3],
        })
    return {
        "task_type": case.metadata["task_type"],
        "required_capability": case.metadata["required_capability"],
        "task_requirements": dict(zip(TASK_QOS_METRICS, case.task.values.tolist())),
        "task_nodes": {
            "complex_task_id": case.task.complex_task_id,
            "subtask_count": len(case.task.dag_descriptions) or 1,
            "dag_edge_index": (
                case.task.dag_edge_index.tolist() if case.task.dag_edge_index is not None else [[], []]
            ),
            "current_subtask_index": case.task.dag_current_index,
            "representation": "transient task nodes connected to every candidate agent",
            "subtask_assignments": output["subtask_assignments"].detach().cpu().tolist(),
            "selected_agent_ids": output["selected_agent_ids"].detach().cpu().tolist(),
            "complex_agent_attention": output["complex_agent_attention"].detach().cpu().tolist(),
            "feature_names": list(output["feature_names"]),
            "average_feature_weights": output["modality_weights"].detach().cpu().tolist(),
        },
        "nodes": nodes, "edges": edges,
        "metrics": metrics, "history": history,
        "model_constraints": {
            "queue_batch_used_for_decision": False, "link_path_type_encoded": False,
            "confidence_used": False, "explicit_trend_feature_used": False,
            "node_history_encoder": "GRU", "edge_history_encoder": "GRU",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="训练任务条件化异构 TGN")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "1.data" / "datasets" / "sixg_ma_state",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "checkpoints" / "model.pt",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "artifacts",
    )
    parser.add_argument("--generate-if-missing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-train-cases", type=int, default=0, help="0 表示使用全部训练样本")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if not (args.dataset / "train.jsonl").exists():
        if not args.generate_if_missing:
            raise FileNotFoundError(args.dataset / "train.jsonl")
        foundation_tasks = PROJECT_ROOT / "1.data" / "5.任务数据" / "tasks.json"
        if foundation_tasks.exists():
            subprocess.run([
                sys.executable,
                str(PROJECT_ROOT / "1.data" / "数据集处理与生成" / "build_training_jsonl_from_foundation.py"),
                "--data-root", str(PROJECT_ROOT / "1.data"),
                "--output", str(args.dataset),
                "--seed", str(args.seed),
            ], check=True)
        else:
            generate_dataset(args.dataset, num_agents=100, samples=48, seed=args.seed)
    metadata = json.loads((args.dataset / "metadata.json").read_text(encoding="utf-8"))
    config = ModelConfig(
        num_agents=int(metadata["num_agents"]),
        num_compute_nodes=int(metadata["num_compute_nodes"]),
        dropout=0.05,
    )
    train_cases = list(iter_cases(args.dataset / "train.jsonl"))
    val_cases = list(iter_cases(args.dataset / "val.jsonl"))
    test_cases = list(iter_cases(args.dataset / "test.jsonl"))
    if args.max_train_cases > 0:
        train_cases = train_cases[:args.max_train_cases]
    model = TaskConditionedHeterogeneousTGN(config)
    model.fit_normalizers(train_cases)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    class_loss = nn.CrossEntropyLoss()
    state_loss = nn.BCELoss()
    history: list[dict] = []
    best_selection_score = -1.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train(); loss_sum = 0.0
        for case in train_cases:
            output = run_case(model, case, True)
            loss = class_loss(output["candidate_logits"].unsqueeze(0), case.label.reshape(1))
            loss = loss + 0.40 * state_loss(output["state_predictions"], case.candidate_targets)
            teacher_distribution = torch.softmax(case.teacher_scores / 0.15, dim=0)
            loss = loss + 0.10 * nn.functional.kl_div(
                torch.log_softmax(output["candidate_logits"], dim=0),
                teacher_distribution,
                reduction="batchmean",
            )
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            loss_sum += float(loss.detach())
        validation = evaluate(model, val_cases)
        row = {"epoch": epoch, "train_loss": loss_sum / len(train_cases), **validation}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        selection_score = validation["top1_accuracy"] + 0.25 * validation["ndcg_at_5"]
        if selection_score > best_selection_score:
            best_selection_score = selection_score
            torch.save({"config": config.__dict__, "model": model.state_dict()}, args.output)
    checkpoint = torch.load(args.output, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    test_metrics = evaluate(model, test_cases)
    model.eval()
    with torch.no_grad():
        sample_output = run_case(model, test_cases[0], False)
    args.artifacts.mkdir(parents=True, exist_ok=True)
    (args.artifacts / "training_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.artifacts / "dataset_summary.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.artifacts / "normalization_stats.json").write_text(
        json.dumps(model.normalizer.summary(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    payload = dashboard_payload(config, test_cases[0], sample_output, test_metrics, history)
    (args.artifacts / "latest_inference.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("test=" + json.dumps(test_metrics, ensure_ascii=False))
    print(f"checkpoint={args.output}\nartifacts={args.artifacts}")


if __name__ == "__main__":
    main()
