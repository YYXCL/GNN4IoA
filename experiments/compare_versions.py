"""一键重建同源数据、训练 V1/V2、隔离评估并绘制对比图。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


V2_ROOT = Path(__file__).resolve().parents[1]


def execute(command: list[str], env: dict[str, str]) -> None:
    print("RUN " + " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def plot_report(output: Path, baseline: dict, v2: dict, baseline_history: list[dict], v2_history: list[dict]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    colors = {"V1 MA-TGN": "#68758a", "V2 HGT-GRU": "#007f73"}
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)

    for name, history in (("V1 MA-TGN", baseline_history), ("V2 HGT-GRU", v2_history)):
        axes[0, 0].plot([row["epoch"] for row in history], [row["train_loss"] for row in history], marker="o", label=name, color=colors[name])
        axes[0, 1].plot([row["epoch"] for row in history], [row["top1_accuracy"] for row in history], marker="o", label=name, color=colors[name])
    axes[0, 0].set(title="Training convergence", xlabel="Epoch", ylabel="Loss")
    axes[0, 1].set(title="Validation selection accuracy", xlabel="Epoch", ylabel="Top-1 accuracy", ylim=(0, 1.03))

    rank_metrics = ["top1_accuracy", "top3_hit_rate", "mrr", "ndcg_at_5"]
    rank_labels = ["Top-1", "Top-3", "MRR", "NDCG@5"]
    for result in (baseline, v2):
        axes[0, 2].plot(rank_labels, [result["clean_test"][key] for key in rank_metrics], marker="o", linewidth=2, label=result["name"], color=colors[result["name"]])
    axes[0, 2].set(title="Test ranking quality (higher is better)", ylabel="Score", ylim=(0, 1.03))

    state_metrics = ["success_mae", "execution_time_mae", "tool_failure_mae", "hallucination_mae", "executability_mae"]
    state_labels = ["Success", "Time", "Tool fail", "Halluc.", "Executable"]
    for result in (baseline, v2):
        axes[1, 0].plot(state_labels, [result["clean_test"][key] for key in state_metrics], marker="o", linewidth=2, label=result["name"], color=colors[result["name"]])
    axes[1, 0].set(title="State prediction error (lower is better)", ylabel="MAE")

    for result in (baseline, v2):
        axes[1, 1].plot([row["noise"] * 100 for row in result["robustness"]], [row["ndcg_at_5"] for row in result["robustness"]], marker="o", linewidth=2, label=result["name"], color=colors[result["name"]])
    axes[1, 1].set(title="Robustness to numeric observation noise", xlabel="Relative noise (%)", ylabel="NDCG@5", ylim=(0, 1.03))

    names = [baseline["name"], v2["name"]]
    latency = [baseline["latency_ms_per_case"], v2["latency_ms_per_case"]]
    bars = axes[1, 2].bar(names, latency, color=[colors[name] for name in names], width=0.55)
    axes[1, 2].bar_label(bars, fmt="%.1f ms")
    axes[1, 2].set(title="End-to-end CPU inference latency", ylabel="ms / case")

    for axis in axes.flat:
        axis.grid(alpha=0.22)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(frameon=False, fontsize=8)
    figure.suptitle("GNN4IoA V1 vs V2 — same temporal split, seed and candidate set", fontsize=16, fontweight="bold")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-root", type=Path, default=V2_ROOT.parent / "GNN4IoA-v1")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-train-cases", type=int, default=0)
    parser.add_argument("--rebuild-dataset", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()
    if not (args.original_root / "3.training" / "train.py").is_file():
        parser.error(
            f"V1 checkout not found at {args.original_root}. "
            "Create it with 'git worktree add ../GNN4IoA-v1 v1.0'."
        )
    dataset = V2_ROOT / "1.data" / "datasets" / "sixg_ma_state"
    output = V2_ROOT / "outputs" / "comparison"
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    python = sys.executable
    if args.rebuild_dataset:
        execute([python, str(V2_ROOT / "1.data" / "数据集处理与生成" / "build_training_jsonl_from_foundation.py"), "--output", str(dataset)], env)

    versions = {
        "baseline": (args.original_root, "V1 MA-TGN"),
        "v2": (V2_ROOT, "V2 HGT-GRU"),
    }
    if not args.skip_training:
        for key, (root, _) in versions.items():
            execute([
                python, str(root / "3.training" / "train.py"),
                "--dataset", str(dataset), "--epochs", str(args.epochs),
                "--seed", str(args.seed), "--max-train-cases", str(args.max_train_cases),
                "--output", str(output / key / "model.pt"),
                "--artifacts", str(output / key / "artifacts"),
                "--no-generate-if-missing",
            ], env)

    results = {}
    for key, (root, name) in versions.items():
        result_path = output / key / "evaluation.json"
        execute([
            python, str(V2_ROOT / "experiments" / "evaluate_version.py"),
            "--root", str(root), "--dataset", str(dataset),
            "--checkpoint", str(output / key / "model.pt"),
            "--output", str(result_path), "--name", name,
        ], env)
        results[key] = json.loads(result_path.read_text(encoding="utf-8"))

    histories = {
        key: json.loads((output / key / "artifacts" / "training_history.json").read_text(encoding="utf-8"))
        for key in versions
    }
    plot_path = output / "v1_vs_v2_performance.png"
    plot_report(plot_path, results["baseline"], results["v2"], histories["baseline"], histories["v2"])
    summary = {
        "protocol": {
            "epochs": args.epochs, "seed": args.seed, "max_train_cases": args.max_train_cases,
            "dataset": str(dataset), "same_chronological_split": True,
        },
        "baseline": results["baseline"], "v2": results["v2"], "plot": str(plot_path),
    }
    (output / "comparison_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SUMMARY=" + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
