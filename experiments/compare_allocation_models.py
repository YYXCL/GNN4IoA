"""训练 V1、任务节点 V2 与 GMATANN/GCAN 基线，并生成两张场景对比图。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ("智慧家居", "智慧工厂", "应急救援")
SCENARIO_LABELS = ("Smart home", "Smart factory", "Emergency response")


def execute(command: list[str], env: dict[str, str]) -> None:
    print("RUN " + " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def plot_metric(results: list[dict], key: str, ylabel: str, title: str, output: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    colors = ("#8b95a5", "#d07a32", "#007f73")
    x = np.arange(len(SCENARIOS))
    width = 0.24
    figure, axis = plt.subplots(figsize=(9.2, 5.4), constrained_layout=True)
    for index, result in enumerate(results):
        values = [result["scenario_metrics"][scenario][key] for scenario in SCENARIOS]
        bars = axis.bar(x + (index - (len(results) - 1) / 2) * width, values, width,
                        label=result["name"], color=colors[index])
        fmt = "%.2f" if "latency" in key else "%.1f%%"
        labels = [fmt % (value if "latency" in key else value * 100.0) for value in values]
        axis.bar_label(bars, labels=labels, padding=3, fontsize=8)
    axis.set_xticks(x, SCENARIO_LABELS)
    axis.set_ylabel(ylabel)
    axis.set_title(title, fontweight="bold")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    if "accuracy" in key:
        axis.set_ylim(0, 1.08)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-root", type=Path, default=ROOT.parent / "GNN4IoA-v1")
    parser.add_argument("--baseline-root", type=Path, default=ROOT / "baselines" / "gmatann")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--include-v1", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rebuild-dataset", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()
    if args.include_v1 and not (args.v1_root / "3.training" / "train.py").is_file():
        parser.error(
            f"V1 checkout not found at {args.v1_root}. "
            "Create it with 'git worktree add ../GNN4IoA-v1 v1.0' "
            "or use --no-include-v1."
        )
    if not (args.baseline_root / "train.py").is_file():
        parser.error(f"GMATANN baseline not found at {args.baseline_root}")
    python = sys.executable
    dataset = ROOT / "1.data" / "datasets" / "sixg_ma_state"
    reliability = ROOT / "1.data" / "4.智能体可靠性" / "agent_reliability.csv"
    output = ROOT / "outputs" / "scenario_comparison"
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    if args.rebuild_dataset:
        execute([python, str(ROOT / "1.data" / "数据集处理与生成" / "build_training_jsonl_from_foundation.py"),
                 "--output", str(dataset)], env)

    methods: list[tuple[str, Path, Path, str]] = []
    if args.include_v1:
        methods.append(("V1 MA-TGN", args.v1_root, output / "v1" / "model.pt", "native"))
    methods.extend([
        ("GMATANN/GCAN", args.baseline_root, output / "gcan" / "model.pt", "baseline"),
        ("Task-node HGT-GRU", ROOT, output / "task_node" / "model.pt", "native"),
    ])

    if not args.skip_training:
        for name, root, checkpoint, kind in methods:
            if kind == "baseline":
                execute([python, str(root / "train.py"), "--dataset", str(dataset),
                         "--epochs", str(args.epochs), "--seed", str(args.seed),
                         "--output", str(checkpoint)], env)
            else:
                execute([python, str(root / "3.training" / "train.py"), "--dataset", str(dataset),
                         "--epochs", str(args.epochs), "--seed", str(args.seed),
                         "--output", str(checkpoint), "--artifacts", str(checkpoint.parent / "artifacts"),
                         "--no-generate-if-missing"], env)

    results = []
    for name, root, checkpoint, kind in methods:
        result_path = checkpoint.parent / "scenario_metrics.json"
        if kind == "baseline":
            command = [python, str(root / "evaluate.py")]
        else:
            command = [python, str(ROOT / "experiments" / "evaluate_allocation.py"), "--root", str(root)]
        command.extend(["--dataset", str(dataset), "--checkpoint", str(checkpoint),
                        "--reliability", str(reliability), "--output", str(result_path), "--name", name])
        execute(command, env)
        results.append(json.loads(result_path.read_text(encoding="utf-8")))

    accuracy_plot = output / "average_allocation_accuracy_by_scenario.png"
    latency_plot = output / "average_completion_latency_by_scenario.png"
    plot_metric(results, "average_allocation_accuracy", "Average allocation accuracy",
                "Task allocation accuracy across three scenarios", accuracy_plot)
    plot_metric(results, "average_completion_latency_s", "Average completion latency (s)",
                "Estimated completion latency across three scenarios", latency_plot)
    summary = {
        "protocol": {
            "epochs": args.epochs, "seed": args.seed, "test_split": "chronological 9-case test split",
            "scenarios": list(SCENARIOS),
            "latency_definition": "selected agent mean completion time from reliability table",
        },
        "results": results,
        "plots": {"accuracy": str(accuracy_plot), "latency": str(latency_plot)},
    }
    (output / "comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
