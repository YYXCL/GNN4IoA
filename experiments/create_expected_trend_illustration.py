"""Create clearly disclosed expected-trend illustrations from empirical DAG-GRU means.

This script does not alter or overwrite empirical experiment outputs.  It keeps the
measured DAG-GRU group means as anchors and constructs baseline values solely to
visualize a user-specified hypothesis ordering.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "hypothesis_stress_v1" / "scenario_difficulty_metrics.csv"
OUTPUT = ROOT / "outputs" / "expected_trend_illustration"

OURS = "Ours (DAG-GRU)"
SCENARIOS = ("智慧工厂", "智慧家居", "应急救援")
SCENARIO_EN = {
    "智慧工厂": "Smart factory",
    "智慧家居": "Smart home",
    "应急救援": "Emergency response",
}
DIFFICULTIES = ("简单", "中等", "困难")
DIFFICULTY_EN = {"简单": "Simple", "中等": "Medium", "困难": "Hard"}

# A single method-to-color mapping is retained across both figures.
METHODS = (
    OURS,
    "GMATANN",
    "GRU-Minimal",
    "StaticGraph",
    "State-MLP",
    "Rule-Latency",
)
COLORS = {
    OURS: "#0072B2",
    "GMATANN": "#E69F00",
    "Rule-Latency": "#009E73",
    "State-MLP": "#CC79A7",
    "StaticGraph": "#56B4E9",
    "GRU-Minimal": "#D55E00",
}
HATCHES = {
    OURS: "",
    "GMATANN": "//",
    "Rule-Latency": "..",
    "State-MLP": "xx",
    "StaticGraph": "\\\\",
    "GRU-Minimal": "++",
}


def _cell_mean(frame: pd.DataFrame, scenario: str, difficulty: str, method: str, metric: str) -> float:
    values = frame.loc[
        (frame["scenario"] == scenario)
        & (frame["difficulty"] == difficulty)
        & (frame["method"] == method),
        metric,
    ]
    if values.empty:
        raise ValueError(f"Missing source values for {scenario}/{difficulty}/{method}/{metric}")
    return float(values.mean())


def construct_values(source: pd.DataFrame) -> pd.DataFrame:
    """Build illustrative cell means while enforcing all requested inequalities."""
    rows: list[dict[str, object]] = []
    difficulty_strength = {"简单": 0.90, "中等": 1.00, "困难": 1.15}
    scenario_strength = {"智慧工厂": 1.00, "智慧家居": 0.95, "应急救援": 1.10}

    for scenario in SCENARIOS:
        for difficulty in DIFFICULTIES:
            strength = difficulty_strength[difficulty] * scenario_strength[scenario]
            ours_accuracy = _cell_mean(source, scenario, difficulty, OURS, "completion_accuracy")
            ours_latency = _cell_mean(source, scenario, difficulty, OURS, "completion_latency_s")

            # Accuracy: lower every baseline relative to its original empirical mean,
            # then enforce a minimum 1.2 percentage-point separation in the order:
            # Ours > GMATANN > GRU-Minimal > StaticGraph > State-MLP > Rule-Latency.
            acc_gaps = {
                "GMATANN": 0.014,
                "GRU-Minimal": 0.032,
                "StaticGraph": 0.048,
                "State-MLP": 0.065,
                "Rule-Latency": 0.088,
            }
            accuracy = {OURS: ours_accuracy}
            previous = ours_accuracy
            for method in METHODS[1:]:
                original = _cell_mean(source, scenario, difficulty, method, "completion_accuracy")
                candidate = min(
                    ours_accuracy - acc_gaps[method] * strength,
                    original - 0.008,
                    previous - 0.012,
                )
                accuracy[method] = candidate
                previous = candidate

            # Latency (lower is better): raise every baseline relative to its original
            # mean and enforce a deliberately large GMATANN-to-GRU-Minimal gap:
            # Ours < Rule < GMATANN << GRU-Minimal < StaticGraph < State-MLP.
            original_latency = {
                method: _cell_mean(source, scenario, difficulty, method, "completion_latency_s")
                for method in METHODS[1:]
            }
            latency: dict[str, float] = {OURS: ours_latency}
            latency["Rule-Latency"] = max(
                ours_latency * (1.0 + 0.08 * strength),
                original_latency["Rule-Latency"] * 1.04,
            )
            latency["GMATANN"] = max(
                ours_latency * (1.0 + 0.18 * strength),
                original_latency["GMATANN"] * 1.05,
                latency["Rule-Latency"] + ours_latency * 0.08,
            )
            latency["GRU-Minimal"] = max(
                ours_latency * (1.0 + 0.48 * strength),
                original_latency["GRU-Minimal"] * 1.12,
                latency["GMATANN"] + ours_latency * 0.22,
            )
            latency["StaticGraph"] = max(
                ours_latency * (1.0 + 0.66 * strength),
                original_latency["StaticGraph"] * 1.14,
                latency["GRU-Minimal"] + ours_latency * 0.12,
            )
            latency["State-MLP"] = max(
                ours_latency * (1.0 + 0.88 * strength),
                original_latency["State-MLP"] * 1.16,
                latency["StaticGraph"] + ours_latency * 0.14,
            )

            for method in METHODS:
                rows.append(
                    {
                        "scenario": scenario,
                        "difficulty": difficulty,
                        "method": method,
                        "completion_accuracy": accuracy[method],
                        "completion_latency_s": latency[method],
                        "value_type": (
                            "empirical_mean_anchor" if method == OURS else "constructed_expected_trend"
                        ),
                    }
                )

    result = pd.DataFrame(rows)
    validate_ordering(source, result)
    return result


def validate_ordering(source: pd.DataFrame, result: pd.DataFrame) -> None:
    """Fail loudly if any requested ordering or source-direction constraint is violated."""
    accuracy_order = (
        OURS, "GMATANN", "GRU-Minimal", "StaticGraph", "State-MLP", "Rule-Latency"
    )
    latency_order = (
        OURS, "Rule-Latency", "GMATANN", "GRU-Minimal", "StaticGraph", "State-MLP"
    )
    for scenario in SCENARIOS:
        for difficulty in DIFFICULTIES:
            cell = result[(result.scenario == scenario) & (result.difficulty == difficulty)].set_index("method")
            acc = [float(cell.loc[method, "completion_accuracy"]) for method in accuracy_order]
            lat = [float(cell.loc[method, "completion_latency_s"]) for method in latency_order]
            if not all(left > right for left, right in zip(acc, acc[1:])):
                raise AssertionError(f"Accuracy order failed for {scenario}/{difficulty}: {acc}")
            if not all(left < right for left, right in zip(lat, lat[1:])):
                raise AssertionError(f"Latency order failed for {scenario}/{difficulty}: {lat}")
            for method in METHODS[1:]:
                original_acc = _cell_mean(source, scenario, difficulty, method, "completion_accuracy")
                original_lat = _cell_mean(source, scenario, difficulty, method, "completion_latency_s")
                if not float(cell.loc[method, "completion_accuracy"]) < original_acc:
                    raise AssertionError(f"Accuracy was not lowered for {scenario}/{difficulty}/{method}")
                if not float(cell.loc[method, "completion_latency_s"]) > original_lat:
                    raise AssertionError(f"Latency was not raised for {scenario}/{difficulty}/{method}")


def plot(values: pd.DataFrame) -> None:
    config_dir = OUTPUT / ".matplotlib"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(config_dir)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 14,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    def save(figure, stem: str) -> None:
        figure.savefig(OUTPUT / f"{stem}.png", dpi=360, bbox_inches="tight", facecolor="white")
        figure.savefig(OUTPUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
        plt.close(figure)

    specs = (
        (
            "completion_accuracy",
            "Average completion accuracy (%)",
            "01_expected_accuracy_scenario_difficulty_ILLUSTRATIVE",
            (60.0, 100.0),
        ),
        (
            "completion_latency_s",
            "Average DAG completion latency (s)",
            "02_expected_latency_scenario_difficulty_ILLUSTRATIVE",
            None,
        ),
    )
    x = np.arange(len(DIFFICULTIES))
    width = 0.135
    for metric, ylabel, stem, ylim in specs:
        figure, axes = plt.subplots(1, 3, figsize=(17.2, 5.6), sharey=True)
        for scenario_index, scenario in enumerate(SCENARIOS):
            axis = axes[scenario_index]
            for method_index, method in enumerate(METHODS):
                means = []
                for difficulty in DIFFICULTIES:
                    value = values.loc[
                        (values.scenario == scenario)
                        & (values.difficulty == difficulty)
                        & (values.method == method),
                        metric,
                    ].iloc[0]
                    means.append(float(value) * (100.0 if metric == "completion_accuracy" else 1.0))
                axis.bar(
                    x + (method_index - 2.5) * width,
                    means,
                    width,
                    color=COLORS[method],
                    edgecolor="#2f2f2f",
                    linewidth=0.55,
                    hatch=HATCHES[method],
                    label=method,
                )
            axis.set_title(SCENARIO_EN[scenario], fontweight="bold", pad=9)
            axis.set_xticks(x, [DIFFICULTY_EN[item] for item in DIFFICULTIES])
            axis.grid(axis="y", color="#d5d5d5", linewidth=0.7, alpha=0.8)
            axis.set_axisbelow(True)
            if ylim is not None:
                axis.set_ylim(*ylim)
        axes[0].set_ylabel(ylabel)
        if metric == "completion_latency_s":
            maximum = values[metric].max()
            axes[0].set_ylim(0.0, float(maximum) * 1.12)

        handles, labels = axes[0].get_legend_handles_labels()
        figure.suptitle(
            "Illustrative expected trend — NOT empirical results",
            fontsize=16,
            fontweight="bold",
            color="#9C2F2F",
            y=0.985,
        )
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.915),
            ncol=6,
            frameon=True,
            fancybox=False,
            edgecolor="#777777",
            fontsize=14,
            handlelength=2.0,
            handleheight=1.2,
            columnspacing=1.25,
            borderpad=0.55,
        )
        figure.text(
            0.5,
            0.015,
            "DAG-GRU: empirical group means; all baselines: constructed values for hypothesis illustration only.",
            ha="center",
            va="bottom",
            fontsize=10.5,
            color="#555555",
        )
        figure.subplots_adjust(left=0.065, right=0.992, bottom=0.13, top=0.75, wspace=0.12)
        save(figure, stem)


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(SOURCE, encoding="utf-8-sig")
    values = construct_values(source)
    values.to_csv(OUTPUT / "illustrative_expected_values.csv", index=False, encoding="utf-8-sig")
    plot(values)
    print(f"Wrote disclosed illustrative figures and values to: {OUTPUT}")


if __name__ == "__main__":
    main()
