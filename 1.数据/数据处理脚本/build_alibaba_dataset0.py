"""把 Alibaba GenAI 集群 trace 的多份指标 CSV 聚合/合并成一张宽表 dataset0.csv"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SOURCE_FILES = {
    "gpu_duty_cycle": "pod_gpu_duty_cycle_anon.csv",
    "gpu_memory_used_bytes": "pod_gpu_memory_used_bytes_anon.csv",
    "pod_memory_util": "pod_memory_util_anon.csv",
    "queue_rt": "queue_rt_raw_anon.csv",
    "queue_size": "queue_size_raw_anon.csv",
    "qps": "qps.csv",
    "model_predict": "model_predict_data_anon.csv",
    "pipeline_inference": "pipeline_inference_data_anon.csv",
}

OUTPUT_COLUMNS = [
    "timestamp_anon",
    "container_ip",
    "pod_gpu_duty_cycle_value",
    "pod_gpu_memory_used_bytes_value",
    "pod_memory_util_value",
    "queue_rt_raw_value",
    "qps_value",
    "qps_request_type",
    "queue_size_raw_value",
    "model_predict_data_value",
    "pipeline_inference_data_value",
]


def default_source_dir() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "原始数据"
        / "Alibaba-cluster-trace-v2026-GenAI"
        / "源数据"
    )


def default_output_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "原始数据"
        / "Alibaba-cluster-trace-v2026-GenAI"
        / "处理后的数据"
        / "dataset0.csv"
    )


def load_sources(source_dir: Path) -> dict[str, pd.DataFrame]:
    sources: dict[str, pd.DataFrame] = {}
    for metric, filename in SOURCE_FILES.items():
        path = source_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing source file: {path}")
        sources[metric] = pd.read_csv(path)
    return sources


def unique_metric(
    frame: pd.DataFrame,
    output_name: str,
    *,
    average_duplicate_values: bool = False,
) -> pd.DataFrame:
    """Return one value per timestamp/IP without allowing a merge to expand rows."""
    selected = frame[["timestamp_anon", "container_ip", "value"]].copy()
    duplicate_mask = selected.duplicated(["timestamp_anon", "container_ip"], keep=False)
    if duplicate_mask.any():
        if not average_duplicate_values:
            raise ValueError(f"Duplicate timestamp/IP keys found for {output_name}")
        selected = (
            selected.groupby(
                ["timestamp_anon", "container_ip"],
                as_index=False,
                sort=False,
                dropna=False,
            )["value"]
            .mean()
        )
    return selected.rename(columns={"value": output_name})


def build_dataset0(sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    base = unique_metric(
        sources["gpu_duty_cycle"],
        "pod_gpu_duty_cycle_value",
    )
    result = base

    joins = [
        (
            "gpu_memory_used_bytes",
            "pod_gpu_memory_used_bytes_value",
            False,
        ),
        (
            "pod_memory_util",
            "pod_memory_util_value",
            True,
        ),
        ("queue_rt", "queue_rt_raw_value", False),
    ]
    for source_name, output_name, average_duplicates in joins:
        metric = unique_metric(
            sources[source_name],
            output_name,
            average_duplicate_values=average_duplicates,
        )
        result = result.merge(
            metric,
            on=["timestamp_anon", "container_ip"],
            how="left",
            sort=False,
            validate="one_to_one",
        )

    qps = sources["qps"][
        ["timestamp_anon", "container_ip", "value", "request_type"]
    ].rename(
        columns={
            "value": "qps_value",
            "request_type": "qps_request_type",
        }
    )
    if qps.duplicated(["timestamp_anon", "container_ip"]).any():
        raise ValueError("qps contains duplicate timestamp/IP keys")
    result = result.merge(
        qps,
        on=["timestamp_anon", "container_ip"],
        how="left",
        sort=False,
        validate="one_to_one",
    )

    queue_size = sources["queue_size"][["timestamp_anon", "value"]].rename(
        columns={"value": "queue_size_raw_value"}
    )
    if queue_size.duplicated(["timestamp_anon"]).any():
        raise ValueError("queue_size contains duplicate timestamps")
    result = result.merge(
        queue_size,
        on="timestamp_anon",
        how="left",
        sort=False,
        validate="many_to_one",
    )

    final_joins = [
        ("model_predict", "model_predict_data_value"),
        ("pipeline_inference", "pipeline_inference_data_value"),
    ]
    for source_name, output_name in final_joins:
        metric = unique_metric(sources[source_name], output_name)
        result = result.merge(
            metric,
            on=["timestamp_anon", "container_ip"],
            how="left",
            sort=False,
            validate="one_to_one",
        )

    result = result[OUTPUT_COLUMNS]
    return result


def write_dataset(dataset0: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset0.to_csv(output_path, index=False, na_rep="<null>", lineterminator="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=default_source_dir())
    parser.add_argument("--output", type=Path, default=default_output_path())
    args = parser.parse_args()

    sources = load_sources(args.source_dir)
    dataset0 = build_dataset0(sources)
    write_dataset(dataset0, args.output)


if __name__ == "__main__":
    main()
