"""Inspect and merge Alibaba GenAI cluster trace metrics into dataset0."""

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
        / "raw"
        / "Alibaba-cluster-trace-v2026-GenAI"
        / "源数据"
    )


def default_output_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "raw"
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


def validate_timestamps(name: str, frame: pd.DataFrame) -> None:
    timestamp = pd.to_numeric(frame["timestamp_anon"], errors="raise")
    fractional = (timestamp % 1).abs()
    if (fractional > 1e-9).any():
        examples = timestamp[fractional > 1e-9].head().tolist()
        raise ValueError(f"{name} contains fractional timestamps: {examples}")


def print_analysis(sources: dict[str, pd.DataFrame]) -> None:
    base = sources["gpu_duty_cycle"]
    base_keys = pd.MultiIndex.from_frame(base[["timestamp_anon", "container_ip"]])

    print("metric,rows,timestamps,non_null_ips,duplicate_timestamp_ip_rows,base_key_matches")
    for name, frame in sources.items():
        validate_timestamps(name, frame)
        duplicate_rows = int(frame.duplicated(["timestamp_anon", "container_ip"], keep=False).sum())
        keys = pd.MultiIndex.from_frame(frame[["timestamp_anon", "container_ip"]])
        match_count = int(base_keys.isin(keys).sum())
        print(
            f"{name},{len(frame)},{frame['timestamp_anon'].nunique()},"
            f"{frame['container_ip'].nunique(dropna=True)},{duplicate_rows},{match_count}"
        )
        if duplicate_rows:
            duplicate_detail = frame[
                frame.duplicated(["timestamp_anon", "container_ip"], keep=False)
            ].sort_values(["timestamp_anon", "container_ip"])
            print(f"{name}_duplicate_key_detail:")
            print(duplicate_detail.to_string(index=False))

    base_counts = base.groupby("timestamp_anon", dropna=False).size()
    print(
        "base_rows_per_timestamp: "
        f"min={base_counts.min()}, max={base_counts.max()}, "
        f"distinct_counts={sorted(base_counts.unique().tolist())}"
    )
    print(f"base_duplicate_keys={int(base.duplicated(['timestamp_anon', 'container_ip']).sum())}")

    qps = sources["qps"]
    print("qps_request_type_counts:")
    print(qps["request_type"].value_counts(dropna=False).to_string())
    print("qps_duplicate_timestamp_ip_request_type_rows=")
    print(int(qps.duplicated(["timestamp_anon", "container_ip", "request_type"], keep=False).sum()))
    print("qps_non_null_ip_by_request_type:")
    print(qps.groupby("request_type")["container_ip"].apply(lambda x: int(x.notna().sum())).to_string())

    queue_size = sources["queue_size"]
    queue_size_timestamps = pd.Index(queue_size["timestamp_anon"])
    print(
        "queue_size_rows_per_timestamp: "
        f"min={queue_size.groupby('timestamp_anon').size().min()}, "
        f"max={queue_size.groupby('timestamp_anon').size().max()}, "
        f"non_null_ips={int(queue_size['container_ip'].notna().sum())}, "
        f"base_rows_with_timestamp_match={int(base['timestamp_anon'].isin(queue_size_timestamps).sum())}"
    )


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
    for name, frame in sources.items():
        validate_timestamps(name, frame)

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
    if len(result) != len(base):
        raise AssertionError("The merge changed the number of base rows")
    if result.duplicated(["timestamp_anon", "container_ip"]).any():
        raise AssertionError("dataset0 contains duplicate timestamp/IP keys")
    if not result[["timestamp_anon", "container_ip"]].equals(
        base[["timestamp_anon", "container_ip"]]
    ):
        raise AssertionError("The merge changed the order or identity of base keys")
    return result


def write_and_verify(dataset0: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset0.to_csv(output_path, index=False, na_rep="<null>", lineterminator="\n")

    reread = pd.read_csv(output_path, keep_default_na=False, dtype=str)
    if len(reread) != len(dataset0):
        raise AssertionError("Written CSV row count differs from the in-memory result")
    if reread.columns.tolist() != OUTPUT_COLUMNS:
        raise AssertionError("Written CSV headers differ from the requested schema")
    if reread.duplicated(["timestamp_anon", "container_ip"]).any():
        raise AssertionError("Written CSV contains duplicate timestamp/IP keys")
    if (reread == "").any().any():
        raise AssertionError("Written CSV contains blank fields instead of <null>")
    expected_nulls = dataset0.isna().sum()
    written_null_markers = (reread == "<null>").sum()
    if not written_null_markers.equals(expected_nulls):
        raise AssertionError("Written <null> marker counts do not match missing values")
    qps_types = set(dataset0["qps_request_type"].dropna().unique().tolist())
    if qps_types != {"Generative Requests"}:
        raise AssertionError(f"Unexpected matched qps request types: {qps_types}")

    print(f"output={output_path}")
    print(f"rows={len(dataset0)}")
    print(f"columns={len(dataset0.columns)}")
    print("non_null_counts:")
    print(dataset0.notna().sum().to_string())
    print("null_counts:")
    print(dataset0.isna().sum().to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=default_source_dir())
    parser.add_argument("--output", type=Path, default=default_output_path())
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()

    sources = load_sources(args.source_dir)
    if args.analyze:
        print_analysis(sources)
        return
    dataset0 = build_dataset0(sources)
    write_and_verify(dataset0, args.output)


if __name__ == "__main__":
    main()
