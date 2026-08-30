"""单独重建“基础表 → 滑动窗口 JSONL”训练数据。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "2.模型"))
sys.path.insert(0, str(PROJECT_ROOT / "1.data"))

from ma_tgn_data.foundation import generate_foundation_dataset  # noqa: E402


def main() -> None:
    data_root = PROJECT_ROOT / "1.data"
    parser = argparse.ArgumentParser(description="把基础表转换为 MA-TGN 滑动窗口 JSONL")
    parser.add_argument("--data-root", type=Path, default=data_root)
    parser.add_argument("--output", type=Path, default=data_root / "datasets" / "sixg_ma_state")
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--candidate-count", type=int, default=48)
    parser.add_argument("--events", type=int, default=96)
    parser.add_argument("--neighbor-fanout", type=int, default=8)
    parser.add_argument("--samples", type=int, default=None)
    args = parser.parse_args()
    metadata = generate_foundation_dataset(
        args.data_root,
        args.output,
        seed=args.seed,
        window_size_timestamps=args.window_size,
        candidate_count=args.candidate_count,
        event_count=args.events,
        neighbor_fanout=args.neighbor_fanout,
        samples=args.samples,
    )
    print(json.dumps({
        "output": str(args.output),
        "samples": metadata["samples"],
        "splits": metadata["splits"],
        "window_size_timestamps": metadata["window_size_timestamps"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
