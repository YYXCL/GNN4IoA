"""生成可直接用于本项目 GNN 训练和推理的研究型数据集。"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ma_tgn.dataset import generate_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 6G 多智能体多级状态数据集")
    parser.add_argument("--output", type=Path, default=ROOT / "datasets" / "sixg_ma_state")
    parser.add_argument("--agents", type=int, default=100)
    parser.add_argument("--compute-nodes", type=int, default=12)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--update-ms", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--duration-minutes", type=int, default=30)
    args = parser.parse_args()
    result = generate_dataset(args.output, args.agents, args.compute_nodes, args.samples, args.seed, args.update_ms, args.duration_minutes)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
