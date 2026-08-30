import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "2.模型"), str(ROOT / "1.data")]
from baselines.runtime import train_cli  # noqa: E402

if __name__ == "__main__":
    train_cli("static_graph")
