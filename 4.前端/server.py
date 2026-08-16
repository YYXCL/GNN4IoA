"""无额外 Web 依赖的本地可视化服务。"""

from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

FRONTEND_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = FRONTEND_ROOT.parent
STATIC_FILES = FRONTEND_ROOT / "static"
ARTIFACTS = PROJECT_ROOT / "outputs" / "artifacts"


class DashboardHandler(BaseHTTPRequestHandler):
    def _json(self, path: Path) -> None:
        if not path.exists():
            self.send_response(404); self.send_header("Content-Type", "application/json; charset=utf-8"); self.end_headers()
            self.wfile.write(json.dumps({"error": "请先运行 3.training/train.py 生成训练工件"}, ensure_ascii=False).encode("utf-8")); return
        payload = path.read_bytes()
        self.send_response(200); self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store"); self.send_header("Content-Length", str(len(payload))); self.end_headers(); self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        if route == "/api/inference": return self._json(ARTIFACTS / "latest_inference.json")
        if route == "/api/dataset": return self._json(ARTIFACTS / "dataset_summary.json")
        relative = "index.html" if route == "/" else route.lstrip("/")
        target = (STATIC_FILES / relative).resolve()
        if STATIC_FILES.resolve() not in target.parents and target != STATIC_FILES.resolve():
            self.send_error(403); return
        if not target.exists() or not target.is_file():
            self.send_error(404); return
        content = target.read_bytes(); mime = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self.send_response(200); self.send_header("Content-Type", mime + ("; charset=utf-8" if mime.startswith("text/") else ""))
        self.send_header("Content-Length", str(len(content))); self.end_headers(); self.wfile.write(content)

    def log_message(self, format: str, *args: object) -> None:
        print("[dashboard] " + format % args)


def main() -> None:
    parser = argparse.ArgumentParser(description="启动 6G 多智能体状态建模可视化")
    parser.add_argument("--host", default="127.0.0.1"); parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Dashboard: http://{args.host}:{args.port}")
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()


if __name__ == "__main__":
    main()
