from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn


V3_ROOT = Path(__file__).resolve().parents[1]
if str(V3_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(V3_ROOT / "src"))

from agentmemory_v3.serving.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve V3 retriever.")
    parser.add_argument("--config", default="_V3/configs/default.yaml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8891)
    args = parser.parse_args()
    uvicorn.run(create_app(args.config), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
