from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent


def _run(args: list[str]) -> None:
    subprocess.run(args, cwd=REPO_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V5 coarse-only smoke checks.")
    parser.add_argument("--config", default="_V5/configs/default.yaml")
    parser.add_argument("--query", default="黄山 无人机 消防员")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-queries", type=int, default=5)
    args = parser.parse_args()

    python_exe = str(REPO_ROOT / ".venv" / "Scripts" / "python.exe")
    if not Path(python_exe).exists():
        python_exe = sys.executable

    _run([python_exe, "_V5/scripts/doctor_data_chain.py", "--config", args.config])
    _run([python_exe, "_V5/scripts/runtime_retrieve_demo.py", "--config", args.config, "--query", *args.query.split(), "--top-k", str(args.top_k)])
    _run([python_exe, "_V5/scripts/eval_offline.py", "--config", args.config, "--max-queries", str(args.max_queries), "--split", "test"])


if __name__ == "__main__":
    main()
