"""Runtime helper: select best followup HF run by Recall@5."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_python() -> str:
    venv_py = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_py.exists():
        return str(venv_py)
    return sys.executable


def main() -> None:
    print("[runtime] Tip: use list_runs/select_best_run to inspect or pick best experiments.")
    target = REPO_ROOT / "scripts" / "memory_indexer" / "select_best_run.py"
    if not target.exists():
        print(
            "[runtime] missing scripts/memory_indexer/select_best_run.py. "
            "Please restore this script or switch to your equivalent implementation."
        )
        raise SystemExit(1)

    cmd = [
        resolve_python(),
        "-m",
        "scripts.memory_indexer.select_best_run",
        "--dataset",
        "followup",
        "--encoder-backend",
        "hf",
        "--metric",
        "Recall@5",
    ]
    print("[runtime] exec:", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()



