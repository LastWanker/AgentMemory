"""Runtime helper: run listwise vs pairwise ablation on followup with HF backend."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_python() -> str:
    venv_py = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_py.exists():
        return str(venv_py)
    return sys.executable


def main() -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / "rt" / "ablate_followup" / ts
    print("[runtime] Tip: use list_runs/select_best_run to inspect or pick best experiments.")
    cmd = [
        resolve_python(),
        "-m",
        "scripts.memory_indexer.run_ablation_matrix",
        "--dataset",
        "followup",
        "--encoder-backend",
        "hf",
        "--suite",
        "listwise_vs_pairwise",
        "--seeds",
        "11",
        "--bootstrap",
        "200",
        "--run-dir",
        str(run_dir),
    ]
    print("[runtime] exec:", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()


