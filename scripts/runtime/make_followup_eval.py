"""Runtime helper: generate followup processed eval dataset."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_python() -> str:
    venv_py = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_py.exists():
        return str(venv_py)
    return sys.executable


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate data/Processed/eval_followup.jsonl")
    parser.add_argument("--log-path", help="Path to data_collector jsonl log")
    parser.add_argument("--from-eval-dataset", help="Synthesize from existing eval dataset, e.g. normal")
    parser.add_argument("--dataset", default="followup")
    parser.add_argument("--encoder-backend", choices=("hf", "simple"), default="hf")
    parser.add_argument("--candidate-mode", choices=("coarse", "lexical", "union"), default="union")
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    print("[runtime] Tip: use list_runs/select_best_run to inspect or pick best experiments.")
    if not args.log_path and not args.from_eval_dataset:
        raise ValueError("Either --log-path or --from-eval-dataset is required.")

    cmd = [
        resolve_python(),
        "-m",
        "scripts.generate_training_samples",
        "--export-eval",
        "--out",
        "data/Processed/eval_followup.jsonl",
        "--dataset",
        args.dataset,
        "--encoder-backend",
        args.encoder_backend,
        "--candidate-mode",
        args.candidate_mode,
        "--top-n",
        str(args.top_n),
    ]
    if args.log_path:
        cmd.extend(["--log-path", args.log_path])
    if args.from_eval_dataset:
        cmd.extend(["--from-eval-dataset", args.from_eval_dataset])
    print("[runtime] exec:", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
