"""Wrapper entry for pairwise reranker training."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SAVE_PATH = REPO_ROOT / "data" / "ModelWeights" / "pairwise_reranker.pt"


def backup_if_exists(path: Path) -> None:
    if not path.exists():
        return
    backup = path.parent / f"{path.stem}.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}{path.suffix}"
    shutil.copy2(path, backup)
    print(f"[pairwise-wrapper] backup => {backup}")


def build_cmd(args: argparse.Namespace, passthrough: List[str], save_path: Path) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "scripts.train_reranker",
        "--config",
        args.config,
        "--dataset",
        args.dataset,
        "--encoder-backend",
        args.encoder_backend,
        "--save-path",
        str(save_path),
    ]
    if args.run_dir:
        cmd.extend(["--run-dir", args.run_dir])
    cmd.extend(passthrough)
    # Force pairwise path as the last args.
    cmd.extend(["--loss-type", "pairwise", "--neg-strategy", "ranked"])
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Train pairwise reranker (ranked negatives).")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--dataset", default="followup")
    parser.add_argument("--encoder-backend", choices=("hf", "simple"), default="hf")
    parser.add_argument("--run-dir")
    parser.add_argument("--save-path", default=str(DEFAULT_SAVE_PATH))
    args, passthrough = parser.parse_known_args()

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    backup_if_exists(save_path)

    cmd = build_cmd(args, passthrough, save_path)
    print("[pairwise-wrapper] exec:", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
