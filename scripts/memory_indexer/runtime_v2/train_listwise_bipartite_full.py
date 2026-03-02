"""Train listwise bipartite reranker with production defaults."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default_v2_bipartite.yaml"
DEFAULT_WEIGHT = REPO_ROOT / "data" / "ModelWeights" / "listwise_bipartite_reranker.pt"


def resolve_python() -> str:
    venv_py = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_py.exists():
        return str(venv_py)
    return sys.executable


def main() -> None:
    parser = argparse.ArgumentParser(description="Train listwise bipartite reranker (runtime_v2).")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--dataset", default="followup_plus_chat")
    parser.add_argument("--encoder-backend", choices=("hf", "simple"), default="hf")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--cache-alias", default="users")
    parser.add_argument("--use-cache-signature", action="store_true")
    parser.add_argument("--hf-online", action="store_true")
    parser.add_argument("--save-path", default=str(DEFAULT_WEIGHT))
    parser.add_argument("--bipartite-tau", type=float, default=0.1)
    parser.add_argument("--bipartite-learnable-tau", action="store_true")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = REPO_ROOT / "runs" / "rt_v2" / "listwise_bipartite_train" / ts / "train"
    cmd = [
        resolve_python(),
        "-m",
        "scripts.memory_indexer.train_reranker",
        "--config",
        args.config,
        "--dataset",
        args.dataset,
        "--encoder-backend",
        args.encoder_backend,
        "--run-dir",
        str(run_dir),
        "--save-path",
        str(args.save_path),
        "--epochs",
        str(args.epochs),
        "--top-n",
        str(args.top_n),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--seed",
        str(args.seed),
        "--device",
        str(args.device),
        "--max-queries",
        str(args.max_queries),
        "--cache-alias",
        args.cache_alias,
        "--loss-type",
        "listwise",
        "--neg-strategy",
        "ranked",
        "--model-family",
        "bipartite",
        "--bipartite-tau",
        str(args.bipartite_tau),
        "--listwise-hard-constraint",
    ]
    if args.bipartite_learnable_tau:
        cmd.append("--bipartite-learnable-tau")
    else:
        cmd.append("--no-bipartite-learnable-tau")
    if not args.use_cache_signature:
        cmd.append("--no-cache-signature")
    if not args.hf_online:
        cmd.append("--hf-local-only")

    print("[rt_v2][train] exec:", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    print(f"[rt_v2][train] run_dir={run_dir}")
    print(f"[rt_v2][train] weight={args.save_path}")


if __name__ == "__main__":
    main()

