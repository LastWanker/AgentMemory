"""One-click train + eval pipeline for bipartite reranker (runtime_v2)."""

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


def run(cmd: list[str]) -> None:
    print("[rt_v2][pipeline] exec:", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train+eval followup_plus_chat with bipartite reranker.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--dataset", default="followup_plus_chat")
    parser.add_argument("--encoder-backend", choices=("hf", "simple"), default="hf")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--cache-alias", default="users")
    parser.add_argument("--use-cache-signature", action="store_true")
    parser.add_argument("--bipartite-tau", type=float, default=0.1)
    parser.add_argument("--bipartite-learnable-tau", action="store_true")
    parser.add_argument("--max-eval-queries", type=int, default=1000)
    parser.add_argument("--eval-sample-mode", choices=("random", "head"), default="random")
    parser.add_argument("--eval-sample-seed", type=int, default=11)
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--include-mix", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--hf-online", action="store_true")
    parser.add_argument("--weight-path", default=str(DEFAULT_WEIGHT))
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = REPO_ROOT / "runs" / "rt_v2" / "followup_plus_chat_pipeline" / ts
    train_dir = root / "train"
    eval_dir = root / "eval"

    py = resolve_python()
    if not args.skip_train:
        train_cmd = [
            py,
            "-m",
            "scripts.memory_indexer.train_reranker",
            "--config",
            args.config,
            "--dataset",
            args.dataset,
            "--encoder-backend",
            args.encoder_backend,
            "--run-dir",
            str(train_dir),
            "--save-path",
            str(args.weight_path),
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
            train_cmd.append("--bipartite-learnable-tau")
        else:
            train_cmd.append("--no-bipartite-learnable-tau")
        if not args.use_cache_signature:
            train_cmd.append("--no-cache-signature")
        if not args.hf_online:
            train_cmd.append("--hf-local-only")
        run(train_cmd)

    cache_alias = args.cache_alias
    if args.encoder_backend != "hf" and cache_alias == "users":
        cache_alias = "users_simple"
    ablation_groups = "S-only,mix(auto)" if args.include_mix else "S-only"
    eval_cmd = [
        py,
        "-m",
        "scripts.memory_indexer.eval_router",
        "--dataset",
        args.dataset,
        "--encoder-backend",
        args.encoder_backend,
        "--config",
        args.config,
        "--top-n",
        str(args.top_n),
        "--top-k",
        str(args.top_k),
        "--candidate-mode",
        "coarse",
        "--policies",
        "half_hard",
        "--cache-alias",
        cache_alias,
        "--ablation-groups",
        ablation_groups,
        "--use-learned-scorer",
        "--reranker-path",
        str(args.weight_path),
        "--max-eval-queries",
        str(args.max_eval_queries),
        "--eval-sample-mode",
        args.eval_sample_mode,
        "--eval-sample-seed",
        str(args.eval_sample_seed),
        "--bootstrap",
        str(args.bootstrap),
        "--run-dir",
        str(eval_dir),
    ]
    if not args.use_cache_signature:
        eval_cmd.append("--no-cache-signature")
    if not args.hf_online:
        eval_cmd.append("--hf-local-only")
    run(eval_cmd)
    print(f"[rt_v2][pipeline] root={root}")
    print(f"[rt_v2][pipeline] weight={args.weight_path}")


if __name__ == "__main__":
    main()

