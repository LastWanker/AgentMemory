"""Quick eval for bipartite reranker (runtime_v2)."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
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
    parser = argparse.ArgumentParser(description="Quick eval for followup_plus_chat bipartite reranker.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--dataset", default="followup_plus_chat")
    parser.add_argument("--encoder-backend", choices=("hf", "simple"), default="hf")
    parser.add_argument("--reranker-path", default=str(DEFAULT_WEIGHT))
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-mode", choices=("coarse", "lexical", "union"), default="coarse")
    parser.add_argument("--policies", default="half_hard")
    parser.add_argument("--cache-alias", default="users")
    parser.add_argument("--use-cache-signature", action="store_true")
    parser.add_argument("--max-eval-queries", type=int, default=1000)
    parser.add_argument("--eval-sample-mode", choices=("random", "head"), default="random")
    parser.add_argument("--eval-sample-seed", type=int, default=11)
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--scorer-batch-size", type=int, default=512)
    parser.add_argument("--eval-workers", type=int, default=4)
    parser.add_argument("--torch-num-threads", type=int, default=2)
    parser.add_argument("--include-mix", action="store_true", help="Also run mix(auto) in addition to S-only.")
    parser.add_argument("--hf-online", action="store_true")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = REPO_ROOT / "runs" / "rt_v2" / "followup_plus_chat_eval" / ts / "bipartite" / "eval"
    cache_alias = args.cache_alias
    if args.encoder_backend != "hf" and cache_alias == "users":
        cache_alias = "users_simple"
        print("[rt_v2][eval] info: simple backend detected, cache alias auto-switched to users_simple.")
    ablation_groups = "S-only,mix(auto)" if args.include_mix else "S-only"

    cmd = [
        resolve_python(),
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
        args.candidate_mode,
        "--policies",
        args.policies,
        "--cache-alias",
        cache_alias,
        "--ablation-groups",
        ablation_groups,
        "--use-learned-scorer",
        "--reranker-path",
        args.reranker_path,
        "--scorer-batch-size",
        str(args.scorer_batch_size),
        "--eval-workers",
        str(args.eval_workers),
        "--torch-num-threads",
        str(args.torch_num_threads),
        "--max-eval-queries",
        str(args.max_eval_queries),
        "--eval-sample-mode",
        args.eval_sample_mode,
        "--eval-sample-seed",
        str(args.eval_sample_seed),
        "--bootstrap",
        str(args.bootstrap),
        "--run-dir",
        str(run_dir),
    ]
    if not args.use_cache_signature:
        cmd.append("--no-cache-signature")
    if not args.hf_online:
        cmd.append("--hf-local-only")

    print("[rt_v2][eval] exec:", " ".join(cmd))
    start = time.perf_counter()
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    elapsed_s = time.perf_counter() - start
    print(f"[rt_v2][eval] run_dir={run_dir}")
    print(f"[rt_v2][eval] elapsed_s={elapsed_s:.2f}")


if __name__ == "__main__":
    main()

