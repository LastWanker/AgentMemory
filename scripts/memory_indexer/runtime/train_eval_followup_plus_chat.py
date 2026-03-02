"""One-click pipeline: chat supplemental + followup merged train/eval."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"
DEFAULT_CHAT_CONFIG = REPO_ROOT / "configs" / "chat_memory.yaml"
DEFAULT_SUPPLEMENTAL = REPO_ROOT / "data" / "Processed" / "eval_chat_supplemental.api.jsonl"
PAIRWISE_WEIGHT = REPO_ROOT / "data" / "ModelWeights" / "pairwise_bipartite_reranker.pt"
LISTWISE_WEIGHT = REPO_ROOT / "data" / "ModelWeights" / "listwise_bipartite_reranker.pt"


def resolve_python() -> str:
    venv_py = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_py.exists():
        return str(venv_py)
    return sys.executable


def run(cmd: list[str]) -> None:
    print("[pipeline] exec:", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    cnt = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                cnt += 1
    return cnt


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline for followup + chat supplemental training/eval.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--chat-config", default=str(DEFAULT_CHAT_CONFIG))
    parser.add_argument("--dataset", default="followup_plus_chat")
    parser.add_argument("--supplemental-path", default=str(DEFAULT_SUPPLEMENTAL))
    parser.add_argument("--encoder-backend", choices=("hf", "simple"), default="hf")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--policies", default="half_hard")
    parser.add_argument("--candidate-mode", choices=("coarse", "lexical", "union"), default="coarse")
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--max-queries", type=int, default=0, help="Limit train queries; 0 means full merged eval.")
    parser.add_argument("--hf-online", action="store_true")
    parser.add_argument("--train-listwise", action="store_true")
    parser.add_argument("--eval-listwise", action="store_true")
    parser.add_argument("--skip-chat-refresh", action="store_true")
    parser.add_argument("--skip-dataset-merge", action="store_true")
    parser.add_argument("--consistency-pass", action="store_true", help="Enable second-pass consistency eval.")
    parser.add_argument("--cache-alias", default="users")
    parser.add_argument(
        "--use-cache-signature",
        action="store_true",
        help="Use signature-based cache validation (default off for user-friendly fixed cache files).",
    )
    args = parser.parse_args()

    py = resolve_python()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = REPO_ROOT / "runs" / "rt" / "followup_plus_chat_pipeline" / ts
    pair_train_dir = run_root / "pairwise" / "train"
    list_train_dir = run_root / "listwise" / "train"
    pair_eval_dir = run_root / "pairwise" / "eval"
    list_eval_dir = run_root / "listwise" / "eval"
    cache_alias = args.cache_alias
    if args.encoder_backend != "hf" and cache_alias == "users":
        cache_alias = "users_simple"
        print("[pipeline] info: simple backend detected, cache alias auto-switched to users_simple.")

    supplemental_path = Path(args.supplemental_path)
    if not args.skip_chat_refresh:
        run(
            [
                py,
                "-m",
                "scripts.memory_processer.runtime.build_chat_supplemental_eval",
                "--config",
                args.chat_config,
                "--non-preview-out",
                str(supplemental_path),
                "--skip-merge-queries",
            ]
        )

    if not args.skip_dataset_merge:
        run(
            [
                py,
                "-m",
                "scripts.memory_indexer.build_merged_dataset",
                "--dataset",
                args.dataset,
                "--eval-b",
                str(REPO_ROOT / "data" / "Processed" / "eval_chat_followup.jsonl"),
                "--eval-c",
                str(supplemental_path),
            ]
        )

    pair_cmd = [
        py,
        "-m",
        "scripts.memory_indexer.train_pairwise_reranker",
        "--config",
        args.config,
        "--dataset",
        args.dataset,
        "--encoder-backend",
        args.encoder_backend,
        "--run-dir",
        str(pair_train_dir),
        "--save-path",
        str(PAIRWISE_WEIGHT),
        "--epochs",
        str(args.epochs),
        "--top-n",
        str(args.top_n),
        "--seed",
        str(args.seed),
        "--device",
        str(args.device),
        "--max-queries",
        str(args.max_queries),
        "--cache-alias",
        cache_alias,
    ]
    if not args.use_cache_signature:
        pair_cmd.append("--no-cache-signature")
    if not args.hf_online:
        pair_cmd.append("--hf-local-only")
    run(pair_cmd)

    if args.train_listwise:
        list_cmd = [
            py,
            "-m",
            "scripts.memory_indexer.train_listwise_reranker",
            "--config",
            args.config,
            "--dataset",
            args.dataset,
            "--encoder-backend",
            args.encoder_backend,
            "--run-dir",
            str(list_train_dir),
            "--save-path",
            str(LISTWISE_WEIGHT),
            "--epochs",
            str(args.epochs),
            "--top-n",
            str(args.top_n),
            "--seed",
            str(args.seed),
            "--device",
            str(args.device),
            "--max-queries",
            str(args.max_queries),
            "--cache-alias",
            cache_alias,
        ]
        if not args.use_cache_signature:
            list_cmd.append("--no-cache-signature")
        if not args.hf_online:
            list_cmd.append("--hf-local-only")
        run(list_cmd)

    eval_common = [
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
        "--policies",
        args.policies,
        "--candidate-mode",
        args.candidate_mode,
        "--bootstrap",
        str(args.bootstrap),
        "--use-learned-scorer",
        "--ablation-groups",
        "S-only",
        "--cache-alias",
        cache_alias,
    ]
    if not args.use_cache_signature:
        eval_common.append("--no-cache-signature")
    if args.consistency_pass:
        eval_common.append("--consistency-pass")
    if not args.hf_online:
        eval_common.append("--hf-local-only")

    run(
        [
            py,
            "-m",
            "scripts.memory_indexer.eval_router",
            *eval_common,
            "--reranker-path",
            str(PAIRWISE_WEIGHT),
            "--run-dir",
            str(pair_eval_dir),
        ]
    )

    if args.eval_listwise or args.train_listwise:
        run(
            [
                py,
                "-m",
                "scripts.memory_indexer.eval_router",
                *eval_common,
                "--reranker-path",
                str(LISTWISE_WEIGHT),
                "--run-dir",
                str(list_eval_dir),
            ]
        )

    merged_memory = REPO_ROOT / "data" / "Processed" / f"memory_{args.dataset}.jsonl"
    merged_eval = REPO_ROOT / "data" / "Processed" / f"eval_{args.dataset}.jsonl"
    print("\n[pipeline] done")
    print(f"[pipeline] run_root={run_root}")
    print(f"[pipeline] supplemental_non_preview={supplemental_path} rows={count_jsonl(supplemental_path)}")
    print(f"[pipeline] merged_memory={merged_memory} rows={count_jsonl(merged_memory)}")
    print(f"[pipeline] merged_eval={merged_eval} rows={count_jsonl(merged_eval)}")
    print(f"[pipeline] pairwise_weight={PAIRWISE_WEIGHT}")
    if args.train_listwise or args.eval_listwise:
        print(f"[pipeline] listwise_weight={LISTWISE_WEIGHT}")


if __name__ == "__main__":
    main()
