from __future__ import annotations

import argparse
import sys
from pathlib import Path


V3_ROOT = Path(__file__).resolve().parents[1]
if str(V3_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(V3_ROOT / "src"))

from build_dataset import build_dataset
from build_e5_cache import build_e5_cache
from build_hybrid_index import build_hybrid_index
from export_chat_bundle import export_chat_bundle
from agentmemory_v3.config import cfg_get, load_yaml_config, resolve_path
from agentmemory_v3.evaluation.offline_eval import EvalConfig, evaluate_offline


def main() -> None:
    parser = argparse.ArgumentParser(description="V5 coarse-only runtime pipeline.")
    parser.add_argument("--mode", choices=("prepare", "eval", "all"), default="all")
    parser.add_argument("--config", default="_V5/configs/default.yaml")
    parser.add_argument("--memory-in", default="data/Processed/memory_followup_plus_chat.jsonl")
    parser.add_argument("--eval-in", default="data/Processed/eval_followup_plus_chat.jsonl")
    parser.add_argument("--max-queries", type=int, default=0)
    args = parser.parse_args()
    cfg = load_yaml_config(args.config)
    root_dir = resolve_path(cfg_get(cfg, "data.root_dir", "data/V5"))
    if args.mode in ("prepare", "all"):
        build_dataset(resolve_path(args.memory_in), resolve_path(args.eval_in), root_dir)
        build_e5_cache(args.config)
        build_hybrid_index(root_dir / "processed" / "memory.jsonl", root_dir, config_path=args.config)
        export_chat_bundle(
            root_dir / "processed" / "memory.jsonl",
            root_dir / "exports" / "chat_memory_bundle.jsonl",
        )
        print(f"[v5][runtime] coarse prepare done -> {root_dir}")
    if args.mode in ("eval", "all"):
        result = evaluate_offline(
            EvalConfig(
                config_path=args.config,
                query_path=root_dir / "processed" / "query.jsonl",
                run_dir=resolve_path(Path("_V5") / "runs" / "eval_offline" / "runtime_full"),
                split="test",
                max_queries=args.max_queries,
                split_seed=int(cfg_get(cfg, "training.split_seed", 11)),
                train_ratio=float(cfg_get(cfg, "training.train_ratio", 0.8)),
                valid_ratio=float(cfg_get(cfg, "training.valid_ratio", 0.1)),
                top_ks=tuple(int(item) for item in cfg_get(cfg, "evaluation.top_k_metrics", [1, 3, 5, 10])),
            )
        )
        print(
            f"[v5][runtime] coarse eval done queries={result['query_count']} "
            f"RetrievalRecall@N={result['coarse']['retrieval_recall_at_n']:.4f} "
            f"MRR={result['coarse']['mrr']:.4f}"
        )


if __name__ == "__main__":
    main()
