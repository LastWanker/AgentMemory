from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path


V3_ROOT = Path(__file__).resolve().parents[1]
if str(V3_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(V3_ROOT / "src"))

from agentmemory_v3.config import cfg_get, load_yaml_config, resolve_path
from agentmemory_v3.evaluation.offline_eval import EvalConfig, evaluate_offline


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate V3 offline.")
    parser.add_argument("--config", default="_V3/configs/default.yaml")
    parser.add_argument("--query-in", default="data/V3/processed/query.jsonl")
    parser.add_argument("--split", default="test", choices=("train", "valid", "test", "all"))
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--top-ks", default="")
    args = parser.parse_args()
    cfg = load_yaml_config(args.config)
    top_ks = args.top_ks.strip()
    if top_ks:
        parsed_top_ks = tuple(int(part) for part in top_ks.split(",") if part.strip())
    else:
        parsed_top_ks = tuple(int(item) for item in cfg_get(cfg, "reranker.top_k_metrics", [1, 3, 5, 10]))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_cfg = EvalConfig(
        config_path=args.config,
        query_path=resolve_path(args.query_in),
        run_dir=resolve_path(Path("_V3") / "runs" / "eval_offline" / timestamp),
        split=args.split,
        max_queries=int(args.max_queries),
        split_seed=int(cfg_get(cfg, "training.split_seed", 11)),
        train_ratio=float(cfg_get(cfg, "training.train_ratio", 0.8)),
        valid_ratio=float(cfg_get(cfg, "training.valid_ratio", 0.1)),
        top_ks=parsed_top_ks,
    )
    result = evaluate_offline(eval_cfg)
    print(f"[v3] eval split={args.split} queries={result['query_count']} -> {eval_cfg.run_dir}")
    print(
        f"[v3] coarse RetrievalRecall@N={result['coarse']['retrieval_recall_at_n']:.4f} "
        f"rerank MRR={result['rerank']['mrr']:.4f} Top1={result['rerank']['top1']:.4f}"
    )


if __name__ == "__main__":
    main()
