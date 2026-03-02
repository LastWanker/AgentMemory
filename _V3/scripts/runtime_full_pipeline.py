from __future__ import annotations

import argparse
import sys
from pathlib import Path


V3_ROOT = Path(__file__).resolve().parents[1]
if str(V3_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(V3_ROOT / "src"))

from build_dataset import build_dataset
from build_hybrid_index import build_hybrid_index
from build_query_slots import build_query_slots
from build_slots import build_slots
from build_training_samples import build_training_samples
from agentmemory_v3.config import cfg_get, load_yaml_config, resolve_path
from agentmemory_v3.evaluation.offline_eval import EvalConfig, evaluate_offline
from agentmemory_v3.training.trainer import TrainConfig, train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="V3 full runtime pipeline.")
    parser.add_argument("--mode", choices=("prepare", "train", "eval", "all"), default="all")
    parser.add_argument("--config", default="_V3/configs/default.yaml")
    parser.add_argument("--memory-in", default="data/Processed/memory_followup_plus_chat.jsonl")
    parser.add_argument("--eval-in", default="data/Processed/eval_followup_plus_chat.jsonl")
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--skip-slots", action="store_true")
    parser.add_argument("--slots-limit", type=int, default=0)
    args = parser.parse_args()
    cfg = load_yaml_config(args.config)
    root_dir = resolve_path(cfg_get(cfg, "data.root_dir", "data/V3"))
    if args.mode in ("prepare", "all"):
        build_dataset(resolve_path(args.memory_in), resolve_path(args.eval_in), root_dir)
        slots_path = root_dir / "processed" / "memory_slots.jsonl"
        if not args.skip_slots:
            build_slots(root_dir / "processed" / "memory.jsonl", slots_path, limit=int(args.slots_limit), resume=True)
            build_query_slots(root_dir / "processed" / "query.jsonl", root_dir / "processed" / "query_slots.jsonl", resume=True)
        if not slots_path.exists():
            fallback_slots = root_dir / "processed" / "memory_slots_smoke.jsonl"
            if fallback_slots.exists():
                slots_path = fallback_slots
            else:
                raise RuntimeError(f"slots file not found: {slots_path}")
        build_hybrid_index(root_dir / "processed" / "memory.jsonl", slots_path, root_dir)
        build_training_samples(args.config, root_dir / "processed" / "query.jsonl", root_dir / "training" / "train_samples.jsonl", max_queries=args.max_queries)
        print(f"[v3][runtime] prepare done -> {root_dir}")
    if args.mode in ("train", "all"):
        train_model(
            TrainConfig(
                sample_path=root_dir / "training" / "train_samples.jsonl",
                model_path=resolve_path(cfg_get(cfg, "reranker.model_path", "data/V3/models/reranker.pt")),
                run_dir=resolve_path(Path("_V3") / "runs" / "reranker_train" / "runtime_full"),
                epochs=int(cfg_get(cfg, "training.epochs", 8)),
                batch_size=int(cfg_get(cfg, "training.batch_size", 32)),
                learning_rate=float(cfg_get(cfg, "training.learning_rate", 1e-3)),
                hidden_dim=int(cfg_get(cfg, "reranker.hidden_dim", 64)),
                dropout=float(cfg_get(cfg, "reranker.dropout", 0.1)),
            )
        )
        print("[v3][runtime] train done")
    if args.mode in ("eval", "all"):
        result = evaluate_offline(
            EvalConfig(
                config_path=args.config,
                query_path=root_dir / "processed" / "query.jsonl",
                run_dir=resolve_path(Path("_V3") / "runs" / "eval_offline" / "runtime_full"),
                split="test",
                max_queries=args.max_queries,
                split_seed=int(cfg_get(cfg, "training.split_seed", 11)),
                train_ratio=float(cfg_get(cfg, "training.train_ratio", 0.8)),
                valid_ratio=float(cfg_get(cfg, "training.valid_ratio", 0.1)),
                top_ks=tuple(int(item) for item in cfg_get(cfg, "reranker.top_k_metrics", [1, 3, 5, 10])),
            )
        )
        print(
            f"[v3][runtime] eval done queries={result['query_count']} "
            f"RetrievalRecall@N={result['coarse']['retrieval_recall_at_n']:.4f} "
            f"MRR={result['rerank']['mrr']:.4f}"
        )


if __name__ == "__main__":
    main()
