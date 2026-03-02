from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path


V3_ROOT = Path(__file__).resolve().parents[1]
if str(V3_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(V3_ROOT / "src"))

from agentmemory_v3.config import cfg_get, load_yaml_config, resolve_path
from agentmemory_v3.retrieval.feature_builder import FEATURE_NAMES, build_candidate_feature_map, build_query_context, feature_vector_from_map
from agentmemory_v3.retrieval.hybrid_retriever import HybridRetriever
from agentmemory_v3.training.common import assign_split, resolve_query_text
from agentmemory_v3.utils.io import ensure_parent, read_jsonl, write_jsonl
from agentmemory_v3.utils.text import unique_keep_order


def build_training_samples(config_path: str | Path, query_in: Path, out_path: Path, *, max_queries: int = 0) -> dict:
    cfg = load_yaml_config(config_path)
    retriever = HybridRetriever.from_config(config_path)
    query_rows = list(read_jsonl(query_in))
    query_slots_path = resolve_path(cfg_get(cfg, "query.slots_path", "data/V3/processed/query_slots.jsonl"))
    query_slot_map = {str(row.get("query_id") or ""): row for row in read_jsonl(query_slots_path)} if query_slots_path.exists() else {}
    rng = random.Random(int(cfg_get(cfg, "training.split_seed", 11)))
    all_mem_ids = [row["memory_id"] for row in retriever.artifacts.memory_rows]
    split_seed = int(cfg_get(cfg, "training.split_seed", 11))
    train_ratio = float(cfg_get(cfg, "training.train_ratio", 0.8))
    valid_ratio = float(cfg_get(cfg, "training.valid_ratio", 0.1))
    min_neg_per_source = int(cfg_get(cfg, "training.min_neg_per_source", 4))
    max_neg_per_source = int(cfg_get(cfg, "training.max_neg_per_source", 24))
    near_ratio = float(cfg_get(cfg, "training.neg_ratio_near_miss", 0.5))
    random_ratio = float(cfg_get(cfg, "training.neg_ratio_random", 0.5))
    rows_out = []
    skipped = Counter()
    split_counts = Counter()
    total_list_len = 0
    total_positive_count = 0
    total_coarse_hit = 0
    for idx, row in enumerate(query_rows, start=1):
        query_text = resolve_query_text(row)
        positives = unique_keep_order([str(item) for item in row.get("positives") or [] if str(item).strip()])
        positives = [item for item in positives if item in retriever.artifacts.mem_id_to_idx]
        if not query_text:
            skipped["empty_query"] += 1
            continue
        if not positives:
            skipped["no_positive"] += 1
            continue
        pack = retriever.collect_candidate_pack(query_text)
        query_context = build_query_context(
            retriever.artifacts.dense_index,
            query_text,
            slot_row=query_slot_map.get(str(row.get("query_id") or "")),
        )
        positive_set = set(positives)
        positive_clusters = {retriever.artifacts.mem_id_to_cluster.get(mem_id, mem_id) for mem_id in positives}
        near_pool = [mem_id for mem_id in pack["expanded_ids"] if mem_id not in positive_set]
        random_pool = [
            mem_id
            for mem_id in all_mem_ids
            if mem_id not in positive_set and retriever.artifacts.mem_id_to_cluster.get(mem_id, mem_id) not in positive_clusters
        ]
        positive_count = len(positives)
        total_neg_budget = min(max_neg_per_source * 2, max(min_neg_per_source * 2, positive_count * 2))
        near_budget = min(len(near_pool), max(1, int(round(total_neg_budget * near_ratio))))
        random_budget = min(len(random_pool), max(1, int(round(total_neg_budget * random_ratio))))
        near_ids = near_pool[:near_budget]
        random_ids = rng.sample(random_pool, random_budget) if random_budget < len(random_pool) else list(random_pool)
        selected = set(positives) | set(near_ids) | set(random_ids)
        candidate_ids = [mem_id for mem_id in pack["expanded_ids"] if mem_id in selected]
        for mem_id in positives:
            if mem_id not in candidate_ids:
                candidate_ids.append(mem_id)
        for mem_id in random_ids:
            if mem_id not in candidate_ids:
                candidate_ids.append(mem_id)
        labels = [1 if mem_id in positive_set else 0 for mem_id in candidate_ids]
        feature_rows = []
        for mem_id in candidate_ids:
            feature_map = build_candidate_feature_map(
                retriever.artifacts,
                query_context=query_context,
                mem_id=mem_id,
                source_map=pack["source_map"],
                coarse_scores=pack["coarse_scores"],
            )
            feature_rows.append(feature_vector_from_map(feature_map))
        split = assign_split(
            str(row.get("query_id") or ""),
            seed=split_seed,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
        )
        split_counts[split] += 1
        total_list_len += len(candidate_ids)
        total_positive_count += positive_count
        total_coarse_hit += len(set(pack["expanded_ids"]) & positive_set)
        rows_out.append(
            {
                "query_id": str(row.get("query_id") or ""),
                "split": split,
                "text": query_text,
                "positives": positives,
                "candidate_ids": candidate_ids,
                "labels": labels,
                "features": feature_rows,
                "feature_names": list(FEATURE_NAMES),
                "trace": {
                    **pack["trace"],
                    "coarse_positive_hit_count": len(set(pack["expanded_ids"]) & positive_set),
                    "positive_count": positive_count,
                },
            }
        )
        if idx % 100 == 0:
            print(f"[v3][samples] {idx}/{len(query_rows)}")
        if max_queries > 0 and len(rows_out) >= max_queries:
            break
    write_jsonl(out_path, rows_out)
    summary = {
        "sample_count": len(rows_out),
        "avg_list_len": (total_list_len / len(rows_out)) if rows_out else 0.0,
        "avg_positive_count": (total_positive_count / len(rows_out)) if rows_out else 0.0,
        "avg_coarse_positive_hit_count": (total_coarse_hit / len(rows_out)) if rows_out else 0.0,
        "split_counts": dict(split_counts),
        "skipped": dict(skipped),
        "out": str(out_path),
    }
    ensure_parent(out_path.with_suffix(".summary.json"))
    out_path.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build V3 training samples.")
    parser.add_argument("--config", default="_V3/configs/default.yaml")
    parser.add_argument("--query-in", default="data/V3/processed/query.jsonl")
    parser.add_argument("--out", default="data/V3/training/train_samples.jsonl")
    parser.add_argument("--max-queries", type=int, default=0)
    args = parser.parse_args()
    result = build_training_samples(args.config, resolve_path(args.query_in), resolve_path(args.out), max_queries=args.max_queries)
    print(
        f"[v3] training samples count={result['sample_count']} avg_list_len={result['avg_list_len']:.2f} "
        f"-> {result['out']}"
    )


if __name__ == "__main__":
    main()
