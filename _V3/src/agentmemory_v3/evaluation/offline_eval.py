from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from agentmemory_v3.config import cfg_get, load_yaml_config, resolve_path
from agentmemory_v3.retrieval.hybrid_retriever import HybridRetriever
from agentmemory_v3.training.common import assign_split, resolve_query_text
from agentmemory_v3.utils.io import ensure_parent, read_jsonl, write_jsonl


@dataclass
class EvalConfig:
    config_path: str
    query_path: Path
    run_dir: Path
    split: str = "test"
    max_queries: int = 0
    split_seed: int = 11
    train_ratio: float = 0.8
    valid_ratio: float = 0.1
    top_ks: tuple[int, ...] = (1, 3, 5, 10)


def evaluate_offline(config: EvalConfig) -> dict:
    retriever = HybridRetriever.from_config(config.config_path)
    cfg = load_yaml_config(config.config_path)
    query_slots_path = resolve_path(cfg_get(cfg, "query.slots_path", "data/V3/processed/query_slots.jsonl"))
    query_slot_map = {str(row.get("query_id") or ""): row for row in read_jsonl(query_slots_path)} if query_slots_path.exists() else {}
    query_rows = list(read_jsonl(config.query_path))
    per_query_rows = []
    for row in query_rows:
        if config.split != "all":
            row_split = assign_split(
                str(row.get("query_id") or ""),
                seed=config.split_seed,
                train_ratio=config.train_ratio,
                valid_ratio=config.valid_ratio,
            )
            if row_split != config.split:
                continue
        query_text = resolve_query_text(row)
        if not query_text:
            continue
        positives = [str(item) for item in row.get("positives") or [] if str(item).strip()]
        if not positives:
            continue
        debug = retriever.retrieve_debug(
            query_text,
            top_k=max(config.top_ks) if config.top_ks else 10,
            query_slot_row=query_slot_map.get(str(row.get("query_id") or "")),
        )
        expanded_ids = debug["expanded_ids"]
        reranked_ids = [hit.memory_id for hit in debug["all_hits"]]
        positives_set = set(positives)
        pos_count = len(positives_set)
        record = {
            "query_id": row.get("query_id", ""),
            "split": config.split
            if config.split != "all"
            else assign_split(
                str(row.get("query_id") or ""),
                seed=config.split_seed,
                train_ratio=config.train_ratio,
                valid_ratio=config.valid_ratio,
            ),
            "positive_count": pos_count,
            "retrieval_recall_at_n": _recall(expanded_ids, positives_set, len(expanded_ids)),
            "coarse_rank_recall_at_pos_count": _recall(expanded_ids, positives_set, pos_count),
            "recall_at_pos_count": _recall(reranked_ids, positives_set, pos_count),
            "coarse_mrr": _mrr(expanded_ids, positives_set),
            "mrr": _mrr(reranked_ids, positives_set),
            "coarse_top1": 1.0 if expanded_ids[:1] and expanded_ids[0] in positives_set else 0.0,
            "top1": 1.0 if reranked_ids[:1] and reranked_ids[0] in positives_set else 0.0,
            "trace": debug["trace"],
            "coarse_top_ids": expanded_ids[:10],
            "rerank_top_ids": reranked_ids[:10],
        }
        for top_k in config.top_ks:
            coarse_key = f"coarse_rank_recall_at_{top_k}"
            rerank_key = f"recall_at_{top_k}"
            record[coarse_key] = _recall(expanded_ids, positives_set, top_k)
            record[rerank_key] = _recall(reranked_ids, positives_set, top_k)
            record[f"gain_at_{top_k}"] = record[rerank_key] - record[coarse_key]
        per_query_rows.append(record)
        if config.max_queries > 0 and len(per_query_rows) >= config.max_queries:
            break
        if len(per_query_rows) % 100 == 0:
            print(f"[v3][eval] {len(per_query_rows)} queries")
    if not per_query_rows:
        raise RuntimeError("no queries were evaluated")
    summary = _summarize_records(per_query_rows, config.top_ks)
    ensure_parent(config.run_dir / "metrics.json")
    write_jsonl(config.run_dir / "per_query.jsonl", per_query_rows)
    (config.run_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _summarize_records(records: list[dict], top_ks: tuple[int, ...]) -> dict:
    count = len(records)
    coarse = {
        "retrieval_recall_at_n": sum(row["retrieval_recall_at_n"] for row in records) / count,
        "coarse_rank_recall_at_pos_count": sum(row["coarse_rank_recall_at_pos_count"] for row in records) / count,
        "coarse_mrr": sum(row["coarse_mrr"] for row in records) / count,
        "coarse_top1": sum(row["coarse_top1"] for row in records) / count,
    }
    rerank = {
        "recall_at_pos_count": sum(row["recall_at_pos_count"] for row in records) / count,
        "mrr": sum(row["mrr"] for row in records) / count,
        "top1": sum(row["top1"] for row in records) / count,
    }
    gain = {}
    for top_k in top_ks:
        coarse_key = f"coarse_rank_recall_at_{top_k}"
        rerank_key = f"recall_at_{top_k}"
        gain_key = f"gain_at_{top_k}"
        coarse[coarse_key] = sum(row[coarse_key] for row in records) / count
        rerank[rerank_key] = sum(row[rerank_key] for row in records) / count
        gain[gain_key] = sum(row[gain_key] for row in records) / count
    return {"query_count": count, "coarse": coarse, "rerank": rerank, "gain": gain}


def _recall(ranked_ids: list[str], positives: set[str], top_k: int) -> float:
    if not positives:
        return 0.0
    return len(set(ranked_ids[: max(1, top_k)]) & positives) / len(positives)


def _mrr(ranked_ids: list[str], positives: set[str]) -> float:
    for rank, mem_id in enumerate(ranked_ids, start=1):
        if mem_id in positives:
            return 1.0 / rank
    return 0.0
