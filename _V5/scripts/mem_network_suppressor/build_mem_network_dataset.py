from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def _resolve_v5_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "src").exists() and (parent / "chat" / "src").exists():
            return parent
    return here.parents[1]


V5_ROOT = _resolve_v5_root()
if str(V5_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(V5_ROOT / "src"))

from agentmemory_v3.suppressor.data_utils import (
    collect_user_queries,
    load_dense_index_from_config,
    load_feedback_rows,
    load_memory_text_by_id,
    load_v5_encoder,
    search_top_indices,
)
from agentmemory_v3.utils.io import write_jsonl


def _split_name(anchor_key: str, valid_ratio: float) -> str:
    value = int(hashlib.md5(str(anchor_key).encode("utf-8")).hexdigest()[:8], 16) % 10000
    threshold = int(max(0.0, min(1.0, float(valid_ratio))) * 10000.0)
    return "valid" if value < threshold else "train"


def _pick_with_fallback(
    *,
    ordered: list[str],
    fallback_pool: list[str],
    count: int,
    exclude: set[str],
    rng: random.Random,
) -> list[str]:
    need = max(0, int(count))
    out: list[str] = []
    seen: set[str] = set()
    for item in ordered:
        key = str(item or "")
        if not key or key in exclude or key in seen:
            continue
        seen.add(key)
        out.append(key)
        if len(out) >= need:
            return out
    if need <= len(out):
        return out
    sample_k = min(len(fallback_pool), max(0, need * 4))
    for item in rng.sample(fallback_pool, k=sample_k):
        key = str(item or "")
        if not key or key in exclude or key in seen:
            continue
        seen.add(key)
        out.append(key)
        if len(out) >= need:
            break
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build single-head dataset for mem-network suppressor.")
    parser.add_argument("--config", default="_V5/configs/default.yaml")
    parser.add_argument("--feedback-jsonl", default="_V5/chat/data/feedback/feedback_events.jsonl")
    parser.add_argument("--conversation-dir", default="_V5/chat/data/conversations")
    parser.add_argument("--bundle-path", default="data/V5/exports/chat_memory_bundle.jsonl")
    parser.add_argument("--output-dir", default="data/V5/mem_network_suppressor/dataset_v1")
    parser.add_argument("--neg-per-query", type=int, default=10)
    parser.add_argument("--neg-query-per-memory", type=int, default=4)
    parser.add_argument("--hard-neighbor-size", type=int, default=48)
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--max-feedback", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(int(args.seed))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dense_index = load_dense_index_from_config(args.config)
    encoder = load_v5_encoder(args.config)
    memory_ids = [str(item) for item in dense_index.artifact.mem_ids]
    memory_index_by_id = {memory_id: idx for idx, memory_id in enumerate(memory_ids)}
    memory_matrix = np.asarray(dense_index.matrix, dtype=np.float32)
    memory_text_by_id = load_memory_text_by_id(args.config, args.bundle_path)
    feedback_rows = load_feedback_rows(args.feedback_jsonl)
    if not feedback_rows:
        raise RuntimeError("no feedback rows loaded")

    # Single-head target: all feedback types are unified as suppress-positive anchors.
    anchor_map: dict[tuple[str, str, str], dict] = {}
    for src in feedback_rows:
        lane = str(src.get("lane") or "").strip().lower()
        q_text = str(src.get("q_text") or "")
        m_id = str(src.get("m_id") or "")
        if not q_text or not m_id:
            continue
        key = (q_text, m_id, lane)
        anchor_map[key] = {
            "feedback_id": str(src.get("feedback_id") or hashlib.md5(f"{q_text}|{m_id}|{lane}".encode("utf-8")).hexdigest()[:12]),
            "ts": str(src.get("ts") or ""),
            "q_text": q_text,
            "m_id": m_id,
            "m_text": str(src.get("m_text") or memory_text_by_id.get(m_id, "")),
            "feedback_type": "unrelated",
            "lane": lane,
            "effective_candidate_cache_memory_ids": [
                str(item) for item in (src.get("effective_candidate_cache_memory_ids") or []) if str(item)
            ],
        }
    anchors = sorted(
        anchor_map.values(),
        key=lambda item: (str(item.get("ts") or ""), str(item.get("feedback_id") or "")),
    )
    if int(args.max_feedback) > 0:
        anchors = anchors[-int(args.max_feedback) :]
    if not anchors:
        raise RuntimeError("no valid anchors after dedup")

    query_pool = collect_user_queries(args.conversation_dir)
    for item in anchors:
        q_text = str(item.get("q_text") or "")
        if q_text and q_text not in query_pool:
            query_pool.append(q_text)
    query_pool = list(dict.fromkeys(query_pool))
    if not query_pool:
        query_pool = [str(item.get("q_text") or "") for item in anchors if str(item.get("q_text") or "")]
    query_pool = [text for text in query_pool if text]
    if not query_pool:
        raise RuntimeError("query pool is empty")
    query_matrix = encoder.encode_query_texts(query_pool)
    query_index_by_text = {text: idx for idx, text in enumerate(query_pool)}

    all_memory_pool = [mid for mid in memory_ids if mid]
    if not all_memory_pool:
        raise RuntimeError("no memory ids available from dense index")

    rows_map: dict[tuple[str, str, str, str], dict] = {}

    def add_sample(
        *,
        anchor_feedback_id: str,
        split: str,
        lane: str,
        source_kind: str,
        q_text: str,
        m_id: str,
        m_text: str,
        label: int,
        group_id: str,
        pairwise_group: int,
        weight: float,
    ) -> None:
        key = (str(split), str(group_id), str(q_text), str(m_id))
        sample_id = hashlib.md5(
            f"{anchor_feedback_id}|{source_kind}|{q_text}|{m_id}|{group_id}".encode("utf-8")
        ).hexdigest()[:16]
        prev = rows_map.get(key)
        row = {
            "sample_id": sample_id,
            "split": split,
            "feedback_type": "unrelated",
            "label": int(label),
            "q_text": str(q_text),
            "m_id": str(m_id),
            "m_text": str(m_text),
            "source_kind": str(source_kind),
            "anchor_feedback_id": str(anchor_feedback_id),
            "group_id": str(group_id),
            "pairwise_group": int(pairwise_group),
            "lane": str(lane),
            "weight": float(weight),
        }
        if prev is None:
            rows_map[key] = row
            return
        prev["label"] = max(int(prev.get("label") or 0), int(row["label"]))
        prev["weight"] = max(float(prev.get("weight") or 0.0), float(row["weight"]))
        if int(row["pairwise_group"]) == 1:
            prev["pairwise_group"] = 1
        if str(prev.get("source_kind") or "") != "pos_anchor" and row["source_kind"] == "pos_anchor":
            prev["source_kind"] = "pos_anchor"

    for anchor in anchors:
        feedback_id = str(anchor.get("feedback_id") or "")
        q_text = str(anchor.get("q_text") or "")
        m_id = str(anchor.get("m_id") or "")
        lane = str(anchor.get("lane") or "")
        m_text = str(anchor.get("m_text") or memory_text_by_id.get(m_id, ""))
        if not feedback_id or not q_text or not m_id:
            continue
        split = _split_name(feedback_id, float(args.valid_ratio))
        rank_group_id = feedback_id
        query_neg_group_id = f"{feedback_id}::query_neg"

        add_sample(
            anchor_feedback_id=feedback_id,
            split=split,
            lane=lane,
            source_kind="pos_anchor",
            q_text=q_text,
            m_id=m_id,
            m_text=m_text,
            label=1,
            group_id=rank_group_id,
            pairwise_group=1,
            weight=1.0,
        )

        ordered_memory_neg: list[str] = [
            str(item)
            for item in (anchor.get("effective_candidate_cache_memory_ids") or [])
            if str(item) and str(item) != m_id
        ]
        mem_idx = memory_index_by_id.get(m_id)
        if mem_idx is not None:
            hard_indices = search_top_indices(
                memory_matrix,
                memory_matrix[int(mem_idx)],
                top_n=max(1, int(args.hard_neighbor_size) + 1),
                exclude={int(mem_idx)},
            )
            for idx in hard_indices:
                neg_mid = memory_ids[int(idx)]
                if neg_mid and neg_mid != m_id:
                    ordered_memory_neg.append(neg_mid)
        memory_neg_ids = _pick_with_fallback(
            ordered=ordered_memory_neg,
            fallback_pool=all_memory_pool,
            count=max(0, int(args.neg_per_query)),
            exclude={m_id},
            rng=rng,
        )
        for neg_mid in memory_neg_ids:
            add_sample(
                anchor_feedback_id=feedback_id,
                split=split,
                lane=lane,
                source_kind="neg_memory",
                q_text=q_text,
                m_id=neg_mid,
                m_text=str(memory_text_by_id.get(neg_mid, "")),
                label=0,
                group_id=rank_group_id,
                pairwise_group=1,
                weight=1.0,
            )

        ordered_query_neg: list[str] = []
        q_idx = query_index_by_text.get(q_text)
        if q_idx is not None:
            query_neighbors = search_top_indices(
                query_matrix,
                query_matrix[int(q_idx)],
                top_n=max(1, int(args.neg_query_per_memory) * 3),
                exclude={int(q_idx)},
            )
            ordered_query_neg.extend([query_pool[int(i)] for i in query_neighbors if query_pool[int(i)] != q_text])
        query_neg_texts = _pick_with_fallback(
            ordered=ordered_query_neg,
            fallback_pool=query_pool,
            count=max(0, int(args.neg_query_per_memory)),
            exclude={q_text},
            rng=rng,
        )
        for neg_q in query_neg_texts:
            add_sample(
                anchor_feedback_id=feedback_id,
                split=split,
                lane=lane,
                source_kind="neg_query",
                q_text=neg_q,
                m_id=m_id,
                m_text=m_text,
                label=0,
                group_id=query_neg_group_id,
                pairwise_group=0,
                weight=1.0,
            )

    rows = list(rows_map.values())
    train_rows = [row for row in rows if str(row.get("split") or "") == "train"]
    valid_rows = [row for row in rows if str(row.get("split") or "") == "valid"]
    write_jsonl(output_dir / "feedback_samples_train.jsonl", train_rows)
    write_jsonl(output_dir / "feedback_samples_valid.jsonl", valid_rows)

    def build_group_rows(split_rows: list[dict]) -> list[dict]:
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in split_rows:
            if int(row.get("pairwise_group") or 0) != 1:
                continue
            group_id = str(row.get("group_id") or "")
            if group_id:
                grouped[group_id].append(row)
        out: list[dict] = []
        for group_id, members in grouped.items():
            q_text = str(members[0].get("q_text") or "")
            lane = str(members[0].get("lane") or "")
            candidate_map: dict[str, dict] = {}
            for row in members:
                memory_id = str(row.get("m_id") or "")
                if not memory_id:
                    continue
                cand = {
                    "sample_id": str(row.get("sample_id") or ""),
                    "m_id": memory_id,
                    "m_text": str(row.get("m_text") or ""),
                    "q_text": str(row.get("q_text") or ""),
                    "label": int(row.get("label") or 0),
                    "weight": float(row.get("weight") or 1.0),
                    "source_kind": str(row.get("source_kind") or ""),
                }
                prev = candidate_map.get(memory_id)
                if prev is None:
                    candidate_map[memory_id] = cand
                else:
                    prev["label"] = max(int(prev.get("label") or 0), int(cand["label"]))
                    prev["weight"] = max(float(prev.get("weight") or 0.0), float(cand["weight"]))
            candidates = list(candidate_map.values())
            pos_count = sum(1 for item in candidates if int(item.get("label") or 0) == 1)
            neg_count = sum(1 for item in candidates if int(item.get("label") or 0) == 0)
            if pos_count <= 0 or neg_count <= 0:
                continue
            out.append(
                {
                    "group_id": group_id,
                    "anchor_feedback_id": str(members[0].get("anchor_feedback_id") or ""),
                    "feedback_type": "unrelated",
                    "lane": lane,
                    "q_text": q_text,
                    "positive_count": int(pos_count),
                    "negative_count": int(neg_count),
                    "candidates": candidates,
                }
            )
        return out

    train_groups = build_group_rows(train_rows)
    valid_groups = build_group_rows(valid_rows)
    write_jsonl(output_dir / "feedback_groups_train.jsonl", train_groups)
    write_jsonl(output_dir / "feedback_groups_valid.jsonl", valid_groups)

    feedback_store_rows = [
        {
            "feedback_id": str(item.get("feedback_id") or ""),
            "ts": str(item.get("ts") or ""),
            "q_text": str(item.get("q_text") or ""),
            "m_id": str(item.get("m_id") or ""),
            "m_text": str(item.get("m_text") or memory_text_by_id.get(str(item.get("m_id") or ""), "")),
            "feedback_type": "unrelated",
            "lane": str(item.get("lane") or ""),
        }
        for item in anchors
    ]
    write_jsonl(output_dir / "feedback_memory_store.jsonl", feedback_store_rows)

    manifest = {
        "version": "mem_network_dataset_v2",
        "feedback_jsonl": str(args.feedback_jsonl),
        "conversation_dir": str(args.conversation_dir),
        "bundle_path": str(args.bundle_path),
        "train_samples": int(len(train_rows)),
        "valid_samples": int(len(valid_rows)),
        "train_groups": int(len(train_groups)),
        "valid_groups": int(len(valid_groups)),
        "neg_per_query": int(max(0, int(args.neg_per_query))),
        "neg_query_per_memory": int(max(0, int(args.neg_query_per_memory))),
        "hard_neighbor_size": int(max(1, int(args.hard_neighbor_size))),
        "valid_ratio": float(max(0.0, min(1.0, float(args.valid_ratio)))),
        "seed": int(args.seed),
        "anchor_count": int(len(anchors)),
        "query_pool_size": int(len(query_pool)),
        "feedback_memory_rows": int(len(feedback_store_rows)),
        "single_head_unified_feedback_type": True,
    }
    (output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[v5][memnet][dataset]", manifest)


if __name__ == "__main__":
    main()
