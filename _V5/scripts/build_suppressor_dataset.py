from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


V5_ROOT = Path(__file__).resolve().parents[1]
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


def _split_name(feedback_id: str) -> str:
    value = int(hashlib.md5(str(feedback_id).encode("utf-8")).hexdigest()[:8], 16) % 100
    return "valid" if value < 15 else "train"


def _sample_random_pairs(query_pool: list[str], memory_ids: list[str], memory_text_by_id: dict[str, str], rand_size: int, rng: random.Random):
    out = []
    if not query_pool or not memory_ids:
        return out
    for _ in range(max(0, int(rand_size))):
        q_text = query_pool[rng.randrange(len(query_pool))]
        m_id = memory_ids[rng.randrange(len(memory_ids))]
        out.append((q_text, m_id, memory_text_by_id.get(m_id, "")))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build suppressor train/valid dataset from feedback and V5 history.")
    parser.add_argument("--config", default="_V5/configs/default.yaml")
    parser.add_argument("--feedback-jsonl", required=True)
    parser.add_argument("--conversation-dir", default="_V5/chat/data/conversations")
    parser.add_argument("--bundle-path", default="data/V5/exports/chat_memory_bundle.jsonl")
    parser.add_argument("--output-dir", default="data/V5/suppressor")
    parser.add_argument("--nb-size", type=int, default=20)
    parser.add_argument("--rand-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--w-pos", type=float, default=1.0)
    parser.add_argument("--w-neg-out", type=float, default=1.2)
    parser.add_argument("--w-neg-in", type=float, default=1.4)
    parser.add_argument("--w-neg-rand", type=float, default=2.0)
    parser.add_argument("--w-neg-neighbors", type=float, default=1.2)
    parser.add_argument("--w-neg-named", type=float, default=2.5)
    args = parser.parse_args()

    rng = random.Random(int(args.seed))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feedback_rows = load_feedback_rows(args.feedback_jsonl)
    if not feedback_rows:
        raise RuntimeError("no valid feedback rows")

    memory_text_by_id = load_memory_text_by_id(args.config, args.bundle_path)
    dense_index = load_dense_index_from_config(args.config)
    encoder = load_v5_encoder(args.config)
    memory_ids = [str(item) for item in dense_index.artifact.mem_ids]
    memory_index_by_id = {memory_id: idx for idx, memory_id in enumerate(memory_ids)}
    memory_matrix = np.asarray(dense_index.matrix, dtype=np.float32)

    query_pool = collect_user_queries(args.conversation_dir)
    for row in feedback_rows:
        if row["q_text"] not in query_pool:
            query_pool.append(row["q_text"])

    query_pool = list(dict.fromkeys(query_pool))
    query_matrix = encoder.encode_query_texts(query_pool) if query_pool else np.zeros((0, memory_matrix.shape[1]), dtype=np.float32)
    query_index_by_text = {text: idx for idx, text in enumerate(query_pool)}
    source_weight_map = {
        "pos": float(args.w_pos),
        "neg_out": float(args.w_neg_out),
        "neg_in": float(args.w_neg_in),
        "neg_rand": float(args.w_neg_rand),
        "neg_neighbors": float(args.w_neg_neighbors),
        "neg_named": float(args.w_neg_named),
    }

    rows: list[dict] = []
    for item in feedback_rows:
        feedback_id = item["feedback_id"]
        split = _split_name(feedback_id)
        q_text = item["q_text"]
        m_id = item["m_id"]
        m_text = item["m_text"] or memory_text_by_id.get(m_id, "")
        if m_id not in memory_index_by_id:
            continue

        query_idx = query_index_by_text.get(q_text)
        query_vec = query_matrix[int(query_idx)] if query_idx is not None else encoder.encode_query_texts([q_text])[0]
        memory_idx = memory_index_by_id[m_id]
        memory_vec = memory_matrix[int(memory_idx)]

        memory_nb_indices = search_top_indices(memory_matrix, memory_vec, top_n=max(1, int(args.nb_size) + 1), exclude={memory_idx})
        memory_nb_ids = [memory_ids[idx] for idx in memory_nb_indices[: max(1, int(args.nb_size))]]
        candidate_cache_nb_ids = [
            memory_id
            for memory_id in (item.get("effective_candidate_cache_memory_ids") or [])
            if str(memory_id) in memory_index_by_id and str(memory_id) != m_id
        ]
        if candidate_cache_nb_ids:
            merged = []
            seen = set()
            for memory_id in candidate_cache_nb_ids + memory_nb_ids:
                if memory_id in seen:
                    continue
                seen.add(memory_id)
                merged.append(memory_id)
            memory_nb_ids = merged[: max(1, int(args.nb_size))]

        query_nb_indices = search_top_indices(query_matrix, query_vec, top_n=max(1, int(args.nb_size) + 1), exclude={query_idx} if query_idx is not None else None)
        query_nb_texts = [query_pool[idx] for idx in query_nb_indices if query_pool[idx] != q_text][: max(1, int(args.nb_size))]
        if not query_nb_texts:
            query_nb_texts = [q_text]

        def add_sample(label: int, source_kind: str, sample_q: str, sample_m_id: str, sample_m_text: str) -> None:
            rows.append(
                {
                    "sample_id": hashlib.md5(
                        f"{feedback_id}|{source_kind}|{sample_q}|{sample_m_id}".encode("utf-8")
                    ).hexdigest()[:16],
                    "split": split,
                    "feedback_type": item["feedback_type"],
                    "label": int(label),
                    "q_text": sample_q,
                    "m_id": sample_m_id,
                    "m_text": sample_m_text,
                    "source_kind": source_kind,
                    "anchor_feedback_id": feedback_id,
                    "group_id": feedback_id,
                    "lane": str(item.get("lane") or "").strip().lower(),
                    "weight": float(source_weight_map.get(source_kind, 1.0)),
                }
            )

        if item["feedback_type"] == "unrelated":
            add_sample(1, "pos", q_text, m_id, m_text)
            for nb_id in memory_nb_ids:
                add_sample(0, "neg_out", q_text, nb_id, memory_text_by_id.get(nb_id, ""))
            for nb_q in query_nb_texts:
                add_sample(0, "neg_in", nb_q, m_id, m_text)
        else:
            for nb_q in query_nb_texts:
                add_sample(1, "pos", nb_q, m_id, m_text)
            add_sample(0, "neg_named", m_text or q_text, m_id, m_text)
            for nb_q in query_nb_texts:
                for nb_id in memory_nb_ids[: max(1, min(5, int(args.nb_size)))]:
                    add_sample(0, "neg_neighbors", nb_q, nb_id, memory_text_by_id.get(nb_id, ""))

        for rand_q, rand_m_id, rand_m_text in _sample_random_pairs(
            query_pool,
            memory_ids,
            memory_text_by_id,
            rand_size=max(0, int(args.rand_size)),
            rng=rng,
        ):
            add_sample(0, "neg_rand", rand_q, rand_m_id, rand_m_text)

    train_rows = [row for row in rows if row["split"] == "train"]
    valid_rows = [row for row in rows if row["split"] == "valid"]
    write_jsonl(output_dir / "feedback_samples_train.jsonl", train_rows)
    write_jsonl(output_dir / "feedback_samples_valid.jsonl", valid_rows)

    def build_group_rows(split_rows: list[dict]) -> list[dict]:
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in split_rows:
            grouped[str(row.get("group_id") or "")].append(row)
        out: list[dict] = []
        for group_id, members in grouped.items():
            if not group_id or not members:
                continue
            q_text = str(members[0].get("q_text") or "")
            feedback_type = str(members[0].get("feedback_type") or "")
            lane = str(members[0].get("lane") or "")
            candidate_map: dict[tuple[str, str], dict] = {}
            for row in members:
                key = (str(row.get("m_id") or ""), str(row.get("q_text") or ""))
                prev = candidate_map.get(key)
                label = int(row.get("label") or 0)
                weight = float(row.get("weight") or 1.0)
                if prev is None:
                    candidate_map[key] = {
                        "sample_id": str(row.get("sample_id") or ""),
                        "m_id": key[0],
                        "m_text": str(row.get("m_text") or ""),
                        "q_text": key[1],
                        "label": label,
                        "weight": weight,
                        "source_kind": str(row.get("source_kind") or ""),
                    }
                else:
                    prev["label"] = max(int(prev.get("label") or 0), label)
                    prev["weight"] = max(float(prev.get("weight") or 0.0), weight)
            candidates = list(candidate_map.values())
            pos_count = sum(1 for row in candidates if int(row.get("label") or 0) == 1)
            neg_count = sum(1 for row in candidates if int(row.get("label") or 0) == 0)
            if pos_count == 0 or neg_count == 0:
                continue
            out.append(
                {
                    "group_id": group_id,
                    "anchor_feedback_id": str(members[0].get("anchor_feedback_id") or ""),
                    "feedback_type": feedback_type,
                    "lane": lane,
                    "q_text": q_text,
                    "positive_count": pos_count,
                    "negative_count": neg_count,
                    "candidates": candidates,
                }
            )
        return out

    train_groups = build_group_rows(train_rows)
    valid_groups = build_group_rows(valid_rows)
    write_jsonl(output_dir / "feedback_groups_train.jsonl", train_groups)
    write_jsonl(output_dir / "feedback_groups_valid.jsonl", valid_groups)
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "feedback_count": len(feedback_rows),
                "query_pool_size": len(query_pool),
                "memory_count": len(memory_ids),
                "train_samples": len(train_rows),
                "valid_samples": len(valid_rows),
                "train_groups": len(train_groups),
                "valid_groups": len(valid_groups),
                "nb_size": int(args.nb_size),
                "rand_size": int(args.rand_size),
                "seed": int(args.seed),
                "weights": source_weight_map,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        "[v5][suppressor][dataset]",
        {
            "feedback_count": len(feedback_rows),
            "train_samples": len(train_rows),
            "valid_samples": len(valid_rows),
            "train_groups": len(train_groups),
            "valid_groups": len(valid_groups),
            "query_pool_size": len(query_pool),
        },
    )


if __name__ == "__main__":
    main()
