from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


V3_ROOT = Path(__file__).resolve().parents[1]
if str(V3_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(V3_ROOT / "src"))

from agentmemory_v3.config import cfg_get, load_yaml_config, resolve_path
from agentmemory_v3.encoder import HFSentenceEncoder, SentenceEncoderConfig
from agentmemory_v3.retrieval.e5_cache import resolve_cache_paths, write_cache_bundle
from agentmemory_v3.retrieval.feature_builder import SLOT_FIELDS
from agentmemory_v3.slots.extractor import fallback_extract
from agentmemory_v3.training.common import resolve_query_text
from agentmemory_v3.utils.io import read_jsonl


def _slot_text(value: object) -> str:
    if isinstance(value, list):
        return " ".join(str(item) for item in value if str(item).strip())
    return str(value or "")


def _memory_search_text(memory_row: dict, slot_row: dict) -> str:
    parts = [
        str(memory_row.get("raw_text") or ""),
        str(slot_row.get("raw_brief") or ""),
        str(slot_row.get("event") or ""),
        str(slot_row.get("intent") or ""),
        " ".join(str(item) for item in (slot_row.get("entities") or [])),
        str(slot_row.get("status") or ""),
        str(slot_row.get("emotion") or ""),
        str(slot_row.get("context") or ""),
        str(slot_row.get("impact") or ""),
    ]
    return " ".join(part for part in parts if part)


def _query_slot_row(query_row: dict, slot_map: dict[str, dict]) -> dict:
    query_id = str(query_row.get("query_id") or "")
    slot_row = slot_map.get(query_id)
    if slot_row is not None:
        return slot_row
    text = resolve_query_text(query_row)
    return {"query_id": query_id, "text": text, **fallback_extract(prev_raw="", raw=text, next_raw="")}


def build_e5_cache(config_path: str | Path, *, alias: str = "", limit_memory: int = 0, limit_query: int = 0) -> dict:
    cfg = load_yaml_config(config_path)
    root_dir = resolve_path(cfg_get(cfg, "data.root_dir", "data/V4"))
    cache_dir = resolve_path(cfg_get(cfg, "cache.dir", "data/V4/VectorCacheV4"))
    cache_alias = alias or str(cfg_get(cfg, "cache.alias", "users"))
    memory_rows = list(read_jsonl(root_dir / "processed" / "memory.jsonl"))
    query_rows = list(read_jsonl(root_dir / "processed" / "query.jsonl"))
    if limit_memory > 0:
        memory_rows = memory_rows[:limit_memory]
    if limit_query > 0:
        query_rows = query_rows[:limit_query]
    memory_slot_rows = {str(row.get("memory_id") or ""): row for row in read_jsonl(root_dir / "processed" / "memory_slots.jsonl")}
    query_slot_rows = {str(row.get("query_id") or ""): row for row in read_jsonl(root_dir / "processed" / "query_slots.jsonl")}

    encoder = HFSentenceEncoder(
        SentenceEncoderConfig(
            model_name=str(cfg_get(cfg, "encoder.model_name", "intfloat/multilingual-e5-small")),
            use_e5_prefix=bool(cfg_get(cfg, "encoder.use_e5_prefix", True)),
            local_files_only=bool(cfg_get(cfg, "encoder.local_files_only", True)),
            offline=bool(cfg_get(cfg, "encoder.offline", True)),
            device=str(cfg_get(cfg, "encoder.device", "auto")),
            batch_size=int(cfg_get(cfg, "encoder.batch_size_cuda", cfg_get(cfg, "encoder.batch_size", 128))),
        )
    )

    memory_ids = [str(row.get("memory_id") or "") for row in memory_rows]
    query_ids = [str(row.get("query_id") or "") for row in query_rows]
    memory_texts = [_memory_search_text(row, memory_slot_rows.get(str(row.get("memory_id") or ""), {})) for row in memory_rows]
    query_texts = [resolve_query_text(row) for row in query_rows]
    memory_coarse = encoder.encode_passage_texts(memory_texts)
    query_coarse = encoder.encode_query_texts(query_texts)

    memory_slots: dict[str, np.ndarray] = {}
    for field in SLOT_FIELDS:
        texts = [_slot_text(memory_slot_rows.get(mem_id, {}).get(field, "")) for mem_id in memory_ids]
        memory_slots[field] = encoder.encode_passage_texts(texts)

    query_slots: dict[str, np.ndarray] = {}
    resolved_query_slot_rows = [_query_slot_row(row, query_slot_rows) for row in query_rows]
    for field in SLOT_FIELDS:
        texts = [_slot_text(row.get(field, "")) for row in resolved_query_slot_rows]
        query_slots[field] = encoder.encode_query_texts(texts)

    paths = resolve_cache_paths(cache_dir, cache_alias)
    manifest = {
        "cache_version": "v4_e5_sentence_slot_v1",
        "alias": cache_alias,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_name": encoder.config.model_name,
        "use_e5_prefix": encoder.config.use_e5_prefix,
        "local_files_only": encoder.config.local_files_only,
        "offline": encoder.config.offline,
        "device": str(encoder.device),
        "dim": int(encoder.dim),
        "memory_count": len(memory_ids),
        "query_count": len(query_ids),
        "slot_fields": list(SLOT_FIELDS),
    }
    write_cache_bundle(
        paths,
        manifest=manifest,
        memory_ids=memory_ids,
        query_ids=query_ids,
        memory_coarse=memory_coarse,
        query_coarse=query_coarse,
        memory_slots=memory_slots,
        query_slots=query_slots,
    )
    return {"cache_dir": str(paths.manifest.parent), "alias": cache_alias, "memory_count": len(memory_ids), "query_count": len(query_ids)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build V4 E5 vector cache.")
    parser.add_argument("--config", default="_V4/configs/default.yaml")
    parser.add_argument("--alias", default="")
    parser.add_argument("--limit-memory", type=int, default=0)
    parser.add_argument("--limit-query", type=int, default=0)
    args = parser.parse_args()
    result = build_e5_cache(args.config, alias=args.alias, limit_memory=args.limit_memory, limit_query=args.limit_query)
    print(
        f"[v4] e5 cache alias={result['alias']} memory={result['memory_count']} "
        f"query={result['query_count']} -> {result['cache_dir']}"
    )


if __name__ == "__main__":
    main()
