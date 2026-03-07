from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from agentmemory_v3.config import cfg_get, load_yaml_config, resolve_path
from agentmemory_v3.encoder import HFSentenceEncoder, SentenceEncoderConfig
from agentmemory_v3.retrieval.dense_index import DenseIndex
from agentmemory_v3.utils.io import read_jsonl


def load_v5_encoder(config_path: str | Path) -> HFSentenceEncoder:
    cfg = load_yaml_config(config_path)
    return HFSentenceEncoder(
        SentenceEncoderConfig(
            model_name=str(cfg_get(cfg, "encoder.model_name", "intfloat/multilingual-e5-small")),
            use_e5_prefix=bool(cfg_get(cfg, "encoder.use_e5_prefix", True)),
            local_files_only=bool(cfg_get(cfg, "encoder.local_files_only", True)),
            offline=bool(cfg_get(cfg, "encoder.offline", True)),
            device=str(cfg_get(cfg, "encoder.device", "auto")),
            batch_size=int(cfg_get(cfg, "encoder.batch_size", 128)),
        )
    )


def load_dense_index_from_config(config_path: str | Path) -> DenseIndex:
    cfg = load_yaml_config(config_path)
    root_dir = resolve_path(cfg_get(cfg, "data.root_dir", "data/V5"))
    return DenseIndex.load(root_dir / "indexes" / "dense_artifact.pkl", root_dir / "indexes" / "dense_matrix.npy")


def load_memory_text_by_id(config_path: str | Path, bundle_path: str | Path | None = None) -> dict[str, str]:
    cfg = load_yaml_config(config_path)
    root_dir = resolve_path(cfg_get(cfg, "data.root_dir", "data/V5"))
    out: dict[str, str] = {}
    if bundle_path:
        path = Path(bundle_path)
        if path.exists():
            for row in read_jsonl(path):
                memory_id = str(row.get("memory_id") or "")
                if memory_id:
                    out[memory_id] = str(row.get("display_text") or row.get("raw_text") or row.get("search_text") or "")
    processed_path = root_dir / "processed" / "memory.jsonl"
    if processed_path.exists():
        for row in read_jsonl(processed_path):
            memory_id = str(row.get("memory_id") or "")
            if memory_id and memory_id not in out:
                out[memory_id] = str(row.get("raw_text") or row.get("text") or "")
    return out


def collect_user_queries(conversation_dir: str | Path) -> list[str]:
    base = Path(conversation_dir)
    if not base.exists():
        return []
    seen: set[str] = set()
    rows: list[str] = []
    for path in sorted(base.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if str(payload.get("role") or "") != "user":
                continue
            text = str(payload.get("content") or "").strip()
            if text and text not in seen:
                seen.add(text)
                rows.append(text)
    return rows


def load_feedback_rows(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    for row in read_jsonl(Path(path)):
        feedback_type = str(row.get("type") or row.get("feedback_type") or "").strip().lower()
        if feedback_type not in {"unrelated", "toforget"}:
            continue
        q_text = str(row.get("q_text") or row.get("query_text") or row.get("query") or row.get("text") or "").strip()
        m_id = str(row.get("m_id") or row.get("memory_id") or "").strip()
        m_text = str(row.get("m_text") or row.get("memory_text") or row.get("m_text_snapshot") or "").strip()
        if not q_text or not m_id:
            continue
        feedback_id = str(row.get("feedback_id") or row.get("id") or "")
        if not feedback_id:
            feedback_id = hashlib.md5(f"{feedback_type}|{q_text}|{m_id}".encode("utf-8")).hexdigest()[:12]
        rows.append(
            {
                "feedback_id": feedback_id,
                "ts": str(row.get("ts") or row.get("created_at") or ""),
                "feedback_type": feedback_type,
                "q_text": q_text,
                "m_id": m_id,
                "m_text": m_text,
                "lane": str(row.get("lane") or ""),
                "candidate_cache_memory_ids": [str(item) for item in (row.get("candidate_cache_memory_ids") or []) if str(item)],
                "effective_candidate_cache_memory_ids": [
                    str(item) for item in (row.get("effective_candidate_cache_memory_ids") or []) if str(item)
                ],
            }
        )
    return rows


def search_top_indices(matrix: np.ndarray, query_vec: np.ndarray, top_n: int, exclude: set[int] | None = None) -> list[int]:
    data = np.asarray(matrix, dtype=np.float32)
    query = np.asarray(query_vec, dtype=np.float32).reshape(-1)
    if data.size == 0 or query.size == 0:
        return []
    scores = data @ query
    order = np.argsort(scores)[::-1]
    out: list[int] = []
    block = exclude or set()
    for idx in order:
        if int(idx) in block:
            continue
        out.append(int(idx))
        if len(out) >= max(1, int(top_n)):
            break
    return out
