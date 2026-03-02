from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from agentmemory_v3.utils.io import ensure_parent


@dataclass(frozen=True)
class CachePaths:
    manifest: Path
    memory_ids: Path
    query_ids: Path
    memory_coarse: Path
    query_coarse: Path
    memory_slots: Path
    query_slots: Path


def resolve_cache_paths(cache_dir: Path, alias: str) -> CachePaths:
    safe_alias = str(alias or "users").strip() or "users"
    base = cache_dir / safe_alias
    return CachePaths(
        manifest=base / "manifest.json",
        memory_ids=base / "memory_ids.json",
        query_ids=base / "query_ids.json",
        memory_coarse=base / "memory_coarse.npy",
        query_coarse=base / "query_coarse.npy",
        memory_slots=base / "memory_slots.npz",
        query_slots=base / "query_slots.npz",
    )


def write_cache_bundle(
    paths: CachePaths,
    *,
    manifest: dict,
    memory_ids: list[str],
    query_ids: list[str],
    memory_coarse: np.ndarray,
    query_coarse: np.ndarray,
    memory_slots: dict[str, np.ndarray],
    query_slots: dict[str, np.ndarray],
) -> None:
    ensure_parent(paths.manifest)
    paths.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    paths.memory_ids.write_text(json.dumps(memory_ids, ensure_ascii=False, indent=2), encoding="utf-8")
    paths.query_ids.write_text(json.dumps(query_ids, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(paths.memory_coarse, np.asarray(memory_coarse, dtype=np.float32))
    np.save(paths.query_coarse, np.asarray(query_coarse, dtype=np.float32))
    np.savez(paths.memory_slots, **{key: np.asarray(value, dtype=np.float32) for key, value in memory_slots.items()})
    np.savez(paths.query_slots, **{key: np.asarray(value, dtype=np.float32) for key, value in query_slots.items()})


def load_manifest(paths: CachePaths) -> dict:
    return json.loads(paths.manifest.read_text(encoding="utf-8"))


def load_id_list(path: Path) -> list[str]:
    return [str(item) for item in json.loads(path.read_text(encoding="utf-8"))]


def load_slot_npz(path: Path) -> dict[str, np.ndarray]:
    blob = np.load(path)
    return {key: np.asarray(blob[key], dtype=np.float32) for key in blob.files}


def build_row_index(ids: list[str]) -> dict[str, int]:
    return {item: idx for idx, item in enumerate(ids)}


def load_cache_maps(cache_dir: Path, alias: str, *, kind: str) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
    paths = resolve_cache_paths(cache_dir, alias)
    if kind == "memory":
        ids = load_id_list(paths.memory_ids)
        coarse = np.load(paths.memory_coarse)
        slots = load_slot_npz(paths.memory_slots)
    elif kind == "query":
        ids = load_id_list(paths.query_ids)
        coarse = np.load(paths.query_coarse)
        slots = load_slot_npz(paths.query_slots)
    else:
        raise ValueError(f"unsupported cache kind: {kind}")
    coarse_map = {row_id: np.asarray(coarse[idx], dtype=np.float32) for idx, row_id in enumerate(ids)}
    slot_map: dict[str, dict[str, np.ndarray]] = {}
    for idx, row_id in enumerate(ids):
        slot_map[row_id] = {field: np.asarray(matrix[idx], dtype=np.float32) for field, matrix in slots.items()}
    return coarse_map, slot_map
