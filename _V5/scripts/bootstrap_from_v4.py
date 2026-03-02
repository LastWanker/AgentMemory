from __future__ import annotations

import argparse
import shutil
from pathlib import Path


V4_TO_V5_FILES = {
    Path("processed/memory.jsonl"): Path("processed/memory.jsonl"),
    Path("processed/query.jsonl"): Path("processed/query.jsonl"),
    Path("processed/cluster.jsonl"): Path("processed/cluster.jsonl"),
    Path("indexes/dense_artifact.pkl"): Path("indexes/dense_artifact.pkl"),
    Path("indexes/dense_matrix.npy"): Path("indexes/dense_matrix.npy"),
    Path("indexes/bm25.pkl"): Path("indexes/bm25.pkl"),
    Path("exports/chat_memory_bundle.jsonl"): Path("exports/chat_memory_bundle.jsonl"),
    Path("VectorCacheV4/users/manifest.json"): Path("VectorCacheV5/users/manifest.json"),
    Path("VectorCacheV4/users/memory_ids.json"): Path("VectorCacheV5/users/memory_ids.json"),
    Path("VectorCacheV4/users/query_ids.json"): Path("VectorCacheV5/users/query_ids.json"),
    Path("VectorCacheV4/users/memory_coarse.npy"): Path("VectorCacheV5/users/memory_coarse.npy"),
    Path("VectorCacheV4/users/query_coarse.npy"): Path("VectorCacheV5/users/query_coarse.npy"),
}


def bootstrap_from_v4(v4_root: Path, v5_root: Path, *, overwrite: bool = False) -> dict:
    copied = []
    skipped = []
    for src_rel, dst_rel in V4_TO_V5_FILES.items():
        src = v4_root / src_rel
        dst = v5_root / dst_rel
        if not src.exists():
            skipped.append(str(src))
            continue
        if dst.exists() and not overwrite:
            skipped.append(str(dst))
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(str(dst))
    return {"copied_count": len(copied), "skipped_count": len(skipped), "copied": copied, "skipped": skipped}


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap V5 coarse data from V4 artifacts.")
    parser.add_argument("--v4-root", default="data/V4")
    parser.add_argument("--v5-root", default="data/V5")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = bootstrap_from_v4(Path(args.v4_root), Path(args.v5_root), overwrite=args.overwrite)
    print(f"[v5] bootstrap_from_v4 copied={result['copied_count']} skipped={result['skipped_count']}")
    for path in result["copied"]:
        print(f"[copied] {path}")
    for path in result["skipped"]:
        print(f"[skipped] {path}")


if __name__ == "__main__":
    main()
