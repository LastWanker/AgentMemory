"""Selective cache cleaner for data/VectorCache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
VECTOR_CACHE_DIR = REPO_ROOT / "data" / "VectorCache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean vector cache files by signature filters.")
    parser.add_argument("--cache-dir", default=str(VECTOR_CACHE_DIR))
    parser.add_argument("--dataset")
    parser.add_argument("--encoder-backend", choices=("hf", "simple"))
    parser.add_argument("--encoder-id-contains")
    parser.add_argument("--signature-contains", help="Raw signature substring filter.")
    parser.add_argument("--cache-kind", choices=("all", "memory", "eval"), default="all")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def iter_cache_files(cache_dir: Path, cache_kind: str) -> Iterable[Path]:
    if cache_kind in ("all", "memory"):
        yield from sorted(cache_dir.glob("memory_cache_*.jsonl"))
    if cache_kind in ("all", "eval"):
        yield from sorted(cache_dir.glob("eval_cache_*.jsonl"))


def read_cache_signature(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                if isinstance(payload, dict) and "_meta" in payload:
                    meta = payload.get("_meta") or {}
                    return str(meta.get("cache_signature") or "")
                break
    except Exception:
        return ""
    return ""


def match_signature(signature: str, args: argparse.Namespace) -> bool:
    sig = signature or ""
    if args.dataset and (f"dataset_{args.dataset}" not in sig and f"dataset={args.dataset}" not in sig):
        return False
    if args.encoder_backend and (
        f"backend_{args.encoder_backend}" not in sig and f"backend={args.encoder_backend}" not in sig
    ):
        return False
    if args.encoder_id_contains and args.encoder_id_contains not in sig:
        return False
    if args.signature_contains and args.signature_contains not in sig:
        return False
    return True


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        print(f"cache dir not found: {cache_dir}")
        return

    matched = []
    for path in iter_cache_files(cache_dir, args.cache_kind):
        signature = read_cache_signature(path)
        if not match_signature(signature, args):
            continue
        matched.append((path, signature))

    if not matched:
        print("No cache files matched.")
        return

    print(f"Matched {len(matched)} cache file(s):")
    for path, signature in matched:
        print(f"- {path} | signature={signature}")

    if args.dry_run:
        print("Dry-run only, no file deleted.")
        return

    deleted = 0
    for path, _ in matched:
        try:
            path.unlink()
            deleted += 1
        except Exception as exc:
            print(f"failed to delete {path}: {exc}")
    print(f"Deleted {deleted} file(s).")


if __name__ == "__main__":
    main()

