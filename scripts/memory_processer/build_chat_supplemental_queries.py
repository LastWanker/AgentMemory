"""Build LLM supplemental query dataset from memory_chat.jsonl.

This script does NOT overwrite identity queries by itself.
Use build_chat_queries.py --supplemental-queries <this_output> to merge.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.chat_memory_processor import write_jsonl  # noqa: E402
from src.chat_memory_processor.llm_querygen import LLMQueryGenConfig, build_llm_supplemental_rows  # noqa: E402


def _load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise RuntimeError(f"YAML config requested but pyyaml is unavailable: {path}") from exc
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"Config root must be object/dict: {path}")
        return payload
    raise ValueError(f"Unsupported config file type: {path}")


def _cfg(data: dict, key: str, default):
    cur = data
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _load_api_key(*, env_key: str, key_file: Path | None) -> str:
    env_val = os.environ.get(env_key, "").strip()
    if env_val:
        return env_val
    if key_file and key_file.exists():
        for line in key_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() == env_key and v.strip():
                return v.strip().strip("\"").strip("'")
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LLM supplemental chat queries.")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "chat_memory.yaml"),
        help="Optional JSON/YAML config file.",
    )
    parser.add_argument("--memory-in", default=None, help="Input memory jsonl.")
    parser.add_argument("--out", default=None, help="Output supplemental jsonl.")
    parser.add_argument("--base-url", default=None, help="LLM base URL.")
    parser.add_argument("--model", default=None, help="LLM model name.")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--api-key-file", default=None, help="Optional local env file path.")
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=float, default=None)
    parser.add_argument("--max-retries", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-clusters", type=int, default=None)
    parser.add_argument("--min-cluster-size", type=int, default=None)
    parser.add_argument("--max-per-layer", type=int, default=None)
    parser.add_argument("--fallback-only", action="store_true")
    args = parser.parse_args()

    loaded = _load_config(Path(args.config))
    memory_in = Path(
        args.memory_in or _cfg(loaded, "output.memory_out", str(REPO_ROOT / "data" / "Processed" / "memory_chat.jsonl"))
    )
    out = Path(
        args.out or _cfg(loaded, "query.llm_supplemental_out", str(REPO_ROOT / "data" / "Processed" / "eval_chat_supplemental.jsonl"))
    )

    env_key = args.api_key_env
    key_file_path_raw = args.api_key_file
    if key_file_path_raw is None:
        key_file_path_raw = _cfg(loaded, "query.llm.api_key_file", str(REPO_ROOT / "data" / "_secrets" / "deepseek.env"))
    key_file = Path(key_file_path_raw) if key_file_path_raw else None

    api_key = _load_api_key(env_key=env_key, key_file=key_file)
    if not api_key and not args.fallback_only:
        raise RuntimeError(
            f"Missing API key. Set {env_key} or provide key file with `{env_key}=...`."
        )

    cfg = LLMQueryGenConfig(
        base_url=str(args.base_url or _cfg(loaded, "query.llm.base_url", "https://api.deepseek.com")),
        model=str(args.model or _cfg(loaded, "query.llm.model", "deepseek-chat")),
        concurrency=int(args.concurrency if args.concurrency is not None else _cfg(loaded, "query.llm.concurrency", 32)),
        timeout_seconds=float(
            args.timeout_seconds if args.timeout_seconds is not None else _cfg(loaded, "query.llm.timeout_seconds", 60)
        ),
        max_retries=int(args.max_retries if args.max_retries is not None else _cfg(loaded, "query.llm.max_retries", 3)),
        temperature=float(args.temperature if args.temperature is not None else _cfg(loaded, "query.llm.temperature", 1.1)),
        top_p=float(args.top_p if args.top_p is not None else _cfg(loaded, "query.llm.top_p", 0.95)),
        seed=int(args.seed if args.seed is not None else _cfg(loaded, "query.llm.seed", 11)),
        max_clusters=(args.max_clusters if args.max_clusters is not None else _cfg(loaded, "query.llm.max_clusters", None)),
        min_cluster_size=int(
            args.min_cluster_size if args.min_cluster_size is not None else _cfg(loaded, "query.llm.min_cluster_size", 1)
        ),
        max_per_layer=(
            args.max_per_layer
            if args.max_per_layer is not None
            else _cfg(loaded, "query.llm.max_per_layer", 6)
        ),
        fallback_only=bool(args.fallback_only),
    )

    memories = _read_jsonl(memory_in)
    rows = asyncio.run(build_llm_supplemental_rows(memories, api_key=api_key, config=cfg))
    write_jsonl(out, rows)

    print(f"memory_rows={len(memories)} <= {memory_in}")
    print(f"supplemental_rows={len(rows)} => {out}")
    print(f"model={cfg.model} base_url={cfg.base_url} concurrency={cfg.concurrency} fallback_only={cfg.fallback_only}")
    if cfg.max_clusters is not None:
        print(f"max_clusters={cfg.max_clusters}")
    if cfg.max_per_layer is not None:
        print(f"max_per_layer={cfg.max_per_layer}")


if __name__ == "__main__":
    main()
