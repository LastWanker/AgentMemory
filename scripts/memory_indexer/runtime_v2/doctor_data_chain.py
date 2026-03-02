"""Data-chain doctor for runtime_v2.

Checks dataset files, positives integrity, cache alias isolation and key defaults.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from scripts.memory_indexer.runtime_utils import cfg_get, load_config, write_json

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "Processed"
MEMORY_EVAL_DIR = DATA_DIR / "Memory_Eval"
VECTOR_CACHE_DIR = DATA_DIR / "VectorCache"
MODEL_WEIGHTS_DIR = DATA_DIR / "ModelWeights"


def resolve_dataset_paths(dataset: str) -> Tuple[Path, Path]:
    processed_memory = PROCESSED_DIR / f"memory_{dataset}.jsonl"
    processed_eval = PROCESSED_DIR / f"eval_{dataset}.jsonl"
    legacy_memory = MEMORY_EVAL_DIR / f"memory_{dataset}.jsonl"
    legacy_eval = MEMORY_EVAL_DIR / f"eval_{dataset}.jsonl"
    memory_path = processed_memory if processed_memory.exists() else legacy_memory
    eval_path = processed_eval if processed_eval.exists() else legacy_eval
    if not memory_path.exists():
        memory_path = DATA_DIR / f"memory_{dataset}.jsonl"
    if not eval_path.exists():
        eval_path = DATA_DIR / f"eval_{dataset}.jsonl"
    return memory_path, eval_path


def iter_jsonl(path: Path) -> Iterable[Dict[str, object]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def count_jsonl(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def normalize_positives(payload: Dict[str, object]) -> List[str]:
    if isinstance(payload.get("positives"), list):
        return [str(x) for x in payload["positives"]]
    expected = payload.get("expected_mem_ids", [])
    if isinstance(expected, list):
        return [str(x) for x in expected]
    return []


def scan_cache(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    row_count = 0
    has_meta = False
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                if isinstance(payload, dict) and "_meta" in payload:
                    has_meta = True
                    continue
                row_count += 1
    except Exception as exc:
        return {
            "exists": True,
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "parse_error": str(exc),
        }
    return {
        "exists": True,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "row_count": row_count,
        "has_meta": has_meta,
    }


def level_from_issues(fatal_issues: List[str], warn_issues: List[str]) -> str:
    if fatal_issues:
        return "FAIL"
    if warn_issues:
        return "WARN"
    return "PASS"


def main() -> None:
    parser = argparse.ArgumentParser(description="Doctor checks for runtime_v2 data chain.")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default_v2_bipartite.yaml"))
    parser.add_argument("--dataset", help="Override dataset from config")
    parser.add_argument("--cache-alias", default="users")
    parser.add_argument("--smoke-cache-alias", default="users_smoke")
    parser.add_argument("--weight-path", default=str(MODEL_WEIGHTS_DIR / "listwise_bipartite_reranker.pt"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset = args.dataset or str(cfg_get(cfg, "common.dataset", "followup_plus_chat"))
    memory_path, eval_path = resolve_dataset_paths(dataset)

    fatal_issues: List[str] = []
    warn_issues: List[str] = []

    if not memory_path.exists():
        fatal_issues.append(f"memory file missing: {memory_path}")
    if not eval_path.exists():
        fatal_issues.append(f"eval file missing: {eval_path}")

    memory_rows = 0
    eval_rows = 0
    missing_pos_rows = 0
    empty_pos_rows = 0
    unknown_positive_ids = 0
    memory_ids = set()

    if memory_path.exists():
        for row in iter_jsonl(memory_path):
            memory_rows += 1
            mem_id = row.get("mem_id")
            if isinstance(mem_id, str) and mem_id:
                memory_ids.add(mem_id)
            else:
                warn_issues.append("memory row without valid mem_id found")
                break

    if eval_path.exists():
        for row in iter_jsonl(eval_path):
            eval_rows += 1
            positives = normalize_positives(row)
            if "positives" not in row and "expected_mem_ids" not in row:
                missing_pos_rows += 1
                continue
            if not positives:
                empty_pos_rows += 1
                continue
            for mem_id in positives:
                if mem_id not in memory_ids:
                    unknown_positive_ids += 1

    if missing_pos_rows > 0:
        fatal_issues.append(f"eval rows missing positives field: {missing_pos_rows}")
    if empty_pos_rows > 0:
        warn_issues.append(f"eval rows with empty positives: {empty_pos_rows}")
    if unknown_positive_ids > 0:
        warn_issues.append(f"eval positives missing in memory set: {unknown_positive_ids}")

    query_cache = VECTOR_CACHE_DIR / f"eval_cache_{args.cache_alias}.jsonl"
    memory_cache = VECTOR_CACHE_DIR / f"memory_cache_{args.cache_alias}.jsonl"
    smoke_query_cache = VECTOR_CACHE_DIR / f"eval_cache_{args.smoke_cache_alias}.jsonl"
    smoke_memory_cache = VECTOR_CACHE_DIR / f"memory_cache_{args.smoke_cache_alias}.jsonl"

    if args.cache_alias == args.smoke_cache_alias:
        fatal_issues.append("cache alias and smoke cache alias are identical")

    defaults_check = {
        "train.loss_type": cfg_get(cfg, "train_reranker.loss_type", ""),
        "train.model_family": cfg_get(cfg, "train_reranker.model_family", ""),
        "train.neg_strategy": cfg_get(cfg, "train_reranker.neg_strategy", ""),
        "eval.ablation_groups": cfg_get(cfg, "eval_router.ablation_groups", ""),
        "common.candidate_mode": cfg_get(cfg, "common.candidate_mode", ""),
    }
    if defaults_check["train.loss_type"] != "listwise":
        warn_issues.append("config default train.loss_type is not listwise")
    if defaults_check["train.model_family"] != "bipartite":
        warn_issues.append("config default train.model_family is not bipartite")
    if defaults_check["eval.ablation_groups"] not in {"S-only", "S-only,mix(auto)"}:
        warn_issues.append("config default eval.ablation_groups is not S-only")
    if defaults_check["common.candidate_mode"] != "coarse":
        warn_issues.append("config default candidate_mode is not coarse")

    weight_path = Path(args.weight_path)
    weight_info = {
        "path": str(weight_path),
        "exists": weight_path.exists(),
    }
    if weight_path.exists():
        weight_info["size_bytes"] = weight_path.stat().st_size
    else:
        warn_issues.append(f"weight missing (will be created after training): {weight_path}")

    status = level_from_issues(fatal_issues, warn_issues)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = REPO_ROOT / "runs" / "rt_v2" / "doctor" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "status": status,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": str(args.config),
        "dataset": dataset,
        "paths": {
            "memory": str(memory_path),
            "eval": str(eval_path),
        },
        "counts": {
            "memory_rows": memory_rows,
            "eval_rows": eval_rows,
            "missing_pos_rows": missing_pos_rows,
            "empty_pos_rows": empty_pos_rows,
            "unknown_positive_ids": unknown_positive_ids,
        },
        "cache": {
            "formal_query": scan_cache(query_cache),
            "formal_memory": scan_cache(memory_cache),
            "smoke_query": scan_cache(smoke_query_cache),
            "smoke_memory": scan_cache(smoke_memory_cache),
        },
        "defaults_check": defaults_check,
        "weight": weight_info,
        "issues": {
            "fatal": fatal_issues,
            "warn": warn_issues,
        },
    }
    write_json(run_dir / "doctor.report.json", report)

    print(f"[doctor] status={status}")
    print(f"[doctor] dataset={dataset} memory_rows={memory_rows} eval_rows={eval_rows}")
    print(
        "[doctor] cache(formal) "
        f"query={query_cache.exists()} memory={memory_cache.exists()} "
        f"| cache(smoke) query={smoke_query_cache.exists()} memory={smoke_memory_cache.exists()}"
    )
    print(f"[doctor] report={run_dir / 'doctor.report.json'}")
    if fatal_issues:
        for issue in fatal_issues:
            print(f"[doctor][fatal] {issue}")
    if warn_issues:
        for issue in warn_issues:
            print(f"[doctor][warn] {issue}")


if __name__ == "__main__":
    main()

