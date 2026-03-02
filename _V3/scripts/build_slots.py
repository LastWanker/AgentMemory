from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import threading
import time
from pathlib import Path


V3_ROOT = Path(__file__).resolve().parents[1]
if str(V3_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(V3_ROOT / "src"))

from agentmemory_v3.config import get_secret, resolve_path
from agentmemory_v3.slots.extractor import DeepSeekSlotExtractor, SlotExtractorConfig, fallback_extract
from agentmemory_v3.utils.io import read_jsonl


_THREAD_LOCAL = threading.local()


def _get_extractor(config: SlotExtractorConfig) -> DeepSeekSlotExtractor:
    extractor = getattr(_THREAD_LOCAL, "extractor", None)
    if extractor is None:
        extractor = DeepSeekSlotExtractor(config)
        _THREAD_LOCAL.extractor = extractor
    return extractor


def _extract_slot_row(row: dict, config: SlotExtractorConfig, retries: int, retry_delay: float) -> dict:
    prev_raw = str(row.get("prev_raw") or "")
    raw = str(row.get("raw_text") or "")
    next_raw = str(row.get("next_raw") or "")
    last_error = None
    for attempt in range(max(1, int(retries))):
        try:
            slot_payload = _get_extractor(config).extract(prev_raw=prev_raw, raw=raw, next_raw=next_raw)
            break
        except Exception as exc:
            last_error = exc
            if attempt + 1 < max(1, int(retries)):
                time.sleep(max(0.0, float(retry_delay)) * (attempt + 1))
    else:
        slot_payload = fallback_extract(prev_raw=prev_raw, raw=raw, next_raw=next_raw)
        if last_error is not None:
            warnings = slot_payload.get("warnings") if isinstance(slot_payload.get("warnings"), list) else []
            warnings.append(f"deepseek_error:{type(last_error).__name__}")
            slot_payload["warnings"] = warnings
    return {"memory_id": row["memory_id"], "cluster_id": row["cluster_id"], **slot_payload}


def build_slots(
    memory_in: Path,
    out_path: Path,
    *,
    limit: int = 0,
    resume: bool = True,
    flush_every: int = 50,
    workers: int = 1,
    retries: int = 3,
    retry_delay: float = 1.5,
) -> dict:
    memory_rows = list(read_jsonl(memory_in))
    if limit > 0:
        memory_rows = memory_rows[:limit]
    done_ids = set()
    mode = "w"
    if resume and out_path.exists():
        done_ids = {str(row.get("memory_id") or "") for row in read_jsonl(out_path)}
        mode = "a"
    extractor_config = SlotExtractorConfig(
        model=get_secret("DEEPSEEK_MODEL", env_file="data/_secrets/deepseek.env", default="deepseek-chat"),
        base_url=get_secret("DEEPSEEK_BASE_URL", env_file="data/_secrets/deepseek.env", default="https://api.deepseek.com"),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    slot_rows = []
    processed = 0
    pending_rows = [row for row in memory_rows if row["memory_id"] not in done_ids]
    with out_path.open(mode, encoding="utf-8") as f:
        if max(1, int(workers)) == 1:
            for idx, row in enumerate(memory_rows, start=1):
                if row["memory_id"] in done_ids:
                    continue
                slot_rows.append(_extract_slot_row(row, extractor_config, retries, retry_delay))
                processed += 1
                if len(slot_rows) >= max(1, int(flush_every)):
                    for item in slot_rows:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    f.flush()
                    slot_rows = []
                if idx % 20 == 0 or idx == len(memory_rows):
                    print(f"[v3][slots] scanned={idx}/{len(memory_rows)} newly_written={processed}")
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
                future_map = {
                    executor.submit(_extract_slot_row, row, extractor_config, retries, retry_delay): row["memory_id"]
                    for row in pending_rows
                }
                for idx, future in enumerate(concurrent.futures.as_completed(future_map), start=1):
                    slot_rows.append(future.result())
                    processed += 1
                    if len(slot_rows) >= max(1, int(flush_every)):
                        for item in slot_rows:
                            f.write(json.dumps(item, ensure_ascii=False) + "\n")
                        f.flush()
                        slot_rows = []
                    if idx % 20 == 0 or idx == len(pending_rows):
                        print(f"[v3][slots] completed={idx}/{len(pending_rows)} newly_written={processed}")
        for item in slot_rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        f.flush()
    return {"count": len(done_ids) + processed, "written_now": processed, "out": str(out_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build V3 slots via DeepSeek IE.")
    parser.add_argument("--config", default="_V3/configs/default.yaml")
    parser.add_argument("--memory-in", default="data/V3/processed/memory.jsonl")
    parser.add_argument("--out", default="data/V3/processed/memory_slots.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--flush-every", type=int, default=50)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=1.5)
    args = parser.parse_args()
    result = build_slots(
        resolve_path(args.memory_in),
        resolve_path(args.out),
        limit=args.limit,
        resume=not args.no_resume,
        flush_every=args.flush_every,
        workers=args.workers,
        retries=args.retries,
        retry_delay=args.retry_delay,
    )
    print(f"[v3] slots count={result['count']} written_now={result['written_now']} -> {result['out']}")


if __name__ == "__main__":
    main()
