from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


V5_ROOT = Path(__file__).resolve().parents[1]
CHAT_SRC = V5_ROOT / "chat" / "src"
if str(V5_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(V5_ROOT / "src"))
if str(CHAT_SRC) not in sys.path:
    sys.path.insert(0, str(CHAT_SRC))

from agentmemory_v3.suppressor.data_utils import collect_user_queries
from chat_app.config import load_config
from chat_app.retriever_adapter import RetrieverAdapter


def _ref_to_payload(ref: dict, lane: str) -> dict:
    return {
        "memory_id": str(ref.get("memory_id") or ""),
        "lane": str(lane or "").strip().lower(),
        "cluster_id": str(ref.get("cluster_id") or ""),
        "score": float(ref.get("score") or 0.0),
        "base_score": float(ref.get("base_score") or ref.get("score") or 0.0),
        "source": str(ref.get("source") or ""),
        "display_text": str(ref.get("display_text") or ""),
    }


def _build_event(
    *,
    session_id: str,
    query_text: str,
    memory_id: str,
    feedback_type: str,
    lane: str,
    candidate_refs: list[dict],
    prior_self_ids: set[str],
) -> dict:
    dedup_candidate_refs: list[dict] = []
    seen: set[str] = set()
    for row in candidate_refs:
        memory_ref_id = str(row.get("memory_id") or "").strip()
        if not memory_ref_id or memory_ref_id in seen:
            continue
        seen.add(memory_ref_id)
        dedup_candidate_refs.append(
            {
                "memory_id": memory_ref_id,
                "lane": str(row.get("lane") or "").strip().lower(),
                "cluster_id": str(row.get("cluster_id") or ""),
                "score": float(row.get("score") or 0.0),
                "base_score": float(row.get("base_score") or row.get("score") or 0.0),
                "source": str(row.get("source") or ""),
                "display_text": str(row.get("display_text") or ""),
            }
        )

    effective_refs = [
        row
        for row in dedup_candidate_refs
        if str(row.get("memory_id") or "") not in prior_self_ids and str(row.get("memory_id") or "") != memory_id
    ]
    now = datetime.now(timezone.utc)
    next_self_ids = sorted({item for item in prior_self_ids if item} | {memory_id})
    return {
        "feedback_id": now.strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:8],
        "ts": now.isoformat(),
        "session_id": session_id,
        "query_text": query_text,
        "memory_id": memory_id,
        "feedback_type": feedback_type,
        "lane": lane,
        "query_self_memory_ids_before": sorted(item for item in prior_self_ids if item),
        "query_self_memory_ids_after": next_self_ids,
        "candidate_cache_memory_ids": [str(row.get("memory_id") or "") for row in dedup_candidate_refs],
        "effective_candidate_cache_memory_ids": [str(row.get("memory_id") or "") for row in effective_refs],
        "candidate_refs": dedup_candidate_refs,
        "effective_candidate_refs": effective_refs,
    }


def _load_template_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        row = json.loads(line)
        query_text = str(row.get("query_text") or row.get("q_text") or "").strip()
        candidate_refs = list(row.get("candidate_refs") or [])
        if not query_text or not candidate_refs:
            continue
        lanes = {str(item.get("lane") or "").lower() for item in candidate_refs}
        if "association" not in lanes or "coarse" not in lanes:
            continue
        rows.append({"query_text": query_text, "candidate_refs": candidate_refs})
    return rows


def _append_events_for_query(
    *,
    events: list[dict],
    session_id: str,
    query_text: str,
    candidate_refs: list[dict],
) -> bool:
    association_refs = [row for row in candidate_refs if str(row.get("lane") or "").strip().lower() == "association"]
    coarse_refs = [row for row in candidate_refs if str(row.get("lane") or "").strip().lower() == "coarse"]
    if len(association_refs) < 2 or len(coarse_refs) < 2:
        return False

    self_ids: set[str] = set()
    for ref in association_refs[:2]:
        memory_id = str(ref.get("memory_id") or "")
        if not memory_id or memory_id in self_ids:
            continue
        event = _build_event(
            session_id=session_id,
            query_text=query_text,
            memory_id=memory_id,
            feedback_type="toforget",
            lane="association",
            candidate_refs=candidate_refs,
            prior_self_ids=set(self_ids),
        )
        events.append(event)
        self_ids.add(memory_id)

    for ref in coarse_refs[:2]:
        memory_id = str(ref.get("memory_id") or "")
        if not memory_id or memory_id in self_ids:
            continue
        event = _build_event(
            session_id=session_id,
            query_text=query_text,
            memory_id=memory_id,
            feedback_type="unrelated",
            lane="coarse",
            candidate_refs=candidate_refs,
            prior_self_ids=set(self_ids),
        )
        events.append(event)
        self_ids.add(memory_id)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic feedback_events_test.jsonl from retrieval results.")
    parser.add_argument("--conversation-dir", default="_V5/chat/data/conversations")
    parser.add_argument("--template-feedback", default="_V5/chat/data/feedback/feedback_events.jsonl")
    parser.add_argument("--output", default="_V5/chat/data/feedback/feedback_events_test.jsonl")
    parser.add_argument("--max-queries", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--session-id", default="feedback_test_session")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    queries = collect_user_queries(args.conversation_dir)
    if not queries:
        raise RuntimeError("no user queries found from conversations")

    config = load_config()
    retriever = RetrieverAdapter(config)

    events: list[dict] = []
    selected_query_count = 0
    for idx, query_text in enumerate(queries, start=1):
        if selected_query_count >= int(args.max_queries):
            break
        bundle = retriever.retrieve_bundle(query_text, max(2, int(args.top_k)))
        candidate_refs = [_ref_to_payload(ref.model_dump(), "association") for ref in bundle.association_refs] + [
            _ref_to_payload(ref.model_dump(), "coarse") for ref in bundle.coarse_refs
        ]
        ok = _append_events_for_query(
            events=events,
            session_id=f"{args.session_id}_retrieve_{idx:03d}",
            query_text=query_text,
            candidate_refs=candidate_refs,
        )
        if not ok:
            continue
        selected_query_count += 1

    if selected_query_count < int(args.max_queries):
        template_rows = _load_template_rows(Path(args.template_feedback))
        if not template_rows:
            raise RuntimeError(
                "retrieval mode produced too few rows and no valid template feedback rows were found"
            )
        while selected_query_count < int(args.max_queries):
            template = template_rows[selected_query_count % len(template_rows)]
            query_text = str(template.get("query_text") or "")
            candidate_refs = list(template.get("candidate_refs") or [])
            row_index = selected_query_count + 1
            ok = _append_events_for_query(
                events=events,
                session_id=f"{args.session_id}_template_{row_index:03d}",
                query_text=query_text,
                candidate_refs=candidate_refs,
            )
            if not ok:
                selected_query_count += 1
                continue
            selected_query_count += 1

    with output_path.open("w", encoding="utf-8") as f:
        for row in events:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    type_counts: dict[str, int] = {}
    for row in events:
        feedback_type = str(row.get("feedback_type") or "")
        type_counts[feedback_type] = type_counts.get(feedback_type, 0) + 1
    print(
        "[v5][feedback_test]",
        {
            "output": str(output_path),
            "selected_queries": selected_query_count,
            "event_count": len(events),
            "type_counts": type_counts,
        },
    )


if __name__ == "__main__":
    main()
