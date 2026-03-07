from __future__ import annotations

import argparse
import sys
from pathlib import Path


V5_ROOT = Path(__file__).resolve().parents[1]
CHAT_ROOT = V5_ROOT / "chat"
if str(V5_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(V5_ROOT / "src"))
if str(CHAT_ROOT) not in sys.path:
    sys.path.insert(0, str(CHAT_ROOT))

from src.chat_app.config import load_config
from src.chat_app.retriever_adapter import RetrieverAdapter
from src.chat_app.suppressor_adapter import SuppressorAdapter


def _print_refs(title: str, refs: list) -> None:
    print(title)
    if not refs:
        print("  <empty>")
        return
    for idx, ref in enumerate(refs, start=1):
        print(
            f"  [{idx}] {ref.memory_id} score={ref.score:.4f} base={ref.base_score:.4f} "
            f"suppressed={ref.suppressed} s={ref.suppress_score:.4f} lane={ref.suppress_lane}"
        )
        print(f"      {ref.display_text}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect suppressor before/after on V5 chat candidates.")
    parser.add_argument("--query", nargs="+", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    query_text = " ".join(args.query).strip()
    config = load_config()
    config.suppressor_enabled = False
    retriever = RetrieverAdapter(config)
    coarse_refs = retriever._retrieve_coarse(query_text, max(1, int(args.top_k)))
    association_refs = []
    try:
        association_refs, _trace = retriever._retrieve_association(
            query_text,
            top_k=max(1, int(args.top_k)),
            exclude_memory_ids={ref.memory_id for ref in coarse_refs},
        )
    except Exception:
        association_refs = []

    _print_refs("[coarse][before]", coarse_refs)
    _print_refs("[association][before]", association_refs)

    config.suppressor_enabled = True
    suppressor = SuppressorAdapter(config)
    coarse_after, coarse_trace = suppressor.apply(query_text, "coarse", coarse_refs)
    association_after, association_trace = suppressor.apply(query_text, "association", association_refs)

    _print_refs("[coarse][after]", coarse_after)
    _print_refs("[association][after]", association_after)
    print("[trace][coarse]", coarse_trace)
    print("[trace][association]", association_trace)


if __name__ == "__main__":
    main()
