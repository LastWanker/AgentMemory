from __future__ import annotations

import argparse
import sys
from pathlib import Path


V3_ROOT = Path(__file__).resolve().parents[1]
if str(V3_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(V3_ROOT / "src"))

from agentmemory_v3.retrieval.hybrid_retriever import HybridRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one retrieval query against V4.")
    parser.add_argument("--config", default="_V4/configs/default.yaml")
    parser.add_argument("--query", nargs="+", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--coarse-only", action="store_true", help="Disable reranker and show coarse retrieval only.")
    args = parser.parse_args()

    retriever = HybridRetriever.from_config(args.config)
    query_text = " ".join(args.query).strip()
    hits, trace = retriever.retrieve(query_text, args.top_k, coarse_only=args.coarse_only)
    mode = "coarse-only" if args.coarse_only else "reranker"
    print(f"[v4][retrieve] mode: {mode}")
    print("[v4][retrieve] trace:", trace)
    for idx, hit in enumerate(hits, start=1):
        print(f"[{idx}] {hit.memory_id} | cluster={hit.cluster_id} | score={hit.score:.4f} | source={hit.source}")
        print(hit.display_text)


if __name__ == "__main__":
    main()
