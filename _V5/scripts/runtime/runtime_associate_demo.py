from __future__ import annotations

import argparse
import sys
from pathlib import Path


V3_ROOT = Path(__file__).resolve().parents[1]
if str(V3_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(V3_ROOT / "src"))

from agentmemory_v3.association.retriever import AssociationRetriever
from agentmemory_v3.retrieval.hybrid_retriever import HybridRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one association retrieval query against V5.")
    parser.add_argument("--config", default="_V5/configs/default.yaml")
    parser.add_argument("--query", nargs="+", required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--exclude-coarse", action="store_true")
    parser.add_argument("--coarse-top-k", type=int, default=5)
    args = parser.parse_args()

    query_text = " ".join(args.query).strip()
    assoc = AssociationRetriever.from_config(args.config)
    exclude_ids: set[str] = set()
    if args.exclude_coarse:
        coarse = HybridRetriever.from_config(args.config)
        coarse_hits, _ = coarse.retrieve(query_text, top_k=max(1, int(args.coarse_top_k)))
        exclude_ids = {hit.memory_id for hit in coarse_hits}
        print(f"[v5][associate] excluding {len(exclude_ids)} coarse hits")

    hits, trace = assoc.retrieve(query_text, top_k=max(1, int(args.top_k)), exclude_memory_ids=exclude_ids)
    print("[v5][associate] mode: activation")
    print(
        "[v5][associate] summary:",
        {
            "accepted_seed_count": trace.get("accepted_seed_count"),
            "activation_counts": trace.get("activation_counts"),
            "queue_steps": trace.get("queue_steps"),
            "top_l1": trace.get("top_l1", [])[:5],
            "top_l2": trace.get("top_l2", [])[:3],
            "top_l3": trace.get("top_l3", [])[:3],
            "bridge_hits": trace.get("bridge_hits", [])[:5],
            "memory_score_breakdown": trace.get("memory_score_breakdown", [])[:3],
        },
    )
    for idx, hit in enumerate(hits, start=1):
        print(f"[{idx}] {hit.memory_id} | cluster={hit.cluster_id} | score={hit.score:.4f} | source={hit.source}")
        print(hit.display_text)


if __name__ == "__main__":
    main()
