from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


V5_ROOT = Path(__file__).resolve().parents[1]
if str(V5_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(V5_ROOT / "src"))
if str(V5_ROOT / "chat" / "src") not in sys.path:
    sys.path.insert(0, str(V5_ROOT / "chat" / "src"))

from chat_app.config import load_config
from chat_app.retriever_adapter import RetrieverAdapter


def _load_queries(path: Path, sample_size: int, seed: int) -> list[str]:
    names: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        text = str(row.get("canonical_name") or "").strip()
        if text:
            names.append(text)
    unique = list(dict.fromkeys(names))
    random.Random(int(seed)).shuffle(unique)
    return unique[: max(1, int(sample_size))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare top-k retrieval changes with memory_preference toggle on/off.")
    parser.add_argument("--concepts-jsonl", default="data/V5/association/concepts.jsonl")
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-json", default="data/V5/suppressor_newfb/concepts2000_toggle_compare.json")
    args = parser.parse_args()

    query_path = Path(args.concepts_jsonl)
    queries = _load_queries(query_path, sample_size=int(args.sample_size), seed=int(args.seed))
    if not queries:
        raise RuntimeError("no canonical_name queries found")

    cfg = load_config()
    retriever = RetrieverAdapter(cfg)

    coarse_changed = 0
    association_changed = 0
    dual_changed = 0
    coarse_nonempty = 0
    association_nonempty = 0
    samples: list[dict] = []

    for idx, query in enumerate(queries, start=1):
        off = retriever.retrieve_bundle(query, int(args.top_k), memory_preference_enabled=False)
        on = retriever.retrieve_bundle(query, int(args.top_k), memory_preference_enabled=True)

        coarse_off = [item.memory_id for item in off.coarse_refs]
        coarse_on = [item.memory_id for item in on.coarse_refs]
        association_off = [item.memory_id for item in off.association_refs]
        association_on = [item.memory_id for item in on.association_refs]

        coarse_is_changed = coarse_off != coarse_on
        association_is_changed = association_off != association_on
        dual_is_changed = coarse_is_changed or association_is_changed

        if coarse_off:
            coarse_nonempty += 1
        if association_off:
            association_nonempty += 1
        if coarse_is_changed:
            coarse_changed += 1
        if association_is_changed:
            association_changed += 1
        if dual_is_changed:
            dual_changed += 1
            if len(samples) < 20:
                samples.append(
                    {
                        "query": query,
                        "coarse_off": coarse_off,
                        "coarse_on": coarse_on,
                        "association_off": association_off,
                        "association_on": association_on,
                    }
                )
        if idx % 200 == 0:
            print("[progress]", idx)

    count = len(queries)
    payload = {
        "query_source": str(query_path),
        "sample_size": count,
        "seed": int(args.seed),
        "top_k": int(args.top_k),
        "coarse_nonempty": coarse_nonempty,
        "association_nonempty": association_nonempty,
        "coarse_changed_count": coarse_changed,
        "association_changed_count": association_changed,
        "dual_changed_count": dual_changed,
        "coarse_changed_rate": float(coarse_changed) / float(max(1, count)),
        "association_changed_rate": float(association_changed) / float(max(1, count)),
        "dual_changed_rate": float(dual_changed) / float(max(1, count)),
        "samples": samples,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        "[v5][memory_preference][compare]",
        {
            "sample_size": count,
            "coarse_changed_rate": payload["coarse_changed_rate"],
            "association_changed_rate": payload["association_changed_rate"],
            "dual_changed_rate": payload["dual_changed_rate"],
            "output_json": str(output_path),
        },
    )


if __name__ == "__main__":
    main()
