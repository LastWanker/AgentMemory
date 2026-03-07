from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


V5_ROOT = Path(__file__).resolve().parents[1]
if str(V5_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(V5_ROOT / "src"))

from agentmemory_v3.suppressor import SuppressorRuntime, SuppressorRuntimeConfig
from agentmemory_v3.utils.io import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained suppressor on a dataset jsonl.")
    parser.add_argument("--artifact-dir", default="data/V5/suppressor")
    parser.add_argument("--dataset", default="data/V5/suppressor/feedback_samples_valid.jsonl")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    runtime = SuppressorRuntime.load(str(args.artifact_dir))
    if runtime is None:
        raise RuntimeError("failed to load suppressor runtime")
    rows = list(read_jsonl(Path(args.dataset)))
    if not rows:
        raise RuntimeError("dataset is empty")

    cfg = SuppressorRuntimeConfig(enabled=True, threshold=0.0, alpha=0.0, max_drop_per_lane=0, keep_top_per_lane=0)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("q_text") or "")].append(row)

    scored: list[dict] = []
    for query_text, group_rows in grouped.items():
        fake_rows = []
        for row in group_rows:
            fake_rows.append(
                {
                    "memory_id": str(row.get("m_id") or ""),
                    "display_text": str(row.get("m_text") or ""),
                    "score": 1.0,
                    "base_score": 1.0,
                }
            )
        scores, _trace = runtime.score_rows(query_text=query_text, rows=fake_rows, config=cfg)
        by_id = {item.memory_id: item for item in scores}
        for row in group_rows:
            item = by_id.get(str(row.get("m_id") or ""))
            if item is None:
                continue
            scored.append(
                {
                    "source_kind": str(row.get("source_kind") or ""),
                    "label": int(row.get("label") or 0),
                    "score": float(item.suppress_score),
                    "feedback_type": str(row.get("feedback_type") or ""),
                    "pred": 1 if float(item.suppress_score) >= float(args.threshold) else 0,
                }
            )

    metrics: dict[str, float] = {}
    if scored:
        correct = sum(1 for row in scored if int(row["pred"]) == int(row["label"]))
        metrics["accuracy"] = float(correct) / float(len(scored))
    for source_kind in sorted({row["source_kind"] for row in scored}):
        subset = [row for row in scored if row["source_kind"] == source_kind]
        if not subset:
            continue
        positives = [row for row in subset if int(row["label"]) == 1]
        negatives = [row for row in subset if int(row["label"]) == 0]
        if positives:
            metrics[f"{source_kind}_recall"] = float(sum(int(row["pred"]) for row in positives)) / float(len(positives))
        if negatives:
            metrics[f"{source_kind}_false_positive_rate"] = float(sum(int(row["pred"]) for row in negatives)) / float(
                len(negatives)
            )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
