from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


CATEGORY_MAP: dict[str, list[str]] = {
    "ingest": [
        "bootstrap_from_v4.py",
        "build_dataset.py",
        "build_queries.py",
        "build_e5_cache.py",
        "build_hybrid_index.py",
        "build_association_graph.py",
        "export_chat_bundle.py",
        "dedupe_jsonl_by_key.py",
    ],
    "quality": [
        "doctor_data_chain.py",
        "check_e5_similarity.py",
        "eval_offline.py",
        "inspect_association_graph.py",
        "smoke_coarse.py",
    ],
    "runtime": [
        "serve_retriever.py",
        "runtime_prepare_demo.py",
        "runtime_retrieve_demo.py",
        "runtime_associate_demo.py",
        "runtime_full_pipeline.py",
    ],
    "suppressor_legacy": [
        "build_suppressor_dataset.py",
        "train_suppressor.py",
        "eval_suppressor.py",
        "runtime_suppressor_demo.py",
        "generate_feedback_events_test.py",
        "tune_suppressor_by_top5_change.py",
        "compare_memory_preference_top5.py",
        "tune_suppressor_two_cohorts.py",
        "calibrate_suppressor.py",
    ],
    "mem_network_suppressor": [
        "build_mem_network_dataset.py",
        "train_mem_network_suppressor.py",
        "eval_mem_network_suppressor.py",
    ],
}


def _link_or_copy(src: Path, dst: Path, method: str) -> str:
    if dst.exists():
        dst.unlink()
    if method == "hardlink":
        try:
            os.link(src, dst)
            return "hardlink"
        except Exception:
            shutil.copy2(src, dst)
            return "copy_fallback"
    shutil.copy2(src, dst)
    return "copy"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Organize _V5/scripts into categorized folders while keeping root scripts intact."
    )
    parser.add_argument("--method", choices=["hardlink", "copy"], default="hardlink")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    scripts_root = Path(__file__).resolve().parent
    moved = 0
    stats: dict[str, int] = {"hardlink": 0, "copy": 0, "copy_fallback": 0, "missing": 0}

    for category, files in CATEGORY_MAP.items():
        category_dir = scripts_root / category
        if not args.dry_run:
            category_dir.mkdir(parents=True, exist_ok=True)
        for name in files:
            src = scripts_root / name
            dst = category_dir / name
            if not src.exists():
                stats["missing"] += 1
                print(f"[missing] {src}")
                continue
            if args.dry_run:
                print(f"[plan] {src.name} -> {category}/{src.name}")
                moved += 1
                continue
            mode = _link_or_copy(src, dst, args.method)
            stats[mode] = stats.get(mode, 0) + 1
            moved += 1
            print(f"[{mode}] {src.name} -> {category}/{src.name}")

    print(
        "[v5][scripts][organized]",
        {
            "root": str(scripts_root),
            "entries": moved,
            "method": args.method,
            "stats": stats,
            "categories": sorted(CATEGORY_MAP.keys()),
        },
    )


if __name__ == "__main__":
    main()
