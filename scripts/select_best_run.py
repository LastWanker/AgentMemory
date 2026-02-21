"""Select best run from runs/registry.csv and optionally materialize under runs/best/."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY = REPO_ROOT / "runs" / "registry.csv"
DEFAULT_BEST_ROOT = REPO_ROOT / "runs" / "best"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select best run by metric.")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY), help="Path to registry.csv")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--encoder-backend", required=True)
    parser.add_argument("--metric", default="Recall@5")
    parser.add_argument("--best-root", default=str(DEFAULT_BEST_ROOT))
    parser.add_argument(
        "--no-materialize",
        action="store_true",
        help="Only print best row, do not create runs/best artifacts.",
    )
    return parser.parse_args()


def read_registry(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows: List[Dict[str, str]] = []
        for row in csv.DictReader(handle):
            rows.append({(k or "").replace("\ufeff", ""): (v or "") for k, v in row.items()})
        return rows


def parse_metric(value: str) -> Optional[float]:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def resolve_weight_path(row: Dict[str, str]) -> Optional[Path]:
    weight_text = (row.get("weight_path") or "").strip()
    if weight_text:
        candidate = Path(weight_text)
        if candidate.exists():
            return candidate
    run_dir = (row.get("run_dir") or "").strip()
    if run_dir:
        candidate = Path(run_dir) / "tiny_reranker.pt"
        if candidate.exists():
            return candidate
    return None


def pick_best(
    rows: List[Dict[str, str]],
    *,
    dataset: str,
    encoder_backend: str,
    metric: str,
) -> Tuple[Optional[Dict[str, str]], Optional[float], Optional[Path]]:
    best_row: Optional[Dict[str, str]] = None
    best_value: Optional[float] = None
    best_weight: Optional[Path] = None
    for row in rows:
        if (row.get("dataset") or "") != dataset:
            continue
        if (row.get("encoder_backend") or "") != encoder_backend:
            continue
        value = parse_metric(row.get(metric, ""))
        if value is None:
            continue
        weight = resolve_weight_path(row)
        if weight is None:
            continue
        if best_value is None or value > best_value:
            best_value = value
            best_row = row
            best_weight = weight
    return best_row, best_value, best_weight


def materialize_best(
    best_root: Path,
    row: Dict[str, str],
    source_weight: Path,
    *,
    dataset: str,
    encoder_backend: str,
) -> Path:
    target_dir = best_root / dataset / encoder_backend
    target_dir.mkdir(parents=True, exist_ok=True)
    target_weight = target_dir / "tiny_reranker.pt"
    if target_weight.exists():
        target_weight.unlink()
    link_mode = "hardlink"
    try:
        os.link(source_weight, target_weight)
    except Exception:
        shutil.copy2(source_weight, target_weight)
        link_mode = "copy"

    run_dir = Path((row.get("run_dir") or "").strip())
    config_src = run_dir / "config.snapshot.json"
    if config_src.exists():
        shutil.copy2(config_src, target_dir / "config.snapshot.json")
    (target_dir / "best.meta.json").write_text(
        json.dumps(
            {
                "dataset": dataset,
                "encoder_backend": encoder_backend,
                "source_run_dir": row.get("run_dir", ""),
                "source_weight_path": str(source_weight),
                "materialized_weight_path": str(target_weight),
                "materialize_mode": link_mode,
                "registry_row": row,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return target_dir


def main() -> None:
    args = parse_args()
    rows = read_registry(Path(args.registry))
    best_row, best_value, best_weight = pick_best(
        rows,
        dataset=args.dataset,
        encoder_backend=args.encoder_backend,
        metric=args.metric,
    )
    if not best_row or best_value is None or best_weight is None:
        print("No eligible run found.")
        return

    print(f"metric={args.metric}")
    print(f"value={best_value:.6f}")
    print(f"run_dir={best_row.get('run_dir', '')}")
    print(f"weight_path={best_weight}")

    if args.no_materialize:
        return
    target_dir = materialize_best(
        Path(args.best_root),
        best_row,
        best_weight,
        dataset=args.dataset,
        encoder_backend=args.encoder_backend,
    )
    print(f"materialized_dir={target_dir}")


if __name__ == "__main__":
    main()
