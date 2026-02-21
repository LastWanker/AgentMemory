"""List recent run records from runs/registry.csv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY = REPO_ROOT / "runs" / "registry.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List recent runs from registry.")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY), help="Path to registry.csv")
    parser.add_argument("--dataset")
    parser.add_argument("--encoder-backend")
    parser.add_argument("--loss-type")
    parser.add_argument("--neg-strategy")
    parser.add_argument("--config-name")
    parser.add_argument("--limit", type=int, default=20)
    return parser.parse_args()


def read_registry(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows: List[Dict[str, str]] = []
        for row in csv.DictReader(handle):
            rows.append({(k or "").replace("\ufeff", ""): (v or "") for k, v in row.items()})
        return rows


def filter_rows(rows: List[Dict[str, str]], args: argparse.Namespace) -> List[Dict[str, str]]:
    filtered = rows
    if args.dataset:
        filtered = [r for r in filtered if (r.get("dataset") or "") == args.dataset]
    if args.encoder_backend:
        filtered = [r for r in filtered if (r.get("encoder_backend") or "") == args.encoder_backend]
    if args.loss_type:
        filtered = [r for r in filtered if (r.get("loss_type") or "") == args.loss_type]
    if args.neg_strategy:
        filtered = [r for r in filtered if (r.get("neg_strategy") or "") == args.neg_strategy]
    if args.config_name:
        filtered = [r for r in filtered if (r.get("config_name") or "") == args.config_name]
    filtered.sort(key=lambda row: (row.get("timestamp") or "", row.get("run_dir") or ""), reverse=True)
    if args.limit > 0:
        filtered = filtered[: args.limit]
    return filtered


def print_table(rows: List[Dict[str, str]]) -> None:
    if not rows:
        print("No runs found.")
        return
    columns = [
        "timestamp",
        "dataset",
        "encoder_backend",
        "loss_type",
        "neg_strategy",
        "Recall@5",
        "Top1",
        "MRR",
        "weight_path",
        "run_dir",
    ]
    widths = {col: len(col) for col in columns}
    for row in rows:
        for col in columns:
            widths[col] = max(widths[col], len((row.get(col) or "").strip()))
    header = " | ".join(col.ljust(widths[col]) for col in columns)
    sep = "-+-".join("-" * widths[col] for col in columns)
    print(header)
    print(sep)
    for row in rows:
        print(" | ".join((row.get(col) or "").strip().ljust(widths[col]) for col in columns))


def main() -> None:
    args = parse_args()
    rows = read_registry(Path(args.registry))
    filtered = filter_rows(rows, args)
    print_table(filtered)


if __name__ == "__main__":
    main()
