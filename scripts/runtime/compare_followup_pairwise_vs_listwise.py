"""Explicit runtime compare: pairwise vs listwise on followup."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_python() -> str:
    venv_py = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_py.exists():
        return str(venv_py)
    return sys.executable


def run(cmd: list[str]) -> None:
    print("[compare] exec:", " ".join(cmd))
    env = os.environ.copy()
    # Suppress noisy third-party tqdm bars in subprocess logs.
    env["TQDM_DISABLE"] = "1"
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    subprocess.run(cmd, cwd=REPO_ROOT, check=True, env=env)


def parse_eval_metrics(eval_metrics_path: Path) -> Tuple[float, float, float]:
    payload = json.loads(eval_metrics_path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    for record in records:
        if record.get("ablation_group") == "baseline(auto)" and record.get("policy") == "soft":
            metrics = record.get("metrics", {})
            return (
                float(metrics.get("recall_at_k", 0.0)),
                float(metrics.get("top1_acc", 0.0)),
                float(metrics.get("mrr", 0.0)),
            )
    if records:
        metrics = records[0].get("metrics", {})
        return (
            float(metrics.get("recall_at_k", 0.0)),
            float(metrics.get("top1_acc", 0.0)),
            float(metrics.get("mrr", 0.0)),
        )
    return 0.0, 0.0, 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare pairwise vs listwise on followup.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", default="followup")
    parser.add_argument("--encoder-backend", choices=("hf", "simple"), default="hf")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=5)
    # Pitfall note:
    # bootstrap on small/stable datasets mostly adds runtime cost.
    # Keep default at 0 and only enable when you explicitly need CI.
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--seed", type=int, default=11)
    # Pitfall note:
    # by default do NOT retrain here; this script compares existing weights.
    parser.add_argument("--train", action="store_true", help="Train pairwise/listwise before eval.")
    parser.add_argument("--device", default="cuda", help="Training device when --train is enabled.")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = REPO_ROOT / "runs" / "rt" / "compare_followup_pairwise_vs_listwise" / ts
    pair_train_dir = root / "pairwise" / "train"
    list_train_dir = root / "listwise" / "train"
    pair_eval_dir = root / "pairwise" / "eval"
    list_eval_dir = root / "listwise" / "eval"

    pair_weight = REPO_ROOT / "data" / "ModelWeights" / "pairwise_reranker.pt"
    list_weight = REPO_ROOT / "data" / "ModelWeights" / "listwise_reranker.pt"

    py = resolve_python()

    if args.train:
        run(
            [
                py,
                "-m",
                "scripts.train_pairwise_reranker",
                "--config",
                args.config,
                "--dataset",
                args.dataset,
                "--encoder-backend",
                args.encoder_backend,
                "--run-dir",
                str(pair_train_dir),
                "--save-path",
                str(pair_weight),
                "--epochs",
                str(args.epochs),
                "--top-n",
                str(args.top_n),
                "--seed",
                str(args.seed),
                "--device",
                str(args.device),
                "--hf-local-only",
            ]
        )

        run(
            [
                py,
                "-m",
                "scripts.train_listwise_reranker",
                "--config",
                args.config,
                "--dataset",
                args.dataset,
                "--encoder-backend",
                args.encoder_backend,
                "--run-dir",
                str(list_train_dir),
                "--save-path",
                str(list_weight),
                "--epochs",
                str(args.epochs),
                "--top-n",
                str(args.top_n),
                "--seed",
                str(args.seed),
                "--device",
                str(args.device),
                "--hf-local-only",
            ]
        )
    else:
        print("[compare] --train not set, skip training and use existing weight files.")
        if not pair_weight.exists() or not list_weight.exists():
            missing = []
            if not pair_weight.exists():
                missing.append(str(pair_weight))
            if not list_weight.exists():
                missing.append(str(list_weight))
            raise FileNotFoundError(
                "Missing weight file(s) while --train is disabled:\n- "
                + "\n- ".join(missing)
            )

    common_eval_args = [
        "--config",
        args.config,
        "--dataset",
        args.dataset,
        "--encoder-backend",
        args.encoder_backend,
        "--top-n",
        str(args.top_n),
        "--top-k",
        str(args.top_k),
        "--candidate-mode",
        "union",
        "--policies",
        "soft",
        "--ablation-groups",
        "baseline(auto)",
        "--bootstrap",
        str(args.bootstrap),
        "--no-consistency-pass",
        "--use-learned-scorer",
        "--hf-local-only",
    ]

    run(
        [
            py,
            "-m",
            "scripts.eval_router",
            *common_eval_args,
            "--reranker-path",
            str(pair_weight),
            "--run-dir",
            str(pair_eval_dir),
        ]
    )
    run(
        [
            py,
            "-m",
            "scripts.eval_router",
            *common_eval_args,
            "--reranker-path",
            str(list_weight),
            "--run-dir",
            str(list_eval_dir),
        ]
    )

    pair_metrics_path = pair_eval_dir / "eval.metrics.json"
    list_metrics_path = list_eval_dir / "eval.metrics.json"
    if not pair_metrics_path.exists():
        raise FileNotFoundError(
            f"pairwise eval metrics missing: {pair_metrics_path}\n"
            f"Check eval.log.txt under {pair_eval_dir} for failure details."
        )
    if not list_metrics_path.exists():
        raise FileNotFoundError(
            f"listwise eval metrics missing: {list_metrics_path}\n"
            f"Check eval.log.txt under {list_eval_dir} for failure details."
        )
    pair_metrics = parse_eval_metrics(pair_metrics_path)
    list_metrics = parse_eval_metrics(list_metrics_path)
    rows: Dict[str, Tuple[float, float, float]] = {
        "pairwise": pair_metrics,
        "listwise": list_metrics,
    }
    print("\n[compare] followup pairwise vs listwise")
    print("model      | Recall@5 | Top1    | MRR")
    print("-----------+----------+---------+---------")
    for name, (recall, top1, mrr) in rows.items():
        print(f"{name:<10} | {recall:.4f}   | {top1:.4f} | {mrr:.4f}")
    print(f"[compare] artifacts => {root}")


if __name__ == "__main__":
    main()
