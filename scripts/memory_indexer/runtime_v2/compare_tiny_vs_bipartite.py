"""Compare tiny listwise scorer vs bipartite listwise scorer under same eval setup."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default_v2_bipartite.yaml"
TINY_WEIGHT = REPO_ROOT / "data" / "ModelWeights" / "listwise_reranker.pt"
BIPARTITE_WEIGHT = REPO_ROOT / "data" / "ModelWeights" / "listwise_bipartite_reranker.pt"


def resolve_python() -> str:
    venv_py = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_py.exists():
        return str(venv_py)
    return sys.executable


def run(cmd: list[str]) -> None:
    print("[rt_v2][compare] exec:", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def load_primary_metrics(metrics_path: Path) -> Optional[Dict[str, float]]:
    if not metrics_path.exists():
        return None
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if not records:
        return None
    for record in records:
        if record.get("ablation_group") == "S-only" and record.get("policy") == "half_hard":
            metrics = record.get("metrics", {})
            if isinstance(metrics, dict):
                return {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
    first = records[0].get("metrics", {})
    if isinstance(first, dict):
        return {k: float(v) for k, v in first.items() if isinstance(v, (int, float))}
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare tiny vs bipartite scorer.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--dataset", default="followup_plus_chat")
    parser.add_argument("--encoder-backend", choices=("hf", "simple"), default="hf")
    parser.add_argument("--cache-alias", default="users")
    parser.add_argument("--max-eval-queries", type=int, default=1000)
    parser.add_argument("--eval-sample-mode", choices=("random", "head"), default="random")
    parser.add_argument("--eval-sample-seed", type=int, default=11)
    parser.add_argument("--use-cache-signature", action="store_true")
    parser.add_argument("--hf-online", action="store_true")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = REPO_ROOT / "runs" / "rt_v2" / "compare_tiny_vs_bipartite" / ts
    py = resolve_python()
    cache_alias = args.cache_alias
    if args.encoder_backend != "hf" and cache_alias == "users":
        cache_alias = "users_simple"

    def eval_one(tag: str, weight_path: Path) -> Optional[Path]:
        if not weight_path.exists():
            print(f"[rt_v2][compare] skip {tag}: missing weight {weight_path}")
            return None
        run_dir = root / tag / "eval"
        cmd = [
            py,
            "-m",
            "scripts.memory_indexer.eval_router",
            "--dataset",
            args.dataset,
            "--encoder-backend",
            args.encoder_backend,
            "--config",
            args.config,
            "--candidate-mode",
            "coarse",
            "--policies",
            "half_hard",
            "--ablation-groups",
            "S-only",
            "--cache-alias",
            cache_alias,
            "--use-learned-scorer",
            "--reranker-path",
            str(weight_path),
            "--max-eval-queries",
            str(args.max_eval_queries),
            "--eval-sample-mode",
            args.eval_sample_mode,
            "--eval-sample-seed",
            str(args.eval_sample_seed),
            "--run-dir",
            str(run_dir),
        ]
        if not args.use_cache_signature:
            cmd.append("--no-cache-signature")
        if not args.hf_online:
            cmd.append("--hf-local-only")
        run(cmd)
        return run_dir / "eval.metrics.json"

    tiny_metrics_path = eval_one("tiny", TINY_WEIGHT)
    bip_metrics_path = eval_one("bipartite", BIPARTITE_WEIGHT)
    tiny = load_primary_metrics(tiny_metrics_path) if tiny_metrics_path else None
    bip = load_primary_metrics(bip_metrics_path) if bip_metrics_path else None

    print(f"[rt_v2][compare] root={root}")
    if tiny:
        print(
            "[rt_v2][compare] tiny "
            f"Recall@k={tiny.get('recall_at_k', 0.0):.3f} "
            f"Recall@P={tiny.get('recall_at_pos_count', 0.0):.3f} "
            f"Gain@k={tiny.get('gain_at_k', 0.0):.3f} "
            f"Gain@P={tiny.get('gain_at_pos_count', 0.0):.3f}"
        )
    if bip:
        print(
            "[rt_v2][compare] bipartite "
            f"Recall@k={bip.get('recall_at_k', 0.0):.3f} "
            f"Recall@P={bip.get('recall_at_pos_count', 0.0):.3f} "
            f"Gain@k={bip.get('gain_at_k', 0.0):.3f} "
            f"Gain@P={bip.get('gain_at_pos_count', 0.0):.3f}"
        )


if __name__ == "__main__":
    main()

