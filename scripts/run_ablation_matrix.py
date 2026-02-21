"""Run ablation matrix for reranker training + eval with bootstrap CI.

Usage:
    python scripts/run_ablation_matrix.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = REPO_ROOT / "runs"
MODEL_WEIGHTS_DIR = REPO_ROOT / "data" / "ModelWeights"
RUNS_REGISTRY_PATH = RUNS_ROOT / "registry.csv"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.runtime_utils import (
    cfg_get,
    get_git_commit,
    load_config,
    upsert_run_registry,
    write_json,
)

DEFAULT_BOOTSTRAP = 500
DEFAULT_SEEDS = [11, 29, 47]


@dataclass(frozen=True)
class ConfigSpec:
    name: str
    train: bool
    loss_type: Optional[str] = None
    neg_strategy: Optional[str] = None
    use_cardinality_head: bool = False
    use_learned_scorer: bool = False


CONFIGS: List[ConfigSpec] = [
    ConfigSpec(
        name="baseline",
        train=False,
        use_learned_scorer=False,
    ),
    ConfigSpec(
        name="pairwise_random",
        train=True,
        loss_type="pairwise",
        neg_strategy="random",
        use_learned_scorer=True,
    ),
    ConfigSpec(
        name="pairwise_ranked",
        train=True,
        loss_type="pairwise",
        neg_strategy="ranked",
        use_learned_scorer=True,
    ),
    ConfigSpec(
        name="listwise_random",
        train=True,
        loss_type="listwise",
        neg_strategy="random",
        use_learned_scorer=True,
    ),
    ConfigSpec(
        name="listwise_ranked",
        train=True,
        loss_type="listwise",
        neg_strategy="ranked",
        use_learned_scorer=True,
    ),
    ConfigSpec(
        name="listwise_ranked_cardinality",
        train=True,
        loss_type="listwise",
        neg_strategy="ranked",
        use_cardinality_head=True,
        use_learned_scorer=True,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ablation matrix end-to-end.")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "default.yaml"),
        help="配置文件路径（JSON 或 YAML）",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=DEFAULT_BOOTSTRAP,
        help=f"Bootstrap samples for eval_router.py (default: {DEFAULT_BOOTSTRAP})",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help=f"Comma-separated seeds (default: {','.join(str(seed) for seed in DEFAULT_SEEDS)})",
    )
    parser.add_argument(
        "--configs",
        default="all",
        help="Comma-separated config names, or 'all'.",
    )
    parser.add_argument(
        "--suite",
        choices=("listwise_vs_pairwise",),
        help="快捷实验套件名；与 --configs 二选一",
    )
    parser.add_argument(
        "--dataset",
        help="覆盖数据集（如 followup）；未传则使用 config/common.dataset",
    )
    parser.add_argument(
        "--runs-root",
        default=str(RUNS_ROOT),
        help="Output root directory for runs.",
    )
    parser.add_argument(
        "--run-dir",
        help="Optional exact run directory. If provided, do not append timestamp under runs-root.",
    )
    parser.add_argument(
        "--encoder-backend",
        choices=("hf", "simple"),
        help="透传到训练与评测；未传时从 config/common.encoder_backend 读取",
    )
    return parser.parse_args()


def parse_seeds(raw: str) -> List[int]:
    seeds: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        seeds.append(int(part))
    if not seeds:
        raise ValueError("No seeds provided.")
    return seeds


def parse_configs(raw: str) -> List[ConfigSpec]:
    if raw.strip().lower() == "all":
        return CONFIGS
    wanted = {name.strip() for name in raw.split(",") if name.strip()}
    config_map = {config.name: config for config in CONFIGS}
    missing = sorted(name for name in wanted if name not in config_map)
    if missing:
        raise ValueError(f"Unknown config(s): {', '.join(missing)}")
    return [config for config in CONFIGS if config.name in wanted]


def parse_suite(name: Optional[str]) -> Optional[List[ConfigSpec]]:
    if not name:
        return None
    if name == "listwise_vs_pairwise":
        wanted = {"baseline", "pairwise_ranked", "listwise_ranked"}
        return [config for config in CONFIGS if config.name in wanted]
    raise ValueError(f"Unknown suite: {name}")


def run_command(cmd: List[str], output_path: Path) -> str:
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    combined = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    output_path.write_text(combined, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit={proc.returncode}): {' '.join(cmd)}\n"
            f"See log: {output_path}"
        )
    return combined


def parse_eval_metrics(output: str) -> Dict[str, float]:
    metric_matches = re.findall(
        r"Recall@k=([0-9]*\.?[0-9]+)\s+\|\s+MRR=([0-9]*\.?[0-9]+)\s+\|\s+Top1=([0-9]*\.?[0-9]+)",
        output,
    )
    if not metric_matches:
        raise ValueError("Failed to parse core metrics from eval output.")
    recall, mrr, top1 = metric_matches[-1]

    ci_matches = re.findall(
        r"Recall@k CI:\s*([0-9]*\.?[0-9]+)\s*±\s*([0-9]*\.?[0-9]+)\s*"
        r"\(95% CI \[([0-9]*\.?[0-9]+),\s*([0-9]*\.?[0-9]+)\]\)",
        output,
    )
    if ci_matches:
        recall_mean_ci, recall_std_ci, recall_low_ci, recall_high_ci = ci_matches[-1]
    else:
        recall_mean_ci = recall
        recall_std_ci = "0.0"
        recall_low_ci = recall
        recall_high_ci = recall

    return {
        "Recall@5_mean": float(recall_mean_ci),
        "Recall@5_CI_low": float(recall_low_ci),
        "Recall@5_CI_high": float(recall_high_ci),
        "Recall@5_std": float(recall_std_ci),
        "Top1_mean": float(top1),
        "MRR_mean": float(mrr),
    }


def train_one(
    config: ConfigSpec,
    seed: int,
    weight_path: Path,
    train_log_path: Path,
    train_settings: Dict[str, object],
    run_dir: Path,
    config_path: str,
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "scripts.train_reranker",
        "--config",
        config_path,
        "--run-dir",
        str(run_dir),
        "--dataset",
        str(train_settings["dataset"]),
        "--candidate-mode",
        str(train_settings["candidate_mode"]),
        "--top-n",
        str(train_settings["top_n"]),
        "--epochs",
        str(train_settings["epochs"]),
        "--batch-size",
        str(train_settings["batch_size"]),
        "--lr",
        str(train_settings["lr"]),
        "--device",
        str(train_settings["device"]),
        "--encoder-backend",
        str(train_settings["encoder_backend"]),
        "--save-path",
        str(weight_path),
        "--seed",
        str(seed),
    ]
    if config.loss_type:
        cmd.extend(["--loss-type", config.loss_type])
    if config.neg_strategy:
        cmd.extend(["--neg-strategy", config.neg_strategy])
    if config.use_cardinality_head:
        cmd.append("--use-cardinality-head")
    run_command(cmd, train_log_path)


def eval_one(
    config: ConfigSpec,
    bootstrap: int,
    weight_path: Path,
    eval_log_path: Path,
    eval_settings: Dict[str, object],
    run_dir: Path,
    config_path: str,
) -> Dict[str, float]:
    cmd = [
        sys.executable,
        "-m",
        "scripts.eval_router",
        "--config",
        config_path,
        "--run-dir",
        str(run_dir),
        "--dataset",
        str(eval_settings["dataset"]),
        "--top-n",
        str(eval_settings["top_n"]),
        "--top-k",
        str(eval_settings["top_k"]),
        "--candidate-mode",
        str(eval_settings["candidate_mode"]),
        "--policies",
        str(eval_settings["policies"]),
        "--ablation",
        str(eval_settings["ablation"]),
        "--bootstrap",
        str(bootstrap),
        "--encoder-backend",
        str(eval_settings["encoder_backend"]),
        "--no-trace",
    ]
    if config.use_learned_scorer:
        cmd.extend(["--use-learned-scorer", "--reranker-path", str(weight_path)])

    output = run_command(cmd, eval_log_path)
    eval_metrics_path = run_dir / "eval.metrics.json"
    if eval_metrics_path.exists():
        try:
            payload = json.loads(eval_metrics_path.read_text(encoding="utf-8"))
            records = payload.get("records", [])
            if isinstance(records, list) and records:
                for record in records:
                    if record.get("ablation_group") == "baseline(auto)" and record.get("policy") == "soft":
                        metrics = record.get("metrics", {})
                        bs = record.get("bootstrap", {})
                        recall = bs.get("recall_at_k", [metrics.get("recall_at_k", 0.0), 0.0, metrics.get("recall_at_k", 0.0), metrics.get("recall_at_k", 0.0)])
                        return {
                            "Recall@5_mean": float(recall[0]),
                            "Recall@5_CI_low": float(recall[2]),
                            "Recall@5_CI_high": float(recall[3]),
                            "Recall@5_std": float(recall[1]),
                            "Top1_mean": float(metrics.get("top1_acc", 0.0)),
                            "MRR_mean": float(metrics.get("mrr", 0.0)),
                        }
        except Exception:
            pass
    return parse_eval_metrics(output)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    loaded_config = load_config(args.config)
    seeds = parse_seeds(args.seeds)
    if args.suite and args.configs != "all":
        raise ValueError("--suite 与 --configs 不能同时指定")
    suite_configs = parse_suite(args.suite)
    configs = suite_configs if suite_configs is not None else parse_configs(args.configs)
    runs_root = Path(args.runs_root)

    common_dataset = args.dataset or cfg_get(loaded_config, "common.dataset", "normal")
    common_candidate_mode = cfg_get(loaded_config, "common.candidate_mode", "union")
    common_top_n = int(cfg_get(loaded_config, "common.top_n", 20))
    common_top_k = int(cfg_get(loaded_config, "common.top_k", 5))
    common_backend = cfg_get(loaded_config, "common.encoder_backend", "simple")
    encoder_backend = args.encoder_backend or common_backend

    eval_settings: Dict[str, object] = {
        "dataset": common_dataset,
        "top_n": common_top_n,
        "top_k": common_top_k,
        "candidate_mode": common_candidate_mode,
        "policies": cfg_get(loaded_config, "ablation_matrix.policies", "soft"),
        "ablation": cfg_get(loaded_config, "ablation_matrix.ablation", "baseline"),
        "encoder_backend": encoder_backend,
    }
    train_settings: Dict[str, object] = {
        "dataset": common_dataset,
        "candidate_mode": common_candidate_mode,
        "top_n": common_top_n,
        "epochs": int(cfg_get(loaded_config, "train_reranker.epochs", 3)),
        "batch_size": int(cfg_get(loaded_config, "train_reranker.batch_size", 16)),
        "lr": float(cfg_get(loaded_config, "train_reranker.lr", 1e-3)),
        "device": cfg_get(loaded_config, "train_reranker.device", "cpu"),
        "encoder_backend": encoder_backend,
    }

    if args.run_dir:
        run_root = Path(args.run_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_root = runs_root / timestamp
    ensure_dir(run_root)

    print(f"[ablation] run_root={run_root}")
    print(f"[ablation] configs={','.join(config.name for config in configs)}")
    print(
        f"[ablation] seeds={','.join(str(seed) for seed in seeds)} | "
        f"bootstrap={args.bootstrap} | backend={encoder_backend}"
    )
    write_json(
        run_root / "config.snapshot.json",
        {
            "config_path": args.config,
            "loaded_config": loaded_config,
            "effective_train_settings": train_settings,
            "effective_eval_settings": eval_settings,
            "effective_args": vars(args),
        },
    )

    summary_rows: List[Dict[str, object]] = []

    for config in configs:
        group_dir = run_root / config.name
        ensure_dir(group_dir)
        for seed in seeds:
            seed_dir = group_dir / f"seed_{seed}"
            ensure_dir(seed_dir)
            print(f"[ablation] start config={config.name} seed={seed}")

            weight_path = seed_dir / "tiny_reranker.pt"
            train_log = seed_dir / "train.log.txt"
            eval_log = seed_dir / "eval.log.txt"
            eval_json = seed_dir / "eval.metrics.json"
            run_meta = seed_dir / "run.meta.json"
            weight_ref = seed_dir / "weight_path.txt"
            model_weights_copy = MODEL_WEIGHTS_DIR / f"{config.name}__seed_{seed}.pt"
            ensure_dir(MODEL_WEIGHTS_DIR)

            if config.train:
                train_one(
                    config,
                    seed,
                    weight_path,
                    train_log,
                    train_settings=train_settings,
                    run_dir=seed_dir,
                    config_path=args.config,
                )
            else:
                train_log.write_text(
                    "baseline config: no training step, rule-based scorer evaluation only.\n",
                    encoding="utf-8",
                )

            metrics = eval_one(
                config,
                args.bootstrap,
                weight_path,
                eval_log,
                eval_settings=eval_settings,
                run_dir=seed_dir,
                config_path=args.config,
            )
            if config.train and weight_path.exists():
                shutil.copy2(weight_path, model_weights_copy)
            write_json(eval_json, metrics)
            weight_ref.write_text(str(weight_path), encoding="utf-8")
            write_json(
                run_meta,
                {
                    "config_name": config.name,
                    "seed": seed,
                    "bootstrap": args.bootstrap,
                    "train_enabled": config.train,
                    "train_loss_type": config.loss_type,
                    "train_neg_strategy": config.neg_strategy,
                    "train_use_cardinality_head": config.use_cardinality_head,
                    "eval_use_learned_scorer": config.use_learned_scorer,
                    "train_log": str(train_log),
                    "eval_log": str(eval_log),
                    "eval_metrics_json": str(eval_json),
                    "weight_path": str(weight_path),
                    "model_weights_copy": str(model_weights_copy),
                    "encoder_backend": encoder_backend,
                },
            )

            row = {
                "config_name": config.name,
                "seed": seed,
                **metrics,
            }
            summary_rows.append(row)
            upsert_run_registry(
                RUNS_REGISTRY_PATH,
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "run_dir": str(seed_dir),
                    "dataset": common_dataset,
                    "encoder_backend": encoder_backend,
                    "encoder_id": "",
                    "strategy": "token_pool_topk",
                    "top_n": common_top_n,
                    "top_k": common_top_k,
                    "policies": str(eval_settings["policies"]),
                    "candidate_mode": common_candidate_mode,
                    "loss_type": config.loss_type or "",
                    "neg_strategy": config.neg_strategy or "",
                    "Recall@5": metrics.get("Recall@5_mean", ""),
                    "Top1": metrics.get("Top1_mean", ""),
                    "MRR": metrics.get("MRR_mean", ""),
                    "weight_path": str(weight_path) if config.train else "",
                    "git_commit": get_git_commit(REPO_ROOT),
                    "config_name": config.name,
                    "seed": seed,
                },
            )
            print(
                "[ablation] done "
                f"config={config.name} seed={seed} "
                f"Recall@5={metrics['Recall@5_mean']:.3f} "
                f"CI=[{metrics['Recall@5_CI_low']:.3f},{metrics['Recall@5_CI_high']:.3f}] "
                f"Top1={metrics['Top1_mean']:.3f} MRR={metrics['MRR_mean']:.3f}"
            )

    summary_csv = run_root / "summary.csv"
    fieldnames = [
        "config_name",
        "seed",
        "Recall@5_mean",
        "Recall@5_CI_low",
        "Recall@5_CI_high",
        "Top1_mean",
        "MRR_mean",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    print(f"[ablation] summary={summary_csv}")


if __name__ == "__main__":
    main()
