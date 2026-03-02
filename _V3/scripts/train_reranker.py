from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch


V3_ROOT = Path(__file__).resolve().parents[1]
if str(V3_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(V3_ROOT / "src"))

from agentmemory_v3.config import cfg_get, load_yaml_config, resolve_path
from agentmemory_v3.training.trainer import TrainConfig, train_model


def _resolve_device(requested: str, cfg: dict) -> tuple[str, int]:
    raw = (requested or "").strip().lower()
    if not raw or raw == "auto":
        preferred = str(cfg_get(cfg, "training.device", "auto")).strip().lower()
    else:
        preferred = raw
    cuda_ok = bool(torch.cuda.is_available())
    if preferred == "cuda" and not cuda_ok:
        print("[v3][train] cuda requested but unavailable, fallback to cpu")
        return "cpu", int(cfg_get(cfg, "training.batch_size_cpu", cfg_get(cfg, "training.batch_size", 8)))
    if preferred in ("", "auto"):
        if cuda_ok:
            return "cuda", int(cfg_get(cfg, "training.batch_size_cuda", cfg_get(cfg, "training.batch_size", 16)))
        return "cpu", int(cfg_get(cfg, "training.batch_size_cpu", cfg_get(cfg, "training.batch_size", 8)))
    if preferred == "cuda":
        return "cuda", int(cfg_get(cfg, "training.batch_size_cuda", cfg_get(cfg, "training.batch_size", 16)))
    return "cpu", int(cfg_get(cfg, "training.batch_size_cpu", cfg_get(cfg, "training.batch_size", 8)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train V3 reranker.")
    parser.add_argument("--config", default="_V3/configs/default.yaml")
    parser.add_argument("--samples", default="data/V3/training/train_samples.jsonl")
    parser.add_argument("--model-out", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--family", default="")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.0)
    args = parser.parse_args()
    cfg = load_yaml_config(args.config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    resolved_device, default_batch_size = _resolve_device(args.device, cfg)
    family = args.family or str(cfg_get(cfg, "reranker.family", "slot_bipartite_8x8"))
    model_out = args.model_out or cfg_get(
        cfg,
        "reranker.model_path" if family.startswith("slot_bipartite") else "reranker.fallback_model_path",
        "data/V3/models/slot_bipartite_reranker.pt" if family.startswith("slot_bipartite") else "data/V3/models/reranker.pt",
    )
    train_cfg = TrainConfig(
        sample_path=resolve_path(args.samples),
        model_path=resolve_path(model_out),
        run_dir=resolve_path(Path("_V3") / "runs" / "reranker_train" / timestamp),
        epochs=int(args.epochs or cfg_get(cfg, "training.epochs", 8)),
        batch_size=int(args.batch_size or default_batch_size),
        learning_rate=float(args.learning_rate or cfg_get(cfg, "training.learning_rate", 1e-3)),
        hidden_dim=int(cfg_get(cfg, "reranker.hidden_dim", 64)),
        dropout=float(cfg_get(cfg, "reranker.dropout", 0.1)),
        device=resolved_device,
        family=family,
        data_root=resolve_path(cfg_get(cfg, "data.root_dir", "data/V3")),
        query_slots_path=resolve_path(cfg_get(cfg, "query.slots_path", "data/V3/processed/query_slots.jsonl")),
        query_path=resolve_path("data/V3/processed/query.jsonl"),
        input_dim=int(cfg_get(cfg, "reranker.input_dim", 192)),
        seq_len=int(cfg_get(cfg, "reranker.seq_len", 8)),
        proj_dim=int(cfg_get(cfg, "reranker.proj_dim", 128)),
        d_model=int(cfg_get(cfg, "reranker.d_model", 256)),
        num_heads=int(cfg_get(cfg, "reranker.num_heads", 4)),
        num_layers=int(cfg_get(cfg, "reranker.num_layers", 3)),
        tau=float(cfg_get(cfg, "reranker.tau", 0.1)),
        learnable_tau=bool(cfg_get(cfg, "reranker.learnable_tau", False)),
    )
    result = train_model(train_cfg)
    print(
        f"[v3] reranker family={family} device={resolved_device} batch_size={train_cfg.batch_size} "
        f"train rows={result['train_rows']} valid_rows={result['valid_rows']} "
        f"best_valid_mrr={result['best_valid_mrr']:.4f} -> {result['model_path']}"
    )


if __name__ == "__main__":
    main()
