from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from agentmemory_v3.models.reranker import SlotFeatureReranker, save_reranker
from agentmemory_v3.models.slot_bipartite import SlotBipartiteAlignTransformer, save_slot_bipartite
from agentmemory_v3.training.slot_sequences import load_query_slot_sequences, memory_slot_sequences
from agentmemory_v3.utils.io import ensure_parent, read_jsonl


@dataclass
class TrainConfig:
    sample_path: Path
    model_path: Path
    run_dir: Path
    epochs: int = 8
    batch_size: int = 32
    learning_rate: float = 1e-3
    hidden_dim: int = 64
    dropout: float = 0.1
    device: str = "cpu"
    family: str = "feature_mlp"
    data_root: Path | None = None
    query_slots_path: Path | None = None
    query_path: Path | None = None
    input_dim: int = 192
    seq_len: int = 8
    proj_dim: int = 128
    d_model: int = 256
    num_heads: int = 4
    num_layers: int = 3
    tau: float = 0.1
    learnable_tau: bool = False
    cache_dir: Path | None = None
    cache_alias: str = "users"


class ListwiseFeatureDataset(Dataset):
    def __init__(self, rows: list[dict], mean: np.ndarray, std: np.ndarray) -> None:
        self.rows = rows
        self.mean = mean.astype(np.float32)
        self.std = np.where(std == 0.0, 1.0, std).astype(np.float32)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        features = np.asarray(row["features"], dtype=np.float32)
        features = (features - self.mean) / self.std
        labels = np.asarray(row["labels"], dtype=np.float32)
        return {"features": features, "labels": labels}


class ListwiseBipartiteDataset(Dataset):
    def __init__(
        self,
        rows: list[dict],
        *,
        memory_seq: np.ndarray,
        memory_mask: np.ndarray,
        mem_id_to_idx: dict[str, int],
        query_seq_map: dict[str, tuple[np.ndarray, np.ndarray]],
    ) -> None:
        self.rows = rows
        self.memory_seq = memory_seq
        self.memory_mask = memory_mask
        self.mem_id_to_idx = mem_id_to_idx
        self.query_seq_map = query_seq_map

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        query_id = str(row.get("query_id") or "")
        if query_id not in self.query_seq_map:
            raise KeyError(f"query slot sequence missing: {query_id}")
        q_seq, q_mask = self.query_seq_map[query_id]
        candidate_ids = [str(item) for item in row.get("candidate_ids") or []]
        candidate_indices = [self.mem_id_to_idx[mem_id] for mem_id in candidate_ids if mem_id in self.mem_id_to_idx]
        labels = np.asarray(row["labels"], dtype=np.float32)[: len(candidate_indices)]
        return {
            "q_seq": q_seq,
            "q_mask": q_mask,
            "m_seq": self.memory_seq[candidate_indices],
            "m_mask": self.memory_mask[candidate_indices],
            "labels": labels,
        }


def train_model(config: TrainConfig) -> dict[str, Any]:
    if config.family.startswith("slot_bipartite"):
        return train_bipartite_model(config)
    return train_feature_model(config)


def train_feature_model(config: TrainConfig) -> dict[str, Any]:
    rows = list(read_jsonl(config.sample_path))
    if not rows:
        raise RuntimeError(f"training sample file is empty: {config.sample_path}")
    train_rows = [row for row in rows if row.get("split") == "train"]
    valid_rows = [row for row in rows if row.get("split") == "valid"]
    if not train_rows:
        raise RuntimeError("no train rows found in sample file")
    feature_names = [str(item) for item in train_rows[0].get("feature_names") or []]
    feature_mean, feature_std = _compute_feature_stats(train_rows)
    train_loader = DataLoader(
        ListwiseFeatureDataset(train_rows, feature_mean, feature_std),
        batch_size=max(1, int(config.batch_size)),
        shuffle=True,
        collate_fn=_collate_feature_batch,
    )
    valid_loader = (
        DataLoader(
            ListwiseFeatureDataset(valid_rows, feature_mean, feature_std),
            batch_size=max(1, int(config.batch_size)),
            shuffle=False,
            collate_fn=_collate_feature_batch,
        )
        if valid_rows
        else None
    )
    device = torch.device(config.device)
    model = SlotFeatureReranker(len(feature_names), hidden_dim=config.hidden_dim, dropout=config.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history, best_state, best_valid_mrr = _fit_feature_model(model, train_loader, valid_loader, optimizer, device, config.epochs)
    if best_state is not None:
        model.load_state_dict(best_state)
    save_reranker(
        config.model_path,
        model=model,
        feature_names=feature_names,
        feature_mean=feature_mean,
        feature_std=feature_std,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        meta={
            "family": "feature_mlp",
            "epochs": int(config.epochs),
            "batch_size": int(config.batch_size),
            "learning_rate": float(config.learning_rate),
            "train_rows": len(train_rows),
            "valid_rows": len(valid_rows),
        },
    )
    return _write_summary(config.run_dir, config.model_path, train_rows, valid_rows, best_valid_mrr, len(feature_names), history)


def train_bipartite_model(config: TrainConfig) -> dict[str, Any]:
    if config.data_root is None or config.query_slots_path is None:
        raise RuntimeError("slot bipartite training requires data_root and query_slots_path")
    rows = list(read_jsonl(config.sample_path))
    if not rows:
        raise RuntimeError(f"training sample file is empty: {config.sample_path}")
    train_rows = [row for row in rows if row.get("split") == "train"]
    valid_rows = [row for row in rows if row.get("split") == "valid"]
    memory_rows = list(read_jsonl(config.data_root / "processed" / "memory.jsonl"))
    mem_id_to_idx = {row["memory_id"]: idx for idx, row in enumerate(memory_rows)}
    slot_blob = np.load(config.data_root / "indexes" / "slot_vectors.npz")
    slot_vectors = {key: slot_blob[key] for key in slot_blob.files}
    memory_seq, memory_mask = memory_slot_sequences(slot_vectors)
    from agentmemory_v3.retrieval.dense_index import DenseIndex
    from agentmemory_v3.retrieval.e5_cache import load_cache_maps

    dense_index = DenseIndex.load(config.data_root / "indexes" / "dense_artifact.pkl", config.data_root / "indexes" / "dense_matrix.npy")
    query_slot_vector_map = {}
    if config.cache_dir is not None:
        try:
            _query_coarse_map, query_slot_vector_map = load_cache_maps(config.cache_dir, config.cache_alias, kind="query")
        except Exception:
            query_slot_vector_map = {}
    query_seq_map = load_query_slot_sequences(
        config.query_slots_path,
        dense_index,
        query_path=config.query_path,
        query_slot_vector_map=query_slot_vector_map,
    )
    train_loader = DataLoader(
        ListwiseBipartiteDataset(
            train_rows,
            memory_seq=memory_seq,
            memory_mask=memory_mask,
            mem_id_to_idx=mem_id_to_idx,
            query_seq_map=query_seq_map,
        ),
        batch_size=max(1, int(config.batch_size)),
        shuffle=True,
        collate_fn=_collate_bipartite_batch,
    )
    valid_loader = (
        DataLoader(
            ListwiseBipartiteDataset(
                valid_rows,
                memory_seq=memory_seq,
                memory_mask=memory_mask,
                mem_id_to_idx=mem_id_to_idx,
                query_seq_map=query_seq_map,
            ),
            batch_size=max(1, int(config.batch_size)),
            shuffle=False,
            collate_fn=_collate_bipartite_batch,
        )
        if valid_rows
        else None
    )
    device = torch.device(config.device)
    model = SlotBipartiteAlignTransformer(
        input_dim=config.input_dim,
        seq_len=config.seq_len,
        proj_dim=config.proj_dim,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout,
        tau=config.tau,
        learnable_tau=config.learnable_tau,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history, best_state, best_valid_mrr = _fit_bipartite_model(model, train_loader, valid_loader, optimizer, device, config.epochs)
    if best_state is not None:
        model.load_state_dict(best_state)
    save_slot_bipartite(
        config.model_path,
        model=model,
        meta={
            "family": "slot_bipartite_8x8",
            "epochs": int(config.epochs),
            "batch_size": int(config.batch_size),
            "learning_rate": float(config.learning_rate),
            "train_rows": len(train_rows),
            "valid_rows": len(valid_rows),
            "input_dim": int(config.input_dim),
            "seq_len": int(config.seq_len),
            "proj_dim": int(config.proj_dim),
            "d_model": int(config.d_model),
            "num_heads": int(config.num_heads),
            "num_layers": int(config.num_layers),
            "dropout": float(config.dropout),
            "tau": float(model.current_tau().detach().cpu().item()),
            "learnable_tau": bool(config.learnable_tau),
        },
    )
    return _write_summary(config.run_dir, config.model_path, train_rows, valid_rows, best_valid_mrr, config.input_dim, history)


def _fit_feature_model(model, train_loader, valid_loader, optimizer, device, epochs):
    history = []
    best_valid_mrr = -math.inf
    best_state = None
    for epoch in range(1, int(epochs) + 1):
        train_metrics = _run_feature_epoch(model, train_loader, optimizer, device)
        valid_metrics = (
            _run_feature_epoch(model, valid_loader, None, device)
            if valid_loader
            else {"loss": train_metrics["loss"], "mrr": train_metrics["mrr"], "top1": train_metrics["top1"]}
        )
        history.append(_epoch_row(epoch, train_metrics, valid_metrics))
        if valid_metrics["mrr"] >= best_valid_mrr:
            best_valid_mrr = valid_metrics["mrr"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(
            f"[v4][train] epoch={epoch} train_loss={train_metrics['loss']:.4f} train_mrr={train_metrics['mrr']:.4f} "
            f"valid_loss={valid_metrics['loss']:.4f} valid_mrr={valid_metrics['mrr']:.4f}"
        )
    return history, best_state, best_valid_mrr


def _fit_bipartite_model(model, train_loader, valid_loader, optimizer, device, epochs):
    history = []
    best_valid_mrr = -math.inf
    best_state = None
    for epoch in range(1, int(epochs) + 1):
        train_metrics = _run_bipartite_epoch(model, train_loader, optimizer, device)
        valid_metrics = (
            _run_bipartite_epoch(model, valid_loader, None, device)
            if valid_loader
            else {"loss": train_metrics["loss"], "mrr": train_metrics["mrr"], "top1": train_metrics["top1"]}
        )
        history.append(_epoch_row(epoch, train_metrics, valid_metrics))
        if valid_metrics["mrr"] >= best_valid_mrr:
            best_valid_mrr = valid_metrics["mrr"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(
            f"[v4][train] epoch={epoch} train_loss={train_metrics['loss']:.4f} train_mrr={train_metrics['mrr']:.4f} "
            f"valid_loss={valid_metrics['loss']:.4f} valid_mrr={valid_metrics['mrr']:.4f}"
        )
    return history, best_state, best_valid_mrr


def _compute_feature_stats(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    matrices = [np.asarray(row["features"], dtype=np.float32) for row in rows if row.get("features")]
    stacked = np.vstack(matrices)
    return stacked.mean(axis=0), stacked.std(axis=0)


def _collate_feature_batch(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    batch_size = len(batch)
    feature_dim = batch[0]["features"].shape[1]
    max_len = max(item["features"].shape[0] for item in batch)
    features = np.zeros((batch_size, max_len, feature_dim), dtype=np.float32)
    labels = np.zeros((batch_size, max_len), dtype=np.float32)
    mask = np.zeros((batch_size, max_len), dtype=bool)
    for idx, item in enumerate(batch):
        length = item["features"].shape[0]
        features[idx, :length] = item["features"]
        labels[idx, :length] = item["labels"]
        mask[idx, :length] = True
    return {"features": torch.from_numpy(features), "labels": torch.from_numpy(labels), "mask": torch.from_numpy(mask)}


def _collate_bipartite_batch(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    batch_size = len(batch)
    seq_len = batch[0]["q_seq"].shape[0]
    input_dim = batch[0]["q_seq"].shape[1]
    max_candidates = max(item["m_seq"].shape[0] for item in batch)
    q_tensor = np.zeros((batch_size, max_candidates, seq_len, input_dim), dtype=np.float32)
    q_mask_tensor = np.zeros((batch_size, max_candidates, seq_len), dtype=bool)
    m_tensor = np.zeros((batch_size, max_candidates, seq_len, input_dim), dtype=np.float32)
    m_mask_tensor = np.zeros((batch_size, max_candidates, seq_len), dtype=bool)
    labels = np.zeros((batch_size, max_candidates), dtype=np.float32)
    valid_mask = np.zeros((batch_size, max_candidates), dtype=bool)
    for row_idx, item in enumerate(batch):
        n = item["m_seq"].shape[0]
        labels[row_idx, :n] = item["labels"][:n]
        valid_mask[row_idx, :n] = True
        q_tensor[row_idx, :n] = np.repeat(item["q_seq"][None, :, :], n, axis=0)
        q_mask_tensor[row_idx, :n] = np.repeat(item["q_mask"][None, :], n, axis=0)
        m_tensor[row_idx, :n] = item["m_seq"]
        m_mask_tensor[row_idx, :n] = item["m_mask"]
    return {
        "q_tensor": torch.from_numpy(q_tensor),
        "q_mask": torch.from_numpy(q_mask_tensor),
        "m_tensor": torch.from_numpy(m_tensor),
        "m_mask": torch.from_numpy(m_mask_tensor),
        "labels": torch.from_numpy(labels),
        "valid_mask": torch.from_numpy(valid_mask),
    }


def _run_feature_epoch(model, loader, optimizer, device):
    if loader is None:
        return {"loss": 0.0, "mrr": 0.0, "top1": 0.0}
    training = optimizer is not None
    model.train(training)
    total_loss = total_mrr = total_top1 = 0.0
    total_rows = 0
    for batch in loader:
        features = batch["features"].to(device)
        labels = batch["labels"].to(device)
        mask = batch["mask"].to(device)
        batch_size, list_len, feature_dim = features.shape
        scores = model(features.view(batch_size * list_len, feature_dim)).view(batch_size, list_len)
        loss, mrr, top1, row_count = _listwise_step(scores, labels, mask)
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_rows += row_count
        total_loss += float(loss.item()) * max(1, row_count)
        total_mrr += mrr * row_count
        total_top1 += top1 * row_count
    return _aggregate_epoch(total_loss, total_mrr, total_top1, total_rows)


def _run_bipartite_epoch(model, loader, optimizer, device):
    if loader is None:
        return {"loss": 0.0, "mrr": 0.0, "top1": 0.0}
    training = optimizer is not None
    model.train(training)
    total_loss = total_mrr = total_top1 = 0.0
    total_rows = 0
    for batch in loader:
        q_tensor = batch["q_tensor"].to(device)
        q_mask = batch["q_mask"].to(device)
        m_tensor = batch["m_tensor"].to(device)
        m_mask = batch["m_mask"].to(device)
        labels = batch["labels"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        batch_size, max_candidates, seq_len, input_dim = q_tensor.shape
        q_mask_flat = q_mask.view(-1, seq_len)
        m_mask_flat = m_mask.view(-1, seq_len)
        empty_q = q_mask_flat.sum(dim=1) == 0
        empty_m = m_mask_flat.sum(dim=1) == 0
        if torch.any(empty_q):
            q_mask_flat[empty_q, 0] = True
        if torch.any(empty_m):
            m_mask_flat[empty_m, 0] = True
        flat_scores = model(
            q_tensor.view(-1, seq_len, input_dim),
            m_tensor.view(-1, seq_len, input_dim),
            q_mask=q_mask_flat,
            m_mask=m_mask_flat,
        )
        scores = flat_scores.view(batch_size, max_candidates)
        loss, mrr, top1, row_count = _listwise_step(scores, labels, valid_mask)
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_rows += row_count
        total_loss += float(loss.item()) * max(1, row_count)
        total_mrr += mrr * row_count
        total_top1 += top1 * row_count
    return _aggregate_epoch(total_loss, total_mrr, total_top1, total_rows)


def _listwise_step(scores, labels, valid_mask):
    masked_scores = scores.masked_fill(~valid_mask, -1e9)
    target = labels / labels.sum(dim=1, keepdim=True).clamp_min(1.0)
    has_pos = labels.sum(dim=1) > 0
    log_probs = torch.log_softmax(masked_scores, dim=1)
    row_loss = -(target * log_probs).sum(dim=1)
    if int(has_pos.sum().item()) <= 0:
        loss = masked_scores.sum() * 0.0
        mrr, top1 = 0.0, 0.0
        return loss, mrr, top1, 0
    loss = row_loss[has_pos].mean()
    with torch.no_grad():
        mrr, top1 = _batch_rank_metrics(masked_scores, labels, valid_mask)
    row_count = int(has_pos.sum().item())
    return loss, mrr, top1, row_count


def _batch_rank_metrics(scores, labels, mask):
    top1_hits = 0.0
    mrr_total = 0.0
    valid_rows = 0
    order = torch.argsort(scores, dim=1, descending=True)
    for row_idx in range(scores.shape[0]):
        if labels[row_idx].sum().item() <= 0:
            continue
        valid_rows += 1
        ranked = order[row_idx].tolist()
        positives = {idx for idx, value in enumerate(labels[row_idx].tolist()) if value > 0.5}
        if ranked and ranked[0] in positives:
            top1_hits += 1.0
        for rank, candidate_idx in enumerate(ranked, start=1):
            if not mask[row_idx, candidate_idx]:
                continue
            if candidate_idx in positives:
                mrr_total += 1.0 / rank
                break
    if valid_rows == 0:
        return 0.0, 0.0
    return mrr_total / valid_rows, top1_hits / valid_rows


def _aggregate_epoch(total_loss, total_mrr, total_top1, total_rows):
    if total_rows == 0:
        return {"loss": 0.0, "mrr": 0.0, "top1": 0.0}
    return {"loss": total_loss / total_rows, "mrr": total_mrr / total_rows, "top1": total_top1 / total_rows}


def _epoch_row(epoch, train_metrics, valid_metrics):
    return {
        "epoch": epoch,
        "train_loss": train_metrics["loss"],
        "train_mrr": train_metrics["mrr"],
        "train_top1": train_metrics["top1"],
        "valid_loss": valid_metrics["loss"],
        "valid_mrr": valid_metrics["mrr"],
        "valid_top1": valid_metrics["top1"],
    }


def _write_summary(run_dir, model_path, train_rows, valid_rows, best_valid_mrr, feature_dim, history):
    ensure_parent(run_dir / "history.json")
    (run_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "model_path": str(model_path),
        "run_dir": str(run_dir),
        "train_rows": len(train_rows),
        "valid_rows": len(valid_rows),
        "best_valid_mrr": float(best_valid_mrr),
        "feature_dim": int(feature_dim),
        "history": history,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
