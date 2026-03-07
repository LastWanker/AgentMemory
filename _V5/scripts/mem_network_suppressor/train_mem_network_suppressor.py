from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def _resolve_v5_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "src").exists() and (parent / "chat" / "src").exists():
            return parent
    return here.parents[1]


V5_ROOT = _resolve_v5_root()
if str(V5_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(V5_ROOT / "src"))

from agentmemory_v3.mem_network_suppressor.artifacts import MemNetworkManifest
from agentmemory_v3.mem_network_suppressor.features import (
    FEEDBACK_TYPE_VOCAB_SIZE,
    LANE_VOCAB_SIZE,
    build_memory_id_lookup,
    build_feedback_memory_slots,
    build_feedback_type_lane_arrays,
    memory_id_to_index,
)
from agentmemory_v3.mem_network_suppressor.model import MemNetworkScorer
from agentmemory_v3.suppressor.data_utils import load_dense_index_from_config, load_memory_text_by_id, load_v5_encoder
from agentmemory_v3.utils.io import read_jsonl, write_jsonl


def _read_rows(path: Path) -> list[dict]:
    return list(read_jsonl(path)) if path.exists() else []


def _resolve_device(device: str) -> torch.device:
    raw = str(device or "").strip().lower()
    if raw in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if raw in {"gpu", "cuda"} and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(raw)


def _build_pair_indices(group_rows: list[dict], sample_index_by_id: dict[str, int], max_pairs_per_group: int) -> tuple[np.ndarray, np.ndarray]:
    pos_idx: list[int] = []
    neg_idx: list[int] = []
    cap = max(1, int(max_pairs_per_group))
    for group in group_rows:
        pos: list[int] = []
        neg: list[int] = []
        for cand in (group.get("candidates") or []):
            sid = str(cand.get("sample_id") or "")
            idx = sample_index_by_id.get(sid)
            if idx is None:
                continue
            label = int(cand.get("label") or 0)
            if label == 1:
                pos.append(idx)
            else:
                neg.append(idx)
        if not pos or not neg:
            continue
        pair_count = 0
        for p in pos:
            for n in neg:
                pos_idx.append(int(p))
                neg_idx.append(int(n))
                pair_count += 1
                if pair_count >= cap:
                    break
            if pair_count >= cap:
                break
    if not pos_idx:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    return np.asarray(pos_idx, dtype=np.int64), np.asarray(neg_idx, dtype=np.int64)


def _metrics(logits: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict:
    if logits.size == 0:
        return {"count": 0}
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = probs >= float(threshold)
    y = np.asarray(labels, dtype=np.float32).reshape(-1)
    pos = y > 0.5
    neg = y <= 0.5
    out = {
        "count": int(y.shape[0]),
        "accuracy": float(np.mean((preds.astype(np.float32) == y).astype(np.float32))),
        "positive_rate": float(np.mean(y)),
        "pred_positive_rate": float(np.mean(preds.astype(np.float32))),
    }
    if np.any(pos):
        out["recall"] = float(np.mean(preds[pos].astype(np.float32)))
    if np.any(neg):
        out["false_positive_rate"] = float(np.mean(preds[neg].astype(np.float32)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train single-head 1-hop memory-network suppressor.")
    parser.add_argument("--config", default="_V5/configs/default.yaml")
    parser.add_argument("--data-dir", default="data/V5/mem_network_suppressor/dataset_v1")
    parser.add_argument("--artifact-dir", default="data/V5/mem_network_suppressor/current")
    parser.add_argument("--bundle-path", default="data/V5/exports/chat_memory_bundle.jsonl")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--state-dim", type=int, default=256)
    parser.add_argument("--num-hops", type=int, default=1)
    parser.add_argument("--feedback-top-k", type=int, default=8)
    parser.add_argument("--pointwise-weight", type=float, default=0.5)
    parser.add_argument("--pairwise-weight", type=float, default=1.0)
    parser.add_argument("--max-pairs-per-group", type=int, default=256)
    parser.add_argument("--suppress-strength", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-valid-samples", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = _resolve_device(args.device)

    data_dir = Path(args.data_dir)
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    train_rows = _read_rows(data_dir / "feedback_samples_train.jsonl")
    valid_rows = _read_rows(data_dir / "feedback_samples_valid.jsonl")
    train_group_rows = _read_rows(data_dir / "feedback_groups_train.jsonl")
    valid_group_rows = _read_rows(data_dir / "feedback_groups_valid.jsonl")
    feedback_rows = _read_rows(data_dir / "feedback_memory_store.jsonl")
    if not train_rows:
        raise RuntimeError("training rows are empty")
    if not feedback_rows:
        raise RuntimeError("feedback_memory_store.jsonl is empty")

    if int(args.max_train_samples) > 0:
        train_rows = train_rows[: int(args.max_train_samples)]
    if int(args.max_valid_samples) > 0:
        valid_rows = valid_rows[: int(args.max_valid_samples)]

    encoder = load_v5_encoder(args.config)
    dense_index = load_dense_index_from_config(args.config)
    memory_ids = [str(item) for item in dense_index.artifact.mem_ids]
    memory_matrix = np.asarray(dense_index.matrix, dtype=np.float32)
    memory_index_by_id = {memory_id: idx for idx, memory_id in enumerate(memory_ids)}
    memory_text_by_id = load_memory_text_by_id(args.config, args.bundle_path)

    unique_queries = list(
        dict.fromkeys(
            [str(row.get("q_text") or "") for row in train_rows + valid_rows + feedback_rows if str(row.get("q_text") or "")]
        )
    )
    query_matrix = encoder.encode_query_texts(unique_queries) if unique_queries else np.zeros((0, memory_matrix.shape[1]), dtype=np.float32)
    query_vec_by_text = {text: query_matrix[idx] for idx, text in enumerate(unique_queries)}

    missing_texts: dict[str, str] = {}
    for row in train_rows + valid_rows + feedback_rows:
        mid = str(row.get("m_id") or "")
        if mid and mid not in memory_index_by_id:
            missing_texts[mid] = str(row.get("m_text") or memory_text_by_id.get(mid) or "")
    missing_ids = list(missing_texts.keys())
    missing_matrix = (
        encoder.encode_passage_texts([missing_texts[mid] for mid in missing_ids])
        if missing_ids
        else np.zeros((0, memory_matrix.shape[1]), dtype=np.float32)
    )
    missing_index = {mid: idx for idx, mid in enumerate(missing_ids)}

    def memory_vec(memory_id: str, memory_text: str = "") -> np.ndarray:
        idx = memory_index_by_id.get(memory_id)
        if idx is not None:
            return memory_matrix[int(idx)]
        miss = missing_index.get(memory_id)
        if miss is not None:
            return missing_matrix[int(miss)]
        return encoder.encode_passage_texts([memory_text or ""])[0]

    feedback_query_matrix = np.stack(
        [query_vec_by_text.get(str(row.get("q_text") or ""), encoder.encode_query_texts([str(row.get("q_text") or "")])[0]) for row in feedback_rows]
    ).astype(np.float32)
    feedback_memory_matrix = np.stack(
        [memory_vec(str(row.get("m_id") or ""), str(row.get("m_text") or "")) for row in feedback_rows]
    ).astype(np.float32)
    feedback_type_ids, feedback_lane_ids = build_feedback_type_lane_arrays(feedback_rows)
    suppress_memory_ids, suppress_memory_id_to_index = build_memory_id_lookup(feedback_rows + train_rows + valid_rows)
    memory_id_vocab_size = max(2, int(len(suppress_memory_ids) + 1))
    feedback_memory_id_ids = np.asarray(
        [memory_id_to_index(str(row.get("m_id") or ""), suppress_memory_id_to_index) for row in feedback_rows],
        dtype=np.int64,
    )

    def build_split(rows: list[dict]) -> dict:
        q_rows: list[np.ndarray] = []
        c_rows: list[np.ndarray] = []
        slot_q_rows: list[np.ndarray] = []
        slot_m_rows: list[np.ndarray] = []
        candidate_memory_id_rows: list[int] = []
        slot_memory_id_rows: list[np.ndarray] = []
        slot_type_rows: list[np.ndarray] = []
        slot_lane_rows: list[np.ndarray] = []
        slot_mask_rows: list[np.ndarray] = []
        labels: list[float] = []
        weights: list[float] = []
        sample_ids: list[str] = []
        for idx, row in enumerate(rows):
            q_text = str(row.get("q_text") or "")
            q_vec = query_vec_by_text.get(q_text)
            if q_vec is None:
                q_vec = encoder.encode_query_texts([q_text])[0]
                query_vec_by_text[q_text] = q_vec
            m_id = str(row.get("m_id") or "")
            m_text = str(row.get("m_text") or memory_text_by_id.get(m_id) or "")
            c_vec = memory_vec(m_id, m_text)
            candidate_memory_id_id = memory_id_to_index(m_id, suppress_memory_id_to_index)
            slots = build_feedback_memory_slots(
                query_vec=q_vec,
                candidate_vec=c_vec,
                feedback_query_matrix=feedback_query_matrix,
                feedback_memory_matrix=feedback_memory_matrix,
                feedback_type_ids=feedback_type_ids,
                feedback_lane_ids=feedback_lane_ids,
                feedback_memory_id_ids=feedback_memory_id_ids,
                candidate_memory_id_id=candidate_memory_id_id,
                top_k=max(1, int(args.feedback_top_k)),
            )
            q_rows.append(np.asarray(q_vec, dtype=np.float32))
            c_rows.append(np.asarray(c_vec, dtype=np.float32))
            candidate_memory_id_rows.append(int(candidate_memory_id_id))
            slot_q_rows.append(np.asarray(slots["slot_query"], dtype=np.float32))
            slot_m_rows.append(np.asarray(slots["slot_memory"], dtype=np.float32))
            slot_memory_id_rows.append(np.asarray(slots["slot_memory_id_ids"], dtype=np.int64))
            slot_type_rows.append(np.asarray(slots["slot_type_ids"], dtype=np.int64))
            slot_lane_rows.append(np.asarray(slots["slot_lane_ids"], dtype=np.int64))
            slot_mask_rows.append(np.asarray(slots["slot_mask"], dtype=np.float32))
            labels.append(float(row.get("label") or 0.0))
            weights.append(max(0.0, float(row.get("weight") or 1.0)))
            sample_ids.append(str(row.get("sample_id") or f"row_{idx}"))
        if not q_rows:
            return {
                "query": np.zeros((0, 0), dtype=np.float32),
                "candidate": np.zeros((0, 0), dtype=np.float32),
                "candidate_memory_id_ids": np.zeros((0,), dtype=np.int64),
                "slot_query": np.zeros((0, max(1, int(args.feedback_top_k)), 0), dtype=np.float32),
                "slot_memory": np.zeros((0, max(1, int(args.feedback_top_k)), 0), dtype=np.float32),
                "slot_memory_id_ids": np.zeros((0, max(1, int(args.feedback_top_k))), dtype=np.int64),
                "slot_type_ids": np.zeros((0, max(1, int(args.feedback_top_k))), dtype=np.int64),
                "slot_lane_ids": np.zeros((0, max(1, int(args.feedback_top_k))), dtype=np.int64),
                "slot_mask": np.zeros((0, max(1, int(args.feedback_top_k))), dtype=np.float32),
                "labels": np.zeros((0,), dtype=np.float32),
                "weights": np.zeros((0,), dtype=np.float32),
                "sample_index_by_id": {},
            }
        return {
            "query": np.stack(q_rows).astype(np.float32),
            "candidate": np.stack(c_rows).astype(np.float32),
            "candidate_memory_id_ids": np.asarray(candidate_memory_id_rows, dtype=np.int64),
            "slot_query": np.stack(slot_q_rows).astype(np.float32),
            "slot_memory": np.stack(slot_m_rows).astype(np.float32),
            "slot_memory_id_ids": np.stack(slot_memory_id_rows).astype(np.int64),
            "slot_type_ids": np.stack(slot_type_rows).astype(np.int64),
            "slot_lane_ids": np.stack(slot_lane_rows).astype(np.int64),
            "slot_mask": np.stack(slot_mask_rows).astype(np.float32),
            "labels": np.asarray(labels, dtype=np.float32),
            "weights": np.asarray(weights, dtype=np.float32),
            "sample_index_by_id": {sid: i for i, sid in enumerate(sample_ids)},
        }

    train_split = build_split(train_rows)
    valid_split = build_split(valid_rows)
    if train_split["query"].size == 0:
        raise RuntimeError("train split is empty")

    train_pair_pos, train_pair_neg = _build_pair_indices(
        train_group_rows,
        train_split["sample_index_by_id"],
        int(args.max_pairs_per_group),
    )
    valid_pair_pos, valid_pair_neg = _build_pair_indices(
        valid_group_rows,
        valid_split["sample_index_by_id"],
        int(args.max_pairs_per_group),
    )

    model = MemNetworkScorer(
        embedding_dim=int(memory_matrix.shape[1]),
        state_dim=max(32, int(args.state_dim)),
        num_hops=max(1, int(args.num_hops)),
        num_heads=1,
        memory_id_vocab_size=int(memory_id_vocab_size),
        dropout=float(args.dropout),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    def to_tensor(split: dict) -> dict:
        return {
            "query": torch.from_numpy(split["query"]).float().to(device),
            "candidate": torch.from_numpy(split["candidate"]).float().to(device),
            "candidate_memory_id_ids": torch.from_numpy(split["candidate_memory_id_ids"]).long().to(device),
            "slot_query": torch.from_numpy(split["slot_query"]).float().to(device),
            "slot_memory": torch.from_numpy(split["slot_memory"]).float().to(device),
            "slot_memory_id_ids": torch.from_numpy(split["slot_memory_id_ids"]).long().to(device),
            "slot_type_ids": torch.from_numpy(split["slot_type_ids"]).long().to(device),
            "slot_lane_ids": torch.from_numpy(split["slot_lane_ids"]).long().to(device),
            "slot_mask": torch.from_numpy(split["slot_mask"]).float().to(device),
            "labels": torch.from_numpy(split["labels"]).float().to(device),
            "weights": torch.from_numpy(split["weights"]).float().to(device),
        }

    train_t = to_tensor(train_split)
    valid_t = to_tensor(valid_split) if valid_split["query"].size else None
    train_pair_pos_t = torch.from_numpy(train_pair_pos).long().to(device)
    train_pair_neg_t = torch.from_numpy(train_pair_neg).long().to(device)
    valid_pair_pos_t = torch.from_numpy(valid_pair_pos).long().to(device)
    valid_pair_neg_t = torch.from_numpy(valid_pair_neg).long().to(device)

    pos = float(np.sum(train_split["labels"] > 0.5))
    neg = float(np.sum(train_split["labels"] <= 0.5))
    pos_weight = torch.tensor([max(1.0, neg / max(1.0, pos))], dtype=torch.float32, device=device)
    pointwise_weight = max(0.0, float(args.pointwise_weight))
    pairwise_weight = max(0.0, float(args.pairwise_weight))

    best_state = None
    best_valid_total = float("inf")
    history: list[dict] = []

    def forward_logits(split_t: dict) -> torch.Tensor:
        logits, _ = model.forward_logits(
            query_vec=split_t["query"],
            candidate_vec=split_t["candidate"],
            feedback_query_slots=split_t["slot_query"],
            feedback_memory_slots=split_t["slot_memory"],
            candidate_memory_id_ids=split_t["candidate_memory_id_ids"],
            feedback_memory_id_ids=split_t["slot_memory_id_ids"],
            feedback_type_ids=split_t["slot_type_ids"],
            feedback_lane_ids=split_t["slot_lane_ids"],
            feedback_mask=split_t["slot_mask"],
        )
        return logits.reshape(-1)

    def pair_loss(logits_1d: torch.Tensor, pos_idx: torch.Tensor, neg_idx: torch.Tensor) -> torch.Tensor:
        if int(pos_idx.shape[0]) == 0:
            return torch.zeros((), dtype=torch.float32, device=logits_1d.device)
        return torch.mean(F.softplus(-(logits_1d[pos_idx] - logits_1d[neg_idx])))

    def weighted_bce(logits_1d: torch.Tensor, labels_1d: torch.Tensor, weights_1d: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits_1d, labels_1d, pos_weight=pos_weight, reduction="none")
        w = torch.clamp(weights_1d, min=0.0)
        denom = torch.clamp(torch.sum(w), min=1e-6)
        return torch.sum(bce * w) / denom

    for epoch in range(1, max(1, int(args.epochs)) + 1):
        model.train()
        optimizer.zero_grad()
        train_logits = forward_logits(train_t)
        point_loss = weighted_bce(train_logits, train_t["labels"], train_t["weights"])
        pair = pair_loss(train_logits, train_pair_pos_t, train_pair_neg_t)
        total = pointwise_weight * point_loss + pairwise_weight * pair
        total.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            valid_total = total.detach()
            valid_point = point_loss.detach()
            valid_pair = torch.zeros((), dtype=torch.float32, device=device)
            if valid_t is not None:
                valid_logits = forward_logits(valid_t)
                valid_point = weighted_bce(valid_logits, valid_t["labels"], valid_t["weights"])
                valid_pair = pair_loss(valid_logits, valid_pair_pos_t, valid_pair_neg_t)
                valid_total = pointwise_weight * valid_point + pairwise_weight * valid_pair

        if float(valid_total.item()) < best_valid_total:
            best_valid_total = float(valid_total.item())
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        row = {
            "epoch": int(epoch),
            "train_total_loss": float(total.item()),
            "train_point_loss": float(point_loss.item()),
            "train_pair_loss": float(pair.item()),
            "valid_total_loss": float(valid_total.item()),
            "valid_point_loss": float(valid_point.item()),
            "valid_pair_loss": float(valid_pair.item()),
        }
        history.append(row)
        print("[v5][memnet][train]", row)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        train_logits_np = forward_logits(train_t).detach().cpu().numpy()
        valid_logits_np = forward_logits(valid_t).detach().cpu().numpy() if valid_t is not None else np.zeros((0,), dtype=np.float32)

    metrics = {
        "train": _metrics(train_logits_np, train_split["labels"]),
        "valid": _metrics(valid_logits_np, valid_split["labels"]) if valid_split["labels"].size else {},
        "history": history,
        "best_valid_total_loss": float(best_valid_total),
        "pos_weight": float(pos_weight.item()),
        "train_pair_count": int(train_pair_pos.shape[0]),
        "valid_pair_count": int(valid_pair_pos.shape[0]),
        "pointwise_weight": float(pointwise_weight),
        "pairwise_weight": float(pairwise_weight),
        "memory_id_vocab_size": int(memory_id_vocab_size),
    }

    manifest = MemNetworkManifest(
        version="mem_network_suppressor_v4",
        model_type="mem_network_single_head_addr_v4",
        encoder_model_name=str(dense_index.artifact.model_name),
        use_e5_prefix=bool(dense_index.artifact.use_e5_prefix),
        local_files_only=bool(dense_index.artifact.local_files_only),
        offline=bool(dense_index.artifact.offline),
        device=str(dense_index.artifact.device),
        batch_size=int(dense_index.artifact.batch_size),
        embedding_dim=int(memory_matrix.shape[1]),
        feature_dim=0,
        state_dim=max(32, int(args.state_dim)),
        num_hops=max(1, int(args.num_hops)),
        num_heads=1,
        memory_id_vocab_size=int(memory_id_vocab_size),
        use_memory_id_addressing=True,
        type_vocab_size=FEEDBACK_TYPE_VOCAB_SIZE,
        lane_vocab_size=LANE_VOCAB_SIZE,
        type_emb_dim=16,
        lane_emb_dim=8,
        dropout=float(args.dropout),
        feedback_top_k=max(1, int(args.feedback_top_k)),
        suppress_strength=float(args.suppress_strength),
    )

    (artifact_dir / "manifest.json").write_text(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    (artifact_dir / "feature_spec.json").write_text(
        json.dumps(
            {
                "model_type": "mem_network_single_head_addr_v4",
                "head": "suppress",
                "embedding_dim": int(memory_matrix.shape[1]),
                "state_dim": int(max(32, int(args.state_dim))),
                "num_hops": int(max(1, int(args.num_hops))),
                "feedback_top_k": int(max(1, int(args.feedback_top_k))),
                "memory_id_vocab_size": int(memory_id_vocab_size),
                "use_memory_id_addressing": True,
                "loss": {
                    "pointwise_weight": float(pointwise_weight),
                    "pairwise_weight": float(pairwise_weight),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    torch.save({"state_dict": model.state_dict()}, artifact_dir / "model.pt")
    (artifact_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    np.save(artifact_dir / "memory_embeddings.npy", memory_matrix.astype(np.float32))
    (artifact_dir / "memory_ids.json").write_text(json.dumps(memory_ids, ensure_ascii=False), encoding="utf-8")
    write_jsonl(
        artifact_dir / "memory_texts.jsonl",
        [{"memory_id": memory_id, "memory_text": memory_text_by_id.get(memory_id, "")} for memory_id in memory_ids],
    )
    (artifact_dir / "suppress_memory_ids.json").write_text(json.dumps(suppress_memory_ids, ensure_ascii=False), encoding="utf-8")
    write_jsonl(artifact_dir / "feedback_memory_store.jsonl", feedback_rows)
    np.save(artifact_dir / "feedback_query_embeddings.npy", feedback_query_matrix.astype(np.float32))
    np.save(artifact_dir / "feedback_memory_embeddings.npy", feedback_memory_matrix.astype(np.float32))
    np.save(artifact_dir / "feedback_memory_id_ids.npy", feedback_memory_id_ids.astype(np.int64))
    np.save(artifact_dir / "feedback_type_ids.npy", feedback_type_ids.astype(np.int64))
    np.save(artifact_dir / "feedback_lane_ids.npy", feedback_lane_ids.astype(np.int64))

    print(
        "[v5][memnet][train] saved",
        {
            "artifact_dir": str(artifact_dir),
            "train_count": int(train_split["labels"].shape[0]),
            "valid_count": int(valid_split["labels"].shape[0]),
            "feedback_rows": int(len(feedback_rows)),
            "device": str(device),
            "state_dim": int(max(32, int(args.state_dim))),
            "num_hops": int(max(1, int(args.num_hops))),
            "memory_id_vocab_size": int(memory_id_vocab_size),
            "best_valid_total_loss": float(best_valid_total),
        },
    )


if __name__ == "__main__":
    main()
