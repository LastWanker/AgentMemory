from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
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

from agentmemory_v3.suppressor.artifacts import SuppressorManifest
from agentmemory_v3.suppressor.data_utils import load_dense_index_from_config, load_memory_text_by_id, load_v5_encoder
from agentmemory_v3.suppressor.features import FEEDBACK_TYPES, build_feature_vector
from agentmemory_v3.suppressor.model import OreoMemoryMLP, OreoTypeHeadsMLP, PlainSuppressorMLP
from agentmemory_v3.utils.io import read_jsonl, write_jsonl


def _read_rows(path: Path) -> list[dict]:
    return list(read_jsonl(path)) if path.exists() else []


def _resolve_device(device: str) -> torch.device:
    raw = str(device or "").strip().lower()
    if raw in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if raw in {"cuda", "gpu"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def _build_arrays(
    rows: list[dict],
    encoder,
    memory_ids: list[str],
    memory_matrix: np.ndarray,
    memory_text_by_id: dict[str, str],
    *,
    include_query_vec: bool,
    include_memory_vec: bool,
    include_cosine: bool,
    include_product: bool,
    include_abs_diff: bool,
    include_feedback_type: bool,
    include_lane: bool,
    include_lexical_overlap: bool,
    include_explicit_mention: bool,
):
    memory_index_by_id = {memory_id: idx for idx, memory_id in enumerate(memory_ids)}
    unique_queries = list(dict.fromkeys(str(row.get("q_text") or "") for row in rows if str(row.get("q_text") or "")))
    query_matrix = (
        encoder.encode_query_texts(unique_queries) if unique_queries else np.zeros((0, memory_matrix.shape[1]), dtype=np.float32)
    )
    query_index = {text: idx for idx, text in enumerate(unique_queries)}

    missing_memory_texts: dict[str, str] = {}
    for row in rows:
        m_id = str(row.get("m_id") or "")
        if m_id not in memory_index_by_id:
            missing_memory_texts[m_id] = str(row.get("m_text") or memory_text_by_id.get(m_id) or "")
    missing_ids = list(missing_memory_texts.keys())
    missing_matrix = (
        encoder.encode_passage_texts([missing_memory_texts[m_id] for m_id in missing_ids])
        if missing_ids
        else np.zeros((0, memory_matrix.shape[1]), dtype=np.float32)
    )
    missing_index = {memory_id: idx for idx, memory_id in enumerate(missing_ids)}

    feature_rows: list[np.ndarray] = []
    labels: list[float] = []
    weights: list[float] = []
    meta: list[dict] = []
    for row in rows:
        q_text = str(row.get("q_text") or "")
        m_id = str(row.get("m_id") or "")
        m_text = str(row.get("m_text") or memory_text_by_id.get(m_id) or "")
        q_vec = query_matrix[int(query_index[q_text])]
        if m_id in memory_index_by_id:
            m_vec = memory_matrix[int(memory_index_by_id[m_id])]
        else:
            m_vec = missing_matrix[int(missing_index[m_id])]
        feature_rows.append(
            build_feature_vector(
                query_text=q_text,
                query_vec=q_vec,
                memory_text=m_text,
                memory_vec=m_vec,
                feedback_type=str(row.get("feedback_type") or ""),
                lane=str(row.get("lane") or ""),
                include_query_vec=include_query_vec,
                include_memory_vec=include_memory_vec,
                include_cosine=include_cosine,
                include_product=include_product,
                include_abs_diff=include_abs_diff,
                include_feedback_type=include_feedback_type,
                include_lane=include_lane,
                include_lexical_overlap=include_lexical_overlap,
                include_explicit_mention=include_explicit_mention,
            )
        )
        labels.append(float(row.get("label") or 0.0))
        weights.append(float(row.get("weight") or 1.0))
        meta.append(row)
    x = np.stack(feature_rows).astype(np.float32) if feature_rows else np.zeros((0, 0), dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32)
    w = np.asarray(weights, dtype=np.float32)
    return x, y, w, meta


def _build_groups(meta: list[dict], labels: np.ndarray) -> list[dict]:
    grouped: dict[str, dict[str, list[int]]] = defaultdict(lambda: {"all": [], "pos": [], "neg": []})
    for idx, row in enumerate(meta):
        q_text = str(row.get("q_text") or "")
        feedback_type = str(row.get("feedback_type") or "")
        lane = str(row.get("lane") or "")
        group_id = str(row.get("group_id") or row.get("anchor_feedback_id") or f"{q_text}|{feedback_type}|{lane}")
        grouped[group_id]["all"].append(idx)
        if float(labels[idx]) > 0.5:
            grouped[group_id]["pos"].append(idx)
        else:
            grouped[group_id]["neg"].append(idx)
    out: list[dict] = []
    for group_id, item in grouped.items():
        if not item["pos"] or not item["neg"]:
            continue
        out.append({"group_id": group_id, "all": item["all"], "pos": item["pos"], "neg": item["neg"]})
    return out


def _pairwise_loss(logits: torch.Tensor, groups: list[dict], max_pairs_per_group: int) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    max_pairs = max(1, int(max_pairs_per_group))
    for group in groups:
        pos = group["pos"]
        neg = group["neg"]
        if not pos or not neg:
            continue
        pos_tensor = logits[pos]
        neg_tensor = logits[neg]
        diffs = (pos_tensor[:, None] - neg_tensor[None, :]).reshape(-1)
        if diffs.numel() > max_pairs:
            choose = torch.randperm(diffs.numel(), device=diffs.device)[:max_pairs]
            diffs = diffs[choose]
        losses.append(-F.logsigmoid(diffs).mean())
    if not losses:
        return logits.new_tensor(0.0)
    return torch.stack(losses).mean()


def _listwise_loss(logits: torch.Tensor, groups: list[dict]) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for group in groups:
        indices = group["all"]
        pos_indices = set(group["pos"])
        if not indices or not pos_indices:
            continue
        group_logits = logits[indices]
        log_probs = F.log_softmax(group_logits, dim=0)
        target = torch.zeros_like(log_probs)
        pos_positions = [idx for idx, global_idx in enumerate(indices) if global_idx in pos_indices]
        if not pos_positions:
            continue
        target[pos_positions] = 1.0 / float(len(pos_positions))
        losses.append(-(target * log_probs).sum())
    if not losses:
        return logits.new_tensor(0.0)
    return torch.stack(losses).mean()


def _metrics(logits: np.ndarray, labels: np.ndarray, meta: list[dict], threshold: float = 0.5) -> dict:
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = probs >= float(threshold)
    out = {
        "count": int(labels.shape[0]),
        "accuracy": float(np.mean((preds.astype(np.float32) == labels).astype(np.float32))) if labels.size else 0.0,
    }
    by_kind: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(meta):
        by_kind[str(row.get("source_kind") or "")].append(idx)
    for key, indices in by_kind.items():
        idx_arr = np.asarray(indices, dtype=np.int64)
        group_preds = preds[idx_arr]
        group_labels = labels[idx_arr]
        if np.any(group_labels > 0.5):
            value = float(np.mean(group_preds[group_labels > 0.5].astype(np.float32)))
            out[f"{key}_recall"] = value
        else:
            value = float(np.mean(group_preds.astype(np.float32)))
            out[f"{key}_false_positive_rate"] = value
    return out


def _build_memory_bias(
    train_meta: list[dict],
    train_labels: np.ndarray,
    train_weights: np.ndarray,
    *,
    prior_strength: float,
) -> tuple[dict[str, float], dict]:
    total_w = float(np.sum(train_weights))
    if total_w <= 1e-8:
        return {}, {"global_positive_rate": 0.0}
    global_positive = float(np.sum(train_labels * train_weights) / total_w)
    sum_pos_by_id: dict[str, float] = defaultdict(float)
    sum_w_by_id: dict[str, float] = defaultdict(float)
    count_by_id: dict[str, int] = defaultdict(int)
    for idx, row in enumerate(train_meta):
        memory_id = str(row.get("m_id") or "")
        if not memory_id:
            continue
        weight = float(train_weights[idx])
        label = float(train_labels[idx])
        sum_w_by_id[memory_id] += weight
        sum_pos_by_id[memory_id] += weight * label
        count_by_id[memory_id] += 1
    bias_by_id: dict[str, float] = {}
    prior = max(0.0, float(prior_strength))
    for memory_id, w_sum in sum_w_by_id.items():
        if w_sum <= 1e-8:
            continue
        rate = (sum_pos_by_id[memory_id] + prior * global_positive) / (w_sum + prior)
        bias = max(0.0, min(1.0, float(rate) - global_positive))
        if bias > 0:
            bias_by_id[memory_id] = bias
    stats = {
        "global_positive_rate": global_positive,
        "memory_count": len(sum_w_by_id),
        "biased_memory_count": len(bias_by_id),
        "prior_strength": prior,
    }
    return bias_by_id, stats


def _build_model(args, input_dim: int):
    model_type = str(args.model_type or "").strip().lower()
    if model_type in {"oreo_type_heads", "oreo_type_heads_mlp_v1"}:
        return OreoTypeHeadsMLP(
            input_dim=input_dim,
            hidden_dims=tuple(int(item.strip()) for item in str(args.hidden_dims).split(",") if item.strip()),
            dropout=float(args.dropout),
            slots_k=int(args.slots_k),
            top_r=int(args.top_r),
            tau=float(args.tau),
            feedback_types=FEEDBACK_TYPES,
        ), "oreo_type_heads_mlp_v1"
    if model_type in {"oreo", "oreo_memory_mlp_v1"}:
        return OreoMemoryMLP(
            input_dim=input_dim,
            hidden_dims=tuple(int(item.strip()) for item in str(args.hidden_dims).split(",") if item.strip()),
            dropout=float(args.dropout),
            slots_k=int(args.slots_k),
            top_r=int(args.top_r),
            tau=float(args.tau),
        ), "oreo_memory_mlp_v1"
    return PlainSuppressorMLP(
        input_dim=input_dim,
        hidden_dims=tuple(int(item.strip()) for item in str(args.hidden_dims).split(",") if item.strip()),
        dropout=float(args.dropout),
    ), "plain_mlp_v1"


def _feedback_types_from_meta(meta: list[dict]) -> list[str]:
    return [str(row.get("feedback_type") or "").strip().lower() for row in meta]


def _forward_logits(model, x_tensor: torch.Tensor, feedback_types: list[str] | None = None) -> torch.Tensor:
    if isinstance(model, OreoTypeHeadsMLP):
        return model.forward_logits(x_tensor, feedback_types=feedback_types or [])
    return model.forward_logits(x_tensor)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train V5 suppressor (plain/oreo, pointwise/pairwise/listwise).")
    parser.add_argument("--config", default="_V5/configs/default.yaml")
    parser.add_argument("--data-dir", default="data/V5/suppressor")
    parser.add_argument("--artifact-dir", default="data/V5/suppressor")
    parser.add_argument("--bundle-path", default="data/V5/exports/chat_memory_bundle.jsonl")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.18)
    parser.add_argument("--hidden-dims", default="384,128")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--model-type",
        default="oreo_type_heads",
        choices=(
            "plain",
            "oreo",
            "oreo_type_heads",
            "plain_mlp_v1",
            "oreo_memory_mlp_v1",
            "oreo_type_heads_mlp_v1",
        ),
    )
    parser.add_argument("--objective", default="hybrid", choices=("pointwise", "pairwise", "listwise", "hybrid"))
    parser.add_argument("--slots-k", type=int, default=32)
    parser.add_argument("--top-r", type=int, default=2)
    parser.add_argument("--tau", type=float, default=0.7)
    parser.add_argument("--lambda-pointwise", type=float, default=1.0)
    parser.add_argument("--lambda-pairwise", type=float, default=1.0)
    parser.add_argument("--lambda-listwise", type=float, default=0.5)
    parser.add_argument("--max-pairs-per-group", type=int, default=128)
    parser.add_argument("--include-query-vec", type=int, default=0)
    parser.add_argument("--include-memory-vec", type=int, default=0)
    parser.add_argument("--include-cosine", type=int, default=1)
    parser.add_argument("--include-product", type=int, default=1)
    parser.add_argument("--include-abs-diff", type=int, default=1)
    parser.add_argument("--include-feedback-type", type=int, default=1)
    parser.add_argument("--include-lane", type=int, default=1)
    parser.add_argument("--include-lexical-overlap", type=int, default=1)
    parser.add_argument("--include-explicit-mention", type=int, default=1)
    parser.add_argument("--enable-memory-bias", type=int, default=1)
    parser.add_argument("--lambda-memory-bias", type=float, default=0.2)
    parser.add_argument("--memory-bias-prior", type=float, default=8.0)
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    data_dir = Path(args.data_dir)
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    train_rows = _read_rows(data_dir / "feedback_samples_train.jsonl")
    valid_rows = _read_rows(data_dir / "feedback_samples_valid.jsonl")
    if not train_rows:
        raise RuntimeError("training dataset is empty")

    encoder = load_v5_encoder(args.config)
    dense_index = load_dense_index_from_config(args.config)
    memory_ids = [str(item) for item in dense_index.artifact.mem_ids]
    memory_matrix = np.asarray(dense_index.matrix, dtype=np.float32)
    memory_text_by_id = load_memory_text_by_id(args.config, args.bundle_path)

    include_query_vec = bool(int(args.include_query_vec))
    include_memory_vec = bool(int(args.include_memory_vec))
    include_cosine = bool(int(args.include_cosine))
    include_product = bool(int(args.include_product))
    include_abs_diff = bool(int(args.include_abs_diff))
    include_feedback_type = bool(int(args.include_feedback_type))
    include_lane = bool(int(args.include_lane))
    include_lexical_overlap = bool(int(args.include_lexical_overlap))
    include_explicit_mention = bool(int(args.include_explicit_mention))

    x_train, y_train, w_train, train_meta = _build_arrays(
        train_rows,
        encoder,
        memory_ids,
        memory_matrix,
        memory_text_by_id,
        include_query_vec=include_query_vec,
        include_memory_vec=include_memory_vec,
        include_cosine=include_cosine,
        include_product=include_product,
        include_abs_diff=include_abs_diff,
        include_feedback_type=include_feedback_type,
        include_lane=include_lane,
        include_lexical_overlap=include_lexical_overlap,
        include_explicit_mention=include_explicit_mention,
    )
    x_valid, y_valid, w_valid, valid_meta = _build_arrays(
        valid_rows,
        encoder,
        memory_ids,
        memory_matrix,
        memory_text_by_id,
        include_query_vec=include_query_vec,
        include_memory_vec=include_memory_vec,
        include_cosine=include_cosine,
        include_product=include_product,
        include_abs_diff=include_abs_diff,
        include_feedback_type=include_feedback_type,
        include_lane=include_lane,
        include_lexical_overlap=include_lexical_overlap,
        include_explicit_mention=include_explicit_mention,
    )
    if x_train.size == 0:
        raise RuntimeError("no train features built")

    train_groups = _build_groups(train_meta, y_train)
    valid_groups = _build_groups(valid_meta, y_valid) if len(valid_rows) else []
    train_feedback_types = _feedback_types_from_meta(train_meta)
    valid_feedback_types = _feedback_types_from_meta(valid_meta)

    model, model_type = _build_model(args, int(x_train.shape[1]))
    device = _resolve_device(args.device)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    x_train_t = torch.from_numpy(x_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).float().to(device)
    w_train_t = torch.from_numpy(w_train).float().to(device)
    x_valid_t = torch.from_numpy(x_valid).float().to(device) if len(x_valid) else None
    y_valid_t = torch.from_numpy(y_valid).float().to(device) if len(y_valid) else None
    w_valid_t = torch.from_numpy(w_valid).float().to(device) if len(w_valid) else None

    objective = str(args.objective).strip().lower()
    use_pointwise = objective in {"pointwise", "hybrid"}
    use_pairwise = objective in {"pairwise", "hybrid"}
    use_listwise = objective in {"listwise", "hybrid"}

    best_state = None
    best_valid_loss = float("inf")
    history: list[dict] = []

    for epoch in range(1, max(1, int(args.epochs)) + 1):
        model.train()
        optimizer.zero_grad()
        logits = _forward_logits(model, x_train_t, train_feedback_types)
        pointwise_loss = F.binary_cross_entropy_with_logits(logits, y_train_t, reduction="none")
        pointwise_loss = (pointwise_loss * w_train_t).mean()
        pairwise_loss = _pairwise_loss(logits, train_groups, max_pairs_per_group=int(args.max_pairs_per_group))
        listwise_loss = _listwise_loss(logits, train_groups)

        total_loss = logits.new_tensor(0.0)
        if use_pointwise:
            total_loss = total_loss + float(args.lambda_pointwise) * pointwise_loss
        if use_pairwise:
            total_loss = total_loss + float(args.lambda_pairwise) * pairwise_loss
        if use_listwise:
            total_loss = total_loss + float(args.lambda_listwise) * listwise_loss
        total_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            valid_total = float(total_loss.item())
            valid_point = float(pointwise_loss.item())
            valid_pair = float(pairwise_loss.item())
            valid_list = float(listwise_loss.item())
            if x_valid_t is not None:
                valid_logits = _forward_logits(model, x_valid_t, valid_feedback_types)
                valid_point_tensor = F.binary_cross_entropy_with_logits(valid_logits, y_valid_t, reduction="none")
                valid_point_tensor = (valid_point_tensor * w_valid_t).mean()
                valid_pair_tensor = _pairwise_loss(
                    valid_logits,
                    valid_groups,
                    max_pairs_per_group=int(args.max_pairs_per_group),
                )
                valid_list_tensor = _listwise_loss(valid_logits, valid_groups)
                valid_total_tensor = valid_logits.new_tensor(0.0)
                if use_pointwise:
                    valid_total_tensor = valid_total_tensor + float(args.lambda_pointwise) * valid_point_tensor
                if use_pairwise:
                    valid_total_tensor = valid_total_tensor + float(args.lambda_pairwise) * valid_pair_tensor
                if use_listwise:
                    valid_total_tensor = valid_total_tensor + float(args.lambda_listwise) * valid_list_tensor
                valid_total = float(valid_total_tensor.item())
                valid_point = float(valid_point_tensor.item())
                valid_pair = float(valid_pair_tensor.item())
                valid_list = float(valid_list_tensor.item())
        if valid_total < best_valid_loss:
            best_valid_loss = valid_total
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        epoch_row = {
            "epoch": epoch,
            "train_total_loss": float(total_loss.item()),
            "train_pointwise_loss": float(pointwise_loss.item()),
            "train_pairwise_loss": float(pairwise_loss.item()),
            "train_listwise_loss": float(listwise_loss.item()),
            "valid_total_loss": valid_total,
            "valid_pointwise_loss": valid_point,
            "valid_pairwise_loss": valid_pair,
            "valid_listwise_loss": valid_list,
        }
        history.append(epoch_row)
        print("[v5][suppressor][train]", epoch_row)

    if best_state is None:
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        train_logits = _forward_logits(model, x_train_t, train_feedback_types).detach().cpu().numpy()
        valid_logits = (
            _forward_logits(model, x_valid_t, valid_feedback_types).detach().cpu().numpy()
            if x_valid_t is not None
            else np.zeros((0,), dtype=np.float32)
        )
    metrics = {
        "train": _metrics(train_logits, y_train, train_meta),
        "valid": _metrics(valid_logits, y_valid, valid_meta) if len(valid_rows) else {},
        "history": history,
        "objective": objective,
    }

    hidden_dims = tuple(int(item.strip()) for item in str(args.hidden_dims).split(",") if item.strip())
    manifest = SuppressorManifest(
        version=model_type,
        model_type=model_type,
        encoder_model_name=str(dense_index.artifact.model_name),
        use_e5_prefix=bool(dense_index.artifact.use_e5_prefix),
        local_files_only=bool(dense_index.artifact.local_files_only),
        offline=bool(dense_index.artifact.offline),
        device=str(dense_index.artifact.device),
        batch_size=int(dense_index.artifact.batch_size),
        embedding_dim=int(memory_matrix.shape[1]) if memory_matrix.ndim == 2 else 0,
        feature_dim=int(x_train.shape[1]),
        hidden_dims=hidden_dims,
        dropout=float(args.dropout),
        include_query_vec=include_query_vec,
        include_memory_vec=include_memory_vec,
        include_cosine=include_cosine,
        include_product=include_product,
        include_abs_diff=include_abs_diff,
        include_feedback_type=include_feedback_type,
        include_lane=include_lane,
        include_lexical_overlap=include_lexical_overlap,
        include_explicit_mention=include_explicit_mention,
        objective=objective,
        slots_k=int(args.slots_k) if model_type in {"oreo_memory_mlp_v1", "oreo_type_heads_mlp_v1"} else 0,
        top_r=int(args.top_r) if model_type in {"oreo_memory_mlp_v1", "oreo_type_heads_mlp_v1"} else 0,
        tau=float(args.tau) if model_type in {"oreo_memory_mlp_v1", "oreo_type_heads_mlp_v1"} else 1.0,
        feedback_types=tuple(FEEDBACK_TYPES) if model_type == "oreo_type_heads_mlp_v1" else tuple(),
        enable_memory_bias=bool(int(args.enable_memory_bias)),
        lambda_memory_bias=float(args.lambda_memory_bias),
    )
    (artifact_dir / "manifest.json").write_text(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    (artifact_dir / "feature_spec.json").write_text(
        json.dumps(
            {
                "include_query_vec": include_query_vec,
                "include_memory_vec": include_memory_vec,
                "include_cosine": include_cosine,
                "include_product": include_product,
                "include_abs_diff": include_abs_diff,
                "include_feedback_type": include_feedback_type,
                "include_lane": include_lane,
                "include_lexical_overlap": include_lexical_overlap,
                "include_explicit_mention": include_explicit_mention,
                "feature_dim": int(x_train.shape[1]),
                "objective": objective,
                "model_type": model_type,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    torch.save({"state_dict": model.state_dict()}, artifact_dir / "model.pt")
    np.save(artifact_dir / "memory_embeddings.npy", memory_matrix.astype(np.float32))
    (artifact_dir / "memory_ids.json").write_text(json.dumps(memory_ids, ensure_ascii=False), encoding="utf-8")
    write_jsonl(
        artifact_dir / "memory_texts.jsonl",
        [{"memory_id": memory_id, "memory_text": memory_text_by_id.get(memory_id, "")} for memory_id in memory_ids],
    )

    if bool(int(args.enable_memory_bias)):
        bias_by_id, bias_stats = _build_memory_bias(
            train_meta,
            y_train,
            w_train,
            prior_strength=float(args.memory_bias_prior),
        )
    else:
        bias_by_id, bias_stats = {}, {"global_positive_rate": 0.0, "memory_count": 0, "biased_memory_count": 0}
    (artifact_dir / "memory_bias.json").write_text(
        json.dumps(
            {
                "enabled": bool(int(args.enable_memory_bias)),
                "lambda_memory_bias": float(args.lambda_memory_bias),
                "bias_by_id": bias_by_id,
                "stats": bias_stats,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    (artifact_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "[v5][suppressor][train] saved",
        {
            "artifact_dir": str(artifact_dir),
            "best_valid_loss": best_valid_loss,
            "device": str(device),
            "model_type": model_type,
            "objective": objective,
        },
    )


if __name__ == "__main__":
    main()
