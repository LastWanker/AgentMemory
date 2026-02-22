"""训练 TinyReranker：支持 multi-positive、ranked negatives、listwise 与 cardinality head。"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shutil
from typing import Dict, List, Optional, Tuple
import random
import sys
import uuid

import torch
import torch.nn.functional as F

# 兼容直接运行 `python scripts/memory_indexer/train_reranker.py` 的模块搜索路径。
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.memory_indexer import (
    SimpleHashEncoder,
    Vectorizer,
    build_memory_index,
    build_memory_items,
    Query,
    Retriever,
    Router,
)
from src.memory_indexer.scorer import TinyReranker, compute_sim_matrix, FieldScorer
from src.memory_indexer.learned_scorer import CardinalityHead
from src.memory_indexer.utils import normalize
from scripts.memory_indexer.runtime_utils import (
    build_cache_signature,
    build_data_fingerprint,
    cfg_get,
    enable_run_dir_logging,
    get_git_commit,
    load_config,
    upsert_run_registry,
    write_json,
)

DATA_DIR = REPO_ROOT / "data"
MEMORY_EVAL_DIR = DATA_DIR / "Memory_Eval"
PROCESSED_DIR = DATA_DIR / "Processed"
VECTOR_CACHE_DIR = DATA_DIR / "VectorCache"
MODEL_WEIGHTS_DIR = DATA_DIR / "ModelWeights"
RUNS_REGISTRY_PATH = REPO_ROOT / "runs" / "registry.csv"


def resolve_dataset_paths(dataset: str) -> Tuple[Path, Path]:
    processed_memory = PROCESSED_DIR / f"memory_{dataset}.jsonl"
    processed_eval = PROCESSED_DIR / f"eval_{dataset}.jsonl"
    legacy_memory = MEMORY_EVAL_DIR / f"memory_{dataset}.jsonl"
    legacy_eval = MEMORY_EVAL_DIR / f"eval_{dataset}.jsonl"
    memory_path = processed_memory if processed_memory.exists() else legacy_memory
    eval_path = processed_eval if processed_eval.exists() else legacy_eval
    if not memory_path.exists():
        memory_path = DATA_DIR / f"memory_{dataset}.jsonl"
    if not eval_path.exists():
        eval_path = DATA_DIR / f"eval_{dataset}.jsonl"
    return memory_path, eval_path


def resolve_memory_cache_path(signature: str) -> Path:
    VECTOR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return VECTOR_CACHE_DIR / f"memory_cache_{signature}.jsonl"


def load_memory_items(path: Path):
    payloads: List[Dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payloads.append(json.loads(line))
    return build_memory_items(
        payloads,
        source_default="manual",
        chunk_strategy="sentence_window",
        max_sentences=3,
        tags_default=["general"],
    )


def _normalize_positives(payload: Dict[str, object]) -> List[str]:
    if isinstance(payload.get("positives"), list):
        return [str(x) for x in payload["positives"]]
    expected = payload.get("expected_mem_ids", [])
    if isinstance(expected, list):
        return [str(x) for x in expected]
    return []


def load_eval_queries(path: Path) -> List[Dict[str, object]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            rows.append(
                {
                    "query_id": payload.get("query_id") or str(uuid.uuid4()),
                    "query_text": payload["query_text"],
                    "positives": _normalize_positives(payload),
                    "candidates": payload.get("candidates", []),
                    "hard_negatives": payload.get("hard_negatives", []),
                }
            )
    return rows


def is_positive(mem_id: str, expected_ids: List[str]) -> bool:
    expected = set(expected_ids)
    if mem_id in expected:
        return True
    return any(mem_id.startswith(f"{base}#") for base in expected)


def encode_query(query_text: str, sentence_encoder, token_encoder, vectorizer) -> Query:
    token_vecs, model_tokens = token_encoder.encode_tokens(query_text)
    q_vecs, aux = vectorizer.make_group(token_vecs, model_tokens, query_text)
    q_vecs = [normalize(v) for v in q_vecs]
    lex_tokens = sentence_encoder.tokenizer.tokenize(query_text)
    encoder_id = (
        sentence_encoder.encoder_id
        if sentence_encoder.encoder_id == token_encoder.encoder_id
        else f"{sentence_encoder.encoder_id}|{token_encoder.encoder_id}"
    )
    return Query(
        query_id=str(uuid.uuid4()),
        text=query_text,
        encoder_id=encoder_id,
        strategy=vectorizer.strategy,
        q_vecs=q_vecs,
        coarse_vec=normalize(sentence_encoder.encode_query_sentence(query_text)),
        aux={
            **aux,
            "lex_tokens": lex_tokens,
            "model_tokens": model_tokens,
            "tokens": lex_tokens,
            "coarse_role": "query",
        },
    )


def configure_hf_runtime(local_only: bool, offline: bool) -> None:
    if local_only:
        os.environ["HF_LOCAL_FILES_ONLY"] = "1"
    else:
        os.environ["HF_LOCAL_FILES_ONLY"] = "0"
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        os.environ["HF_HUB_OFFLINE"] = "0"
        os.environ["TRANSFORMERS_OFFLINE"] = "0"


def build_query_features(query_text: str, candidate_size: int) -> List[float]:
    token_count = len(query_text.split()) or 1
    return [
        math.log1p(token_count),
        math.log1p(len(query_text)),
        math.log1p(max(1, candidate_size)),
    ]


def build_ranked_negatives(candidates: List[str], positives: List[str], max_negatives: int) -> List[str]:
    pos_set = {p for p in positives}
    pool = [mid for mid in candidates if mid not in pos_set]
    return pool[:max_negatives]


def sample_negatives(
    negatives: List[str],
    fallback_pool: List[str],
    sample_size: int,
) -> List[str]:
    if sample_size <= 0:
        return []
    if len(negatives) >= sample_size:
        return random.sample(negatives, sample_size) if len(negatives) > sample_size else negatives
    merged = list(negatives)
    needed = sample_size - len(merged)
    fallback = [mid for mid in fallback_pool if mid not in set(merged)]
    if fallback:
        merged.extend(random.sample(fallback, min(needed, len(fallback))))
    return merged


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    pre_args, _ = pre_parser.parse_known_args()
    loaded_config = load_config(pre_args.config)

    default_dataset = cfg_get(loaded_config, "common.dataset", "normal")
    default_candidate_mode = cfg_get(loaded_config, "common.candidate_mode", "union")
    default_top_n = int(cfg_get(loaded_config, "common.top_n", 20))
    default_encoder_backend = cfg_get(loaded_config, "common.encoder_backend", "hf")
    default_epochs = int(cfg_get(loaded_config, "train_reranker.epochs", 3))
    default_batch_size = int(cfg_get(loaded_config, "train_reranker.batch_size", 16))
    default_lr = float(cfg_get(loaded_config, "train_reranker.lr", 1e-3))
    default_device = cfg_get(loaded_config, "train_reranker.device", "cuda")
    default_loss_type = cfg_get(loaded_config, "train_reranker.loss_type", "pairwise")
    default_neg_strategy = cfg_get(loaded_config, "train_reranker.neg_strategy", "random")

    parser = argparse.ArgumentParser(description="训练 TinyReranker")
    parser.add_argument("--config", default=pre_args.config, help="配置文件路径（JSON 或 YAML）")
    parser.add_argument(
        "--run-dir",
        help="运行产物目录；传入后会写 train.log.txt / train.metrics.json / config.snapshot.json",
    )
    parser.add_argument("--dataset", default=default_dataset)
    parser.add_argument("--candidate-mode", choices=("coarse", "lexical", "union"), default=default_candidate_mode)
    parser.add_argument("--top-n", type=int, default=default_top_n)
    parser.add_argument("--epochs", type=int, default=default_epochs)
    parser.add_argument("--batch-size", type=int, default=default_batch_size)
    parser.add_argument("--lr", type=float, default=default_lr)
    parser.add_argument("--device", default=default_device)
    parser.add_argument(
        "--save-path",
        help="可选权重输出路径。传入 --run-dir 时，仅在显式传入本参数时才会额外导出。",
    )
    parser.add_argument(
        "--export-weights-to",
        help="可选导出目录（例如 data/ModelWeights），按带配置后缀文件名写入，避免覆盖固定文件名。",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--encoder-backend", choices=("hf", "simple"), default=default_encoder_backend)
    parser.add_argument("--neg-per-pos", type=int, default=3)
    parser.add_argument("--max-queries", type=int, default=0, help="限制训练 query 数量，0 表示全量")
    parser.add_argument("--neg-strategy", choices=("random", "ranked"), default=default_neg_strategy)
    parser.add_argument("--loss-type", choices=("pairwise", "listwise"), default=default_loss_type)
    parser.add_argument("--list-size", type=int, default=50)
    parser.add_argument("--use-cardinality-head", action="store_true")
    parser.add_argument("--cardinality-loss-weight", type=float, default=0.1)
    parser.add_argument("--cardinality-k-max", type=int, default=20)
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="快速自检模式：自动切换 easy+simple+小样本参数",
    )
    parser.add_argument("--hf-local-only", action="store_true", help="仅使用本地 HF 模型缓存")
    parser.add_argument("--hf-offline", action="store_true", help="强制 HF 离线模式")
    parser.add_argument("--hf-online", action="store_true", help="允许 HF 在线模式（会关闭 offline/local-only）")

    args = parser.parse_args()
    run_root = enable_run_dir_logging(
        args.run_dir,
        log_filename="train.log.txt",
        argv=sys.argv,
        config_snapshot={
            "config_path": args.config,
            "loaded_config": loaded_config,
            "effective_args": vars(args),
        },
    )

    if not args.hf_online:
        args.hf_local_only = True
        args.hf_offline = True

    # Pitfall note:
    # prefer GPU by default, but auto-downgrade to CPU when CUDA is unavailable.
    requested_device = str(args.device).strip().lower()
    if requested_device in {"auto", "gpu", "cuda"}:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    elif requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("[train] warn: CUDA requested but unavailable, fallback to CPU.")
        args.device = "cpu"
    else:
        args.device = str(args.device)

    if args.fast_dev_run:
        args.dataset = "easy"
        args.encoder_backend = "simple"
        args.top_n = min(args.top_n, 20)
        args.epochs = 1
        args.batch_size = min(args.batch_size, 8)
        args.neg_per_pos = min(args.neg_per_pos, 2)
        if args.max_queries <= 0:
            args.max_queries = 32

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    memory_path, eval_path = resolve_dataset_paths(args.dataset)
    items = load_memory_items(memory_path)
    eval_queries = load_eval_queries(eval_path)
    if args.max_queries > 0:
        eval_queries = eval_queries[: args.max_queries]

    if args.encoder_backend == "simple":
        sentence_encoder = SimpleHashEncoder(dims=64)
        token_encoder = sentence_encoder
    else:
        configure_hf_runtime(local_only=args.hf_local_only, offline=args.hf_offline)
        try:
            from src.memory_indexer.encoder.hf_sentence import HFSentenceEncoder
            from src.memory_indexer.encoder.e5_token import E5TokenEncoder

            sentence_encoder = HFSentenceEncoder(
                model_name="intfloat/multilingual-e5-small",
                tokenizer="jieba",
            )
            token_encoder = E5TokenEncoder(model_name="intfloat/multilingual-e5-small")
        except Exception as exc:
            raise RuntimeError(
                "HF/E5 编码器初始化失败。当前默认是 --hf-local-only + --hf-offline，"
                "请先手动下载模型缓存后重试；或显式使用 --hf-online。"
            ) from exc

    data_fingerprint = build_data_fingerprint(memory_path, eval_path)
    vectorizer = Vectorizer(strategy="token_pool_topk", k=8)
    encoder_id = (
        sentence_encoder.encoder_id
        if sentence_encoder.encoder_id == token_encoder.encoder_id
        else f"{sentence_encoder.encoder_id}|{token_encoder.encoder_id}"
    )
    cache_signature = build_cache_signature(
        {
            "dataset": args.dataset,
            "backend": args.encoder_backend,
            "encoder": encoder_id,
            "strategy": vectorizer.strategy,
            "k": vectorizer.k,
            "datafp": data_fingerprint,
            "cache_v": "v2",
        }
    )
    memory_cache_path = resolve_memory_cache_path(cache_signature)
    store, index, lexical_index = build_memory_index(
        items,
        sentence_encoder,
        vectorizer,
        token_encoder=token_encoder,
        cache_path=memory_cache_path,
        cache_signature=cache_signature,
        return_lexical=True,
    )

    retriever = Retriever(
        store,
        index,
        FieldScorer(),
        router=Router(policy="soft", top_k=args.top_n),
        lexical_index=lexical_index,
    )

    model = TinyReranker().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    cardinality_head: Optional[CardinalityHead] = None
    cardinality_optimizer = None
    if args.use_cardinality_head:
        cardinality_head = CardinalityHead(k_max=args.cardinality_k_max).to(args.device)
        cardinality_optimizer = torch.optim.Adam(cardinality_head.parameters(), lr=args.lr)

    pair_buffer: List[Tuple[torch.Tensor, torch.Tensor]] = []
    listwise_buffer: List[Dict[str, object]] = []
    cardinality_samples: List[Tuple[List[float], int]] = []

    query_count = 0
    query_with_training_pairs = 0
    pos_count_total = 0

    for row in eval_queries:
        query_text = str(row["query_text"])
        expected_ids = list(row.get("positives", []))
        if not expected_ids:
            continue
        query_count += 1
        q = encode_query(query_text, sentence_encoder, token_encoder, vectorizer)
        provided_candidates = [
            str(mem_id)
            for mem_id in (row.get("candidates") or [])
            if isinstance(mem_id, str) and mem_id in store.embs
        ]
        if provided_candidates:
            candidates = provided_candidates
        else:
            results = retriever.retrieve(
                q,
                top_n=args.top_n,
                top_k=args.top_n,
                candidate_mode=args.candidate_mode,
            )
            candidates = [r.mem_id for r in results]

        seen_candidates = set()
        dedup_candidates: List[str] = []
        for mem_id in candidates:
            if mem_id in seen_candidates:
                continue
            seen_candidates.add(mem_id)
            dedup_candidates.append(mem_id)
        candidates = dedup_candidates

        if not candidates:
            continue

        # 若显式 candidates 未覆盖 positives，尽量把可映射正例补进来，避免样本被静默丢弃。
        if not any(is_positive(mem_id, expected_ids) for mem_id in candidates):
            for mem_id in store.embs:
                if is_positive(mem_id, expected_ids) and mem_id not in seen_candidates:
                    seen_candidates.add(mem_id)
                    candidates.append(mem_id)

        positives = [mid for mid in candidates if is_positive(mid, expected_ids)]
        if not positives:
            continue

        pos_count_total += len(positives)
        all_negatives = [mid for mid in candidates if mid not in positives]
        if not all_negatives:
            continue
        hard_negatives = [
            str(mem_id)
            for mem_id in (row.get("hard_negatives") or [])
            if isinstance(mem_id, str) and mem_id in store.embs and mem_id not in positives
        ]

        candidate_features = build_query_features(query_text, len(candidates))
        k_target = min(len(positives), args.cardinality_k_max)
        cardinality_samples.append((candidate_features, k_target))

        if args.loss_type == "pairwise":
            if args.neg_strategy == "ranked":
                ranked_pool = build_ranked_negatives(
                    candidates,
                    positives,
                    max_negatives=max(1, args.neg_per_pos * len(positives)),
                )
                negative_pool = []
                for mem_id in hard_negatives + ranked_pool:
                    if mem_id not in negative_pool:
                        negative_pool.append(mem_id)
                if not negative_pool:
                    negative_pool = all_negatives
            else:
                negative_pool = all_negatives

            for p in positives:
                sampled_negs = sample_negatives(
                    negative_pool,
                    all_negatives,
                    sample_size=args.neg_per_pos,
                )
                for n in sampled_negs:
                    if n == p:
                        continue
                    p_mat = compute_sim_matrix(q.q_vecs or [], store.embs[p].vecs)
                    n_mat = compute_sim_matrix(q.q_vecs or [], store.embs[n].vecs)
                    pair_buffer.append((p_mat, n_mat))
            if positives:
                query_with_training_pairs += 1
        else:
            capped_candidates = candidates[: max(1, args.list_size)]
            matrices = [compute_sim_matrix(q.q_vecs or [], store.embs[mid].vecs) for mid in capped_candidates]
            if not matrices:
                continue
            pos_mask = [1.0 if mid in positives else 0.0 for mid in capped_candidates]
            if sum(pos_mask) <= 0:
                continue
            listwise_buffer.append(
                {
                    "matrices": torch.stack(matrices),
                    "pos_mask": torch.tensor(pos_mask, dtype=torch.float32),
                    "query_features": torch.tensor(candidate_features, dtype=torch.float32),
                    "k_target": k_target,
                }
            )
            query_with_training_pairs += 1

    if args.loss_type == "pairwise" and not pair_buffer:
        raise RuntimeError("没有构造出训练样本，请检查数据集、候选参数或编码器设置。")
    if args.loss_type == "listwise" and not listwise_buffer:
        raise RuntimeError("listwise 模式没有构造出训练样本，请检查 positives/candidates。")

    avg_pos = (pos_count_total / query_count) if query_count else 0.0
    print(
        f"[train] dataset={args.dataset} backend={args.encoder_backend} "
        f"queries={query_count} trained_queries={query_with_training_pairs} avg_positives={avg_pos:.3f}"
    )
    print(
        f"[train] loss_type={args.loss_type} neg_strategy={args.neg_strategy} "
        f"pair_samples={len(pair_buffer)} listwise_samples={len(listwise_buffer)}"
    )
    print(f"[train] epochs={args.epochs} batch_size={args.batch_size} lr={args.lr}")

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        rank_steps = 0
        card_loss_total = 0.0

        if args.loss_type == "pairwise":
            random.shuffle(pair_buffer)
            for i in range(0, len(pair_buffer), args.batch_size):
                batch = pair_buffer[i : i + args.batch_size]
                p_batch = torch.stack([x[0] for x in batch]).to(args.device)
                n_batch = torch.stack([x[1] for x in batch]).to(args.device)
                p_score = model(p_batch)
                n_score = model(n_batch)
                rank_loss = F.softplus(-(p_score - n_score)).mean()
                optimizer.zero_grad()
                rank_loss.backward()
                optimizer.step()
                total_loss += float(rank_loss.item())
                rank_steps += 1

            if args.use_cardinality_head and cardinality_head is not None and cardinality_optimizer is not None:
                random.shuffle(cardinality_samples)
                for i in range(0, len(cardinality_samples), args.batch_size):
                    batch = cardinality_samples[i : i + args.batch_size]
                    feat = torch.tensor([x[0] for x in batch], dtype=torch.float32, device=args.device)
                    tgt = torch.tensor([x[1] for x in batch], dtype=torch.long, device=args.device)
                    logits = cardinality_head(feat)
                    card_loss = F.cross_entropy(logits, tgt)
                    cardinality_optimizer.zero_grad()
                    card_loss.backward()
                    cardinality_optimizer.step()
                    card_loss_total += float(card_loss.item())
        else:
            random.shuffle(listwise_buffer)
            for i in range(0, len(listwise_buffer), args.batch_size):
                batch = listwise_buffer[i : i + args.batch_size]
                max_candidates = max(sample["matrices"].shape[0] for sample in batch)
                mats = torch.zeros((len(batch), max_candidates, 8, 8), dtype=torch.float32)
                pos_mask = torch.zeros((len(batch), max_candidates), dtype=torch.float32)
                valid_mask = torch.zeros((len(batch), max_candidates), dtype=torch.bool)
                feat = torch.zeros((len(batch), 3), dtype=torch.float32)
                tgt = torch.zeros((len(batch),), dtype=torch.long)

                for row_idx, sample in enumerate(batch):
                    n = sample["matrices"].shape[0]
                    mats[row_idx, :n] = sample["matrices"]
                    pos_mask[row_idx, :n] = sample["pos_mask"]
                    valid_mask[row_idx, :n] = True
                    feat[row_idx] = sample["query_features"]
                    tgt[row_idx] = int(sample["k_target"])

                mats = mats.to(args.device)
                pos_mask = pos_mask.to(args.device)
                valid_mask = valid_mask.to(args.device)
                feat = feat.to(args.device)
                tgt = tgt.to(args.device)

                flat_logits = model(mats.view(-1, 8, 8)).view(len(batch), max_candidates)
                masked_logits = flat_logits.masked_fill(~valid_mask, float("-inf"))
                pos_logits = masked_logits.masked_fill(pos_mask <= 0, float("-inf"))
                rank_loss = -(torch.logsumexp(pos_logits, dim=1) - torch.logsumexp(masked_logits, dim=1)).mean()

                loss = rank_loss
                if args.use_cardinality_head and cardinality_head is not None:
                    logits = cardinality_head(feat)
                    card_loss = F.cross_entropy(logits, tgt)
                    card_loss_total += float(card_loss.item())
                    loss = loss + args.cardinality_loss_weight * card_loss

                optimizer.zero_grad()
                if cardinality_optimizer is not None:
                    cardinality_optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if cardinality_optimizer is not None:
                    cardinality_optimizer.step()
                total_loss += float(rank_loss.item())
                rank_steps += 1

        avg_rank_loss = total_loss / max(rank_steps, 1)
        if args.use_cardinality_head:
            card_steps = max(1, math.ceil(len(cardinality_samples) / max(1, args.batch_size)))
            avg_card_loss = card_loss_total / card_steps
            print(f"[train] epoch={epoch} rank_loss={avg_rank_loss:.6f} cardinality_loss={avg_card_loss:.6f}")
        else:
            print(f"[train] epoch={epoch} rank_loss={avg_rank_loss:.6f}")

    default_weight_path = MODEL_WEIGHTS_DIR / "tiny_reranker.pt"
    requested_save_path = Path(args.save_path) if args.save_path else None
    if run_root:
        save_path = run_root / "tiny_reranker.pt"
    else:
        save_path = requested_save_path or default_weight_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state": model.state_dict(), "meta": vars(args)}
    if cardinality_head is not None:
        payload["cardinality_state"] = cardinality_head.state_dict()
        payload["cardinality_meta"] = {
            "input_dim": 3,
            "hidden_dim": 16,
            "k_max": args.cardinality_k_max,
        }
    if save_path.resolve() == default_weight_path.resolve() and save_path.exists():
        backup_name = f"tiny_reranker.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        backup_path = save_path.parent / backup_name
        shutil.copy2(save_path, backup_path)
        print(f"[train] backup => {backup_path}")
    torch.save(payload, save_path)
    print(f"[train] saved => {save_path}")
    copied_weight_path: Optional[Path] = None
    if requested_save_path and save_path.resolve() != requested_save_path.resolve():
        requested_save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, requested_save_path)
        print(f"[train] copied => {requested_save_path}")
        copied_weight_path = requested_save_path

    exported_weight_path: Optional[Path] = None
    if args.export_weights_to:
        export_dir = Path(args.export_weights_to)
        export_dir.mkdir(parents=True, exist_ok=True)
        stem = (
            f"tiny_reranker__{args.dataset}__{args.loss_type}__"
            f"{args.neg_strategy}__seed_{args.seed}"
        )
        candidate = export_dir / f"{stem}.pt"
        suffix = 1
        while candidate.exists():
            candidate = export_dir / f"{stem}__{suffix}.pt"
            suffix += 1
        torch.save(payload, candidate)
        exported_weight_path = candidate
        print(f"[train] exported => {candidate}")

    if run_root:
        train_metrics_payload = {
            "dataset": args.dataset,
            "cache_signature": cache_signature,
            "memory_cache_path": str(memory_cache_path),
            "query_count": query_count,
            "trained_queries": query_with_training_pairs,
            "avg_positives": avg_pos,
            "loss_type": args.loss_type,
            "neg_strategy": args.neg_strategy,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "encoder_backend": args.encoder_backend,
            "encoder_id": encoder_id,
            "vectorizer_strategy": vectorizer.strategy,
            "vectorizer_k": vectorizer.k,
            "candidate_mode": args.candidate_mode,
            "top_n": args.top_n,
            "data_fingerprint": data_fingerprint,
            "saved_weight_path": str(save_path),
            "copied_weight_path": str(copied_weight_path) if copied_weight_path else "",
            "exported_weight_path": str(exported_weight_path) if exported_weight_path else "",
        }
        write_json(run_root / "train.metrics.json", train_metrics_payload)
        upsert_run_registry(
            RUNS_REGISTRY_PATH,
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "run_dir": str(run_root),
                "dataset": args.dataset,
                "encoder_backend": args.encoder_backend,
                "encoder_id": encoder_id,
                "strategy": vectorizer.strategy,
                "top_n": args.top_n,
                "top_k": "",
                "policies": "",
                "candidate_mode": args.candidate_mode,
                "loss_type": args.loss_type,
                "neg_strategy": args.neg_strategy,
                "Recall@5": "",
                "Top1": "",
                "MRR": "",
                "weight_path": str(save_path),
                "git_commit": get_git_commit(REPO_ROOT),
            },
        )


if __name__ == "__main__":
    main()


