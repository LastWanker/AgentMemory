"""训练 TinyReranker：只基于 8x8 相似度矩阵学习 pairwise 排序。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import random
import uuid

import torch
import torch.nn.functional as F

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
from src.memory_indexer.utils import normalize

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MEMORY_EVAL_DIR = DATA_DIR / "Memory_Eval"
VECTOR_CACHE_DIR = DATA_DIR / "VectorCache"
MODEL_WEIGHTS_DIR = DATA_DIR / "ModelWeights"


def resolve_dataset_paths(dataset: str) -> Tuple[Path, Path, Path]:
    memory_path = MEMORY_EVAL_DIR / f"memory_{dataset}.jsonl"
    eval_path = MEMORY_EVAL_DIR / f"eval_{dataset}.jsonl"
    memory_cache_path = VECTOR_CACHE_DIR / f"memory_cache_{dataset}.jsonl"
    if not memory_path.exists():
        memory_path = DATA_DIR / f"memory_{dataset}.jsonl"
    if not eval_path.exists():
        eval_path = DATA_DIR / f"eval_{dataset}.jsonl"
    memory_cache_path.parent.mkdir(parents=True, exist_ok=True)
    return memory_path, eval_path, memory_cache_path


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


def load_eval_queries(path: Path) -> List[Tuple[str, List[str]]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            rows.append((payload["query_text"], payload["expected_mem_ids"]))
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
    """配置 HF 运行时：默认离线+只本地缓存。"""

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


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 TinyReranker")
    # 正式训练默认：normal + hf + 全量
    parser.add_argument("--dataset", choices=("normal", "easy"), default="normal")
    parser.add_argument("--candidate-mode", choices=("coarse", "lexical", "union"), default="union")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-path", default=str(MODEL_WEIGHTS_DIR / "tiny_reranker.pt"))
    parser.add_argument("--encoder-backend", choices=("hf", "simple"), default="hf")
    parser.add_argument("--neg-per-pos", type=int, default=3)
    parser.add_argument("--max-queries", type=int, default=0, help="限制训练 query 数量，0 表示全量")

    # 快速开发模式：easy + simple + 小样本
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="快速自检模式：自动切换 easy+simple+小样本参数",
    )

    # HF 默认离线 + 本地缓存；可显式切到在线
    parser.add_argument("--hf-local-only", action="store_true", help="仅使用本地 HF 模型缓存")
    parser.add_argument("--hf-offline", action="store_true", help="强制 HF 离线模式")
    parser.add_argument("--hf-online", action="store_true", help="允许 HF 在线模式（会关闭 offline/local-only）")

    args = parser.parse_args()

    # 默认离线策略
    if not args.hf_online:
        args.hf_local_only = True
        args.hf_offline = True

    if args.fast_dev_run:
        args.dataset = "easy"
        args.encoder_backend = "simple"
        args.top_n = min(args.top_n, 20)
        args.epochs = 1
        args.batch_size = min(args.batch_size, 8)
        args.neg_per_pos = min(args.neg_per_pos, 2)
        if args.max_queries <= 0:
            args.max_queries = 32

    random.seed(42)
    torch.manual_seed(42)

    memory_path, eval_path, memory_cache_path = resolve_dataset_paths(args.dataset)
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

    vectorizer = Vectorizer(strategy="token_pool_topk", k=8)
    store, index, lexical_index = build_memory_index(
        items,
        sentence_encoder,
        vectorizer,
        token_encoder=token_encoder,
        cache_path=memory_cache_path,
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

    pair_buffer: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for query_text, expected_ids in eval_queries:
        q = encode_query(query_text, sentence_encoder, token_encoder, vectorizer)
        results = retriever.retrieve(
            q,
            top_n=args.top_n,
            top_k=args.top_n,
            candidate_mode=args.candidate_mode,
        )
        candidates = [r.mem_id for r in results]
        positives = [mid for mid in candidates if is_positive(mid, expected_ids)]
        negatives = [mid for mid in candidates if mid not in positives]
        if not positives or not negatives:
            continue
        for p in positives:
            sampled_negs = negatives[: args.neg_per_pos]
            if len(negatives) > args.neg_per_pos:
                sampled_negs = random.sample(negatives, args.neg_per_pos)
            for n in sampled_negs:
                p_mat = compute_sim_matrix(q.q_vecs or [], store.embs[p].vecs)
                n_mat = compute_sim_matrix(q.q_vecs or [], store.embs[n].vecs)
                pair_buffer.append((p_mat, n_mat))

    if not pair_buffer:
        raise RuntimeError("没有构造出训练样本，请检查数据集、候选参数或编码器设置。")

    print(
        f"[train] dataset={args.dataset} backend={args.encoder_backend} "
        f"queries={len(eval_queries)} total_pairs={len(pair_buffer)}"
    )
    print(f"[train] epochs={args.epochs} batch_size={args.batch_size} lr={args.lr}")
    for epoch in range(1, args.epochs + 1):
        random.shuffle(pair_buffer)
        total_loss = 0.0
        steps = 0
        for i in range(0, len(pair_buffer), args.batch_size):
            batch = pair_buffer[i : i + args.batch_size]
            p_batch = torch.stack([x[0] for x in batch]).to(args.device)
            n_batch = torch.stack([x[1] for x in batch]).to(args.device)
            p_score = model(p_batch)
            n_score = model(n_batch)
            loss = F.softplus(-(p_score - n_score)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            steps += 1
        print(f"[train] epoch={epoch} loss={total_loss / max(steps, 1):.6f}")

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "meta": vars(args)}, save_path)
    print(f"[train] saved => {save_path}")


if __name__ == "__main__":
    main()
