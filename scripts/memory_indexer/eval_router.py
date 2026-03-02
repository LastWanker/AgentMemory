"""小评测集：验证路由策略 + Recall@k/MRR + 路由指标。

说明：
- 这个脚本只依赖本仓库最小可用编码器，用于稳定、可复现的对比。
- 默认每条 query 会“重复跑两次”，第二次的路由输出用于一致性指标。
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple
import uuid

# 兼容直接运行 `python scripts/memory_indexer/eval_router.py` 的模块搜索路径。
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.memory_indexer import (
    Encoder,
    MemoryItem,
    Query,
    Router,
    Retriever,
    FieldScorer,
    LearnedFieldScorer,
    SimpleHashEncoder,
    Vectorizer,
    build_memory_items,
    build_memory_index,
    set_trace,
)
from src.memory_indexer.utils import normalize
from scripts.memory_indexer.runtime_utils import (
    build_cache_signature,
    build_data_fingerprint,
    cfg_get,
    enable_run_dir_logging,
    get_git_commit,
    load_config,
    sanitize_slug,
    upsert_run_registry,
    write_json,
)

DATA_DIR = REPO_ROOT / "data"
MEMORY_EVAL_DIR = DATA_DIR / "Memory_Eval"
PROCESSED_DIR = DATA_DIR / "Processed"
VECTOR_CACHE_DIR = DATA_DIR / "VectorCache"
MODEL_WEIGHTS_DIR = DATA_DIR / "ModelWeights"
RUNS_REGISTRY_PATH = REPO_ROOT / "runs" / "registry.csv"
HF_MODEL_NAME = "intfloat/multilingual-e5-small"
SIMPLE_DIMS = 64


class _CacheOnlyTokenizer:
    def tokenize(self, text: str) -> List[str]:
        return text.split()


class _CacheOnlyEncoder(Encoder):
    """Encoder stub used when both memory/query caches are fully reusable."""

    def __init__(self, encoder_id: str) -> None:
        super().__init__(encoder_id=encoder_id)
        self.tokenizer = _CacheOnlyTokenizer()

    def _raise(self) -> None:
        raise RuntimeError("cache-only encoder called due to cache miss")

    def encode_tokens(self, text: str) -> Tuple[List[List[float]], List[str]]:
        self._raise()

    def encode_query_sentence(self, text: str) -> List[float]:
        self._raise()

    def encode_passage_sentence(self, text: str) -> List[float]:
        self._raise()


def expected_encoder_ids(backend: str) -> Tuple[str, str, str]:
    if backend == "simple":
        sentence_id = f"simple-hash@{SIMPLE_DIMS}"
        token_id = sentence_id
    else:
        sentence_id = f"hf-sentence@{HF_MODEL_NAME}"
        token_id = f"e5-token@{HF_MODEL_NAME}"
    if sentence_id == token_id:
        return sentence_id, token_id, sentence_id
    return sentence_id, token_id, f"{sentence_id}|{token_id}"


def resolve_dataset_paths(dataset: str) -> Tuple[Path, Path]:
    """根据数据集名称解析数据文件路径。"""

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


def resolve_query_cache_path(signature: str) -> Path:
    VECTOR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:16]
    return VECTOR_CACHE_DIR / f"eval_cache_{digest}.jsonl"


def resolve_memory_cache_path(signature: str) -> Path:
    VECTOR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:16]
    return VECTOR_CACHE_DIR / f"memory_cache_{digest}.jsonl"


def find_legacy_memory_cache(
    *,
    dataset: str,
    backend: str,
    encoder_id: str,
    strategy: str,
    k: int,
) -> Optional[Path]:
    """Find a reusable legacy memory cache from old naming conventions."""

    dataset_tag = sanitize_slug(dataset)
    encoder_tag = sanitize_slug(encoder_id)
    strategy_tag = sanitize_slug(strategy)
    prefix = f"memory_cache_backend_{sanitize_slug(backend)}__"

    candidates: List[Path] = []
    for path in VECTOR_CACHE_DIR.glob("memory_cache_*.jsonl"):
        name = path.name
        if not name.startswith(prefix):
            continue
        if f"__dataset_{dataset_tag}__" not in name:
            continue
        if f"__encoder_{encoder_tag}__" not in name:
            continue
        if f"__k_{k}__" not in name:
            continue
        if f"__strategy_{strategy_tag}" not in name:
            continue
        candidates.append(path)
    if not candidates:
        return None
    # Prefer larger and newer file (typically richer historical cache).
    candidates.sort(key=lambda p: (p.stat().st_size, p.stat().st_mtime), reverse=True)
    return candidates[0]


def load_memory_items(path: Path) -> List[MemoryItem]:
    """读取 memory_xxx.jsonl，转换为 MemoryItem 列表。"""

    payloads: List[Dict[str, object]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        payloads.append(json.loads(line))

    # 记忆切块与来源标记：入口处统一补全元信息
    return build_memory_items(
        payloads,
        source_default="manual",
        chunk_strategy="sentence_window",
        max_sentences=3,
        tags_default=["general"],
    )


def load_eval_queries(path: Path) -> List[Tuple[str, List[str]]]:
    """读取 eval_xxx.jsonl，返回 (query_text, expected_mem_ids/positives)。"""

    queries: List[Tuple[str, List[str]]] = []
    legacy_field_rows = 0
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if "candidates" in payload or "hard_negatives" in payload:
            legacy_field_rows += 1
        if isinstance(payload.get("positives"), list):
            expected_ids = [str(mid) for mid in payload["positives"]]
        else:
            expected_ids = [str(mid) for mid in payload.get("expected_mem_ids", [])]
        queries.append((payload["query_text"], expected_ids))
    if legacy_field_rows > 0:
        print(
            "[warn] eval 查询文件含旧字段 candidates/hard_negatives，"
            f"已忽略（rows={legacy_field_rows}）"
        )
    return queries


def recall_at_k(hit_ids: Iterable[str], expected_ids: Iterable[str]) -> float:
    """计算 Recall@k：命中比例 / 期望数量。"""

    expected = set(expected_ids)
    if not expected:
        return 0.0
    hits = set(hit_ids) & expected
    return len(hits) / len(expected)


def mrr(hit_ids: List[str], expected_ids: Iterable[str]) -> float:
    """计算 MRR：命中第一个相关项的倒数排名。"""

    expected = set(expected_ids)
    for idx, mem_id in enumerate(hit_ids, start=1):
        if mem_id in expected:
            return 1.0 / idx
    return 0.0


def _compose_encoder_id(sentence_encoder: Encoder, token_encoder: Encoder) -> str:
    if sentence_encoder.encoder_id == token_encoder.encoder_id:
        return sentence_encoder.encoder_id
    return f"{sentence_encoder.encoder_id}|{token_encoder.encoder_id}"


def build_cached_queries(
    queries: List[Tuple[str, List[str]]],
    sentence_encoder: Encoder,
    token_encoder: Encoder,
    vectorizer: Vectorizer,
) -> Iterator[Dict[str, object]]:
    """预编码查询，避免在评测循环里重复编码。"""

    encoder_id = _compose_encoder_id(sentence_encoder, token_encoder)
    for query_text, expected_ids in queries:
        token_vecs, model_tokens = token_encoder.encode_tokens(query_text)
        q_vecs, aux = vectorizer.make_group(token_vecs, model_tokens, query_text)
        q_vecs = [normalize(v) for v in q_vecs]
        lex_tokens = sentence_encoder.tokenizer.tokenize(query_text)
        yield {
            "query_id": str(uuid.uuid4()),
            "query_text": query_text,
            "expected_mem_ids": expected_ids,
            "encoder_id": encoder_id,
            "strategy": vectorizer.strategy,
            "q_vecs": q_vecs,
            "coarse_vec": normalize(sentence_encoder.encode_query_sentence(query_text)),
            "aux": {
                **aux,
                "lex_tokens": lex_tokens,
                "model_tokens": model_tokens,
                "tokens": lex_tokens,
                "coarse_role": "query",
            },
        }


def write_cached_queries(
    path: Path,
    queries: List[Tuple[str, List[str]]],
    sentence_encoder: Encoder,
    token_encoder: Encoder,
    vectorizer: Vectorizer,
    *,
    cache_signature: Optional[str],
) -> None:
    """将预编码的 query 写入磁盘缓存文件。"""

    with path.open("w", encoding="utf-8") as handle:
        if cache_signature:
            handle.write(
                json.dumps({"_meta": {"cache_signature": cache_signature}}, ensure_ascii=False)
                + "\n"
            )
        for payload in build_cached_queries(
            queries, sentence_encoder, token_encoder, vectorizer
        ):
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def iter_cached_queries(
    path: Path,
    *,
    cache_signature: Optional[str],
) -> Iterator[Tuple[Query, List[str]]]:
    """从磁盘缓存中读取 query，避免占用大量内存。"""

    signature_checked = cache_signature is None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict) and "_meta" in payload:
                meta = payload.get("_meta") or {}
                if cache_signature and meta.get("cache_signature") != cache_signature:
                    raise ValueError(
                        f"query cache 签名不匹配: expected={cache_signature} "
                        f"got={meta.get('cache_signature')}"
                    )
                signature_checked = True
                continue
            if not signature_checked:
                raise ValueError("query cache 缺少签名头，需重建缓存。")
            yield (
                Query(
                    query_id=payload["query_id"],
                    text=payload["query_text"],
                    encoder_id=payload["encoder_id"],
                    strategy=payload["strategy"],
                    q_vecs=payload["q_vecs"],
                    coarse_vec=payload["coarse_vec"],
                    aux=payload.get("aux", {}),
                ),
                payload["expected_mem_ids"],
            )


def count_query_rows_in_cache(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict) and "_meta" in payload:
                continue
            count += 1
    return count


def query_cache_compatible(
    path: Path,
    cache_signature: Optional[str],
    *,
    expected_query_rows: Optional[int] = None,
) -> bool:
    if not path.exists():
        return False
    if cache_signature is None:
        # 用户手动管理缓存模式：只要文件存在且非空即复用，不再要求条数与本次评测一致。
        return path.stat().st_size > 0
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict) and "_meta" in payload:
            meta = payload.get("_meta") or {}
            return meta.get("cache_signature") == cache_signature
        return False
    return False


def memory_cache_fully_reusable(
    path: Path,
    items: List[MemoryItem],
    *,
    encoder_id: str,
    strategy: str,
    cache_signature: Optional[str],
) -> bool:
    if not path.exists():
        return False
    required = {item.mem_id: item.text for item in items}
    if not required:
        return True

    matched_ids = set()
    signature_checked = cache_signature is None
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                if isinstance(payload, dict) and "_meta" in payload:
                    meta = payload.get("_meta") or {}
                    if cache_signature and meta.get("cache_signature") != cache_signature:
                        return False
                    signature_checked = True
                    continue
                mem_id = payload.get("mem_id")
                if not isinstance(mem_id, str) or mem_id not in required:
                    continue
                if payload.get("encoder_id") != encoder_id or payload.get("strategy") != strategy:
                    continue
                if payload.get("text") != required[mem_id]:
                    continue
                aux = payload.get("aux") or {}
                coarse_role = payload.get("coarse_role") or (
                    aux.get("coarse_role") if isinstance(aux, dict) else None
                )
                if coarse_role != "passage":
                    continue
                matched_ids.add(mem_id)
        if cache_signature and not signature_checked:
            return False
        return len(matched_ids) == len(required)
    except Exception:
        return False


def sample_eval_queries(
    queries: List[Tuple[str, List[str]]],
    *,
    max_eval_queries: int,
    sample_mode: str,
    sample_seed: int,
) -> List[Tuple[str, List[str]]]:
    if max_eval_queries <= 0 or len(queries) <= max_eval_queries:
        return queries
    if sample_mode == "head":
        return queries[:max_eval_queries]
    if sample_mode == "random":
        picked = list(queries)
        rng = random.Random(sample_seed)
        rng.shuffle(picked)
        return picked[:max_eval_queries]
    raise ValueError(f"未知 eval 采样模式: {sample_mode}")


def sample_cached_query_rows(
    query_rows: List[Tuple[Query, List[str]]],
    *,
    max_eval_queries: int,
    sample_mode: str,
    sample_seed: int,
) -> List[Tuple[Query, List[str]]]:
    if max_eval_queries <= 0 or len(query_rows) <= max_eval_queries:
        return query_rows
    if sample_mode == "head":
        return query_rows[:max_eval_queries]
    if sample_mode == "random":
        picked = list(query_rows)
        rng = random.Random(sample_seed)
        rng.shuffle(picked)
        return picked[:max_eval_queries]
    raise ValueError(f"未知 eval 采样模式: {sample_mode}")


def select_registry_metrics(
    records: List[Dict[str, object]],
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    target: Optional[Dict[str, object]] = None
    for record in records:
        if record.get("ablation_group") == "S-only" and record.get("policy") == "half_hard":
            target = record
            break
    for record in records:
        if target is None and record.get("ablation_group") == "mix(auto)" and record.get("policy") == "half_hard":
            target = record
            break
    if target is None:
        for record in records:
            if record.get("ablation_group") in {"mix(auto)", "baseline(auto)"} and record.get(
                "policy"
            ) == "soft":
                target = record
                break
    if target is None and records:
        target = records[0]
    if not target:
        return None, None, None
    metrics = target.get("metrics", {})
    if not isinstance(metrics, dict):
        return None, None, None
    recall_value = metrics.get("recall_at_k")
    bootstrap = target.get("bootstrap")
    if isinstance(bootstrap, dict):
        recall_ci = bootstrap.get("recall_at_k")
        if isinstance(recall_ci, list) and len(recall_ci) >= 1:
            recall_value = recall_ci[0]
    top1_value = metrics.get("top1_acc")
    mrr_value = metrics.get("mrr")
    return (
        float(recall_value) if recall_value is not None else None,
        float(top1_value) if top1_value is not None else None,
        float(mrr_value) if mrr_value is not None else None,
    )


def print_reranker_meta(reranker_path: str) -> None:
    path = Path(reranker_path)
    if not path.exists():
        print(f"[eval] learned scorer weight not found: {path}")
        return
    print(f"[eval] learned scorer weight: {path}")
    try:
        import torch  # type: ignore

        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(path, map_location="cpu")
        if isinstance(state, dict):
            meta = state.get("meta", {})
            if isinstance(meta, dict):
                keys = (
                    "model_family",
                    "loss_type",
                    "neg_strategy",
                    "epochs",
                    "seed",
                    "dataset",
                    "bipartite_tau",
                    "bipartite_learnable_tau",
                    "bipartite_final_tau",
                )
                picked = {key: meta.get(key) for key in keys if key in meta}
                if picked:
                    print(f"[eval] learned scorer meta: {picked}")
    except Exception as exc:
        print(f"[eval] learned scorer meta unavailable: {exc}")


def evaluate_policy(
    policy: str,
    cached_queries: Iterator[Tuple[Query, List[str]]],
    store,
    index,
    lexical_index,
    top_n: int,
    top_k: int,
    candidate_mode: str = "coarse",
    consistency_pass: bool = False,
    fixed_channel_weights: Optional[Dict[str, float]] = None,
    use_learned_scorer: bool = False,
    reranker_path: Optional[str] = None,
    scorer_batch_size: int = 512,
    eval_workers: int = 1,
    ablation_name: str = "baseline",
    debug_score_flat: bool = False,
    debug_max_queries: int = 5,
    debug_topm: int = 10,
) -> Dict[str, float]:
    """评估单个路由策略，返回平均指标。"""

    rows = list(cached_queries)
    if not rows:
        return {
            "recall_at_k": 0.0,
            "coarse_rank_recall_at_k": 0.0,
            "coarse_rank_recall_at_pos_count": 0.0,
            "recall_at_pos_count": 0.0,
            "retrieval_recall_at_n": 0.0,
            "mrr": 0.0,
            "top1_acc": 0.0,
            "entropy": 0.0,
            "mass_at_k": 0.0,
            "consistency": 0.0,
            "counterfactual_drop_top1": 0.0,
            "counterfactual_drop_topk": 0.0,
            "candidate_size": 0.0,
            "coarse_lexical_jaccard": 0.0,
            "coarse_elapsed_s": 0.0,
            "lexical_elapsed_s": 0.0,
            "scorer_elapsed_s": 0.0,
            "routing_elapsed_s": 0.0,
            "retrieve_elapsed_s": 0.0,
            "coarse_elapsed_s_total": 0.0,
            "lexical_elapsed_s_total": 0.0,
            "scorer_elapsed_s_total": 0.0,
            "routing_elapsed_s_total": 0.0,
            "retrieve_elapsed_s_total": 0.0,
        }

    def summarize_scores(values: List[float]) -> Tuple[float, float]:
        if not values:
            return 0.0, 0.0
        mean_value = sum(values) / len(values)
        std_value = math.sqrt(sum((value - mean_value) ** 2 for value in values) / len(values))
        return max(values) - min(values), std_value

    def build_retriever() -> Retriever:
        router = Router(policy=policy, top_k=top_k, fixed_channel_weights=fixed_channel_weights)
        scorer = FieldScorer()
        if use_learned_scorer:
            if not reranker_path:
                raise ValueError("启用 learned scorer 时必须提供 --reranker-path")
            scorer = LearnedFieldScorer(
                reranker_path=reranker_path,
                batch_size=scorer_batch_size,
            )
        return Retriever(
            store,
            index,
            scorer,
            router=router,
            lexical_index=lexical_index,
        )

    def evaluate_rows(
        worker_rows: List[Tuple[Query, List[str]]],
        worker_id: int,
    ) -> Dict[str, object]:
        retriever = build_retriever()
        local_total_recall = 0.0
        local_total_coarse_rank_recall = 0.0
        local_total_coarse_rank_recall_at_pos_count = 0.0
        local_total_recall_at_pos_count = 0.0
        local_total_retrieval_recall_at_n = 0.0
        local_total_mrr = 0.0
        local_total_top1 = 0.0
        local_count = 0
        local_metric_sums = {
            "entropy": 0.0,
            "mass_at_k": 0.0,
            "consistency": 0.0,
            "counterfactual_drop_top1": 0.0,
            "counterfactual_drop_topk": 0.0,
            "candidate_size": 0.0,
            "coarse_lexical_jaccard": 0.0,
            "coarse_elapsed_s": 0.0,
            "lexical_elapsed_s": 0.0,
            "scorer_elapsed_s": 0.0,
            "routing_elapsed_s": 0.0,
            "retrieve_elapsed_s": 0.0,
        }
        local_debug_printed = 0
        for query, expected_mem_ids in worker_rows:
            local_count += 1
            expected_set = set(expected_mem_ids)
            score_top_k = max(top_k, len(expected_set))

            if consistency_pass:
                retriever.retrieve(
                    query,
                    top_n=top_n,
                    top_k=score_top_k,
                    candidate_mode=candidate_mode,
                )

            results = retriever.retrieve(
                query,
                top_n=top_n,
                top_k=score_top_k,
                candidate_mode=candidate_mode,
            )
            ranked_ids = [result.mem_id for result in results]
            hit_ids = ranked_ids[:top_k]
            local_total_recall += recall_at_k(hit_ids, expected_set)
            local_total_mrr += mrr(hit_ids, expected_set)
            local_total_top1 += 1.0 if hit_ids and hit_ids[0] in expected_set else 0.0

            route_output = results[0].route_output if results else None
            if route_output and route_output.scores:
                candidate_ids = list(route_output.scores.keys())
            else:
                candidate_ids = list(ranked_ids)
            coarse_ranked_ids: List[str] = []
            if route_output:
                coarse_ranked_ids = list(route_output.explain.get("coarse_ranked_ids", []))
            if not coarse_ranked_ids:
                coarse_ranked_ids = candidate_ids
            local_total_coarse_rank_recall += recall_at_k(
                coarse_ranked_ids[:top_k], expected_set
            )
            local_total_coarse_rank_recall_at_pos_count += recall_at_k(
                coarse_ranked_ids[: len(expected_set)], expected_set
            )
            local_total_retrieval_recall_at_n += recall_at_k(candidate_ids, expected_set)
            local_total_recall_at_pos_count += recall_at_k(
                ranked_ids[: len(expected_set)], expected_set
            )

            if route_output:
                for key in local_metric_sums:
                    local_metric_sums[key] += route_output.metrics.get(key, 0.0)

                if (
                    debug_score_flat
                    and worker_id == 0
                    and ablation_name == "S-only"
                    and local_debug_printed < max(0, debug_max_queries)
                ):
                    topm = max(1, debug_topm)
                    ranked_results = sorted(
                        results,
                        key=lambda item: route_output.scores.get(item.mem_id, item.score),
                        reverse=True,
                    )
                    top_results = ranked_results[:topm]
                    top_semantic_scores = [
                        float(item.features.get("semantic_score", 0.0)) for item in top_results
                    ]
                    top_combined_scores = [
                        float(route_output.scores.get(item.mem_id, item.score))
                        for item in top_results
                    ]
                    top_final_scores = [float(item.score) for item in top_results]
                    top_raw_scores = [
                        float((item.debug.get("raw_score", [0.0]) if item.debug else [0.0])[0])
                        for item in top_results
                    ]

                    semantic_range, semantic_std = summarize_scores(top_semantic_scores)
                    combined_range, combined_std = summarize_scores(top_combined_scores)
                    final_range, final_std = summarize_scores(top_final_scores)
                    raw_range, raw_std = summarize_scores(top_raw_scores)

                    query_id_or_prefix = query.query_id if query.query_id else query.text[:30]
                    print(
                        "[debug-score-flat] "
                        f"query={query_id_or_prefix} | "
                        f"ablation={ablation_name} | "
                        f"policy={policy}"
                    )
                    print(
                        f"  top{topm}_semantic={','.join(f'{value:.6f}' for value in top_semantic_scores)}"
                        f" | max-min={semantic_range:.6f} | std={semantic_std:.6f}"
                    )
                    print(
                        f"  top{topm}_combined={','.join(f'{value:.6f}' for value in top_combined_scores)}"
                        f" | max-min={combined_range:.6f} | std={combined_std:.6f}"
                    )
                    print(
                        f"  top{topm}_final={','.join(f'{value:.6f}' for value in top_final_scores)}"
                        f" | max-min={final_range:.6f} | std={final_std:.6f}"
                    )
                    print(
                        f"  top{topm}_raw={','.join(f'{value:.6f}' for value in top_raw_scores)}"
                        f" | max-min={raw_range:.6f} | std={raw_std:.6f}"
                    )
                    local_debug_printed += 1

        return {
            "total_recall": local_total_recall,
            "total_coarse_rank_recall": local_total_coarse_rank_recall,
            "total_coarse_rank_recall_at_pos_count": local_total_coarse_rank_recall_at_pos_count,
            "total_recall_at_pos_count": local_total_recall_at_pos_count,
            "total_retrieval_recall_at_n": local_total_retrieval_recall_at_n,
            "total_mrr": local_total_mrr,
            "total_top1": local_total_top1,
            "count": local_count,
            "metric_sums": local_metric_sums,
        }

    worker_count = max(1, int(eval_workers))
    if worker_count > 1 and debug_score_flat:
        print("[warn] eval_workers>1 时 debug-score-flat 仅在 worker0 上输出。")

    worker_results: List[Dict[str, object]] = []
    if worker_count == 1 or len(rows) <= 1:
        worker_results.append(evaluate_rows(rows, worker_id=0))
    else:
        shard_size = math.ceil(len(rows) / worker_count)
        shards = [rows[i : i + shard_size] for i in range(0, len(rows), shard_size)]
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(evaluate_rows, shard, shard_id)
                for shard_id, shard in enumerate(shards)
                if shard
            ]
            for future in futures:
                worker_results.append(future.result())

    total_recall = 0.0
    total_coarse_rank_recall = 0.0
    total_coarse_rank_recall_at_pos_count = 0.0
    total_recall_at_pos_count = 0.0
    total_retrieval_recall_at_n = 0.0
    total_mrr = 0.0
    total_top1 = 0.0
    count = 0
    metric_sums = {
        "entropy": 0.0,
        "mass_at_k": 0.0,
        "consistency": 0.0,
        "counterfactual_drop_top1": 0.0,
        "counterfactual_drop_topk": 0.0,
        "candidate_size": 0.0,
        "coarse_lexical_jaccard": 0.0,
        "coarse_elapsed_s": 0.0,
        "lexical_elapsed_s": 0.0,
        "scorer_elapsed_s": 0.0,
        "routing_elapsed_s": 0.0,
        "retrieve_elapsed_s": 0.0,
    }
    for result in worker_results:
        total_recall += float(result["total_recall"])
        total_coarse_rank_recall += float(result["total_coarse_rank_recall"])
        total_coarse_rank_recall_at_pos_count += float(
            result["total_coarse_rank_recall_at_pos_count"]
        )
        total_recall_at_pos_count += float(result["total_recall_at_pos_count"])
        total_retrieval_recall_at_n += float(result["total_retrieval_recall_at_n"])
        total_mrr += float(result["total_mrr"])
        total_top1 += float(result["total_top1"])
        count += int(result["count"])
        partial_metric_sums = result["metric_sums"]
        if isinstance(partial_metric_sums, dict):
            for key in metric_sums:
                metric_sums[key] += float(partial_metric_sums.get(key, 0.0))

    count = max(1, count)
    averages = {
        "recall_at_k": total_recall / count,
        "coarse_rank_recall_at_k": total_coarse_rank_recall / count,
        "coarse_rank_recall_at_pos_count": total_coarse_rank_recall_at_pos_count / count,
        "recall_at_pos_count": total_recall_at_pos_count / count,
        "retrieval_recall_at_n": total_retrieval_recall_at_n / count,
        "mrr": total_mrr / count,
        "top1_acc": total_top1 / count,
    }
    averages["gain_at_k"] = averages["recall_at_k"] - averages["coarse_rank_recall_at_k"]
    averages["gain_at_pos_count"] = (
        averages["recall_at_pos_count"] - averages["coarse_rank_recall_at_pos_count"]
    )
    averages.update({key: value / count for key, value in metric_sums.items()})
    for key in (
        "coarse_elapsed_s",
        "lexical_elapsed_s",
        "scorer_elapsed_s",
        "routing_elapsed_s",
        "retrieve_elapsed_s",
    ):
        averages[f"{key}_total"] = metric_sums[key]
    return averages



def bootstrap_ci(values: List[float]) -> Tuple[float, float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    sorted_values = sorted(values)
    n = len(sorted_values)
    lo = sorted_values[int(0.025 * (n - 1))]
    hi = sorted_values[int(0.975 * (n - 1))]
    mean = sum(sorted_values) / n
    std = math.sqrt(sum((v - mean) ** 2 for v in sorted_values) / n)
    return mean, std, lo, hi


def run_bootstrap(
    *,
    query_rows: List[Tuple[Query, List[str]]],
    n_samples: int,
    policy: str,
    store,
    index,
    lexical_index,
    top_n: int,
    top_k: int,
    candidate_mode: str,
    consistency_pass: bool,
    fixed_channel_weights: Optional[Dict[str, float]],
    use_learned_scorer: bool,
    reranker_path: Optional[str],
    scorer_batch_size: int,
    eval_workers: int,
    ablation_name: str,
) -> Dict[str, Tuple[float, float, float, float]]:
    if n_samples <= 0 or not query_rows:
        return {}
    recall_draws: List[float] = []
    coarse_rank_recall_pos_count_draws: List[float] = []
    recall_pos_count_draws: List[float] = []
    gain_at_k_draws: List[float] = []
    gain_at_pos_count_draws: List[float] = []
    retrieval_recall_draws: List[float] = []
    mrr_draws: List[float] = []
    top1_draws: List[float] = []
    for _ in range(n_samples):
        sampled = [random.choice(query_rows) for _ in range(len(query_rows))]
        metrics = evaluate_policy(
            policy,
            iter(sampled),
            store,
            index,
            lexical_index,
            top_n,
            top_k,
            candidate_mode=candidate_mode,
            consistency_pass=consistency_pass,
            fixed_channel_weights=fixed_channel_weights,
            use_learned_scorer=use_learned_scorer,
            reranker_path=reranker_path,
            scorer_batch_size=scorer_batch_size,
            eval_workers=eval_workers,
            ablation_name=ablation_name,
        )
        recall_draws.append(metrics["recall_at_k"])
        coarse_rank_recall_pos_count_draws.append(metrics["coarse_rank_recall_at_pos_count"])
        recall_pos_count_draws.append(metrics["recall_at_pos_count"])
        gain_at_k_draws.append(metrics["gain_at_k"])
        gain_at_pos_count_draws.append(metrics["gain_at_pos_count"])
        retrieval_recall_draws.append(metrics["retrieval_recall_at_n"])
        mrr_draws.append(metrics["mrr"])
        top1_draws.append(metrics["top1_acc"])
    return {
        "recall_at_k": bootstrap_ci(recall_draws),
        "coarse_rank_recall_at_pos_count": bootstrap_ci(coarse_rank_recall_pos_count_draws),
        "recall_at_pos_count": bootstrap_ci(recall_pos_count_draws),
        "gain_at_k": bootstrap_ci(gain_at_k_draws),
        "gain_at_pos_count": bootstrap_ci(gain_at_pos_count_draws),
        "retrieval_recall_at_n": bootstrap_ci(retrieval_recall_draws),
        "mrr": bootstrap_ci(mrr_draws),
        "top1_acc": bootstrap_ci(top1_draws),
    }

def main() -> None:
    run_start = time.perf_counter()
    """入口：运行软/半硬/硬路由对比评测。"""
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    pre_args, _ = pre_parser.parse_known_args()
    loaded_config = load_config(pre_args.config)

    default_dataset = cfg_get(loaded_config, "common.dataset", "normal")
    default_top_n = int(cfg_get(loaded_config, "common.top_n", 30))
    default_top_k = int(cfg_get(loaded_config, "common.top_k", 5))
    default_candidate_mode = cfg_get(loaded_config, "common.candidate_mode", "coarse")
    default_encoder_backend = cfg_get(loaded_config, "common.encoder_backend", "hf")
    default_policies = cfg_get(loaded_config, "eval_router.policies", "half_hard")
    default_bootstrap = int(cfg_get(loaded_config, "eval_router.bootstrap", 0))
    default_max_eval_queries = int(cfg_get(loaded_config, "eval_router.max_eval_queries", 1000))
    default_eval_sample_seed = int(cfg_get(loaded_config, "eval_router.eval_sample_seed", 11))
    default_eval_sample_mode = cfg_get(loaded_config, "eval_router.eval_sample_mode", "random")
    default_eval_workers = int(cfg_get(loaded_config, "eval_router.eval_workers", 1))
    default_torch_num_threads = int(cfg_get(loaded_config, "eval_router.torch_num_threads", 0))
    default_ablation_groups = cfg_get(loaded_config, "eval_router.ablation_groups", "S-only")

    parser = argparse.ArgumentParser(description="运行路由评测集")
    parser.add_argument(
        "--config",
        default=pre_args.config,
        help="配置文件路径（JSON 或 YAML）",
    )
    parser.add_argument(
        "--run-dir",
        help="运行产物目录；传入后会写 eval.log.txt / eval.metrics.json / config.snapshot.json",
    )
    parser.add_argument(
        "--dataset",
        default=default_dataset,
        help="选择评测集 (default: normal)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=default_top_n,
        help=f"粗召回数量 top_n (default: {default_top_n})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=default_top_k,
        help="最终输出 top_k (default: 5)",
    )
    parser.add_argument(
        "--candidate-mode",
        choices=("coarse", "lexical", "union"),
        default=default_candidate_mode,
        help="候选池策略（当前推荐 coarse）：coarse / lexical / union",
    )
    # 下面这些参数用于开关耗时点，便于组合使用。
    consistency_group = parser.add_mutually_exclusive_group()
    consistency_group.add_argument(
        "--consistency-pass",
        action="store_true",
        help="开启每条 query 的第二次检索（计算一致性对比）",
    )
    consistency_group.add_argument(
        "--no-consistency-pass",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--policies",
        default=default_policies,
        help="只评估指定 policy，逗号分隔 (soft/half_hard/hard)",
    )
    parser.add_argument(
        "--ablation",
        choices=("baseline", "coarse"),
        help="快捷：baseline=>mix(auto)；coarse=>C-only",
    )
    parser.add_argument(
        "--ablation-groups",
        help="精确挑选 ablation 组名，逗号分隔",
    )
    parser.add_argument(
        "--no-lexical",
        action="store_true",
        help="关闭词法通道（不构建 lexical_index）",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="强制重建 query/memory 缓存",
    )
    parser.add_argument(
        "--cache-alias",
        default=None,
        help="固定缓存别名（例如 users/easy/normal）；启用后使用 eval_cache_<alias>.jsonl 与 memory_cache_<alias>.jsonl。",
    )
    parser.add_argument(
        "--no-cache-signature",
        action="store_true",
        help="关闭缓存签名校验（适合手动管理固定缓存文件）。",
    )
    parser.add_argument(
        "--encoder-timing",
        action="store_true",
        help="输出 encoder 构建与首轮编码的耗时",
    )
    trace_group = parser.add_mutually_exclusive_group()
    trace_group.add_argument("--trace", action="store_true", help="开启运行 trace 输出")
    trace_group.add_argument("--no-trace", action="store_true", help="关闭运行 trace 输出")
    hf_group = parser.add_mutually_exclusive_group()
    hf_group.add_argument("--hf-offline", action="store_true", help="强制 HF 离线加载")
    hf_group.add_argument("--hf-online", action="store_true", help="强制 HF 在线加载")
    hf_group.add_argument("--hf-local-only", action="store_true", help="仅使用本地 HF 模型缓存")
    parser.add_argument(
        "--encoder-backend",
        choices=("hf", "simple"),
        default=default_encoder_backend,
        help="编码器后端：hf(默认) 或 simple（用于快速回归）",
    )
    parser.add_argument(
        "--use-learned-scorer",
        action="store_true",
        help="启用 learned semantic scorer",
    )
    parser.add_argument(
        "--reranker-path",
        default=str(MODEL_WEIGHTS_DIR / "listwise_bipartite_reranker.pt"),
        help="learned scorer 权重路径 (.pt)",
    )
    parser.add_argument(
        "--scorer-batch-size",
        type=int,
        default=512,
        help="learned scorer 批量打分 batch 大小",
    )
    parser.add_argument(
        "--eval-workers",
        type=int,
        default=default_eval_workers,
        help="query 级并行 worker 数（线程并行）",
    )
    parser.add_argument(
        "--torch-num-threads",
        type=int,
        default=default_torch_num_threads,
        help="PyTorch CPU 线程数；0 表示保持默认",
    )
    parser.add_argument(
        "--debug-score-flat",
        action="store_true",
        help="开启分数扁平分布调试打印（仅 S-only 组生效）",
    )
    parser.add_argument(
        "--debug-max-queries",
        type=int,
        default=5,
        help="调试时最多打印多少个 query (default: 5)",
    )
    parser.add_argument(
        "--debug-topm",
        type=int,
        default=10,
        help="调试时每个 query 打印多少个 top combined score (default: 10)",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=default_bootstrap,
        help="bootstrap 重采样次数，0 表示关闭",
    )
    parser.add_argument(
        "--max-eval-queries",
        type=int,
        default=default_max_eval_queries,
        help="评测 query 子采样上限；0 表示全量",
    )
    parser.add_argument(
        "--eval-sample-seed",
        type=int,
        default=default_eval_sample_seed,
        help="评测 query 采样随机种子",
    )
    parser.add_argument(
        "--eval-sample-mode",
        choices=("random", "head"),
        default=default_eval_sample_mode,
        help="评测 query 采样策略：random 或 head",
    )
    args = parser.parse_args()
    if not args.ablation_groups:
        args.ablation_groups = default_ablation_groups
    if args.ablation and args.ablation_groups == default_ablation_groups:
        # Allow --ablation shortcuts to override default ablation group.
        args.ablation_groups = None
    args.eval_workers = max(1, int(args.eval_workers))

    if args.torch_num_threads > 0:
        try:
            import torch  # type: ignore

            torch.set_num_threads(int(args.torch_num_threads))
            if hasattr(torch, "set_num_interop_threads"):
                try:
                    torch.set_num_interop_threads(1)
                except RuntimeError:
                    # set_num_interop_threads 只允许进程生命周期内设置一次。
                    pass
            print(
                "[runtime] torch thread config: "
                f"num_threads={args.torch_num_threads}, num_interop_threads=1"
            )
        except Exception as exc:
            print(f"[warn] 设置 torch-num-threads 失败，继续默认线程策略: {exc}")

    run_root = enable_run_dir_logging(
        args.run_dir,
        log_filename="eval.log.txt",
        argv=sys.argv,
        config_snapshot={
            "config_path": args.config,
            "loaded_config": loaded_config,
            "effective_args": vars(args),
        },
    )

    if args.use_learned_scorer:
        print_reranker_meta(args.reranker_path)
    else:
        print(f"[eval] learned scorer disabled; reranker_path={args.reranker_path}")

    if args.trace:
        set_trace(True)
    elif args.no_trace:
        set_trace(False)
    else:
        set_trace(False)

    if args.encoder_timing:
        os.environ["MEMORY_ENCODER_TIMING"] = "1"

    if args.hf_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_LOCAL_FILES_ONLY"] = "1"
    elif args.hf_online:
        os.environ["HF_HUB_ONLINE"] = "1"
        os.environ["TRANSFORMERS_ONLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "0"
        os.environ["TRANSFORMERS_OFFLINE"] = "0"
        os.environ["HF_LOCAL_FILES_ONLY"] = "0"
    elif args.hf_local_only:
        os.environ["HF_LOCAL_FILES_ONLY"] = "1"

    memory_path, eval_path = resolve_dataset_paths(args.dataset)
    if not memory_path.exists() or not eval_path.exists():
        raise FileNotFoundError(f"缺少数据集文件: {memory_path.name} / {eval_path.name}")

    if args.candidate_mode != "coarse":
        print("[warn] 当前评测已切换为“仅粗召回候选池”，candidate_mode 自动切换为 coarse")
        args.candidate_mode = "coarse"

    if args.no_lexical and args.candidate_mode != "coarse":
        print("[warn] --no-lexical 已开启，candidate_mode 自动切换为 coarse")
        args.candidate_mode = "coarse"

    items = load_memory_items(memory_path)
    all_queries = load_eval_queries(eval_path)
    total_queries = len(all_queries)
    requested_queries = sample_eval_queries(
        all_queries,
        max_eval_queries=args.max_eval_queries,
        sample_mode=args.eval_sample_mode,
        sample_seed=args.eval_sample_seed,
    )
    if len(requested_queries) != total_queries:
        print(
            f"[eval] requested queries sampled: {len(requested_queries)}/{total_queries} "
            f"(mode={args.eval_sample_mode}, seed={args.eval_sample_seed})"
        )

    data_fingerprint = build_data_fingerprint(memory_path, eval_path)
    vectorizer = Vectorizer(strategy="token_pool_topk", k=8)
    sentence_encoder_id, token_encoder_id, encoder_id = expected_encoder_ids(
        args.encoder_backend
    )
    query_cache_signature: Optional[str] = build_cache_signature(
        {
            "dataset": args.dataset,
            "backend": args.encoder_backend,
            "encoder": encoder_id,
            "strategy": vectorizer.strategy,
            "k": vectorizer.k,
            "datafp": data_fingerprint,
            "max_eval_queries": args.max_eval_queries,
            "eval_sample_seed": args.eval_sample_seed,
            "eval_sample_mode": args.eval_sample_mode,
            "cache_v": "v3",
        }
    )
    memory_cache_signature: Optional[str] = build_cache_signature(
        {
            # memory 向量缓存只绑定“记忆侧”配置，避免 eval 抽样参数触发重建。
            "dataset": args.dataset,
            "backend": args.encoder_backend,
            "encoder": encoder_id,
            "strategy": vectorizer.strategy,
            "k": vectorizer.k,
            "cache_v": "mem_simple_v1",
        }
    )
    if args.no_cache_signature:
        query_cache_signature = None
        memory_cache_signature = None

    VECTOR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if args.cache_alias:
        alias = sanitize_slug(str(args.cache_alias))
        cache_path = VECTOR_CACHE_DIR / f"eval_cache_{alias}.jsonl"
        memory_cache_path = VECTOR_CACHE_DIR / f"memory_cache_{alias}.jsonl"
    else:
        cache_path = resolve_query_cache_path(query_cache_signature or "query_nosig")
        memory_cache_path = resolve_memory_cache_path(memory_cache_signature or "memory_nosig")

    # 兼容历史缓存文件命名：若新命名路径不存在，则回退到可复用的 legacy memory cache。
    selected_memory_cache_path = memory_cache_path
    if not selected_memory_cache_path.exists():
        legacy_memory_cache = find_legacy_memory_cache(
            dataset=args.dataset,
            backend=args.encoder_backend,
            encoder_id=encoder_id,
            strategy=vectorizer.strategy,
            k=vectorizer.k,
        )
        if legacy_memory_cache is not None:
            selected_memory_cache_path = legacy_memory_cache
            print(f"[cache] 复用 legacy memory cache: {legacy_memory_cache}")
            if args.cache_alias and memory_cache_path != legacy_memory_cache:
                try:
                    shutil.copy2(legacy_memory_cache, memory_cache_path)
                    selected_memory_cache_path = memory_cache_path
                    print(f"[cache] 已迁移为固定缓存名: {memory_cache_path}")
                except Exception as exc:
                    print(f"[warn] legacy 缓存迁移失败，继续使用 legacy 路径: {exc}")

    if args.rebuild_cache:
        if cache_path.exists():
            cache_path.unlink()
            print(f"已删除 query 缓存: {cache_path}")
        if selected_memory_cache_path.exists():
            selected_memory_cache_path.unlink()
            print(f"已删除 memory 缓存: {selected_memory_cache_path}")
    memory_cache_exists = selected_memory_cache_path.exists()
    if memory_cache_exists:
        print(f"检测到缓存文件，直接读取: {selected_memory_cache_path}")
    else:
        print(f"未检测到缓存文件，将构建并写入: {selected_memory_cache_path}")

    query_cache_ok = query_cache_compatible(
        cache_path,
        query_cache_signature,
        expected_query_rows=len(requested_queries),
    )
    memory_cache_reusable = False
    if query_cache_ok and memory_cache_exists and not args.rebuild_cache:
        memory_cache_reusable = memory_cache_fully_reusable(
            selected_memory_cache_path,
            items,
            encoder_id=encoder_id,
            strategy=vectorizer.strategy,
            cache_signature=memory_cache_signature,
        )
    can_skip_encoder_load = memory_cache_reusable and query_cache_ok and not args.rebuild_cache

    def build_runtime_encoders() -> Tuple[Encoder, Encoder]:
        if args.encoder_backend == "simple":
            sentence = SimpleHashEncoder(dims=SIMPLE_DIMS)
            return sentence, sentence
        from src.memory_indexer.encoder.hf_sentence import HFSentenceEncoder
        from src.memory_indexer.encoder.e5_token import E5TokenEncoder

        sentence = HFSentenceEncoder(model_name=HF_MODEL_NAME, tokenizer="jieba")
        token = E5TokenEncoder(model_name=HF_MODEL_NAME)
        return sentence, token

    if can_skip_encoder_load:
        print("[cache] query/memory 缓存完整命中，跳过 HF 编码器初始化。")
        sentence_encoder = _CacheOnlyEncoder(sentence_encoder_id)
        token_encoder = _CacheOnlyEncoder(token_encoder_id)
    else:
        sentence_encoder, token_encoder = build_runtime_encoders()

    def build_index_with_encoders(
        sentence_encoder_obj: Encoder, token_encoder_obj: Encoder
    ) -> Tuple[object, object, Optional[object]]:
        if args.no_lexical:
            store_obj, index_obj = build_memory_index(
                items,
                sentence_encoder_obj,
                vectorizer,
                token_encoder=token_encoder_obj,
                cache_path=selected_memory_cache_path,
                cache_signature=memory_cache_signature,
                return_lexical=False,
            )
            return store_obj, index_obj, None
        store_obj, index_obj, lexical_obj = build_memory_index(
            items,
            sentence_encoder_obj,
            vectorizer,
            token_encoder=token_encoder_obj,
            cache_path=selected_memory_cache_path,
            cache_signature=memory_cache_signature,
            return_lexical=True,
        )
        return store_obj, index_obj, lexical_obj

    try:
        store, index, lexical_index = build_index_with_encoders(sentence_encoder, token_encoder)
    except RuntimeError as exc:
        if can_skip_encoder_load and "cache-only encoder called" in str(exc):
            print("[cache] 检测到缓存未完全命中，回退为真实编码器初始化。")
            sentence_encoder, token_encoder = build_runtime_encoders()
            store, index, lexical_index = build_index_with_encoders(
                sentence_encoder, token_encoder
            )
        else:
            raise

    if not memory_cache_exists and selected_memory_cache_path.exists():
        print(f"已生成缓存文件: {selected_memory_cache_path}")
    if query_cache_ok:
        if query_cache_signature is None:
            print(f"检测到 query 缓存（无签名模式），直接读取: {cache_path}")
        else:
            print(f"检测到签名匹配 query 缓存，直接读取: {cache_path}")
    else:
        if cache_path.exists():
            if query_cache_signature is None:
                print(f"query 缓存不可用（无签名模式），重建: {cache_path}")
            else:
                print(f"query 缓存签名不匹配，重建: {cache_path}")
        query_cache_build_rows = (
            all_queries if query_cache_signature is None else requested_queries
        )
        write_cached_queries(
            cache_path,
            query_cache_build_rows,
            sentence_encoder,
            token_encoder,
            vectorizer,
            cache_signature=query_cache_signature,
        )
        print(f"已生成缓存文件: {cache_path}")

    top_n = args.top_n
    top_k = args.top_k
    cached_query_rows = list(iter_cached_queries(cache_path, cache_signature=query_cache_signature))
    eval_query_rows = cached_query_rows
    if query_cache_signature is None:
        cached_total = len(cached_query_rows)
        eval_query_rows = sample_cached_query_rows(
            cached_query_rows,
            max_eval_queries=args.max_eval_queries,
            sample_mode=args.eval_sample_mode,
            sample_seed=args.eval_sample_seed,
        )
        if args.max_eval_queries > 0 and cached_total < args.max_eval_queries:
            print(
                f"[cache] warn: 缓存行数不足请求样本数，按缓存全量评测: "
                f"cached={cached_total}, requested={args.max_eval_queries}"
            )
        if len(eval_query_rows) != cached_total:
            print(
                f"[eval] cached queries sampled: {len(eval_query_rows)}/{cached_total} "
                f"(mode={args.eval_sample_mode}, seed={args.eval_sample_seed})"
            )

    print("\n=== 小评测集 ===")
    print(f"样本数: {len(eval_query_rows)} | top_n={top_n} | top_k={top_k} | dataset={args.dataset}")
    print(
        f"candidate_mode={args.candidate_mode} | "
        f"consistency_pass={args.consistency_pass and not args.no_consistency_pass}"
    )
    print("阶段 1/3: 构建索引与编码器完成，开始跑 ablation。")
    metrics_records: List[Dict[str, object]] = []
    scoring_start = time.perf_counter()

    # ablation 组：固定通道权重来观察证据价值与变硬风险。
    ablation_groups = [
        ("mix(auto)", None),
        ("S-only", {"semantic": 1.0, "lexical": 0.0, "meta": 0.0, "coarse": 0.0}),
        ("S+L", {"semantic": 0.7, "lexical": 0.3, "meta": 0.0, "coarse": 0.0}),
        ("S+M", {"semantic": 0.7, "lexical": 0.0, "meta": 0.3, "coarse": 0.0}),
        ("S+L+M", {"semantic": 0.6, "lexical": 0.25, "meta": 0.15, "coarse": 0.0}),
        ("L-only", {"semantic": 0.0, "lexical": 1.0, "meta": 0.0, "coarse": 0.0}),
        ("C-only", {"semantic": 0.0, "lexical": 0.0, "meta": 0.0, "coarse": 1.0}),
    ]
    ablation_lookup = {name: weights for name, weights in ablation_groups}
    ablation_alias = {
        "baseline(auto)": "mix(auto)",
        "baseline": "mix(auto)",
        "mix": "mix(auto)",
        "baseline_coarse": "C-only",
        "coarse": "C-only",
    }
    if args.ablation and args.ablation_groups:
        raise ValueError("--ablation 与 --ablation-groups 不能同时使用")
    if args.ablation == "baseline":
        ablation_groups = [("mix(auto)", ablation_lookup["mix(auto)"])]
    elif args.ablation == "coarse":
        ablation_groups = [("C-only", ablation_lookup["C-only"])]
    if args.ablation_groups:
        requested = []
        for name in [n.strip() for n in args.ablation_groups.split(",") if n.strip()]:
            requested.append(ablation_alias.get(name, name))
        missing = [name for name in requested if name not in ablation_lookup]
        if missing:
            raise ValueError(f"未知 ablation 组: {', '.join(missing)}")
        ablation_groups = [(name, ablation_lookup[name]) for name in requested]

    policies = [name.strip() for name in args.policies.split(",") if name.strip()]
    allowed_policies = {"soft", "half_hard", "hard"}
    invalid_policies = [name for name in policies if name not in allowed_policies]
    if invalid_policies:
        raise ValueError(f"未知 policy: {', '.join(invalid_policies)}")
    consistency_pass = args.consistency_pass and not args.no_consistency_pass

    for group_name, fixed_weights in ablation_groups:
        print("\n" + "=" * 72)
        print(f"阶段 2/3: ablation={group_name} | fixed_weights={fixed_weights}")
        print("=" * 72)
        for policy in policies:
            print(f"  -> 开始评估 policy={policy}")
            metrics = evaluate_policy(
                policy,
                iter(eval_query_rows),
                store,
                index,
                lexical_index,
                top_n,
                top_k,
                candidate_mode=args.candidate_mode,
                consistency_pass=consistency_pass,
                fixed_channel_weights=fixed_weights,
                use_learned_scorer=args.use_learned_scorer,
                reranker_path=args.reranker_path,
                scorer_batch_size=args.scorer_batch_size,
                eval_workers=args.eval_workers,
                ablation_name=group_name,
                debug_score_flat=args.debug_score_flat,
                debug_max_queries=args.debug_max_queries,
                debug_topm=args.debug_topm,
            )
            print(f"  [policy={policy}]")
            print(
                "    RetrievalRecall@N={retrieval_recall_at_n:.3f} | "
                "Recall@k={recall_at_k:.3f} | Recall@P={recall_at_pos_count:.3f}\n"
                "    CoarseRankRecall@k={coarse_rank_recall_at_k:.3f} | "
                "CoarseRankRecall@P={coarse_rank_recall_at_pos_count:.3f} | "
                "Gain@k={gain_at_k:.3f} | Gain@P={gain_at_pos_count:.3f}\n"
                "    MRR={mrr:.3f} | Top1={top1_acc:.3f}\n"
                "    entropy={entropy:.3f} | mass@k={mass_at_k:.3f} | "
                "consistency={consistency:.3f} | "
                "cf_drop_top1={counterfactual_drop_top1:.3f} | "
                "cf_drop_topk={counterfactual_drop_topk:.3f}\n"
                "    candidate_mode={candidate_mode} | candidate_size={candidate_size:.1f}"
                " | coarse_lexical_jaccard={coarse_lexical_jaccard:.3f}".format(
                    candidate_mode=args.candidate_mode,
                    **metrics,
                )
            )
            print(
                "    timing/query_ms coarse={coarse_ms:.2f} | lexical={lexical_ms:.2f} | "
                "scorer={scorer_ms:.2f} | routing={routing_ms:.2f} | total={total_ms:.2f}".format(
                    coarse_ms=metrics.get("coarse_elapsed_s", 0.0) * 1000.0,
                    lexical_ms=metrics.get("lexical_elapsed_s", 0.0) * 1000.0,
                    scorer_ms=metrics.get("scorer_elapsed_s", 0.0) * 1000.0,
                    routing_ms=metrics.get("routing_elapsed_s", 0.0) * 1000.0,
                    total_ms=metrics.get("retrieve_elapsed_s", 0.0) * 1000.0,
                )
            )
            record: Dict[str, object] = {
                "ablation_group": group_name,
                "policy": policy,
                "metrics": metrics,
            }
            if args.bootstrap > 0:
                bs = run_bootstrap(
                    query_rows=eval_query_rows,
                    n_samples=args.bootstrap,
                    policy=policy,
                    store=store,
                    index=index,
                    lexical_index=lexical_index,
                    top_n=top_n,
                    top_k=top_k,
                    candidate_mode=args.candidate_mode,
                    consistency_pass=consistency_pass,
                    fixed_channel_weights=fixed_weights,
                    use_learned_scorer=args.use_learned_scorer,
                    reranker_path=args.reranker_path,
                    scorer_batch_size=args.scorer_batch_size,
                    eval_workers=args.eval_workers,
                    ablation_name=group_name,
                )
                r = bs.get("recall_at_k")
                crp = bs.get("coarse_rank_recall_at_pos_count")
                rp = bs.get("recall_at_pos_count")
                gk = bs.get("gain_at_k")
                gp = bs.get("gain_at_pos_count")
                rr = bs.get("retrieval_recall_at_n")
                t = bs.get("top1_acc")
                if r and crp and rp and gk and gp and rr and t:
                    print(
                        f"    Recall@k CI: {r[0]:.3f} ± {r[1]:.3f} (95% CI [{r[2]:.3f}, {r[3]:.3f}]) | "
                        f"CoarseRankRecall@P CI: {crp[0]:.3f} ± {crp[1]:.3f} "
                        f"(95% CI [{crp[2]:.3f}, {crp[3]:.3f}]) | "
                        f"Recall@P CI: {rp[0]:.3f} ± {rp[1]:.3f} (95% CI [{rp[2]:.3f}, {rp[3]:.3f}]) | "
                        f"Gain@k CI: {gk[0]:.3f} ± {gk[1]:.3f} (95% CI [{gk[2]:.3f}, {gk[3]:.3f}]) | "
                        f"Gain@P CI: {gp[0]:.3f} ± {gp[1]:.3f} (95% CI [{gp[2]:.3f}, {gp[3]:.3f}]) | "
                        f"RetrievalRecall@N CI: {rr[0]:.3f} ± {rr[1]:.3f} "
                        f"(95% CI [{rr[2]:.3f}, {rr[3]:.3f}]) | "
                        f"Top1 CI: {t[0]:.3f} ± {t[1]:.3f} (95% CI [{t[2]:.3f}, {t[3]:.3f}])"
                    )
                record["bootstrap"] = bs
            metrics_records.append(record)
        print("  完成该组 ablation。")

    print("\n阶段 3/3: 全部 ablation 结束，可对比证据价值与变硬风险。\n")
    scoring_elapsed_s = time.perf_counter() - scoring_start
    total_elapsed_s = time.perf_counter() - run_start
    print(
        f"[timing] scoring_elapsed_s={scoring_elapsed_s:.2f} "
        f"({scoring_elapsed_s/60.0:.2f} min)"
    )
    print(
        f"[timing] total_elapsed_s={total_elapsed_s:.2f} "
        f"({total_elapsed_s/60.0:.2f} min)"
    )
    if run_root:
        eval_metrics_payload = {
            "dataset": args.dataset,
            "query_cache_signature": query_cache_signature,
            "memory_cache_signature": memory_cache_signature,
            "cache_path": str(cache_path),
            "memory_cache_path": str(selected_memory_cache_path),
            "data_fingerprint": data_fingerprint,
            "top_n": top_n,
            "top_k": top_k,
            "scorer_batch_size": args.scorer_batch_size,
            "eval_workers": args.eval_workers,
            "torch_num_threads": args.torch_num_threads,
            "candidate_mode": args.candidate_mode,
            "records": metrics_records,
            "timing": {
                "scoring_elapsed_s": scoring_elapsed_s,
                "total_elapsed_s": total_elapsed_s,
            },
        }
        write_json(run_root / "eval.metrics.json", eval_metrics_payload)
        recall, top1, mrr_score = select_registry_metrics(metrics_records)
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
                "top_k": args.top_k,
                "policies": args.policies,
                "candidate_mode": args.candidate_mode,
                "loss_type": "",
                "neg_strategy": "",
                "Recall@5": recall if recall is not None else "",
                "Top1": top1 if top1 is not None else "",
                "MRR": mrr_score if mrr_score is not None else "",
                "weight_path": args.reranker_path if args.use_learned_scorer else "",
                "git_commit": get_git_commit(REPO_ROOT),
            },
        )


if __name__ == "__main__":
    main()



