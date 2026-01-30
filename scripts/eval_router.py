"""小评测集：验证路由策略 + Recall@k/MRR + 路由指标。

说明：
- 这个脚本只依赖本仓库最小可用编码器，用于稳定、可复现的对比。
- 每条 query 会“重复跑两次”，第二次的路由输出用于一致性指标。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple
import uuid

from src.memory_indexer import (
    Encoder,
    MemoryItem,
    Query,
    Router,
    Retriever,
    FieldScorer,
    Vectorizer,
    build_memory_items,
    build_memory_index,
    set_trace,
)
from src.memory_indexer.encoder.hf_sentence import HFSentenceEncoder
from src.memory_indexer.utils import normalize

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MEMORY_PATH = DATA_DIR / "memory.jsonl"
EVAL_PATH = DATA_DIR / "eval.jsonl"
EVAL_CACHE_PATH = DATA_DIR / "eval_cache.jsonl"


def load_memory_items(path: Path) -> List[MemoryItem]:
    """读取 memory.jsonl，转换为 MemoryItem 列表。"""

    payloads: List[Dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
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
    """读取 eval.jsonl，返回 (query_text, expected_mem_ids)。"""

    queries: List[Tuple[str, List[str]]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        queries.append((payload["query_text"], payload["expected_mem_ids"]))
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


def build_cached_queries(
    queries: List[Tuple[str, List[str]]],
    encoder: Encoder,
    vectorizer: Vectorizer,
) -> Iterator[Dict[str, object]]:
    """预编码查询，避免在评测循环里重复编码。"""

    texts = [query_text for query_text, _ in queries]

    if isinstance(encoder, HFSentenceEncoder):
        placeholder_vecs = encoder.encode_query_placeholder_batch(texts)
        coarse_vecs = encoder.encode_sentence_batch(texts)
        for (query_text, expected_ids), placeholder_vec, coarse_vec in zip(
            queries, placeholder_vecs, coarse_vecs
        ):
            tokens = encoder.tokenizer.tokenize(query_text)
            token_vecs = [placeholder_vec for _ in tokens]
            q_vecs, aux = vectorizer.make_group(token_vecs, tokens, query_text)
            q_vecs = [normalize(v) for v in q_vecs]
            yield {
                "query_id": str(uuid.uuid4()),
                "query_text": query_text,
                "expected_mem_ids": expected_ids,
                "encoder_id": encoder.encoder_id,
                "strategy": vectorizer.strategy,
                "q_vecs": q_vecs,
                "coarse_vec": normalize(coarse_vec),
                "aux": {**aux, "tokens": tokens},
            }
        return

    for query_text, expected_ids in queries:
        token_vecs, tokens = encoder.encode_tokens(query_text)
        q_vecs, aux = vectorizer.make_group(token_vecs, tokens, query_text)
        q_vecs = [normalize(v) for v in q_vecs]
        yield {
            "query_id": str(uuid.uuid4()),
            "query_text": query_text,
            "expected_mem_ids": expected_ids,
            "encoder_id": encoder.encoder_id,
            "strategy": vectorizer.strategy,
            "q_vecs": q_vecs,
            "coarse_vec": normalize(encoder.encode_sentence(query_text)),
            "aux": {**aux, "tokens": tokens},
        }


def write_cached_queries(
    path: Path,
    queries: List[Tuple[str, List[str]]],
    encoder: Encoder,
    vectorizer: Vectorizer,
) -> None:
    """将预编码的 query 写入磁盘缓存文件。"""

    with path.open("w", encoding="utf-8") as handle:
        for payload in build_cached_queries(queries, encoder, vectorizer):
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def iter_cached_queries(path: Path) -> Iterator[Tuple[Query, List[str]]]:
    """从磁盘缓存中读取 query，避免占用大量内存。"""

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
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


def evaluate_policy(
    policy: str,
    cached_queries: Iterator[Tuple[Query, List[str]]],
    store,
    index,
    lexical_index,
    top_n: int,
    top_k: int,
    fixed_channel_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """评估单个路由策略，返回平均指标。"""

    router = Router(policy=policy, top_k=top_k, fixed_channel_weights=fixed_channel_weights)
    total_recall = 0.0
    total_mrr = 0.0
    total_top1 = 0.0
    count = 0
    metric_sums = {
        "entropy": 0.0,
        "mass_at_k": 0.0,
        "consistency": 0.0,
        "counterfactual_drop_top1": 0.0,
        "counterfactual_drop_topk": 0.0,
    }

    retriever = Retriever(
        store,
        index,
        FieldScorer(),
        router=router,
        lexical_index=lexical_index,
    )
    for query, expected_mem_ids in cached_queries:
        count += 1
        # 第一次跑：用于触发路由缓存，准备一致性对比。
        retriever.retrieve(query, top_n=top_n, top_k=top_k)

        # 第二次跑：读取 route_output，得到“同一 query 重复跑”的一致性指标。
        results = retriever.retrieve(query, top_n=top_n, top_k=top_k)
        hit_ids = [result.mem_id for result in results]
        total_recall += recall_at_k(hit_ids, expected_mem_ids)
        total_mrr += mrr(hit_ids, expected_mem_ids)
        total_top1 += 1.0 if hit_ids and hit_ids[0] in set(expected_mem_ids) else 0.0

        route_output = results[0].route_output if results else None
        if route_output:
            for key in metric_sums:
                metric_sums[key] += route_output.metrics.get(key, 0.0)

    count = max(1, count)
    averages = {
        "recall_at_k": total_recall / count,
        "mrr": total_mrr / count,
        "top1_acc": total_top1 / count,
    }
    averages.update({key: value / count for key, value in metric_sums.items()})
    return averages


def main() -> None:
    """入口：运行软/半硬/硬路由对比评测。"""

    if not MEMORY_PATH.exists() or not EVAL_PATH.exists():
        raise FileNotFoundError("缺少 data/memory.jsonl 或 data/eval.jsonl")

    set_trace(False)
    items = load_memory_items(MEMORY_PATH)
    queries = load_eval_queries(EVAL_PATH)

    encoder = HFSentenceEncoder(model_name="intfloat/multilingual-e5-small", tokenizer="jieba")
    vectorizer = Vectorizer(strategy="token_pool_topk", k=8)
    store, index, lexical_index = build_memory_index(items, encoder, vectorizer, return_lexical=True)
    if EVAL_CACHE_PATH.exists():
        print(f"检测到缓存文件，直接读取: {EVAL_CACHE_PATH}")
    else:
        write_cached_queries(EVAL_CACHE_PATH, queries, encoder, vectorizer)
        print(f"已生成缓存文件: {EVAL_CACHE_PATH}")

    top_n = 10
    top_k = 5

    print("\n=== 小评测集 ===")
    print(f"样本数: {len(queries)} | top_n={top_n} | top_k={top_k}")
    print("阶段 1/3: 构建索引与编码器完成，开始跑 ablation。")

    # ablation 组：固定通道权重来观察证据价值与变硬风险。
    ablation_groups = [
        ("baseline(auto)", None),
        ("S-only", {"semantic": 1.0, "lexical": 0.0, "meta": 0.0, "coarse": 0.0}),
        ("S+L", {"semantic": 0.7, "lexical": 0.3, "meta": 0.0, "coarse": 0.0}),
        ("S+M", {"semantic": 0.7, "lexical": 0.0, "meta": 0.3, "coarse": 0.0}),
        ("S+L+M", {"semantic": 0.6, "lexical": 0.25, "meta": 0.15, "coarse": 0.0}),
        ("L-only", {"semantic": 0.0, "lexical": 1.0, "meta": 0.0, "coarse": 0.0}),
    ]

    for group_name, fixed_weights in ablation_groups:
        print("\n" + "=" * 72)
        print(f"阶段 2/3: ablation={group_name} | fixed_weights={fixed_weights}")
        print("=" * 72)
        for policy in ("soft", "half_hard", "hard"):
            print(f"  -> 开始评估 policy={policy}")
            metrics = evaluate_policy(
                policy,
                iter_cached_queries(EVAL_CACHE_PATH),
                store,
                index,
                lexical_index,
                top_n,
                top_k,
                fixed_channel_weights=fixed_weights,
            )
            print(f"  [policy={policy}]")
            print(
                "    Recall@k={recall_at_k:.3f} | MRR={mrr:.3f} | Top1={top1_acc:.3f}\n"
                "    entropy={entropy:.3f} | mass@k={mass_at_k:.3f} | "
                "consistency={consistency:.3f} | "
                "cf_drop_top1={counterfactual_drop_top1:.3f} | "
                "cf_drop_topk={counterfactual_drop_topk:.3f}".format(**metrics)
            )
        print("  完成该组 ablation。")

    print("\n阶段 3/3: 全部 ablation 结束，可对比证据价值与变硬风险。\n")


if __name__ == "__main__":
    main()
