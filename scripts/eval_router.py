"""小评测集：验证路由策略 + Recall@k/MRR + 路由指标。

说明：
- 这个脚本只依赖本仓库最小可用编码器，用于稳定、可复现的对比。
- 每条 query 会“重复跑两次”，第二次的路由输出用于一致性指标。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from src.memory_indexer import (
    MemoryItem,
    Router,
    SimpleHashEncoder,
    Vectorizer,
    build_memory_items,
    build_memory_index,
    retrieve_top_k,
    set_trace,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MEMORY_PATH = DATA_DIR / "memory.jsonl"
EVAL_PATH = DATA_DIR / "eval.jsonl"


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


def evaluate_policy(
    policy: str,
    queries: List[Tuple[str, List[str]]],
    encoder: SimpleHashEncoder,
    vectorizer: Vectorizer,
    store,
    index,
    top_n: int,
    top_k: int,
) -> Dict[str, float]:
    """评估单个路由策略，返回平均指标。"""

    router = Router(policy=policy, top_k=top_k)
    total_recall = 0.0
    total_mrr = 0.0
    metric_sums = {
        "entropy": 0.0,
        "mass_at_k": 0.0,
        "consistency": 0.0,
        "counterfactual_drop_top1": 0.0,
        "counterfactual_drop_topk": 0.0,
    }

    for query_text, expected_mem_ids in queries:
        # 第一次跑：用于触发路由缓存，准备一致性对比。
        retrieve_top_k(
            query_text,
            encoder,
            vectorizer,
            store,
            index,
            top_n=top_n,
            top_k=top_k,
            router=router,
        )

        # 第二次跑：读取 route_output，得到“同一 query 重复跑”的一致性指标。
        results = retrieve_top_k(
            query_text,
            encoder,
            vectorizer,
            store,
            index,
            top_n=top_n,
            top_k=top_k,
            router=router,
        )
        hit_ids = [result.mem_id for result in results]
        total_recall += recall_at_k(hit_ids, expected_mem_ids)
        total_mrr += mrr(hit_ids, expected_mem_ids)

        route_output = results[0].route_output if results else None
        if route_output:
            for key in metric_sums:
                metric_sums[key] += route_output.metrics.get(key, 0.0)

    count = len(queries) or 1
    averages = {
        "recall_at_k": total_recall / count,
        "mrr": total_mrr / count,
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

    encoder = SimpleHashEncoder(dims=16)
    vectorizer = Vectorizer(strategy="token_pool_topk", k=8)
    store, index = build_memory_index(items, encoder, vectorizer)

    top_n = 10
    top_k = 5

    print("\n=== 小评测集 ===")
    print(f"样本数: {len(queries)} | top_n={top_n} | top_k={top_k}\n")

    for policy in ("soft", "half_hard", "hard"):
        metrics = evaluate_policy(
            policy,
            queries,
            encoder,
            vectorizer,
            store,
            index,
            top_n,
            top_k,
        )
        print(f"[policy={policy}]")
        print(
            "  Recall@k={recall_at_k:.3f} | MRR={mrr:.3f} | "
            "entropy={entropy:.3f} | mass@k={mass_at_k:.3f} | "
            "consistency={consistency:.3f} | "
            "cf_drop_top1={counterfactual_drop_top1:.3f} | "
            "cf_drop_topk={counterfactual_drop_topk:.3f}".format(**metrics)
        )
        print("-")


if __name__ == "__main__":
    main()
