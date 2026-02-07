"""小评测集：验证路由策略 + Recall@k/MRR + 路由指标。

说明：
- 这个脚本只依赖本仓库最小可用编码器，用于稳定、可复现的对比。
- 默认每条 query 会“重复跑两次”，第二次的路由输出用于一致性指标。
"""

from __future__ import annotations

import argparse
import json
import os
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
from src.memory_indexer.encoder.e5_token import E5TokenEncoder
from src.memory_indexer.utils import normalize

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def resolve_dataset_paths(dataset: str) -> Tuple[Path, Path, Path, Path]:
    """根据数据集名称解析路径，默认采用 normal。"""

    memory_path = DATA_DIR / f"memory_{dataset}.jsonl"
    eval_path = DATA_DIR / f"eval_{dataset}.jsonl"
    cache_path = DATA_DIR / f"eval_cache_{dataset}.jsonl"
    memory_cache_path = DATA_DIR / f"memory_cache_{dataset}.jsonl"
    return memory_path, eval_path, cache_path, memory_cache_path


def load_memory_items(path: Path) -> List[MemoryItem]:
    """读取 memory_xxx.jsonl，转换为 MemoryItem 列表。"""

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
    """读取 eval_xxx.jsonl，返回 (query_text, expected_mem_ids)。"""

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
) -> None:
    """将预编码的 query 写入磁盘缓存文件。"""

    with path.open("w", encoding="utf-8") as handle:
        for payload in build_cached_queries(
            queries, sentence_encoder, token_encoder, vectorizer
        ):
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
    candidate_mode: str = "coarse",
    consistency_pass: bool = True,
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
        "candidate_size": 0.0,
        "coarse_lexical_jaccard": 0.0,
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
        if consistency_pass:
            retriever.retrieve(query, top_n=top_n, top_k=top_k, candidate_mode=candidate_mode)

        # 第二次跑：读取 route_output，得到“同一 query 重复跑”的一致性指标。
        results = retriever.retrieve(query, top_n=top_n, top_k=top_k, candidate_mode=candidate_mode)
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

    parser = argparse.ArgumentParser(description="运行路由评测集")
    parser.add_argument(
        "--dataset",
        choices=("normal", "easy"),
        default="normal",
        help="选择评测集 (default: normal)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="粗召回数量 top_n (default: 50)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="最终输出 top_k (default: 5)",
    )
    parser.add_argument(
        "--candidate-mode",
        choices=("coarse", "lexical", "union"),
        default="union",
        help="候选池策略：coarse / lexical / union (default: union)",
    )
    # 下面这些参数用于开关耗时点，便于组合使用。
    parser.add_argument(
        "--no-consistency-pass",
        action="store_true",
        help="关闭每条 query 的第二次检索（不计算一致性对比）",
    )
    parser.add_argument(
        "--policies",
        default="soft,half_hard,hard",
        help="只评估指定 policy，逗号分隔 (soft/half_hard/hard)",
    )
    parser.add_argument(
        "--ablation",
        choices=("baseline",),
        help="快捷：只跑 baseline(auto) 组",
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
    args = parser.parse_args()

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

    memory_path, eval_path, cache_path, memory_cache_path = resolve_dataset_paths(args.dataset)
    if not memory_path.exists() or not eval_path.exists():
        raise FileNotFoundError(f"缺少数据集文件: {memory_path.name} / {eval_path.name}")

    if args.rebuild_cache:
        if cache_path.exists():
            cache_path.unlink()
            print(f"已删除 query 缓存: {cache_path}")
        if memory_cache_path.exists():
            memory_cache_path.unlink()
            print(f"已删除 memory 缓存: {memory_cache_path}")

    items = load_memory_items(memory_path)
    queries = load_eval_queries(eval_path)

    sentence_encoder = HFSentenceEncoder(
        model_name="intfloat/multilingual-e5-small", tokenizer="jieba"
    )
    token_encoder = E5TokenEncoder(model_name="intfloat/multilingual-e5-small")
    vectorizer = Vectorizer(strategy="token_pool_topk", k=8)
    if args.no_lexical:
        store, index = build_memory_index(
            items,
            sentence_encoder,
            vectorizer,
            token_encoder=token_encoder,
            cache_path=memory_cache_path,
            return_lexical=False,
        )
        lexical_index = None
    else:
        store, index, lexical_index = build_memory_index(
            items,
            sentence_encoder,
            vectorizer,
            token_encoder=token_encoder,
            cache_path=memory_cache_path,
            return_lexical=True,
        )
    if cache_path.exists():
        print(f"检测到缓存文件，直接读取: {cache_path}")
    else:
        write_cached_queries(
            cache_path, queries, sentence_encoder, token_encoder, vectorizer
        )
        print(f"已生成缓存文件: {cache_path}")

    top_n = args.top_n
    top_k = args.top_k

    print("\n=== 小评测集 ===")
    print(f"样本数: {len(queries)} | top_n={top_n} | top_k={top_k} | dataset={args.dataset}")
    print(f"candidate_mode={args.candidate_mode}")
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
    ablation_lookup = {name: weights for name, weights in ablation_groups}
    if args.ablation and args.ablation_groups:
        raise ValueError("--ablation 与 --ablation-groups 不能同时使用")
    if args.ablation == "baseline":
        ablation_groups = [("baseline(auto)", ablation_lookup["baseline(auto)"])]
    if args.ablation_groups:
        requested = [name.strip() for name in args.ablation_groups.split(",") if name.strip()]
        missing = [name for name in requested if name not in ablation_lookup]
        if missing:
            raise ValueError(f"未知 ablation 组: {', '.join(missing)}")
        ablation_groups = [(name, ablation_lookup[name]) for name in requested]

    policies = [name.strip() for name in args.policies.split(",") if name.strip()]
    allowed_policies = {"soft", "half_hard", "hard"}
    invalid_policies = [name for name in policies if name not in allowed_policies]
    if invalid_policies:
        raise ValueError(f"未知 policy: {', '.join(invalid_policies)}")

    for group_name, fixed_weights in ablation_groups:
        print("\n" + "=" * 72)
        print(f"阶段 2/3: ablation={group_name} | fixed_weights={fixed_weights}")
        print("=" * 72)
        for policy in policies:
            print(f"  -> 开始评估 policy={policy}")
            metrics = evaluate_policy(
                policy,
                iter_cached_queries(cache_path),
                store,
                index,
                lexical_index,
                top_n,
                top_k,
                candidate_mode=args.candidate_mode,
                consistency_pass=not args.no_consistency_pass,
                fixed_channel_weights=fixed_weights,
            )
            print(f"  [policy={policy}]")
            print(
                "    Recall@k={recall_at_k:.3f} | MRR={mrr:.3f} | Top1={top1_acc:.3f}\n"
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
        print("  完成该组 ablation。")

    print("\n阶段 3/3: 全部 ablation 结束，可对比证据价值与变硬风险。\n")


if __name__ == "__main__":
    main()
