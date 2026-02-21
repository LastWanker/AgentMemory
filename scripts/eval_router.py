"""小评测集：验证路由策略 + Recall@k/MRR + 路由指标。

说明：
- 这个脚本只依赖本仓库最小可用编码器，用于稳定、可复现的对比。
- 默认每条 query 会“重复跑两次”，第二次的路由输出用于一致性指标。
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple
import uuid

# 兼容直接运行 `python scripts/eval_router.py` 的模块搜索路径。
REPO_ROOT = Path(__file__).resolve().parent.parent
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
from scripts.runtime_utils import (
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


def resolve_cache_paths(signature: str) -> Tuple[Path, Path]:
    VECTOR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return (
        VECTOR_CACHE_DIR / f"eval_cache_{signature}.jsonl",
        VECTOR_CACHE_DIR / f"memory_cache_{signature}.jsonl",
    )


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
    """读取 eval_xxx.jsonl，返回 (query_text, expected_mem_ids/positives)。"""

    queries: List[Tuple[str, List[str]]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload.get("positives"), list):
            expected_ids = [str(mid) for mid in payload["positives"]]
        else:
            expected_ids = [str(mid) for mid in payload.get("expected_mem_ids", [])]
        queries.append((payload["query_text"], expected_ids))
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
    cache_signature: str,
) -> None:
    """将预编码的 query 写入磁盘缓存文件。"""

    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps({"_meta": {"cache_signature": cache_signature}}, ensure_ascii=False)
            + "\n"
        )
        for payload in build_cached_queries(
            queries, sentence_encoder, token_encoder, vectorizer
        ):
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def iter_cached_queries(path: Path, *, cache_signature: str) -> Iterator[Tuple[Query, List[str]]]:
    """从磁盘缓存中读取 query，避免占用大量内存。"""

    signature_checked = False
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict) and "_meta" in payload:
                meta = payload.get("_meta") or {}
                if meta.get("cache_signature") != cache_signature:
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


def query_cache_compatible(path: Path, cache_signature: str) -> bool:
    if not path.exists():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict) and "_meta" in payload:
            meta = payload.get("_meta") or {}
            return meta.get("cache_signature") == cache_signature
        return False
    return False


def select_registry_metrics(
    records: List[Dict[str, object]],
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    target: Optional[Dict[str, object]] = None
    for record in records:
        if record.get("ablation_group") == "baseline(auto)" and record.get("policy") == "soft":
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
                keys = ("loss_type", "neg_strategy", "epochs", "seed", "dataset")
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
    candidate_mode: str = "union",
    consistency_pass: bool = True,
    fixed_channel_weights: Optional[Dict[str, float]] = None,
    use_learned_scorer: bool = False,
    reranker_path: Optional[str] = None,
    ablation_name: str = "baseline",
    debug_score_flat: bool = False,
    debug_max_queries: int = 5,
    debug_topm: int = 10,
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

    scorer = FieldScorer()
    if use_learned_scorer:
        if not reranker_path:
            raise ValueError("启用 learned scorer 时必须提供 --reranker-path")
        scorer = LearnedFieldScorer(reranker_path=reranker_path)

    retriever = Retriever(
        store,
        index,
        scorer,
        router=router,
        lexical_index=lexical_index,
    )

    def summarize_scores(values: List[float]) -> Tuple[float, float]:
        if not values:
            return 0.0, 0.0
        mean_value = sum(values) / len(values)
        std_value = math.sqrt(sum((value - mean_value) ** 2 for value in values) / len(values))
        return max(values) - min(values), std_value

    debug_printed = 0
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

            if (
                debug_score_flat
                and ablation_name == "S-only"
                and debug_printed < max(0, debug_max_queries)
            ):
                topm = max(1, debug_topm)
                ranked_results = sorted(
                    results,
                    key=lambda item: route_output.scores.get(item.mem_id, item.score),
                    reverse=True,
                )
                top_results = ranked_results[:topm]
                top_semantic_scores = [float(item.features.get("semantic_score", 0.0)) for item in top_results]
                top_combined_scores = [float(route_output.scores.get(item.mem_id, item.score)) for item in top_results]
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
                debug_printed += 1

    count = max(1, count)
    averages = {
        "recall_at_k": total_recall / count,
        "mrr": total_mrr / count,
        "top1_acc": total_top1 / count,
    }
    averages.update({key: value / count for key, value in metric_sums.items()})
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
    ablation_name: str,
) -> Dict[str, Tuple[float, float, float, float]]:
    if n_samples <= 0 or not query_rows:
        return {}
    recall_draws: List[float] = []
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
            ablation_name=ablation_name,
        )
        recall_draws.append(metrics["recall_at_k"])
        mrr_draws.append(metrics["mrr"])
        top1_draws.append(metrics["top1_acc"])
    return {
        "recall_at_k": bootstrap_ci(recall_draws),
        "mrr": bootstrap_ci(mrr_draws),
        "top1_acc": bootstrap_ci(top1_draws),
    }

def main() -> None:
    """入口：运行软/半硬/硬路由对比评测。"""
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    pre_args, _ = pre_parser.parse_known_args()
    loaded_config = load_config(pre_args.config)

    default_dataset = cfg_get(loaded_config, "common.dataset", "normal")
    default_top_n = int(cfg_get(loaded_config, "common.top_n", 20))
    default_top_k = int(cfg_get(loaded_config, "common.top_k", 5))
    default_candidate_mode = cfg_get(loaded_config, "common.candidate_mode", "union")
    default_encoder_backend = cfg_get(loaded_config, "common.encoder_backend", "hf")
    default_policies = cfg_get(loaded_config, "eval_router.policies", "soft,half_hard,hard")
    default_bootstrap = int(cfg_get(loaded_config, "eval_router.bootstrap", 0))

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
        help="粗召回数量 top_n (default: 20)",
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
        default=default_policies,
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
    parser.add_argument(
        "--encoder-backend",
        choices=("hf", "simple"),
        default=default_encoder_backend,
        help="编码器后端：hf(默认) 或 simple（用于快速回归）",
    )
    parser.add_argument(
        "--use-learned-scorer",
        action="store_true",
        help="启用 TinyReranker learned semantic scorer",
    )
    parser.add_argument(
        "--reranker-path",
        default=str(MODEL_WEIGHTS_DIR / "tiny_reranker.pt"),
        help="learned scorer 权重路径 (.pt)",
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
    args = parser.parse_args()

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

    if args.no_lexical and args.candidate_mode != "coarse":
        print("[warn] --no-lexical 已开启，candidate_mode 自动切换为 coarse")
        args.candidate_mode = "coarse"

    items = load_memory_items(memory_path)
    queries = load_eval_queries(eval_path)

    data_fingerprint = build_data_fingerprint(memory_path, eval_path)
    if args.encoder_backend == "simple":
        sentence_encoder = SimpleHashEncoder(dims=64)
        token_encoder = sentence_encoder
    else:
        from src.memory_indexer.encoder.hf_sentence import HFSentenceEncoder
        from src.memory_indexer.encoder.e5_token import E5TokenEncoder

        sentence_encoder = HFSentenceEncoder(
            model_name="intfloat/multilingual-e5-small", tokenizer="jieba"
        )
        token_encoder = E5TokenEncoder(model_name="intfloat/multilingual-e5-small")
    vectorizer = Vectorizer(strategy="token_pool_topk", k=8)
    encoder_id = _compose_encoder_id(sentence_encoder, token_encoder)
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
    cache_path, memory_cache_path = resolve_cache_paths(cache_signature)

    if args.rebuild_cache:
        if cache_path.exists():
            cache_path.unlink()
            print(f"已删除 query 缓存: {cache_path}")
        if memory_cache_path.exists():
            memory_cache_path.unlink()
            print(f"已删除 memory 缓存: {memory_cache_path}")
    memory_cache_exists = memory_cache_path.exists()
    if memory_cache_exists:
        print(f"检测到缓存文件，直接读取: {memory_cache_path}")
    else:
        print(f"未检测到缓存文件，将构建并写入: {memory_cache_path}")

    if args.no_lexical:
        store, index = build_memory_index(
            items,
            sentence_encoder,
            vectorizer,
            token_encoder=token_encoder,
            cache_path=memory_cache_path,
            cache_signature=cache_signature,
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
            cache_signature=cache_signature,
            return_lexical=True,
        )

    if not memory_cache_exists and memory_cache_path.exists():
        print(f"已生成缓存文件: {memory_cache_path}")

    if query_cache_compatible(cache_path, cache_signature):
        print(f"检测到签名匹配 query 缓存，直接读取: {cache_path}")
    else:
        if cache_path.exists():
            print(f"query 缓存签名不匹配，重建: {cache_path}")
        write_cached_queries(
            cache_path,
            queries,
            sentence_encoder,
            token_encoder,
            vectorizer,
            cache_signature=cache_signature,
        )
        print(f"已生成缓存文件: {cache_path}")

    top_n = args.top_n
    top_k = args.top_k
    cached_query_rows = list(iter_cached_queries(cache_path, cache_signature=cache_signature))

    print("\n=== 小评测集 ===")
    print(f"样本数: {len(queries)} | top_n={top_n} | top_k={top_k} | dataset={args.dataset}")
    print(f"candidate_mode={args.candidate_mode}")
    print("阶段 1/3: 构建索引与编码器完成，开始跑 ablation。")
    metrics_records: List[Dict[str, object]] = []

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
                iter(cached_query_rows),
                store,
                index,
                lexical_index,
                top_n,
                top_k,
                candidate_mode=args.candidate_mode,
                consistency_pass=not args.no_consistency_pass,
                fixed_channel_weights=fixed_weights,
                use_learned_scorer=args.use_learned_scorer,
                reranker_path=args.reranker_path,
                ablation_name=group_name,
                debug_score_flat=args.debug_score_flat,
                debug_max_queries=args.debug_max_queries,
                debug_topm=args.debug_topm,
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
            record: Dict[str, object] = {
                "ablation_group": group_name,
                "policy": policy,
                "metrics": metrics,
            }
            if args.bootstrap > 0:
                bs = run_bootstrap(
                    query_rows=cached_query_rows,
                    n_samples=args.bootstrap,
                    policy=policy,
                    store=store,
                    index=index,
                    lexical_index=lexical_index,
                    top_n=top_n,
                    top_k=top_k,
                    candidate_mode=args.candidate_mode,
                    consistency_pass=not args.no_consistency_pass,
                    fixed_channel_weights=fixed_weights,
                    use_learned_scorer=args.use_learned_scorer,
                    reranker_path=args.reranker_path,
                    ablation_name=group_name,
                )
                r = bs.get("recall_at_k")
                t = bs.get("top1_acc")
                if r and t:
                    print(
                        f"    Recall@k CI: {r[0]:.3f} ± {r[1]:.3f} (95% CI [{r[2]:.3f}, {r[3]:.3f}]) | "
                        f"Top1 CI: {t[0]:.3f} ± {t[1]:.3f} (95% CI [{t[2]:.3f}, {t[3]:.3f}])"
                    )
                record["bootstrap"] = bs
            metrics_records.append(record)
        print("  完成该组 ablation。")

    print("\n阶段 3/3: 全部 ablation 结束，可对比证据价值与变硬风险。\n")
    if run_root:
        eval_metrics_payload = {
            "dataset": args.dataset,
            "cache_signature": cache_signature,
            "cache_path": str(cache_path),
            "memory_cache_path": str(memory_cache_path),
            "data_fingerprint": data_fingerprint,
            "records": metrics_records,
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
