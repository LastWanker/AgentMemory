"""训练 TinyReranker：支持 multi-positive、ranked negatives、listwise 与 cardinality head。"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
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
from src.memory_indexer.scorer import (
    BipartiteAlignTransformer,
    TinyReranker,
    compute_sim_matrix,
    FieldScorer,
)
from src.memory_indexer.learned_scorer import CardinalityHead
from src.memory_indexer.learned_scorer_bipartite import vector_group_to_fixed
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


class _CacheOnlyEncoder:
    def __init__(self, encoder_id: str) -> None:
        self.encoder_id = encoder_id
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
    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:16]
    return VECTOR_CACHE_DIR / f"memory_cache_{digest}.jsonl"


def resolve_query_cache_path(signature: str) -> Path:
    VECTOR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:16]
    return VECTOR_CACHE_DIR / f"train_query_cache_{digest}.jsonl"


def _compose_encoder_id(sentence_encoder, token_encoder) -> str:
    if sentence_encoder.encoder_id == token_encoder.encoder_id:
        return sentence_encoder.encoder_id
    return f"{sentence_encoder.encoder_id}|{token_encoder.encoder_id}"


def load_memory_items(path: Path):
    payloads: List[Dict[str, object]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
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
    legacy_field_rows = 0
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            payload = json.loads(line)
            if "candidates" in payload or "hard_negatives" in payload:
                legacy_field_rows += 1
            rows.append(
                {
                    "query_id": payload.get("query_id") or str(uuid.uuid4()),
                    "query_text": payload["query_text"],
                    "positives": _normalize_positives(payload),
                }
            )
    if legacy_field_rows > 0:
        print(
            "[train] warn: eval 查询文件含旧字段 candidates/hard_negatives，"
            f"训练已忽略（rows={legacy_field_rows})"
        )
    return rows


def build_cached_train_queries(
    rows: List[Dict[str, object]],
    sentence_encoder,
    token_encoder,
    vectorizer: Vectorizer,
) -> List[Dict[str, object]]:
    encoder_id = _compose_encoder_id(sentence_encoder, token_encoder)
    payloads: List[Dict[str, object]] = []
    for row in rows:
        query_text = str(row["query_text"])
        positives = [str(mid) for mid in row.get("positives", [])]
        token_vecs, model_tokens = token_encoder.encode_tokens(query_text)
        q_vecs, aux = vectorizer.make_group(token_vecs, model_tokens, query_text)
        q_vecs = [normalize(v) for v in q_vecs]
        lex_tokens = sentence_encoder.tokenizer.tokenize(query_text)
        payloads.append(
            {
                "query_id": str(row.get("query_id") or uuid.uuid4()),
                "query_text": query_text,
                "expected_mem_ids": positives,
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
        )
    return payloads


def write_cached_train_queries(
    path: Path,
    rows: List[Dict[str, object]],
    sentence_encoder,
    token_encoder,
    vectorizer: Vectorizer,
    *,
    cache_signature: Optional[str],
) -> None:
    payloads = build_cached_train_queries(rows, sentence_encoder, token_encoder, vectorizer)
    with path.open("w", encoding="utf-8") as handle:
        if cache_signature:
            handle.write(
                json.dumps({"_meta": {"cache_signature": cache_signature}}, ensure_ascii=False)
                + "\n"
            )
        for payload in payloads:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def iter_cached_train_queries(
    path: Path,
    *,
    cache_signature: Optional[str],
) -> List[Tuple[Query, List[str]]]:
    rows: List[Tuple[Query, List[str]]] = []
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
                        f"train query cache 签名不匹配: expected={cache_signature} "
                        f"got={meta.get('cache_signature')}"
                    )
                signature_checked = True
                continue
            if not signature_checked:
                raise ValueError("train query cache 缺少签名头，需重建缓存。")
            rows.append(
                (
                    Query(
                        query_id=str(payload.get("query_id", uuid.uuid4())),
                        text=str(payload["query_text"]),
                        encoder_id=str(payload["encoder_id"]),
                        strategy=str(payload["strategy"]),
                        q_vecs=payload["q_vecs"],
                        coarse_vec=payload["coarse_vec"],
                        aux=payload.get("aux", {}),
                    ),
                    [str(mid) for mid in payload.get("expected_mem_ids", [])],
                )
            )
    return rows


def query_cache_compatible(path: Path, cache_signature: Optional[str]) -> bool:
    if not path.exists():
        return False
    if cache_signature is None:
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
    items,
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


def sample_train_queries(
    rows: List[Dict[str, object]],
    *,
    max_queries: int,
) -> List[Dict[str, object]]:
    if max_queries <= 0 or len(rows) <= max_queries:
        return rows
    return rows[:max_queries]


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
    encoder_id = _compose_encoder_id(sentence_encoder, token_encoder)
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


def resolve_listwise_target_size(list_size: int, positives_count: int, neg_per_pos: int) -> int:
    if list_size > 0:
        return max(1, list_size)
    # 动态长度：按当前 query 的正例规模扩展负例容量。
    return max(positives_count + 1, positives_count * (1 + max(1, neg_per_pos)))


def build_bipartite_fixed(
    vecs: List[List[float]],
    *,
    seq_len: int,
    input_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return vector_group_to_fixed(
        vecs,
        size=seq_len,
        target_dim=input_dim,
    )


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    pre_args, _ = pre_parser.parse_known_args()
    loaded_config = load_config(pre_args.config)

    default_dataset = cfg_get(loaded_config, "common.dataset", "normal")
    default_candidate_mode = cfg_get(loaded_config, "common.candidate_mode", "coarse")
    default_top_n = int(cfg_get(loaded_config, "common.top_n", 20))
    default_encoder_backend = cfg_get(loaded_config, "common.encoder_backend", "hf")
    default_epochs = int(cfg_get(loaded_config, "train_reranker.epochs", 3))
    default_batch_size = int(cfg_get(loaded_config, "train_reranker.batch_size", 16))
    default_lr = float(cfg_get(loaded_config, "train_reranker.lr", 1e-3))
    default_device = cfg_get(loaded_config, "train_reranker.device", "cuda")
    default_loss_type = cfg_get(loaded_config, "train_reranker.loss_type", "listwise")
    default_neg_strategy = cfg_get(loaded_config, "train_reranker.neg_strategy", "ranked")
    default_model_family = str(cfg_get(loaded_config, "train_reranker.model_family", "bipartite"))
    default_bipartite_tau = float(cfg_get(loaded_config, "train_reranker.bipartite_tau", 0.1))
    default_bipartite_learnable_tau = bool(
        cfg_get(loaded_config, "train_reranker.bipartite_learnable_tau", False)
    )

    parser = argparse.ArgumentParser(description="训练 reranker（tiny / bipartite）")
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
    parser.add_argument("--cache-alias", default="users", help="固定缓存别名（例如 users/users_simple）")
    parser.add_argument("--no-cache-signature", action="store_true", help="关闭缓存签名校验，按缓存文件名复用")
    parser.add_argument("--neg-per-pos", type=int, default=3)
    parser.add_argument("--max-queries", type=int, default=0, help="限制训练 query 数量，0 表示全量")
    parser.add_argument("--neg-strategy", choices=("random", "ranked"), default=default_neg_strategy)
    parser.add_argument("--loss-type", choices=("pairwise", "listwise"), default=default_loss_type)
    parser.add_argument(
        "--model-family",
        choices=("tiny", "bipartite"),
        default=default_model_family,
        help="reranker 模型族：tiny 或 bipartite",
    )
    parser.add_argument("--list-size", type=int, default=0, help="listwise 列表长度；0 表示按 positives 动态扩展")
    parser.add_argument("--use-cardinality-head", action="store_true")
    parser.add_argument("--cardinality-loss-weight", type=float, default=0.1)
    parser.add_argument("--cardinality-k-max", type=int, default=20)
    listwise_hard_group = parser.add_mutually_exclusive_group()
    listwise_hard_group.add_argument(
        "--listwise-hard-constraint",
        dest="listwise_hard_constraint",
        action="store_true",
        help="启用 top-|P| 约束近似：惩罚负例分数超过正例分数",
    )
    listwise_hard_group.add_argument(
        "--no-listwise-hard-constraint",
        dest="listwise_hard_constraint",
        action="store_false",
        help="关闭 listwise top-|P| 约束近似",
    )
    parser.set_defaults(listwise_hard_constraint=True)
    parser.add_argument("--listwise-hard-weight", type=float, default=1.0)
    parser.add_argument("--bipartite-input-dim", type=int, default=384)
    parser.add_argument("--bipartite-seq-len", type=int, default=8)
    parser.add_argument("--bipartite-proj-dim", type=int, default=192)
    parser.add_argument("--bipartite-d-model", type=int, default=256)
    parser.add_argument("--bipartite-num-heads", type=int, default=4)
    parser.add_argument("--bipartite-num-layers", type=int, default=3)
    parser.add_argument("--bipartite-dropout", type=float, default=0.1)
    parser.add_argument("--bipartite-tau", type=float, default=default_bipartite_tau)
    tau_group = parser.add_mutually_exclusive_group()
    tau_group.add_argument(
        "--bipartite-learnable-tau",
        dest="bipartite_learnable_tau",
        action="store_true",
        help="将 tau 作为可学习参数",
    )
    tau_group.add_argument(
        "--no-bipartite-learnable-tau",
        dest="bipartite_learnable_tau",
        action="store_false",
        help="将 tau 固定为常数（推荐先用固定 tau）",
    )
    parser.set_defaults(bipartite_learnable_tau=default_bipartite_learnable_tau)
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="快速自检模式：自动切换 easy+simple+小样本参数",
    )
    parser.add_argument("--hf-local-only", action="store_true", help="仅使用本地 HF 模型缓存")
    parser.add_argument("--hf-offline", action="store_true", help="强制 HF 离线模式")
    parser.add_argument("--hf-online", action="store_true", help="允许 HF 在线模式（会关闭 offline/local-only）")

    args = parser.parse_args()
    args.model_family = str(args.model_family).strip().lower()
    if args.bipartite_tau <= 0:
        raise ValueError("--bipartite-tau 必须 > 0")
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

    if args.candidate_mode != "coarse":
        print("[train] warn: 当前训练候选池固定为粗召回 top_n，candidate_mode 自动切换为 coarse")
        args.candidate_mode = "coarse"

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    memory_path, eval_path = resolve_dataset_paths(args.dataset)
    items = load_memory_items(memory_path)
    all_eval_queries = load_eval_queries(eval_path)
    requested_eval_queries = sample_train_queries(all_eval_queries, max_queries=args.max_queries)
    if len(requested_eval_queries) != len(all_eval_queries):
        print(
            f"[train] requested queries sampled: {len(requested_eval_queries)}/{len(all_eval_queries)} "
            "(mode=head)"
        )

    data_fingerprint = build_data_fingerprint(memory_path, eval_path)
    vectorizer = Vectorizer(strategy="token_pool_topk", k=8)
    sentence_encoder_id, token_encoder_id, expected_encoder_id = expected_encoder_ids(
        args.encoder_backend
    )
    query_cache_signature: Optional[str] = build_cache_signature(
        {
            "dataset": args.dataset,
            "backend": args.encoder_backend,
            "encoder": expected_encoder_id,
            "strategy": vectorizer.strategy,
            "k": vectorizer.k,
            "datafp": data_fingerprint,
            "max_queries": args.max_queries,
            "cache_v": "train_query_v1",
        }
    )
    memory_cache_signature: Optional[str] = build_cache_signature(
        {
            "dataset": args.dataset,
            "backend": args.encoder_backend,
            "encoder": expected_encoder_id,
            "strategy": vectorizer.strategy,
            "k": vectorizer.k,
            "datafp": data_fingerprint,
            "cache_v": "v2",
        }
    )
    if args.no_cache_signature:
        query_cache_signature = None
        memory_cache_signature = None

    cache_alias = sanitize_slug(str(args.cache_alias)) if args.cache_alias else ""
    if args.encoder_backend != "hf" and cache_alias == "users":
        cache_alias = "users_simple"
        print("[train] info: simple backend detected, cache alias auto-switched to users_simple.")

    if cache_alias:
        query_cache_path = VECTOR_CACHE_DIR / f"eval_cache_{cache_alias}.jsonl"
        memory_cache_path = VECTOR_CACHE_DIR / f"memory_cache_{cache_alias}.jsonl"
    else:
        query_cache_path = resolve_query_cache_path(query_cache_signature or "train_query_nosig")
        memory_cache_path = resolve_memory_cache_path(memory_cache_signature or "memory_nosig")

    query_cache_ok = query_cache_compatible(query_cache_path, query_cache_signature)
    memory_cache_exists = memory_cache_path.exists()
    memory_cache_reusable = False
    if query_cache_ok and memory_cache_exists:
        memory_cache_reusable = memory_cache_fully_reusable(
            memory_cache_path,
            items,
            encoder_id=expected_encoder_id,
            strategy=vectorizer.strategy,
            cache_signature=memory_cache_signature,
        )
    can_skip_encoder_load = query_cache_ok and memory_cache_reusable

    if memory_cache_exists:
        print(f"[train] memory cache detected: {memory_cache_path}")
    else:
        print(f"[train] memory cache missing, will build: {memory_cache_path}")

    def build_runtime_encoders():
        if args.encoder_backend == "simple":
            sentence = SimpleHashEncoder(dims=SIMPLE_DIMS)
            return sentence, sentence
        configure_hf_runtime(local_only=args.hf_local_only, offline=args.hf_offline)
        try:
            from src.memory_indexer.encoder.hf_sentence import HFSentenceEncoder
            from src.memory_indexer.encoder.e5_token import E5TokenEncoder

            sentence = HFSentenceEncoder(model_name=HF_MODEL_NAME, tokenizer="jieba")
            token = E5TokenEncoder(model_name=HF_MODEL_NAME)
            return sentence, token
        except Exception as exc:
            raise RuntimeError(
                "HF/E5 编码器初始化失败。当前默认是 --hf-local-only + --hf-offline，"
                "请先手动下载模型缓存后重试；或显式使用 --hf-online。"
            ) from exc

    if can_skip_encoder_load:
        print("[train] query/memory 缓存完整命中，跳过 HF/E5 编码器初始化。")
        sentence_encoder = _CacheOnlyEncoder(sentence_encoder_id)
        token_encoder = _CacheOnlyEncoder(token_encoder_id)
    else:
        sentence_encoder, token_encoder = build_runtime_encoders()

    try:
        store, index, lexical_index = build_memory_index(
            items,
            sentence_encoder,
            vectorizer,
            token_encoder=token_encoder,
            cache_path=memory_cache_path,
            cache_signature=memory_cache_signature,
            return_lexical=True,
        )
    except RuntimeError as exc:
        if can_skip_encoder_load and "cache-only encoder called" in str(exc):
            print("[train] cache miss detected, fallback to real encoder initialization.")
            sentence_encoder, token_encoder = build_runtime_encoders()
            store, index, lexical_index = build_memory_index(
                items,
                sentence_encoder,
                vectorizer,
                token_encoder=token_encoder,
                cache_path=memory_cache_path,
                cache_signature=memory_cache_signature,
                return_lexical=True,
            )
        else:
            raise

    if query_cache_ok:
        if query_cache_signature is None:
            print(f"[train] query cache hit (no-signature): {query_cache_path}")
        else:
            print(f"[train] query cache signature hit: {query_cache_path}")
    else:
        query_cache_build_rows = (
            all_eval_queries if query_cache_signature is None else requested_eval_queries
        )
        print(f"[train] query cache rebuild => {query_cache_path} (rows={len(query_cache_build_rows)})")
        write_cached_train_queries(
            query_cache_path,
            query_cache_build_rows,
            sentence_encoder,
            token_encoder,
            vectorizer,
            cache_signature=query_cache_signature,
        )

    cached_train_queries = iter_cached_train_queries(
        query_cache_path,
        cache_signature=query_cache_signature,
    )
    train_query_rows = cached_train_queries
    if args.max_queries > 0:
        train_query_rows = train_query_rows[: args.max_queries]
    if len(train_query_rows) != len(cached_train_queries):
        print(
            f"[train] cached queries sampled: {len(train_query_rows)}/{len(cached_train_queries)} "
            "(mode=head)"
        )
    encoder_id = _compose_encoder_id(sentence_encoder, token_encoder)

    retriever = Retriever(
        store,
        index,
        FieldScorer(),
        router=Router(policy="soft", top_k=args.top_n),
        lexical_index=lexical_index,
    )

    if args.model_family == "tiny":
        model = TinyReranker().to(args.device)
        model_input_mode = "sim_matrix"
    else:
        model = BipartiteAlignTransformer(
            input_dim=args.bipartite_input_dim,
            seq_len=args.bipartite_seq_len,
            proj_dim=args.bipartite_proj_dim,
            d_model=args.bipartite_d_model,
            num_heads=args.bipartite_num_heads,
            num_layers=args.bipartite_num_layers,
            dropout=args.bipartite_dropout,
            tau=args.bipartite_tau,
            learnable_tau=args.bipartite_learnable_tau,
        ).to(args.device)
        model_input_mode = "vector_pair"
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    cardinality_head: Optional[CardinalityHead] = None
    cardinality_optimizer = None
    if args.use_cardinality_head:
        cardinality_head = CardinalityHead(k_max=args.cardinality_k_max).to(args.device)
        cardinality_optimizer = torch.optim.Adam(cardinality_head.parameters(), lr=args.lr)

    pair_buffer: List[Tuple[object, object]] = []
    listwise_buffer: List[Dict[str, object]] = []
    cardinality_samples: List[Tuple[List[float], int]] = []
    all_mem_ids = list(store.embs.keys())

    query_count = 0
    query_with_training_pairs = 0
    pos_count_total = 0

    for q, expected_ids in train_query_rows:
        query_text = str(q.text)
        expected_ids = [str(mid) for mid in expected_ids]
        if not expected_ids:
            continue
        query_count += 1
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

        # 若粗召回未覆盖 positives，尽量把可映射正例补进来，避免样本被静默丢弃。
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
        global_negative_pool = [mid for mid in all_mem_ids if not is_positive(mid, expected_ids)]
        if not all_negatives:
            if not global_negative_pool:
                continue
            need = min(
                max(1, args.neg_per_pos * max(1, len(positives))),
                len(global_negative_pool),
            )
            sampled_global_negs = random.sample(global_negative_pool, need)
            for mem_id in sampled_global_negs:
                if mem_id not in seen_candidates:
                    seen_candidates.add(mem_id)
                    candidates.append(mem_id)
            all_negatives = [mid for mid in candidates if mid not in positives]
            if not all_negatives:
                continue

        k_target = min(len(positives), args.cardinality_k_max)

        if args.loss_type == "pairwise":
            if args.neg_strategy == "ranked":
                ranked_pool = build_ranked_negatives(
                    candidates,
                    positives,
                    max_negatives=max(1, args.neg_per_pos * len(positives)),
                )
                negative_pool = []
                for mem_id in ranked_pool:
                    if mem_id not in negative_pool:
                        negative_pool.append(mem_id)
                if not negative_pool:
                    negative_pool = all_negatives
            else:
                negative_pool = all_negatives

            for p in positives:
                sampled_negs = sample_negatives(
                    negative_pool,
                    global_negative_pool,
                    sample_size=args.neg_per_pos,
                )
                for n in sampled_negs:
                    if n == p:
                        continue
                    if model_input_mode == "sim_matrix":
                        p_mat = compute_sim_matrix(q.q_vecs or [], store.embs[p].vecs)
                        n_mat = compute_sim_matrix(q.q_vecs or [], store.embs[n].vecs)
                        pair_buffer.append((p_mat, n_mat))
                    else:
                        pair_buffer.append((q.q_vecs or [], (p, n)))
            if positives:
                candidate_features = build_query_features(query_text, len(candidates))
                cardinality_samples.append((candidate_features, k_target))
                query_with_training_pairs += 1
        else:
            target_list_size = min(
                resolve_listwise_target_size(args.list_size, len(positives), args.neg_per_pos),
                len(all_mem_ids),
            )
            neg_quota = max(1, target_list_size - len(positives))
            listwise_negative_pool = all_negatives if all_negatives else global_negative_pool
            sampled_listwise_negs = sample_negatives(
                listwise_negative_pool,
                global_negative_pool,
                sample_size=neg_quota,
            )
            capped_candidates: List[str] = []
            seen_listwise = set()
            for mem_id in positives + sampled_listwise_negs:
                if mem_id in seen_listwise:
                    continue
                seen_listwise.add(mem_id)
                capped_candidates.append(mem_id)
            if len(capped_candidates) < 2:
                continue
            random.shuffle(capped_candidates)
            pos_mask = [1.0 if mid in positives else 0.0 for mid in capped_candidates]
            if sum(pos_mask) <= 0:
                continue
            candidate_features = build_query_features(query_text, len(capped_candidates))
            cardinality_samples.append((candidate_features, k_target))
            if model_input_mode == "sim_matrix":
                matrices = [
                    compute_sim_matrix(q.q_vecs or [], store.embs[mid].vecs)
                    for mid in capped_candidates
                ]
                if not matrices:
                    continue
                listwise_buffer.append(
                    {
                        "matrices": torch.stack(matrices),
                        "candidate_ids": list(capped_candidates),
                        "pos_mask": torch.tensor(pos_mask, dtype=torch.float32),
                        "query_features": torch.tensor(candidate_features, dtype=torch.float32),
                        "k_target": k_target,
                    }
                )
            else:
                listwise_buffer.append(
                    {
                        "q_vecs": q.q_vecs or [],
                        "candidate_ids": list(capped_candidates),
                        "pos_mask": torch.tensor(pos_mask, dtype=torch.float32),
                        "query_features": torch.tensor(candidate_features, dtype=torch.float32),
                        "k_target": k_target,
                    }
                )
            query_with_training_pairs += 1

    if args.loss_type == "pairwise" and not pair_buffer:
        raise RuntimeError("没有构造出训练样本，请检查数据集、候选参数或编码器设置。")
    if args.loss_type == "listwise" and not listwise_buffer:
        raise RuntimeError("listwise 模式没有构造出训练样本，请检查 positives 与负例采样配置。")

    avg_pos = (pos_count_total / query_count) if query_count else 0.0
    print(
        f"[train] dataset={args.dataset} backend={args.encoder_backend} "
        f"queries={query_count} trained_queries={query_with_training_pairs} avg_positives={avg_pos:.3f}"
    )
    print(
        f"[train] loss_type={args.loss_type} neg_strategy={args.neg_strategy} "
        f"pair_samples={len(pair_buffer)} listwise_samples={len(listwise_buffer)}"
    )
    if args.model_family == "bipartite":
        tau_mode = "learnable" if args.bipartite_learnable_tau else "constant"
        print(
            f"[train] model_family=bipartite tau_mode={tau_mode} "
            f"tau_init={float(args.bipartite_tau):.6f}"
        )
    else:
        print("[train] model_family=tiny")
    print(f"[train] epochs={args.epochs} batch_size={args.batch_size} lr={args.lr}")

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        rank_steps = 0
        card_loss_total = 0.0
        hard_constraint_loss_total = 0.0

        if args.loss_type == "pairwise":
            random.shuffle(pair_buffer)
            for i in range(0, len(pair_buffer), args.batch_size):
                batch = pair_buffer[i : i + args.batch_size]
                if model_input_mode == "sim_matrix":
                    p_batch = torch.stack([x[0] for x in batch]).to(args.device)
                    n_batch = torch.stack([x[1] for x in batch]).to(args.device)
                    p_score = model(p_batch)
                    n_score = model(n_batch)
                else:
                    q_batch_list: List[torch.Tensor] = []
                    q_mask_list: List[torch.Tensor] = []
                    p_batch_list: List[torch.Tensor] = []
                    p_mask_list: List[torch.Tensor] = []
                    n_batch_list: List[torch.Tensor] = []
                    n_mask_list: List[torch.Tensor] = []
                    for q_vecs_raw, pair_ids in batch:
                        pos_id, neg_id = pair_ids
                        q_fixed, q_mask = build_bipartite_fixed(
                            q_vecs_raw,
                            seq_len=args.bipartite_seq_len,
                            input_dim=args.bipartite_input_dim,
                        )
                        p_fixed, p_mask = build_bipartite_fixed(
                            store.embs[pos_id].vecs,
                            seq_len=args.bipartite_seq_len,
                            input_dim=args.bipartite_input_dim,
                        )
                        n_fixed, n_mask = build_bipartite_fixed(
                            store.embs[neg_id].vecs,
                            seq_len=args.bipartite_seq_len,
                            input_dim=args.bipartite_input_dim,
                        )
                        q_batch_list.append(q_fixed)
                        q_mask_list.append(q_mask)
                        p_batch_list.append(p_fixed)
                        p_mask_list.append(p_mask)
                        n_batch_list.append(n_fixed)
                        n_mask_list.append(n_mask)
                    q_batch = torch.stack(q_batch_list).to(args.device)
                    q_mask = torch.stack(q_mask_list).to(args.device)
                    p_batch = torch.stack(p_batch_list).to(args.device)
                    p_mask = torch.stack(p_mask_list).to(args.device)
                    n_batch = torch.stack(n_batch_list).to(args.device)
                    n_mask = torch.stack(n_mask_list).to(args.device)
                    p_score = model(q_batch, p_batch, q_mask=q_mask, m_mask=p_mask)
                    n_score = model(q_batch, n_batch, q_mask=q_mask, m_mask=n_mask)
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
                if model_input_mode == "sim_matrix":
                    max_candidates = max(sample["matrices"].shape[0] for sample in batch)
                    pos_mask = torch.zeros((len(batch), max_candidates), dtype=torch.float32)
                    valid_mask = torch.zeros((len(batch), max_candidates), dtype=torch.bool)
                    feat = torch.zeros((len(batch), 3), dtype=torch.float32)
                    tgt = torch.zeros((len(batch),), dtype=torch.long)
                    mats = torch.zeros((len(batch), max_candidates, 8, 8), dtype=torch.float32)
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
                else:
                    max_candidates = max(len(sample["candidate_ids"]) for sample in batch)
                    pos_mask = torch.zeros((len(batch), max_candidates), dtype=torch.float32)
                    valid_mask = torch.zeros((len(batch), max_candidates), dtype=torch.bool)
                    feat = torch.zeros((len(batch), 3), dtype=torch.float32)
                    tgt = torch.zeros((len(batch),), dtype=torch.long)
                    q_tensor = torch.zeros(
                        (len(batch), max_candidates, args.bipartite_seq_len, args.bipartite_input_dim),
                        dtype=torch.float32,
                    )
                    q_mask_tensor = torch.zeros(
                        (len(batch), max_candidates, args.bipartite_seq_len),
                        dtype=torch.bool,
                    )
                    m_tensor = torch.zeros(
                        (len(batch), max_candidates, args.bipartite_seq_len, args.bipartite_input_dim),
                        dtype=torch.float32,
                    )
                    m_mask_tensor = torch.zeros(
                        (len(batch), max_candidates, args.bipartite_seq_len),
                        dtype=torch.bool,
                    )
                    for row_idx, sample in enumerate(batch):
                        candidate_ids = sample["candidate_ids"]
                        n = len(candidate_ids)
                        pos_mask[row_idx, :n] = sample["pos_mask"]
                        valid_mask[row_idx, :n] = True
                        feat[row_idx] = sample["query_features"]
                        tgt[row_idx] = int(sample["k_target"])
                        q_fixed, q_mask_fixed = build_bipartite_fixed(
                            sample["q_vecs"],
                            seq_len=args.bipartite_seq_len,
                            input_dim=args.bipartite_input_dim,
                        )
                        for col_idx, mem_id in enumerate(candidate_ids):
                            m_fixed, m_mask_fixed = build_bipartite_fixed(
                                store.embs[mem_id].vecs,
                                seq_len=args.bipartite_seq_len,
                                input_dim=args.bipartite_input_dim,
                            )
                            q_tensor[row_idx, col_idx] = q_fixed
                            q_mask_tensor[row_idx, col_idx] = q_mask_fixed
                            m_tensor[row_idx, col_idx] = m_fixed
                            m_mask_tensor[row_idx, col_idx] = m_mask_fixed
                    pos_mask = pos_mask.to(args.device)
                    valid_mask = valid_mask.to(args.device)
                    feat = feat.to(args.device)
                    tgt = tgt.to(args.device)
                    q_tensor = q_tensor.to(args.device)
                    q_mask_tensor = q_mask_tensor.to(args.device)
                    m_tensor = m_tensor.to(args.device)
                    m_mask_tensor = m_mask_tensor.to(args.device)
                    flat_logits = model(
                        q_tensor.view(-1, args.bipartite_seq_len, args.bipartite_input_dim),
                        m_tensor.view(-1, args.bipartite_seq_len, args.bipartite_input_dim),
                        q_mask=q_mask_tensor.view(-1, args.bipartite_seq_len),
                        m_mask=m_mask_tensor.view(-1, args.bipartite_seq_len),
                    ).view(len(batch), max_candidates)
                masked_logits = flat_logits.masked_fill(~valid_mask, float("-inf"))
                pos_logits = masked_logits.masked_fill(pos_mask <= 0, float("-inf"))
                rank_loss = -(torch.logsumexp(pos_logits, dim=1) - torch.logsumexp(masked_logits, dim=1)).mean()

                loss = rank_loss
                if args.listwise_hard_constraint:
                    hard_terms: List[torch.Tensor] = []
                    for row_idx in range(len(batch)):
                        row_logits = masked_logits[row_idx]
                        row_pos_mask = (pos_mask[row_idx] > 0) & valid_mask[row_idx]
                        row_neg_mask = (pos_mask[row_idx] <= 0) & valid_mask[row_idx]
                        pos_scores = row_logits[row_pos_mask]
                        neg_scores = row_logits[row_neg_mask]
                        if pos_scores.numel() == 0 or neg_scores.numel() == 0:
                            continue
                        # 近似 top-|P| 约束：惩罚任意负例分数超过正例分数。
                        hard_terms.append(
                            F.softplus(
                                neg_scores.unsqueeze(0) - pos_scores.unsqueeze(1)
                            ).mean()
                        )
                    if hard_terms:
                        hard_constraint_loss = torch.stack(hard_terms).mean()
                        hard_constraint_loss_total += float(hard_constraint_loss.item())
                        loss = loss + args.listwise_hard_weight * hard_constraint_loss

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
            extra = ""
            if args.loss_type == "listwise" and args.listwise_hard_constraint:
                avg_hard_loss = hard_constraint_loss_total / max(rank_steps, 1)
                extra = f" listwise_hard_loss={avg_hard_loss:.6f}"
            print(
                f"[train] epoch={epoch} rank_loss={avg_rank_loss:.6f} "
                f"cardinality_loss={avg_card_loss:.6f}{extra}"
            )
        else:
            if args.loss_type == "listwise" and args.listwise_hard_constraint:
                avg_hard_loss = hard_constraint_loss_total / max(rank_steps, 1)
                print(
                    f"[train] epoch={epoch} rank_loss={avg_rank_loss:.6f} "
                    f"listwise_hard_loss={avg_hard_loss:.6f}"
                )
            else:
                print(f"[train] epoch={epoch} rank_loss={avg_rank_loss:.6f}")

    if args.model_family == "bipartite":
        print(
            f"[train] bipartite_tau_final={float(model.current_tau().detach().cpu().item()):.6f} "
            f"(learnable={bool(args.bipartite_learnable_tau)})"
        )

    if args.model_family == "bipartite":
        default_weight_name = (
            "pairwise_bipartite_reranker.pt"
            if args.loss_type == "pairwise"
            else "listwise_bipartite_reranker.pt"
        )
    else:
        default_weight_name = (
            "pairwise_reranker.pt" if args.loss_type == "pairwise" else "listwise_reranker.pt"
        )
    default_weight_path = MODEL_WEIGHTS_DIR / default_weight_name
    requested_save_path = Path(args.save_path) if args.save_path else None
    if run_root:
        save_path = run_root / default_weight_name
    else:
        save_path = requested_save_path or default_weight_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    meta_payload = dict(vars(args))
    meta_payload["model_family"] = args.model_family
    if args.model_family == "bipartite":
        meta_payload["bipartite_final_tau"] = float(model.current_tau().detach().cpu().item())
    payload = {"model_state": model.state_dict(), "meta": meta_payload}
    if cardinality_head is not None:
        payload["cardinality_state"] = cardinality_head.state_dict()
        payload["cardinality_meta"] = {
            "input_dim": 3,
            "hidden_dim": 16,
            "k_max": args.cardinality_k_max,
        }
    if save_path.resolve() == default_weight_path.resolve() and save_path.exists():
        backup_name = f"{default_weight_path.stem}.old_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
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
            f"{default_weight_path.stem}__{args.dataset}__{args.loss_type}__"
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
            "query_cache_signature": query_cache_signature,
            "memory_cache_signature": memory_cache_signature,
            "query_cache_path": str(query_cache_path),
            "memory_cache_path": str(memory_cache_path),
            "query_count": query_count,
            "trained_queries": query_with_training_pairs,
            "avg_positives": avg_pos,
            "loss_type": args.loss_type,
            "model_family": args.model_family,
            "neg_strategy": args.neg_strategy,
            "listwise_hard_constraint": bool(args.listwise_hard_constraint),
            "listwise_hard_weight": float(args.listwise_hard_weight),
            "bipartite_tau": float(args.bipartite_tau),
            "bipartite_learnable_tau": bool(args.bipartite_learnable_tau),
            "bipartite_final_tau": float(model.current_tau().detach().cpu().item())
            if args.model_family == "bipartite"
            else None,
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
                "model_family": args.model_family,
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



