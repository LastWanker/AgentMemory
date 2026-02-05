"""端到端流程：建库与检索。"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple
import json
from pathlib import Path
import uuid

from .encoder.base import Encoder
from .models import EmbeddingRecord, MemoryItem, Query, RetrieveResult
from .index import CoarseIndex, LexicalIndex
from .retriever import Retriever, Router
from .scorer import FieldScorer
from .store import MemoryStore
from .utils import normalize
from .vectorizer import Vectorizer
from .trace import trace, trace_progress
from .tokenizers import TokenizerInput, resolve_tokenizer


def _compose_encoder_id(sentence_encoder: Encoder, token_encoder: Encoder) -> str:
    if sentence_encoder.encoder_id == token_encoder.encoder_id:
        return sentence_encoder.encoder_id
    return f"{sentence_encoder.encoder_id}|{token_encoder.encoder_id}"


def _resolve_lex_tokenizer(
    encoder: Encoder, lex_tokenizer: TokenizerInput
) -> object:
    if lex_tokenizer is not None:
        return resolve_tokenizer(lex_tokenizer)
    if hasattr(encoder, "tokenizer"):
        encoder_tokenizer = getattr(encoder, "tokenizer")
        if hasattr(encoder_tokenizer, "tokenize"):
            return encoder_tokenizer
    return resolve_tokenizer(None)


def _load_memory_cache(
    cache_path: Optional[Path],
    encoder_id: str,
    strategy: str,
) -> Dict[str, Dict[str, object]]:
    if not cache_path or not cache_path.exists():
        return {}
    payloads: Dict[str, Dict[str, object]] = {}
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("encoder_id") != encoder_id or payload.get("strategy") != strategy:
            continue
        mem_id = payload.get("mem_id")
        if isinstance(mem_id, str):
            payloads[mem_id] = payload
    if payloads:
        trace(f"检测到记忆缓存: {cache_path} | entries={len(payloads)}")
    return payloads


def _build_memory_cache_payload(
    item: MemoryItem,
    encoder_id: str,
    strategy: str,
    m_vecs: List[List[float]],
    coarse_vec: Optional[List[float]],
    aux: Dict[str, List[str]],
    lex_tokens: List[str],
    model_tokens: List[str],
) -> Dict[str, object]:
    return {
        "mem_id": item.mem_id,
        "text": item.text,
        "encoder_id": encoder_id,
        "strategy": strategy,
        "vecs": m_vecs,
        "coarse_vec": coarse_vec,
        "aux": {
            **aux,
            "lex_tokens": lex_tokens,
            "model_tokens": model_tokens,
            "tokens": lex_tokens,
        },
        "tokens": lex_tokens,
        "lex_tokens": lex_tokens,
        "model_tokens": model_tokens,
    }


def _write_memory_cache(
    cache_path: Path,
    items: List[MemoryItem],
    cached_payloads: Dict[str, Dict[str, object]],
) -> None:
    with cache_path.open("w", encoding="utf-8") as handle:
        for item in items:
            payload = cached_payloads.get(item.mem_id)
            if not payload:
                continue
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def build_memory_index(
    memory_items: Iterable[MemoryItem],
    encoder: Encoder,
    vectorizer: Vectorizer,
    *,
    token_encoder: Optional[Encoder] = None,
    lex_tokenizer: TokenizerInput = None,
    cache_path: Optional[Path] = None,
    return_lexical: bool = False,
) -> Tuple[MemoryStore, CoarseIndex] | Tuple[MemoryStore, CoarseIndex, LexicalIndex]:
    """构建记忆库索引。

    默认保持兼容：只返回 (store, index)。
    当 return_lexical=True 时返回 (store, index, lexical_index)。
    """

    items = list(memory_items)
    sentence_encoder = encoder
    token_encoder = token_encoder or encoder
    lex_tokenizer_resolved = _resolve_lex_tokenizer(sentence_encoder, lex_tokenizer)
    store = MemoryStore()
    index = CoarseIndex()
    lexical_index = LexicalIndex() if return_lexical else None
    encoder_id = _compose_encoder_id(sentence_encoder, token_encoder)
    cached_payloads = _load_memory_cache(cache_path, encoder_id, vectorizer.strategy)
    cache_hits = 0
    cache_misses = 0

    trace("开始构建记忆库索引")
    progress = trace_progress("建库进度", total=len(items))
    for idx, item in enumerate(items, start=1):
        cached = cached_payloads.get(item.mem_id)
        cache_ok = bool(cached and cached.get("text") == item.text)
        if cache_ok:
            token_vecs = []
            payload_tokens = cached.get("tokens") or []
            payload_aux = cached.get("aux") or {}
            lex_tokens = (
                cached.get("lex_tokens")
                or payload_aux.get("lex_tokens")
                or payload_tokens
                or payload_aux.get("tokens")
                or []
            )
            model_tokens = (
                cached.get("model_tokens")
                or payload_aux.get("model_tokens")
                or payload_tokens
                or payload_aux.get("tokens")
                or []
            )
            aux = {
                k: v
                for k, v in payload_aux.items()
                if k not in {"tokens", "lex_tokens", "model_tokens"}
            }
            m_vecs = cached.get("vecs") or []
            coarse_vec = cached.get("coarse_vec")
            cache_hits += 1
        else:
            token_vecs, model_tokens = token_encoder.encode_tokens(item.text)
            m_vecs, aux = vectorizer.make_group(token_vecs, model_tokens, item.text)
            m_vecs = [normalize(v) for v in m_vecs]
            coarse_vec = normalize(sentence_encoder.encode_sentence(item.text))
            lex_tokens = lex_tokenizer_resolved.tokenize(item.text)
            cached_payloads[item.mem_id] = _build_memory_cache_payload(
                item,
                encoder_id,
                vectorizer.strategy,
                m_vecs,
                coarse_vec,
                aux,
                lex_tokens,
                model_tokens,
            )
            cache_misses += 1
        emb = EmbeddingRecord(
            emb_id=str(uuid.uuid4()),
            mem_id=item.mem_id,
            encoder_id=encoder_id,
            strategy=vectorizer.strategy,
            dims=len(m_vecs[0]) if m_vecs else 0,
            n_vecs=len(m_vecs),
            vecs=m_vecs,
            coarse_vec=coarse_vec,
            aux={
                **aux,
                "lex_tokens": lex_tokens,
                "model_tokens": model_tokens,
                "tokens": lex_tokens,
            },
        )
        store.add(item, emb, lex_tokens)
        index.add(item.mem_id, coarse_vec)
        if lexical_index:
            lexical_index.add(item.mem_id, lex_tokens)
        progress.update(idx)

    progress.finish()
    if cache_path:
        _write_memory_cache(cache_path, items, cached_payloads)
        trace(
            f"记忆缓存更新完成: {cache_path} | hit={cache_hits} | miss={cache_misses}"
        )
    trace("记忆库索引构建完成")
    if return_lexical and lexical_index:
        return store, index, lexical_index
    return store, index


def retrieve_top_k(
    query_text: str,
    encoder: Encoder,
    vectorizer: Vectorizer,
    store: MemoryStore,
    index: CoarseIndex,
    lexical_index: LexicalIndex | None = None,
    top_n: int = 1000,
    top_k: int = 10,
    router: Router | None = None,
    token_encoder: Optional[Encoder] = None,
    lex_tokenizer: TokenizerInput = None,
) -> List[RetrieveResult]:
    """执行检索，并返回 top-k 结果。"""

    trace("开始构建查询向量")
    sentence_encoder = encoder
    token_encoder = token_encoder or encoder
    lex_tokenizer_resolved = _resolve_lex_tokenizer(sentence_encoder, lex_tokenizer)
    token_vecs, model_tokens = token_encoder.encode_tokens(query_text)
    q_vecs, aux = vectorizer.make_group(token_vecs, model_tokens, query_text)
    q_vecs = [normalize(v) for v in q_vecs]
    lex_tokens = lex_tokenizer_resolved.tokenize(query_text)
    encoder_id = _compose_encoder_id(sentence_encoder, token_encoder)
    q = Query(
        query_id=str(uuid.uuid4()),
        text=query_text,
        encoder_id=encoder_id,
        strategy=vectorizer.strategy,
        q_vecs=q_vecs,
        coarse_vec=normalize(sentence_encoder.encode_sentence(query_text)),
        aux={
            **aux,
            "lex_tokens": lex_tokens,
            "model_tokens": model_tokens,
            "tokens": lex_tokens,
        },
    )
    trace("查询向量构建完成，进入检索")
    # 【关键修复】检索流程必须走 Retriever.retrieve()，否则 router 只是“装上方向盘”。
    # - Retriever 内部会统一处理粗召回、精排、路由与 route_output 填充。
    # - pipeline 只负责构建 Query 并委托检索，避免重复一遍粗召回/精排逻辑。
    retriever = Retriever(
        store,
        index,
        FieldScorer(),
        router=router,
        lexical_index=lexical_index,
    )
    results = retriever.retrieve(q, top_n=top_n, top_k=top_k)
    trace(f"检索完成，输出 {len(results)} 条结果")
    return results
