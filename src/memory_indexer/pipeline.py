"""端到端流程：建库与检索。"""

from __future__ import annotations

from typing import Iterable, List, Tuple
import uuid

from .encoder.base import Encoder
from .models import EmbeddingRecord, MemoryItem, Query, RetrieveResult
from .index import CoarseIndex
from .retriever import Retriever, Router
from .scorer import FieldScorer
from .store import MemoryStore
from .utils import normalize
from .vectorizer import Vectorizer
from .trace import trace, trace_progress


def build_memory_index(
    memory_items: Iterable[MemoryItem],
    encoder: Encoder,
    vectorizer: Vectorizer,
) -> Tuple[MemoryStore, CoarseIndex]:
    """构建记忆库索引。"""

    items = list(memory_items)
    store = MemoryStore()
    index = CoarseIndex()

    trace("开始构建记忆库索引")
    progress = trace_progress("建库进度", total=len(items))
    for idx, item in enumerate(items, start=1):
        token_vecs, tokens = encoder.encode_tokens(item.text)
        m_vecs, aux = vectorizer.make_group(token_vecs, tokens, item.text)
        m_vecs = [normalize(v) for v in m_vecs]
        coarse_vec = normalize(encoder.encode_sentence(item.text))
        emb = EmbeddingRecord(
            emb_id=str(uuid.uuid4()),
            mem_id=item.mem_id,
            encoder_id=encoder.encoder_id,
            strategy=vectorizer.strategy,
            dims=len(m_vecs[0]) if m_vecs else 0,
            n_vecs=len(m_vecs),
            vecs=m_vecs,
            coarse_vec=coarse_vec,
            aux=aux,
        )
        store.add(item, emb)
        index.add(item.mem_id, coarse_vec)
        progress.update(idx)

    progress.finish()
    trace("记忆库索引构建完成")
    return store, index


def retrieve_top_k(
    query_text: str,
    encoder: Encoder,
    vectorizer: Vectorizer,
    store: MemoryStore,
    index: CoarseIndex,
    top_n: int = 1000,
    top_k: int = 10,
    router: Router | None = None,
) -> List[RetrieveResult]:
    """执行检索，并返回 top-k 结果。"""

    trace("开始构建查询向量")
    token_vecs, tokens = encoder.encode_tokens(query_text)
    q_vecs, aux = vectorizer.make_group(token_vecs, tokens, query_text)
    q_vecs = [normalize(v) for v in q_vecs]
    q = Query(
        query_id=str(uuid.uuid4()),
        text=query_text,
        encoder_id=encoder.encoder_id,
        strategy=vectorizer.strategy,
        q_vecs=q_vecs,
        coarse_vec=normalize(encoder.encode_sentence(query_text)),
        aux=aux,
    )
    trace("查询向量构建完成，进入检索")
    # 【新增】router 可在后续阶段切换软/半硬/硬策略，保持接口不变。
    retriever = Retriever(store, index, FieldScorer(), router=router)
    candidates = index.search(q.coarse_vec, top_n=top_n)
    progress = trace_progress("精排进度", total=len(candidates))
    results: List[RetrieveResult] = []
    for idx, (mem_id, coarse_score) in enumerate(candidates, start=1):
        emb = store.embs[mem_id]
        score, debug = retriever.scorer.score(q.q_vecs, emb.vecs)
        results.append(
            RetrieveResult(
                mem_id=mem_id,
                score=score,
                coarse_score=coarse_score,
                debug=debug,
            )
        )
        progress.update(idx)
    progress.finish()
    results.sort(key=lambda r: r.score, reverse=True)
    final_results = results[:top_k]
    trace(f"检索完成，输出 {len(final_results)} 条结果")
    return final_results
