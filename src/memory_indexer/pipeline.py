"""端到端流程：建库与检索。"""

from __future__ import annotations

from typing import Iterable, List, Tuple
import uuid

from .encoder.base import Encoder
from .models import EmbeddingRecord, MemoryItem, Query, RetrieveResult
from .index import CoarseIndex
from .retriever import Retriever
from .scorer import FieldScorer
from .store import MemoryStore
from .utils import normalize
from .vectorizer import Vectorizer


def build_memory_index(
    memory_items: Iterable[MemoryItem],
    encoder: Encoder,
    vectorizer: Vectorizer,
) -> Tuple[MemoryStore, CoarseIndex]:
    """构建记忆库索引。"""

    store = MemoryStore()
    index = CoarseIndex()

    for item in memory_items:
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

    return store, index


def retrieve_top_k(
    query_text: str,
    encoder: Encoder,
    vectorizer: Vectorizer,
    store: MemoryStore,
    index: CoarseIndex,
    top_n: int = 1000,
    top_k: int = 10,
) -> List[RetrieveResult]:
    """执行检索，并返回 top-k 结果。"""

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
    retriever = Retriever(store, index, FieldScorer())
    return retriever.retrieve(q, top_n=top_n, top_k=top_k)
