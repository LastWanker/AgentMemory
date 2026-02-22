"""最小回归脚本：验证 coarse query/passage 角色没有混用。"""

from __future__ import annotations

import uuid

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.memory_indexer import (
    FieldScorer,
    MemoryItem,
    Query,
    Retriever,
    Vectorizer,
    build_memory_index,
)
from src.memory_indexer.encoder.simple import SimpleHashEncoder
from src.memory_indexer.utils import normalize


def main() -> None:
    # 1) 构造最小数据：2 条 memory + 1 条 query。
    memories = [
        MemoryItem(mem_id="m1", text="我喜欢周末爬山，顺便拍风景照。"),
        MemoryItem(mem_id="m2", text="项目例会定在周三上午十点。"),
    ]
    query_text = "这周例会几点开始？"

    encoder = SimpleHashEncoder(dims=16)
    vectorizer = Vectorizer(strategy="token_pool_topk", k=8)

    # 2) 建库：内部会强制 coarse_vec 走 encode_passage_sentence，并写入 coarse_role 标记。
    store, index = build_memory_index(memories, encoder, vectorizer)

    memory_roles = {mem_id: emb.aux.get("coarse_role") for mem_id, emb in store.embs.items()}
    print("memory coarse_role:", memory_roles)
    assert all(role == "passage" for role in memory_roles.values())

    # 3) 构造 query：coarse_vec 显式走 encode_query_sentence，并附带 coarse_role=query。
    token_vecs, model_tokens = encoder.encode_tokens(query_text)
    q_vecs, aux = vectorizer.make_group(token_vecs, model_tokens, query_text)
    q_vecs = [normalize(v) for v in q_vecs]
    query = Query(
        query_id=str(uuid.uuid4()),
        text=query_text,
        encoder_id=encoder.encoder_id,
        strategy=vectorizer.strategy,
        q_vecs=q_vecs,
        coarse_vec=normalize(encoder.encode_query_sentence(query_text)),
        aux={**aux, "coarse_role": "query", "tokens": model_tokens, "lex_tokens": model_tokens},
    )
    print("query coarse_role:", query.aux.get("coarse_role"))

    # 4) 跑一次 CoarseIndex.add + search（通过 Retriever 触发），验证流程可用。
    retriever = Retriever(store, index, FieldScorer())
    results = retriever.retrieve(query, top_n=2, top_k=2)
    print("search mem_ids:", [item.mem_id for item in results])
    print("OK: coarse roles verified")


if __name__ == "__main__":
    main()

