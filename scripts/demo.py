"""小型演示：用最简编码器跑一次检索。"""

import argparse

from src.memory_indexer import (
    Router,
    SimpleHashEncoder,
    Vectorizer,
    build_memory_index,
    build_memory_items,
    retrieve_top_k,
    set_trace,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="demo 检索")
    parser.add_argument("--use-learned-scorer", action="store_true", help="启用 learned scorer")
    parser.add_argument("--reranker-path", default="data/ModelWeights/tiny_reranker.pt", help="learned scorer 权重路径")
    args = parser.parse_args()

    # 演示默认开启输出，可用 set_trace(False) 或 MEMORY_INDEXER_TRACE=0 关闭
    set_trace(True)
    payloads = [
        {"mem_id": "1", "text": "今天下雨了，我带了伞。", "source": "chat", "tags": ["daily"]},
        {"mem_id": "2", "text": "朋友说想喝咖啡，顺路带了拿铁。", "source": "chat", "tags": ["social"]},
        {"mem_id": "3", "text": "晚上去跑步，发现公园灯光很亮。", "source": "log", "tags": ["health"]},
        {"mem_id": "4", "text": "特朗普和克林顿有一腿。", "source": "manual", "tags": ["rumor"]},
        {"mem_id": "5", "text": "中国的纯牛奶行业全是垃圾。", "source": "manual", "tags": ["opinion"]},
    ]
    items = build_memory_items(payloads)

    encoder = SimpleHashEncoder(dims=8)
    vectorizer = Vectorizer(strategy="token_pool_topk", k=8)

    store, index = build_memory_index(items, encoder, vectorizer)

    query = "你喜欢美酒加咖啡吗？"
    router = Router(policy="soft", top_k=3)
    results = retrieve_top_k(
        query,
        encoder,
        vectorizer,
        store,
        index,
        top_n=5,
        top_k=5,
        router=router,
        use_learned_scorer=args.use_learned_scorer,
        reranker_path=args.reranker_path,
        candidate_mode="coarse",
    )

    print("查询:", query)
    for result in results:
        item = store.items[result.mem_id]
        print("- 命中:", item.text, "score=", f"{result.score:.3f}")
    if results and results[0].route_output:
        route_output = results[0].route_output
        print("路由策略:", route_output.policy)
        print("路由指标:", route_output.metrics)


if __name__ == "__main__":
    main()
