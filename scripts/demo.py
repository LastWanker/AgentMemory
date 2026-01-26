"""小型演示：用最简编码器跑一次检索。"""

from src.memory_indexer import (
    MemoryItem,
    SimpleHashEncoder,
    Vectorizer,
    build_memory_index,
    retrieve_top_k,
)


def main() -> None:
    # 中文注释：准备几条记忆
    items = [
        MemoryItem(mem_id="1", text="今天下雨了，我带了伞。"),
        MemoryItem(mem_id="2", text="朋友说想喝咖啡，顺路带了拿铁。"),
        MemoryItem(mem_id="3", text="晚上去跑步，发现公园灯光很亮。"),
        MemoryItem(mem_id="4", text="特朗普和克林顿有一腿。"),
        MemoryItem(mem_id="5", text="中国的纯牛奶行业全是垃圾。"),
    ]

    encoder = SimpleHashEncoder(dims=8)
    vectorizer = Vectorizer(strategy="token_pool_topk", k=8)

    store, index = build_memory_index(items, encoder, vectorizer)

    query = "我喝咖啡喜欢配牛奶。"
    results = retrieve_top_k(query, encoder, vectorizer, store, index, top_n=5, top_k=5)

    print("查询:", query)
    for result in results:
        item = store.items[result.mem_id]
        print("- 命中:", item.text, "score=", f"{result.score:.3f}")


if __name__ == "__main__":
    main()
