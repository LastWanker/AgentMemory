from __future__ import annotations

import asyncio
import json
import math
import random
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import aiohttp


META_BANNED_TOKENS = (
    "训练数据",
    "样本",
    "cluster",
    "召回",
    "模型",
    "向量",
    "embedding",
    "indexer",
    "pipeline",
)

GENERIC_OBJECT_WORDS = {
    "这件事",
    "那个事",
    "那个问题",
    "这个问题",
    "这个话题",
    "那个话题",
    "它",
    "这个",
    "那个",
}

EN_STOPWORDS = {
    "we",
    "you",
    "they",
    "them",
    "he",
    "she",
    "it",
    "ai",
    "llm",
    "gpt",
    "chatgpt",
    "feel",
    "think",
    "like",
    "with",
    "from",
    "that",
    "this",
    "there",
    "here",
    "what",
    "when",
    "where",
    "which",
    "how",
    "why",
    "then",
    "also",
    "just",
    "good",
    "okay",
    "yes",
    "no",
    "not",
    "your",
    "my",
    "our",
    "their",
    "another",
    "other",
    "about",
}

CN_ANCHOR_STOPWORDS = {
    "哈哈",
    "呵呵",
    "嘿嘿",
    "不不不",
    "等等",
    "对吗",
    "另外",
    "而已",
    "就是",
    "其实",
    "然后",
    "那个",
    "这个",
    "我们",
    "你们",
    "你说",
    "我说",
    "我觉得",
    "感觉",
    "可能",
    "可以",
    "一下",
    "一些",
    "很多",
    "什么",
    "怎么",
    "为啥",
    "不是",
    "没有",
    "有点",
    "感受",
    "状态",
    "我就是说",
    "还是那句话",
    "一个有比较好的记",
    "一个神经元结构和",
    "一体两面吗",
    "一大堆信息",
    "一种理论罢了",
}


LAYER_DESCRIPTIONS = {
    "object": "对象层：围绕讨论对象/事实语义内容，允许遗忘指代与模糊提法。",
    "stance": "立场层：围绕用户对对象的态度变化、动摇、反驳与纠错请求。",
    "intent": "意图层：围绕用户希望 AI 做什么（解释、论证、方案、安慰、辩论等）。",
}


@dataclass
class LLMQueryGenConfig:
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    concurrency: int = 32
    timeout_seconds: float = 60.0
    max_retries: int = 3
    temperature: float = 1.1
    top_p: float = 0.95
    seed: int = 11
    max_clusters: int | None = None
    min_cluster_size: int = 1
    max_per_layer: int | None = 6
    fallback_only: bool = False


@dataclass
class ClusterContext:
    cluster_id: str
    mem_ids: List[str]
    texts: List[str]


async def build_llm_supplemental_rows(
    memories: Sequence[Dict[str, object]],
    *,
    api_key: str,
    config: LLMQueryGenConfig,
) -> List[Dict[str, object]]:
    clusters = _group_clusters(memories, min_cluster_size=config.min_cluster_size)
    if config.max_clusters is not None:
        clusters = clusters[: max(0, int(config.max_clusters))]

    if not clusters:
        return []

    rows: List[Dict[str, object]] = []
    sem = asyncio.Semaphore(max(1, int(config.concurrency)))
    timeout = aiohttp.ClientTimeout(total=config.timeout_seconds)
    connector = aiohttp.TCPConnector(limit=max(100, config.concurrency * 4))
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async with aiohttp.ClientSession(timeout=timeout, connector=connector, headers=headers) as session:
        tasks = []
        for cluster_idx, cluster in enumerate(clusters):
            n = len(cluster.mem_ids)
            k = max(1, math.ceil(n / 3))
            if config.max_per_layer is not None:
                k = min(k, max(1, int(config.max_per_layer)))
            for layer in ("object", "stance", "intent"):
                tasks.append(
                    asyncio.create_task(
                        _generate_layer_rows(
                            session=session,
                            sem=sem,
                            cluster=cluster,
                            layer=layer,
                            k=k,
                            cluster_idx=cluster_idx,
                            config=config,
                        )
                    )
                )
        for fut in asyncio.as_completed(tasks):
            rows.extend(await fut)
    rows.sort(key=lambda x: str(x.get("query_id", "")))
    return rows


async def _generate_layer_rows(
    *,
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    cluster: ClusterContext,
    layer: str,
    k: int,
    cluster_idx: int,
    config: LLMQueryGenConfig,
) -> List[Dict[str, object]]:
    if config.fallback_only:
        texts = _fallback_queries(cluster, layer, k, seed=config.seed + cluster_idx)
        generation_mode = "fallback"
    else:
        texts, generation_mode = await _generate_queries_with_retry(
            session=session,
            sem=sem,
            cluster=cluster,
            layer=layer,
            k=k,
            cluster_idx=cluster_idx,
            config=config,
        )

    rows: List[Dict[str, object]] = []
    for i, query_text in enumerate(texts, start=1):
        rows.append(
            {
                "query_id": f"llmq-{cluster.cluster_id}-{layer}-{i:03d}",
                "query_text": query_text,
                "positives": list(cluster.mem_ids),
                "candidates": list(cluster.mem_ids),
                "hard_negatives": [],
                "meta": {
                    "source": "chat_llm_supplemental",
                    "channel": "supplemental",
                    "layer": layer,
                    "cluster_id": cluster.cluster_id,
                    "memory_count": len(cluster.mem_ids),
                    "k_target": k,
                    "model": config.model,
                    "generation_mode": generation_mode,
                },
            }
        )
    return rows


async def _generate_queries_with_retry(
    *,
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    cluster: ClusterContext,
    layer: str,
    k: int,
    cluster_idx: int,
    config: LLMQueryGenConfig,
) -> tuple[List[str], str]:
    anchors = _anchor_terms(cluster.texts, topn=8)
    pool: List[str] = []
    pool_seen = set()
    for attempt in range(1, max(1, config.max_retries) + 1):
        try:
            query_texts = await _request_layer_queries(
                session=session,
                sem=sem,
                cluster=cluster,
                layer=layer,
                k=k,
                cluster_idx=cluster_idx,
                config=config,
            )
            cleaned = _normalize_queries(query_texts, k)
            cleaned = _enforce_anchor_and_style(
                cleaned,
                k=k,
                anchors=anchors,
            )
            for text in cleaned:
                key = re.sub(r"[，。！？、,.!?\\s]", "", text.lower())
                if not key or key in pool_seen:
                    continue
                pool_seen.add(key)
                pool.append(text)
                if len(pool) >= k:
                    return pool[:k], "llm"
        except Exception:
            pass
        await asyncio.sleep(min(2.5, 0.4 * attempt))

    fallback = _fallback_queries(cluster, layer, k, seed=config.seed + cluster_idx)
    if pool:
        for text in fallback:
            key = re.sub(r"[，。！？、,.!?\\s]", "", text.lower())
            if not key or key in pool_seen:
                continue
            pool_seen.add(key)
            pool.append(text)
            if len(pool) >= k:
                break
        return pool[:k], "mixed"
    return fallback[:k], "fallback"


async def _request_layer_queries(
    *,
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    cluster: ClusterContext,
    layer: str,
    k: int,
    cluster_idx: int,
    config: LLMQueryGenConfig,
) -> List[str]:
    prompt = _build_prompt(cluster, layer=layer, k=k, seed=config.seed + cluster_idx)
    payload = {
        "model": config.model,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是中文对话里的“用户提问生成器”。"
                    "你的任务是生成用户会对 AI 说的话。输出必须是 JSON。"
                    "禁止出现训练数据、样本、cluster、召回、模型等元话术。"
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }
    endpoint = config.base_url.rstrip("/") + "/chat/completions"

    async with sem:
        async with session.post(endpoint, json=payload) as resp:
            text = await resp.text()
            if resp.status >= 400:
                raise RuntimeError(f"llm_http_error status={resp.status}: {text[:400]}")
            data = json.loads(text)
    content = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    return _parse_queries_from_content(content)


def _group_clusters(memories: Sequence[Dict[str, object]], *, min_cluster_size: int) -> List[ClusterContext]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in memories:
        mem_id = str(row.get("mem_id", "")).strip()
        text = str(row.get("text", "")).strip()
        if not mem_id or not text:
            continue
        cluster_id = str(row.get("cluster_id", "")).strip() or mem_id
        grouped.setdefault(cluster_id, []).append(row)

    clusters: List[ClusterContext] = []
    for cluster_id, rows in grouped.items():
        mem_ids = [str(x.get("mem_id", "")).strip() for x in rows if str(x.get("mem_id", "")).strip()]
        texts = [str(x.get("text", "")).strip() for x in rows if str(x.get("text", "")).strip()]
        if len(mem_ids) < max(1, int(min_cluster_size)):
            continue
        clusters.append(ClusterContext(cluster_id=cluster_id, mem_ids=mem_ids, texts=texts))
    clusters.sort(key=lambda c: (-len(c.mem_ids), c.cluster_id))
    return clusters


def _build_prompt(cluster: ClusterContext, *, layer: str, k: int, seed: int) -> str:
    rng = random.Random(seed * 131 + len(cluster.mem_ids) * 17 + len(layer))
    keywords = _anchor_terms(cluster.texts, topn=10)
    snippets = _pick_snippets(cluster.texts, max_count=6)
    template_refs = _template_references(layer, rng=rng)
    style_guides = _style_guides(layer, rng=rng)
    pre_rules = [
        f"层级定义：{LAYER_DESCRIPTIONS[layer]}",
        f"目标数量：{k} 条。",
        "你生成的是“用户对 AI 说的话”，要有对话感，像真实提问而不是摘要。",
        "参考模板仅提供方向，禁止直接套用模板原句。",
        "硬性要求：每条 query 必须明确出现至少 1 个“对象锚点”词，不允许空泛代词糊过去。",
        "建议：尽量不要只用“这件事/那个问题/它”承载对象，优先补一个具体线索。",
        "硬性要求：不允许出现训练数据/样本/cluster/召回/模型/向量等元话术。",
        "输出 JSON 对象格式：",
        '{"queries":[{"query_text":"..."}, ...]}',
    ]
    per_item_rules = [
        "每条 query 要求：",
        "- 单条 12~60 字，中文为主，口语自然。",
        "- 不写长段解释，不写项目术语。",
        "- 允许模糊指代、记忆断裂、隔天重提、改口。",
        "- 必须体现原 memory 的具体内容，不得出现“把什么/关于什么”这类空槽位提问。",
        "- 尽量“同内核换说法”，避免模板痕迹。",
    ]
    context_part = [
        f"对象锚点（至少命中其一）={keywords if keywords else ['这件事']}",
        f"cluster_id={cluster.cluster_id}",
        f"cluster_memory_count={len(cluster.mem_ids)}",
        "片段提示：",
    ]
    for idx, text in enumerate(snippets, start=1):
        context_part.append(f"{idx}. {text}")

    refs_part = [
        f"参考骨架（只看意图，不要复用原句）：\n{template_refs}",
        f"风格引导卡片（仅作灵感，允许自由发挥）：\n{style_guides}",
    ]
    rng.shuffle(refs_part)

    parts = []
    parts.extend(pre_rules)
    parts.extend(per_item_rules)
    parts.extend(refs_part)
    parts.extend(context_part)
    parts.append("请只返回 JSON，不要输出任何额外说明。")
    return "\n".join(parts)


def _template_references(layer: str, *, rng: random.Random) -> str:
    templates = {
        "object": [
            "[遗忘指代] 先承认叫不出名称，再抛出对象锚点并追问。",
            "[特征锚点] 用关键机制或推理链定位对象，再确认是否记对。",
            "[类比定位] 用两个近似概念做参照，再落回当前对象。",
            "[直接点题] 直接报出主题对象，再追加一个具体问题。",
            "[历史回钩] 先提“我们之前聊过”，再要求补齐缺失线索。",
            "[误差修正] 先说自己可能记混，再请求对齐正确对象。",
        ],
        "stance": [
            "[直觉不对] 表达“逻辑顺但心里不踏实”，并点名哪一段不稳。",
            "[反驳不服] 点名一个常见反驳，再说为何觉得力度不足。",
            "[摇摆改口] 展示立场变化，明确“我可能记错/想岔”。",
            "[要求纠错] 明确请 AI 不要顺着说，直接给硬反驳。",
            "[自检偏见] 承认自己可能站队，再要求指出盲点。",
            "[前后对照] 提及上次立场，与这次态度做对比。",
        ],
        "intent": [
            "[求解释] 明确希望拆步骤说明，不急着下结论。",
            "[求对辩] 明确让 AI 站反方，聚焦打漏洞。",
            "[求方案] 明确需要可执行动作，而非抽象概念。",
            "[求梳理] 明确希望整理为可选路径或决策分支。",
            "[求下一问] 明确希望得到“下一步最该问什么”。",
            "[求安稳] 明确希望先给不惊吓的解释版本。",
        ],
    }[layer]
    pool = list(templates)
    rng.shuffle(pool)
    return "\n".join(f"- {x}" for x in pool)


def _style_guides(layer: str, *, rng: random.Random) -> str:
    guides = {
        "object": [
            "把“名词忘了但机制记得”作为入口。",
            "把“类比到相近概念再纠正”作为入口。",
            "把“直接点题 + 一句追问”作为入口。",
            "把“我们上次聊到哪了”作为入口。",
        ],
        "stance": [
            "可用“先认同再怀疑”的转折。",
            "可用“我可能记错了”来自我修正。",
            "可用“请直接反驳我”提高对抗性。",
            "可用“态度变化”体现时间跨度。",
        ],
        "intent": [
            "明确希望 AI 做什么，不要只表达感受。",
            "可指定输出形态：步骤、清单、对辩、比较表。",
            "可指定约束：短、硬核、温和、直白。",
            "可体现阶段目标：先理解，再判断，再行动。",
            "每条都要出现对象锚点词，禁止空槽位问法。",
        ],
    }[layer]
    pool = list(guides)
    rng.shuffle(pool)
    return "\n".join(f"- {x}" for x in pool)


def _parse_queries_from_content(content: str) -> List[str]:
    payload = _load_json_payload(content)
    if isinstance(payload, dict):
        queries = payload.get("queries")
        if isinstance(queries, list):
            out: List[str] = []
            for item in queries:
                if isinstance(item, dict):
                    text = str(item.get("query_text", "")).strip()
                else:
                    text = str(item).strip()
                if text:
                    out.append(text)
            return out
    return []


def _load_json_payload(content: str):
    text = content.strip()
    if not text:
        return {}
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def _normalize_queries(texts: Iterable[str], k: int) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in texts:
        text = str(raw).strip()
        if not text:
            continue
        text = re.sub(r"\s+", " ", text)
        if len(text) < 8 or len(text) > 90:
            continue
        lowered = text.lower()
        if any(tok in lowered for tok in META_BANNED_TOKENS):
            continue
        dedup_key = re.sub(r"[，。！？、,.!?\\s]", "", lowered)
        if not dedup_key or dedup_key in seen:
            continue
        seen.add(dedup_key)
        out.append(text)
        if len(out) >= k:
            break
    return out


def _enforce_anchor_and_style(texts: Iterable[str], *, k: int, anchors: Sequence[str]) -> List[str]:
    normalized_anchors = [a for a in anchors if a and a not in GENERIC_OBJECT_WORDS]
    out: List[str] = []
    seen = set()
    for text in texts:
        if not text:
            continue
        lower = text.lower()
        if any(bad in text for bad in ("【", "】", "{", "}")):
            continue
        if normalized_anchors:
            if not any(anchor in text for anchor in normalized_anchors):
                continue
        else:
            if all(g in text for g in ("这件事", "那个问题")):
                continue
        key = re.sub(r"[，。！？、,.!?\\s]", "", lower)
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= k:
            break
    return out


def _fallback_queries(cluster: ClusterContext, layer: str, k: int, *, seed: int) -> List[str]:
    rng = random.Random(seed * 97 + len(cluster.mem_ids) * 13)
    anchors = _anchor_terms(cluster.texts, topn=8)
    if not anchors:
        anchors = [_topic_hint(cluster.texts)]

    object_templates = [
        "我们之前聊过那个{a}，名字我有点卡住了，你帮我对齐一下？",
        "核心是不是在说{a}背后的推理链？我怕我记偏了。",
        "{a}是不是跟某个近似概念混了？你帮我定位下真正对象。",
        "我想把{a}这个主题重新捋一遍，先从定义开始行吗？",
        "上次关于{a}我们聊到哪一步了，你接着往下讲？",
    ]
    stance_templates = [
        "{a}这套说法听着顺，但我总觉得某个环节不踏实。",
        "拿常见反驳去打{a}，我感觉还是不够硬，你怎么看？",
        "我对{a}的立场有点摇摆了，可能我前面想岔了。",
        "别顺着我，直接指出我在{a}上的关键漏洞。",
        "我现在对{a}的态度和之前不一致，你帮我对照下。",
    ]
    intent_templates = [
        "围绕{a}你先给我一个分步骤解释，不急着下结论。",
        "请基于{a}给我可执行方案，不要只讲抽象概念。",
        "你站在反方围绕{a}和我辩一轮，重点打我的薄弱点。",
        "我想把{a}整理成几条可选路径，你先给框架。",
        "如果继续深挖{a}，下一步最值得问的问题是什么？",
    ]
    bank = {"object": object_templates, "stance": stance_templates, "intent": intent_templates}[layer]

    generated: List[str] = []
    rounds = max(2, math.ceil(k / max(1, len(bank))))
    for _ in range(rounds + 1):
        local_bank = list(bank)
        rng.shuffle(local_bank)
        local_anchors = list(anchors)
        rng.shuffle(local_anchors)
        for tpl in local_bank:
            a = local_anchors[len(generated) % len(local_anchors)]
            generated.append(tpl.format(a=a))
            if len(generated) >= k * 3:
                break
        if len(generated) >= k * 3:
            break

    cleaned = _normalize_queries(generated, k * 3)
    enforced = _enforce_anchor_and_style(cleaned, k=k, anchors=anchors)
    if len(enforced) >= k:
        return enforced[:k]

    # Last resort: deterministic short forms with explicit anchors.
    topup: List[str] = list(enforced)
    for idx in range(k * 2):
        a = anchors[idx % len(anchors)]
        extra = {
            "object": f"我们之前提到过{a}，你帮我把关键点再说一遍？",
            "stance": f"关于{a}我现在立场有点变了，你帮我挑最容易错的点。",
            "intent": f"围绕{a}我下一步该怎么问，才能更快推进？",
        }[layer]
        topup.append(extra)
    topup = _normalize_queries(topup, k * 2)
    topup = _enforce_anchor_and_style(topup, k=k, anchors=anchors)
    return topup[:k]


def _top_keywords(texts: Sequence[str], *, topn: int) -> List[str]:
    token_re = re.compile(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9_]{2,}")
    stop = {
        "这个",
        "那个",
        "我们",
        "你们",
        "然后",
        "感觉",
        "就是",
        "因为",
        "所以",
        "哎呀",
        "欧耶",
        "哈哈",
        "不不不",
        "等等",
        "呃",
        "额",
        "嗯",
        "啊",
        "还有",
        "可以",
        "怎么",
        "什么",
    }
    freq: Dict[str, int] = {}
    for text in texts:
        for tok in token_re.findall(text.lower()):
            if tok in stop:
                continue
            if re.fullmatch(r"[a-z0-9_]+", tok):
                if len(tok) < 4:
                    continue
                if tok in EN_STOPWORDS:
                    continue
            freq[tok] = freq.get(tok, 0) + 1
    ranked = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    return [tok for tok, _ in ranked[:topn]]


def _pick_snippets(texts: Sequence[str], *, max_count: int) -> List[str]:
    clean = [re.sub(r"\s+", " ", t).strip() for t in texts if t.strip()]
    if len(clean) <= max_count:
        return clean
    head = clean[: max_count // 2]
    tail = clean[-(max_count - len(head)) :]
    out = head + tail
    deduped: List[str] = []
    seen = set()
    for x in out:
        if x in seen:
            continue
        seen.add(x)
        deduped.append(x)
    return deduped[:max_count]


def _topic_hint(texts: Sequence[str]) -> str:
    kws = _anchor_terms(texts, topn=3)
    if kws:
        return " / ".join(kws[:2])
    if texts:
        text = texts[0].strip()
        return text[:16] + ("..." if len(text) > 16 else "")
    return "这件事"


def _anchor_terms(texts: Sequence[str], *, topn: int) -> List[str]:
    chinese_re = re.compile(r"[\u4e00-\u9fff]{2,8}")
    latin_re = re.compile(r"[A-Za-z0-9_]{4,}")
    freq_cn: Dict[str, int] = {}
    freq_en: Dict[str, int] = {}

    for text in texts:
        lower = text.lower()
        for tok in chinese_re.findall(lower):
            if tok in GENERIC_OBJECT_WORDS:
                continue
            if tok in CN_ANCHOR_STOPWORDS:
                continue
            if re.fullmatch(r"(.)\1+", tok):
                continue
            if _is_discourse_marker(tok):
                continue
            freq_cn[tok] = freq_cn.get(tok, 0) + 1
        for tok in latin_re.findall(lower):
            if tok in EN_STOPWORDS:
                continue
            freq_en[tok] = freq_en.get(tok, 0) + 1

    ranked_cn = [tok for tok, _ in sorted(freq_cn.items(), key=lambda x: (-x[1], x[0]))]
    ranked_en = [tok for tok, _ in sorted(freq_en.items(), key=lambda x: (-x[1], x[0]))]

    out: List[str] = []
    for kw in ranked_cn + ranked_en:
        if kw in GENERIC_OBJECT_WORDS:
            continue
        if len(kw) < 2:
            continue
        out.append(kw)
        if len(out) >= topn:
            break

    if not out:
        kws = _top_keywords(texts, topn=max(12, topn))
        for kw in kws:
            if kw in GENERIC_OBJECT_WORDS:
                continue
            if _is_discourse_marker(kw):
                continue
            out.append(kw)
            if len(out) >= topn:
                break
    return out


def _is_discourse_marker(tok: str) -> bool:
    if tok in CN_ANCHOR_STOPWORDS:
        return True
    if re.match(r"^(我|你|他|她|它).{0,6}(说|觉得|感觉|意思|认为)", tok):
        return True
    if tok.startswith(("还是那句", "一大堆", "一个有比较好的记", "一个神经元", "一体两面", "一种理论")):
        return True
    return False
