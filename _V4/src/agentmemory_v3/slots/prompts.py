SLOT_FIELDS = (
    "raw_brief",
    "event",
    "intent",
    "entities",
    "status",
    "emotion",
    "context",
    "impact",
)


IE_SYSTEM_PROMPT = """你是信息抽取器（IE），只允许基于输入文本抽取信息。禁止编造、禁止补全常识、禁止推断未出现的事实。输出必须是严格 JSON，字段齐全。"""


IE_USER_PROMPT_TEMPLATE = """输入包含三段文本：prev_raw、raw、next_raw。
请输出严格 JSON，字段必须齐全，结构如下：
{{
  "raw_brief": "",
  "event": "",
  "intent": "",
  "entities": [],
  "status": "",
  "emotion": "",
  "context": "",
  "impact": "",
  "evidence": {{
    "raw_brief": "",
    "event": "",
    "intent": "",
    "entities": "",
    "status": "",
    "emotion": "",
    "context": "",
    "impact": ""
  }},
  "warnings": [],
  "confidence": 0.0
}}

字段边界：
- raw_brief：一句话“这条在说什么”，最泛化，但不能包含具体实体列表。
- event：发生了什么动作/事件，必须包含动词。
- intent：当事人的目标/请求。
- entities：只列名词实体，不写句子。
- status：结果/现状，尽量可判定。
- emotion：情绪标签 + 极短原因短语。
- context：prev_raw 提供的背景条件或前置状态。
- impact：next_raw 表现的后续变化或后果。

硬规则：
- 所有 text 字段必须短：<=24 个中文字符或 <=16 个英文词。
- entities 每个 item <=12 个中文字符或 <=8 个英文词，最多 6 个。
- 8 个字段必须差异化，不能复述同一句话。
- entities 只能是名词列表；event 必须包含动词；status 必须是结果/状态；intent 必须是目标/请求。
- context 只能引用 prev_raw；impact 只能引用 next_raw。
- 如果 prev_raw / next_raw 与 raw 不相关或为空，context / impact 留空，并在 warnings 写原因。
- evidence 必须是从对应文本中直接复制的片段，每条 <=35 字。
- 如果不确定，就留空，不要猜。
- confidence 为 0~1 之间的小数。

prev_raw:
{prev_raw}

raw:
{raw}

next_raw:
{next_raw}
"""
