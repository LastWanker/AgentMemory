# AgentMemory 项目现状审计（2026-02-17）

这份文档回答你这轮提出的核心问题：
- 缓存会不会互相冲突
- 现在到底在用哪些文件（尤其 `VectorCache`）
- `pairwise` 是不是“原本那套”
- 现在的 memory/eval 数据是否满足当前目标
- 哪些是临时补丁、哪些是主链路
- 是否需要整理，怎么整理

---

## 1. 先给结论（TL;DR）

1. 你“项目很乱”的感受是准确的。  
不是代码完全没结构，而是“实验入口多、默认参数多、缓存策略不统一”导致认知负担很高。

2. 存在**真实冲突风险**，最关键的是 `eval_cache_*.jsonl`。  
`eval_router.py` 只要发现 query cache 存在就直接读，不校验 `encoder_id/strategy`。  
这会导致切换 backend（`hf`/`simple`）时可能复用旧缓存，结果“能跑但不可信”。

3. `pairwise` 确实是当前训练主默认。  
`train_reranker.py` 默认 `--loss-type pairwise --neg-strategy random`，这就是“原本那种”。

4. 你看到的 `{"query_text": "...", "expected_mem_ids": ["n040"]}` 是**评测标注格式**，不是训练格式。  
它天然是单正例（或少量正例）评测；多正例训练用的是 `positives` 字段（训练脚本兼容两者）。

5. “处理后的文件”确实不显眼。  
memory/query 的“处理后结果”主要在 `data/VectorCache/*.jsonl`（缓存），不是独立规范产物目录。  
这就是你觉得“过程不透明”的核心原因之一。

6. 现在最该做的是一次工程整理，而不是继续加功能。  
建议先做“配置统一 + 缓存隔离 + 产物统一落盘”三件事。

---

## 2. 当前脚本职责图（按主链路）

### 2.1 主链路脚本

- `scripts/eval_router.py`
  - 读取 `data/Memory_Eval/memory_{dataset}.jsonl` 与 `eval_{dataset}.jsonl`
  - 构建/加载 `data/VectorCache/memory_cache_{dataset}.jsonl`
  - 构建/加载 `data/VectorCache/eval_cache_{dataset}.jsonl`
  - 跑检索评测（Recall/MRR/Top1 + route metrics）

- `scripts/train_reranker.py`
  - 读取同一套 memory/eval 数据
  - 使用 `Retriever + FieldScorer` 先取候选，再构造 pairwise/listwise 训练样本
  - 输出权重（默认 `data/ModelWeights/tiny_reranker.pt`，可 `--save-path` 覆盖）

- `scripts/run_ablation_matrix.py`（新增）
  - 串联训练与评测
  - 结果统一落盘到 `runs/<timestamp>/...`（这一点是好方向）

### 2.2 辅助脚本

- `scripts/data_collector.py`：采集交互日志
- `scripts/generate_training_samples.py`：从交互日志生成 `positives/hard_negatives`
- `scripts/regression_learned_scorer.py`：最小回归验证
- `scripts/demo.py`：演示用途

---

## 3. 缓存与冲突分析（重点）

## 3.1 `memory_cache_{dataset}.jsonl`（相对安全）

位置：`data/VectorCache/memory_cache_normal.jsonl` 等  
由 `build_memory_index(..., cache_path=...)` 使用。

安全点：
- 读取时会过滤 `encoder_id + strategy`
- 还校验 `text` 与 `coarse_role=passage`

风险：
- 同文件长期存多版本 payload，最终会被“当前 items”覆盖重写，过程不够透明。

## 3.2 `eval_cache_{dataset}.jsonl`（高风险）

位置：`data/VectorCache/eval_cache_normal.jsonl` 等  
由 `eval_router.py` 的 `write_cached_queries/iter_cached_queries` 使用。

问题点：
- 文件存在即直接读
- **不校验** cache 内 `encoder_id/strategy` 与当前运行配置是否一致

后果：
- 比如之前用 HF 写过缓存，后续用 simple backend 时仍可能直接读取旧 query 向量
- 程序未必报错，但评测结果不可信（静默污染）

## 3.3 权重文件冲突

- 默认权重路径是 `data/ModelWeights/tiny_reranker.pt`
- 多次训练会覆盖同名文件
- `run_ablation_matrix.py` 已规避：每个 seed 单独写到 `runs/<timestamp>/<config>/seed_x/tiny_reranker.pt`

结论：
- `runs/` 新机制是正确方向，但只覆盖了 ablation 链路，其他脚本还在旧路径体系。

---

## 4. 你问的几个关键问题

## 4.1 “pairwise 是原本方法吗？”

是。  
`train_reranker.py` 默认：
- `--loss-type pairwise`
- `--neg-strategy random`

所以你可以把“pairwise+random”视为当前训练基线。

## 4.2 “现在的记忆满足需求吗？”

要分两层看：

1) 对“评测闭环可跑”来说：满足。  
`memory_normal.jsonl + eval_normal.jsonl` 是一套可运行基准集。

2) 对“多正例、更真实路由学习”来说：不够。  
目前 `eval_normal.jsonl` 主要是 `expected_mem_ids`（常见单正例），更像检索正确性验证，不是复杂训练数据。

## 4.3 “有脚本生成新 memory 吗？”

当前没有成熟“原始语料 -> 标准 memory_{dataset}.jsonl”专用脚本。  
现有是：
- `build_memory_items()` 在运行时做切块/元信息补全（内存中处理）
- `data_collector.py + generate_training_samples.py` 更偏生成训练 query 样本（`positives`），不是生成新 memory 库。

## 4.4 “为什么没见到处理后文件？”

因为“处理后的东西”主要进了 cache：
- `data/VectorCache/memory_cache_*.jsonl`
- `data/VectorCache/eval_cache_*.jsonl`

这不是“清晰的数据版本产物目录”，所以你会感觉链路不透明。

---

## 5. 哪些是主干，哪些是权宜之计

### 主干（现在依赖它们）

- `src/memory_indexer/pipeline.py`（建库+检索入口）
- `src/memory_indexer/retriever.py`（候选/打分/路由）
- `scripts/eval_router.py`（主评测）
- `scripts/train_reranker.py`（主训练）

### 权宜之计/实验性较强

- `scripts/demo.py`（演示用，不是严谨实验入口）
- `scripts/regression_learned_scorer.py`（回归脚本）
- `scripts/data_collector.py`, `scripts/generate_training_samples.py`（新加的数据闭环雏形）
- `scripts/run_ablation_matrix.py`（是整理方向，但目前仍依赖旧脚本默认行为）

---

## 6. 为什么你会感觉“配置散、脑子乱”

这是结构性问题，不是你的问题。当前有三个“配置中心”并存：

1) `eval_router.py` 命令参数（很多）
2) `train_reranker.py` 命令参数（很多）
3) `run_ablation_matrix.py` 顶部常量 + CLI

再叠加：
- cache 路径默认复用
- weights 路径在不同脚本下策略不一致
- backend（hf/simple）切换会影响缓存有效性

自然会造成“我以为在 A 配置，实际混了 B”的风险。

---

## 7. 建议的整理方案（按优先级）

## P0（先做，1-2 次迭代）

1. 统一缓存隔离策略  
- 把 `eval_cache` 改为按签名命名，至少包含：`dataset + encoder_backend + encoder_id + strategy + chunk_strategy`  
- 或强制 `eval_router.py` 默认 `--rebuild-cache`（更暴力但可靠）

2. 统一输出根目录  
- 所有训练/评测脚本都支持 `--run-dir`，默认写 `runs/<timestamp>/...`
- `data/ModelWeights` 只保留“手工固定基线权重”，不再当实验临时目录

3. 建立“单一配置文件”  
- 例如 `configs/ablation_matrix.yaml`（或 json）
- 训练/评测都从这个配置展开，CLI 只做覆盖

## P1（随后做）

4. 明确数据分层目录  
- `data/raw/` 原始输入
- `data/processed/` 结构化 memory/eval/train-samples
- `data/cache/` 可重建缓存
- `runs/` 实验产物（日志、指标、权重、summary）

5. 统一字段规范  
- 评测：允许 `expected_mem_ids` 多值
- 训练：统一 `positives/hard_negatives/candidates`
- 文档写明兼容关系与优先级

## P2（稳定后）

6. 把 `run_ablation_matrix.py` 升级为“唯一实验入口”  
- 其他脚本变为内部子命令或库函数
- 避免多入口长期分叉

---

## 8. 对你当前工作的直接建议

如果你现在就要跑可信实验，建议暂时这样做：

1. 固定 backend（先 `simple`）  
2. 每次切配置先清 cache（或使用独立 cache 文件）  
3. 权重全部写到 `runs/` 下，不覆盖 `data/ModelWeights/tiny_reranker.pt`  
4. 每次实验记录完整命令行和 git commit（至少在 `run.meta.json`）

---

## 9. 最后一句（判断）

项目并非“没有条理”，而是“条理分散在多个脚本里，没有统一调度层”。  
你现在的直觉是对的：继续堆功能只会更乱；先做一轮整理，会让后续开发效率直接翻倍。

