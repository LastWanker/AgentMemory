# AgentMemory 文档总览（对齐当前代码）

更新时间：2026-02-26

## 1. 项目定位（当前实现）
- 目标是可复现的工程链路，而不是抽象记忆理论验证：
  - 聊天导出数据 -> 结构化 memory + eval/followup query
  - 统一候选池口径下的检索路由评测
  - pairwise/listwise tiny reranker 训练与对照

## 2. 代码结构
- `src/chat_memory_processor/`
  - `extractor.py`：从 DeepSeek conversations mapping 抽取 user fragments
  - `cleaning.py`：清洗与排除（空文本、超长、代码块、重复 dump 等）
  - `segmentation.py`：会话内 local cluster（adaptive/fixed）
  - `clustering.py`：跨会话 optional global cluster 合并
  - `querygen.py`：identity query + supplemental query 合并
  - `llm_querygen.py`：object/stance/intent 三层 supplemental（LLM + fallback）
- `src/memory_indexer/`
  - `ingest.py`：memory payload -> `MemoryItem`（`sentence_window` 切块）
  - `pipeline.py`：建库与检索装配（含缓存读写）
  - `index.py`：`CoarseIndex` + `LexicalIndex`
  - `retriever.py`：候选构建、特征聚合、路由打分、指标落点
  - `learned_scorer.py`：PyTorch tiny reranker、batch scorer、cardinality head
  - `encoder/`：`SimpleHashEncoder`、`HFSentenceEncoder`、`E5TokenEncoder`

## 3. 数据产物（当前约定）
- 聊天处理输出（`scripts.memory_processer.build_chat_memory`）
  - `data/Processed/user_turns_raw.jsonl`
  - `data/Processed/user_turns_dedup.jsonl`
  - `data/Processed/memory_chat.jsonl`
- query 输出（`scripts.memory_processer.build_chat_queries`）
  - `data/Processed/eval_chat.jsonl`
  - `data/Processed/eval_chat_followup.jsonl`
- supplemental 输出（`scripts.memory_processer.build_chat_supplemental_queries`）
  - 默认 `data/Processed/eval_chat_supplemental.api.jsonl`
- 合并数据集（`scripts.memory_indexer.build_merged_dataset`）
  - `data/Processed/memory_<dataset>.jsonl`
  - `data/Processed/eval_<dataset>.jsonl`
  - 可选同步 `data/Processed/groups/<dataset>/memory.jsonl|eval.jsonl`

## 4. 评测与训练口径（当前关键点）
- `eval_router.py`
  - 支持 `soft/half_hard/hard`、ablation 组、bootstrap、run-dir 落盘
  - `candidate_mode` 会被强制收敛到 `coarse`
  - 指标包括：`RetrievalRecall@N`、`Recall@k`、`Recall@P`、`CoarseRankRecall@k`、`MRR`、`Top1` 及路由稳定性/耗时指标
- `train_reranker.py`
  - 支持 `pairwise/listwise`、`random/ranked` 负例、可选 cardinality head
  - 同样强制 `candidate_mode=coarse`
  - `positives/expected_mem_ids` 均可读取，旧字段 `candidates/hard_negatives` 仅兼容提示
  - 训练正例匹配支持 `base_mem_id -> base_mem_id#cN` 切块映射
- registry/caches
  - `runs/registry.csv` 记录训练与评测 run 元信息
  - `data/VectorCache/` 提供 query/memory 缓存；支持签名模式和 alias 模式

## 5. 推荐执行链路（当前）
1. 生成聊天 memory：
```bash
python -m scripts.memory_processer.build_chat_memory --config configs/chat_memory.yaml
```
2. 生成 supplemental 并重建 chat eval/followup：
```bash
python -m scripts.memory_processer.runtime.build_chat_supplemental_eval --config configs/chat_memory.yaml
```
3. 合并 followup + chat：
```bash
python -m scripts.memory_indexer.build_merged_dataset --dataset followup_plus_chat
```
4. 训练 pairwise/listwise：
```bash
python -m scripts.memory_indexer.train_pairwise_reranker --dataset followup_plus_chat
python -m scripts.memory_indexer.train_listwise_reranker --dataset followup_plus_chat
```
5. 评测：
```bash
python -m scripts.memory_indexer.eval_router --dataset followup_plus_chat --use-learned-scorer
```

## 6. 进一步阅读
- `README/伪代码框架.md`：按当前实现组织的伪代码骨架
- `README/接口设计.md`：JSONL 字段、Python 接口、脚本契约
- `README/开发日志.md`：按日期记录变更与口径修正
