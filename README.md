# AgentMemory（工程现状快照）

更新时间：2026-02-26

这是一个“聊天记忆构建 + 检索路由评测/训练”的工程化仓库，当前重点是：
- 把聊天导出数据整理成 `memory_*.jsonl / eval_*.jsonl`
- 在统一候选池口径下做路由检索评测
- 用 pairwise/listwise 训练 tiny reranker 并沉淀 run 产物

## 当前主模块
- `src/chat_memory_processor/`：聊天数据抽取、清洗、会话切分、跨会话聚类、identity/supplemental query 构建
- `src/memory_indexer/`：记忆切块、向量化建库、粗召回、词法证据、语义精排、路由融合
- `scripts/memory_processer/`：聊天侧数据生产与 runtime 封装
- `scripts/memory_indexer/`：评测、训练、数据集合并、run 管理脚本

## 当前推荐入口（命令行）
- 生成聊天 memory：
```bash
python -m scripts.memory_processer.build_chat_memory --config configs/chat_memory.yaml
```
- 生成 supplemental 并重建 chat eval/followup：
```bash
python -m scripts.memory_processer.runtime.build_chat_supplemental_eval --config configs/chat_memory.yaml
```
- 合并 followup + chat：
```bash
python -m scripts.memory_indexer.build_merged_dataset --dataset followup_plus_chat
```
- 训练 reranker（wrapper）：
```bash
python -m scripts.memory_indexer.train_pairwise_reranker --config configs/default.yaml --dataset followup_plus_chat
python -m scripts.memory_indexer.train_listwise_reranker --config configs/default.yaml --dataset followup_plus_chat
```
- 评测路由（run-dir、缓存、指标落盘）：
```bash
python -m scripts.memory_indexer.eval_router --config configs/default.yaml --dataset followup_plus_chat --use-learned-scorer
```

## 运行口径（当前已固定）
- `eval_router/train_reranker` 会将非 `coarse` 的 `candidate_mode` 自动纠正为 `coarse`。
- query 中旧字段 `candidates/hard_negatives` 只做兼容提示，不进入候选池逻辑。
- 建库时会按 `sentence_window` 对 memory 文本切块，运行态 `mem_id` 可能变为 `原mem_id#cN`。
- 训练侧正例匹配支持 `expected_mem_id` 前缀（可命中 `mem_id#cN`）。
- 缓存支持两种模式：
  - 签名缓存（默认；query 签名含数据指纹与采样参数）
  - 固定别名缓存（`--cache-alias`，如 `users` / `users_simple`）

## 文档索引
- 总览与执行链路：`README/README.md`
- 伪代码骨架：`README/伪代码框架.md`
- 接口与数据契约：`README/接口设计.md`
- 一键 runtime：`README/runtime_一键入口.md`
- 最新变更记录：`README/开发日志.md`
