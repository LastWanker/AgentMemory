# codex注意事项

## 0. 先看这个再动手
- 当前用户环境是 Windows + PyCharm，且用户明确要求命令统一走 `cmd`。
- 当前工作偏“开发可用性与可控提速”，不是论文式全量评测。
- 涉及数据与缓存写入时，默认先判断是否会污染用户常用缓存。

## 1. 终端硬规则（已取代旧 PowerShell 习惯）
- 一律使用 `cmd /c <command>`。
- 不使用 PowerShell 语法，不使用 bash/WSL/Linux 命令。
- 旧版“Get-Content 防卡”条目不再作为主规则；现在以 `cmd` 直跑为准。

## 2. 命令防卡与防误判
- 禁止无范围 `git diff`（尤其对 `data/Processed/*.jsonl`）。
- 先用 `git status --short` 看改动面，再按文件小范围 diff。
- `rg` 查询在 `cmd` 下优先用多 `-e`，避免复杂引号/管道误解析：
  - 推荐：`rg -n -e foo -e bar path`
  - 不推荐：单条里混入 `|` 和复杂转义。

## 3. 缓存规则（现在是用户手动管理优先）
- 正式缓存名：
  - `data/VectorCache/memory_cache_users.jsonl`
  - `data/VectorCache/eval_cache_users.jsonl`
- 默认参数：
  - `--cache-alias users`
  - `--no-cache-signature`
- `simple` 后端默认用 `users_simple`，不要覆盖 HF 的 `users`。
- 烟测请用独立 alias（例如 `users_smoke`），不要污染正式缓存。

## 4. 评测默认与性能参数（当前可用）
- quick 入口：`scripts/memory_indexer/runtime/eval_followup_plus_chat_quick.py`
- 当前默认：
  - `top_n=30`
  - `top_k=5`
  - `policy=half_hard`
  - `ablation=mix(auto)`
  - `bootstrap=0`
  - `eval_workers=4`
  - `torch_num_threads=2`
  - `scorer_batch_size=512`
- 看日志时优先抓三行：
  - `[cache] ... 跳过 HF 编码器初始化`
  - `[timing] scoring_elapsed_s=...`
  - `[runtime] elapsed_s=...`

## 5. 本轮已知坑（经常导致“看起来像卡死”）
- `cmd` 下正则/引号写法不对，会导致命令直接被 shell 误拆，不是代码问题。
- 先跑了 `--max-eval-queries 200` 会把 `eval_cache_users.jsonl` 改成 200 条；之后跑默认 1000 会触发重建，属正常行为。
- 长日志里 `Loading weights ...` 是第三方输出噪声，不是业务卡死。

## 6. 失败恢复流程（必须执行）
1. `cmd /c git status --short`
2. 确认最近 run 目录的 `eval.log.txt` 是否完整落盘
3. 先 `py_compile` 再继续长任务
4. 若是缓存问题，优先切换 alias，不要直接删正式 users 缓存

## 7. 语义边界（别再混）
- `P`：positives（正例集合）
- `C`：coarse recall top_n（检索候选池）
- `R`：最终排序结果 top_k
- 评测不再用 query 行内 `candidates/hard_negatives` 参与候选池构建。

## 8. 数据与敏感信息
- secrets 目录：`data/_secrets/`（已忽略提交）。
- key 读取失败先检查编码/BOM，再查值本身。

## 9. 开发过程要求
- 大改前先定位最小改动点，优先小步验证（`py_compile` + 小样本）。
- 若用户说“先讨论”，不要直接开长跑。
- 每次关键变更都要同步追加 `README/开发日志.md`。
