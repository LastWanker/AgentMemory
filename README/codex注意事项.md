# codex注意事项

## 1. 搜索命令不要全盘乱扫
- 优先用 `rg`，不要递归 `Get-ChildItem -Recurse`。
- 尽量限制搜索范围到项目子目录：
  - `scripts`
  - `src`
  - `README*`
- 推荐命令：
```powershell
rg -n -S "关键词1|关键词2" scripts src README*
```

## 2. 避免卡死的 rg 写法
- 对大仓库搜索时，加排除规则：
```powershell
rg -n -S --glob '!data/**' --glob '!.venv/**' "关键词" .
```
- 若只查脚本路径相关，直接指定目录，不要 `.` 起手。

## 3. 执行前先确认当前目录
- 必须先确认在仓库根目录：
```powershell
pwd
```
- 预期目录：`F:\GitHub\AgentMemory`

## 4. 先看结构，再跑重命令
- 先用轻命令确认文件存在：
```powershell
rg --files scripts src README
```
- 再跑精确搜索，避免一次打满输出。

## 5. 本项目当前脚本入口约定
- 聊天记忆处理：`scripts/memory_processer/build_chat_memory.py`
- memory_indexer 主入口：`scripts/memory_indexer/`
- 旧路径脚本多数是兼容 wrapper，可用但不建议作为长期入口。

## 6. 调试策略（快）
- 先 `--help` 验证脚本可启动。
- 先跑小范围参数，再跑全量数据。
- 先确认输出文件生成，再做下一步训练/评测。

## 7. 聊天记忆处理推荐命令
- 默认配置跑法：
```powershell
python scripts/memory_processer/build_chat_memory.py --config configs/chat_memory.yaml
```
- 生成 eval/query（已拆分独立脚本）：
```powershell
python scripts/memory_processer/build_chat_queries.py --config configs/chat_memory.yaml
```
- 生成 LLM 补充 query（可并发，输出独立文件）：
```powershell
python scripts/memory_processer/build_chat_supplemental_queries.py --config configs/chat_memory.yaml
```
- 合并 identity + 补充 query（不覆盖 identity）：
```powershell
python scripts/memory_processer/build_chat_queries.py --config configs/chat_memory.yaml --supplemental-queries data/Processed/eval_chat_supplemental.jsonl
```
- 覆盖配置跑法（示例）：
```powershell
python scripts/memory_processer/build_chat_memory.py --config configs/chat_memory.yaml --segmentation-mode adaptive --no-cross-session-merge
```
