# AgentMemory V5

这是一个与母项目、`_V3`、`_V4` 并列隔离的新工作区。V5 当前只保留一条可运行主链：

- `E5-small dense`
- `BM25`
- `cluster 连带扩召`
- `chat coarse-only 接入`

先看这两份：

- `readme/窗口交接_V5启动前说明.md`
- `readme/阶段性计划书_V5启动阶段.md`

实现细节看：

- `readme/实现架构.md`

## 当前口径

- V5 默认路径已经切到 `_V5/...` 与 `data/V5/...`
- `data/V5/VectorCacheV5/` 是 V5 正式缓存目录
- `_V5/chat/` 是 V5 专用聊天器，默认固定使用 coarse-only
- V5 当前最小资产集合只保留 coarse 路径必需项：
  - `data/V5/processed/{memory,query,cluster}.jsonl`
  - `data/V5/VectorCacheV5/users/{manifest,memory_ids,query_ids,memory_coarse,query_coarse}`
  - `data/V5/indexes/{dense_artifact,dense_matrix,bm25}`
  - `data/V5/exports/chat_memory_bundle.jsonl`
- `_V5/chat/data/conversations/` 默认保持空目录，避免把历史测试会话当成现状
- `_V5/runs/` 只作为 V5 自己的临时输出目录使用

## 命令习惯

优先用 `cmd /c`，不要在 PowerShell 里直接写复杂管道。大文件搜索优先 `rg`，不要一次性 `Get-Content` 整个大文件。

例子：

```bat
cmd /c "rg -n coarse _V5"
cmd /c ".venv\Scripts\python.exe _V5\scripts\doctor_data_chain.py"
cmd /c ".venv\Scripts\python.exe _V5\scripts\smoke_coarse.py"
cmd /c ".venv\Scripts\python.exe _V5\scripts\runtime_full_pipeline.py --mode prepare --config _V5\configs\default.yaml"
```

## 现阶段不宜做的事

- 不要一上来继续大改 `_V4`
- 不要重新引入已经删除的旧链路数据或代码
- 不要忘记 E5 默认 `local_files_only + offline`
- 不要把历史测试输出留在 `_V5/chat/data/conversations` 里
