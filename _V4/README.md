# AgentMemory V4

这是一个与母项目隔离的新工作区。目标不是停留在 demo，而是把 `_V4` 做成一套可训练、可评测、可本地部署的独立记忆检索链路。

建议先看：

- `readme/问题分析.md`
- `readme/计划书.md`
- `readme/阶段性计划书_主链补全阶段.md`
- `readme/阶段性计划书_E5迁移阶段.md`
- `readme/实现架构.md`

## 当前推荐命令

### 0. 体检

```bat
cmd /c "python _V4\scripts\doctor_data_chain.py"
```

### 1. 全链准备

```bat
cmd /c "python _V4\scripts\runtime_full_pipeline.py --mode prepare"
```

单独补 query slot：

```bat
cmd /c "python _V4\scripts\build_query_slots.py --workers 4 --flush-every 20"
```

单独补 memory slot：

```bat
cmd /c "python _V4\scripts\build_slots.py --workers 4 --flush-every 20"
```

如果还不想全量跑槽抽取，可以先临时 smoke：

```bat
cmd /c "python _V4\scripts\runtime_full_pipeline.py --mode prepare --slots-limit 100 --max-queries 300"
```

### 2. 训练

```bat
cmd /c "python _V4\scripts\train_reranker.py"
```

### 3. 离线评测

```bat
cmd /c "python _V4\scripts\eval_offline.py --split test"
```

### 4. 本地聊天

```bat
cmd /c "python _chat\app.py"
```

说明：

- `data/` 允许与母项目共享，但 V4 产物统一写入 `data/V4/...`
- `_V4/src`、`_V4/scripts`、`_V4/configs`、`_V4/runs` 与母项目隔离
- `_chat/` 是并列的本地聊天器，会优先读取训练后的 V4 检索器
- 当前 reranker 主线已切到 `slot_bipartite_8x8`
- 在全量 slot 尚未跑完前，在线推理会采用“bipartite 分数 + 特征分兜底”的过渡形态
