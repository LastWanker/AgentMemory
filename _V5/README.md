# AgentMemory V5 Workspace

`_V5/` 是当前主线工作区，负责维护可运行的记忆检索链路、评测脚本与本地服务能力。

## 简介

V5 旨在解决大模型多轮对话中的短记忆问题，构建跨会话长期记忆能力。系统采用双通路检索：`coarse` 主通路保障高精度召回，`association` 联想通路补足语义跳跃场景，并通过 activation trace 提供可解释检索路径。

当前离线结果快照：在 3728 条 memory、489 条 query 上，coarse 通路 Recall@N 为 0.83；association 通路在抽象概念场景带来补充召回。

## 目标与范围

V5 当前聚焦于稳定的 coarse-only 主链：

- Dense：`intfloat/multilingual-e5-small`
- Lexical：BM25
- 扩召：cluster 连带扩召
- 交互：本地 retriever + 本地 chat 应用

暂不将 `_V3/`、`_V4/` 作为默认开发入口。

## 工作区结构

- `src/agentmemory_v3/`
  - `retrieval/`：coarse 检索主链
  - `evaluation/`：离线评测
  - `serving/`：retriever API
  - `association/`：联想图谱能力
  - `suppressor/`、`mem_network_suppressor/`：候选抑制能力
- `scripts/`
  - `ingest/`：数据准备
  - `runtime/`：运行链路
  - `quality/`：诊断与评测
  - `suppressor/`：抑制器专项脚本
- `configs/default.yaml`：默认配置
- `chat/`：V5 聊天应用

## 标准运行流程

### 1) 准备数据与索引

```bat
cmd /c ".venv\Scripts\python.exe _V5\scripts\runtime_full_pipeline.py --mode prepare --config _V5\configs\default.yaml"
```

### 2) 验证链路健康

```bat
cmd /c ".venv\Scripts\python.exe _V5\scripts\doctor_data_chain.py --config _V5\configs\default.yaml"
cmd /c ".venv\Scripts\python.exe _V5\scripts\smoke_coarse.py --config _V5\configs\default.yaml"
```

### 3) 执行离线评测

```bat
cmd /c ".venv\Scripts\python.exe _V5\scripts\eval_offline.py --config _V5\configs\default.yaml --split test"
```

### 4) 启动在线服务

```bat
cmd /c ".venv\Scripts\python.exe _V5\scripts\serve_retriever.py --config _V5\configs\default.yaml"
cmd /c ".venv\Scripts\python.exe _V5\chat\app.py"
```

## 关键资产约定

- `data/V5/processed/{memory,query,cluster}.jsonl`
- `data/V5/VectorCacheV5/users/{manifest,memory_ids,query_ids,memory_coarse,query_coarse}`
- `data/V5/indexes/{dense_artifact.pkl,dense_matrix.npy,bm25.pkl}`
- `data/V5/exports/chat_memory_bundle.jsonl`
- `_V5/runs/eval_offline/*`

## 配置与环境

- 默认配置文件：`_V5/configs/default.yaml`
- E5 默认策略：`local_files_only: true`、`offline: true`
- DeepSeek 密钥默认读取：`data/_secrets/deepseek.env`

## 参考文档

- `readme/实现架构.md`
- `readme/阶段性计划书_V5启动阶段.md`
- `readme/窗口聊天接续/窗口交接_V5启动前说明.md`
