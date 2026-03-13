# AgentMemory V5

面向聊天记忆检索的 V5 工程主线。当前仓库以 `_V5/` 为默认开发与运行入口，`_V3/`、`_V4/` 仅作为历史版本留档。

## 项目简介

AgentMemory V5 面向大模型多轮对话中的“短记忆”问题，构建可持续增量更新的长期记忆模块，使系统能够跨会话理解并召回用户历史语义，形成完整的个人记忆数据链路。

系统采用双通路检索架构：`coarse` 主通路（E5 dense + BM25 + cluster 连带）负责高精度召回；`association` 联想通路基于语义关联扩散，补足抽象概念和语义跳跃场景。联想能力通过分层概念图激活机制实现，以节点点亮和 bridge 扩散完成检索，同时输出 activation trace 提升可解释性。

工程侧基于 FastAPI 提供检索 API，形成“检索 -> 组装 -> 生成”的 RAG 工作流，并配套反馈学习机制与 Web 可视化。数据侧支持聊天记录导入、清洗、切分、聚类与结构化，统一沉淀为 `memory/query/cluster` 数据契约。

## 指标快照

- 数据规模：3728 条 memory
- 测试规模：489 条 query
- coarse 通路：Recall@N = 0.83
- association 通路：在抽象概念与语义跳跃场景补充召回，增强“抽象输入 -> 具体记忆”联想检索能力

## 项目定位

V5 的目标是维护一条稳定、可复现、可评测的检索主链：

- 检索主链：`E5 dense + BM25 + cluster 连带扩召（coarse-only）`
- 数据主链：外部数据规范化后沉淀到 `data/V5/processed/`
- 评测主链：离线评测结果统一落盘到 `_V5/runs/eval_offline/`
- 服务主链：本地 retriever 服务 + V5 聊天前端

## V5 架构概览

1. 数据准备：`build_dataset.py` 将标准输入转为 `memory/query/cluster` 三份 V5 数据集。
2. 表征与索引：`build_e5_cache.py` + `build_hybrid_index.py` 生成向量缓存与混合索引。
3. 检索与评测：`runtime_retrieve_demo.py` / `eval_offline.py` 进行检索验证与指标评估。
4. 服务与交互：`serve_retriever.py` 提供 API，`_V5/chat/` 提供本地聊天界面。
5. 扩展模块：`association`（联想图谱）与 `suppressor`（候选抑制）作为可选增强层。

## 目录结构（V5）

- `_V5/src/agentmemory_v3/`：V5 核心实现（retrieval、association、suppressor、serving）
- `_V5/scripts/`：V5 脚本入口（ingest、runtime、quality、suppressor）
- `_V5/configs/default.yaml`：V5 默认配置
- `_V5/chat/`：本地聊天应用（FastAPI + 静态前端）
- `data/V5/`：V5 数据资产目录（processed、indexes、VectorCacheV5、exports、association）
- `_V5/runs/`：V5 评测与实验输出

## 快速开始

### 1) 基础准备

- 建议使用仓库现有虚拟环境：`.venv`
- 默认配置：`_V5/configs/default.yaml`
- 如需聊天与联想相关能力，准备 `data/_secrets/deepseek.env`（`DEEPSEEK_API_KEY` 等）

### 2) 一键准备 V5 资产

```bat
cmd /c ".venv\Scripts\python.exe _V5\scripts\runtime_full_pipeline.py --mode prepare --config _V5\configs\default.yaml"
```

该步骤会生成：

- `data/V5/processed/{memory,query,cluster}.jsonl`
- `data/V5/VectorCacheV5/users/*`
- `data/V5/indexes/{dense_artifact.pkl,dense_matrix.npy,bm25.pkl}`
- `data/V5/exports/chat_memory_bundle.jsonl`

### 3) 冒烟检查

```bat
cmd /c ".venv\Scripts\python.exe _V5\scripts\doctor_data_chain.py --config _V5\configs\default.yaml"
cmd /c ".venv\Scripts\python.exe _V5\scripts\smoke_coarse.py --config _V5\configs\default.yaml"
```

### 4) 离线评测

```bat
cmd /c ".venv\Scripts\python.exe _V5\scripts\eval_offline.py --config _V5\configs\default.yaml --split test"
```

结果默认输出到 `_V5/runs/eval_offline/<timestamp>/`。

## 服务运行

### Retriever API

```bat
cmd /c ".venv\Scripts\python.exe _V5\scripts\serve_retriever.py --config _V5\configs\default.yaml --host 127.0.0.1 --port 8891"
```

### V5 Chat

```bat
cmd /c ".venv\Scripts\python.exe _V5\chat\app.py"
```

默认地址：`http://127.0.0.1:7861`

## 可选增强能力

- 联想图谱构建：
```bat
cmd /c ".venv\Scripts\python.exe _V5\scripts\build_association_graph.py --config _V5\configs\default.yaml"
```
- 抑制器训练与评估：
```bat
cmd /c ".venv\Scripts\python.exe _V5\scripts\train_suppressor.py --config _V5\configs\default.yaml"
cmd /c ".venv\Scripts\python.exe _V5\scripts\eval_suppressor.py --artifact-dir data/V5/suppressor"
```

## 版本说明

- 当前默认开发目标：V5
- 历史目录 `_V3/`、`_V4/` 不作为当前主链运行入口
