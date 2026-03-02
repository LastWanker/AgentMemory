# V5 Chat

`_V5/chat/` 是 V5 专用的本地聊天器。

目标：
- 默认只使用 V5 coarse-only 检索
- 默认优先走本地 `_V5` 检索器
- 展示“本次参考记忆”
- 保存会话到本地 JSONL

当前形态：
- 后端：FastAPI
- 前端：原生 HTML/CSS/JS
- 检索接入：
  - 默认走本地 `_V5` coarse retriever
  - 若手动把 `CHAT_RETRIEVAL_MODE=http`，则走 `http://127.0.0.1:8891`
  - 再不行才回退到 `data/V5/exports/chat_memory_bundle.jsonl`

## 运行

1. 默认会自动读取：

`data/_secrets/deepseek.env`

至少包含：

```env
DEEPSEEK_API_KEY=...
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

2. 直接启动：

```bat
cmd /c ".venv\Scripts\python.exe _V5\chat\app.py"
```

3. 打开：

`http://127.0.0.1:7861`

可选：如果你想单独启动 V5 检索服务：

```bat
cmd /c ".venv\Scripts\python.exe _V5\scripts\serve_retriever.py"
```

## 会话落盘

- 目录：`_V5/chat/data/conversations/`
- 文件：`<session_id>.jsonl`
