# _chat

`_chat/` 是与 `_V3/` 并列的本地聊天器。

目标：
- 使用 DeepSeek API 完成对话
- 把 `_V3` 当作记忆找回器
- 展示“本次参考记忆”
- 保存会话到本地 JSONL

当前形态：
- 后端：FastAPI
- 前端：原生 HTML/CSS/JS
- 检索接入：
  - 优先走 `_V3` HTTP retriever service
  - 无服务时回退到本地 `_V3` 检索器
  - 若本地已有 `data/V3/models/reranker.pt`，会自动加载训练后的 reranker
  - 再不行才回退到导出 bundle 的轻量检索

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
cmd /c "python _chat\app.py"
```

3. 打开：

`http://127.0.0.1:7860`

可选：如果你想把 `_V3` 检索服务单独开起来：

```bat
cmd /c "python _V3\scripts\serve_retriever.py"
```

即使不单独启动它，`_chat` 也会回退到本地 `_V3` 检索。

## 会话落盘

- 目录：`_chat/data/conversations/`
- 文件：`<session_id>.jsonl`

## DeepSeek API 参考

- https://api-docs.deepseek.com/
- https://api-docs.deepseek.com/api/create-chat-completion
