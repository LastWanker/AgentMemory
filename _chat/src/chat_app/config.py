from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ROOT.parent


def _load_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out


@dataclass
class ChatAppConfig:
    api_key: str
    base_url: str
    model: str
    system_prompt: str
    data_dir: Path
    retrieval_mode: str
    retrieval_url: str
    retrieval_project: str
    retrieval_config: Path
    retrieval_bundle: Path
    top_k: int
    history_window: int


def load_config() -> ChatAppConfig:
    secret_env = _load_env_file(REPO_ROOT / "data" / "_secrets" / "deepseek.env")
    data_dir = ROOT / "data" / "conversations"
    retrieval_project = os.getenv("CHAT_RETRIEVAL_PROJECT", "V4").strip().upper() or "V4"
    default_project = "_V4" if retrieval_project == "V4" else "_V3"
    retrieval_bundle = Path(
        os.getenv(
            "CHAT_RETRIEVAL_BUNDLE",
            str(ROOT.parent / "data" / ("V4" if retrieval_project == "V4" else "V3") / "exports" / "chat_memory_bundle.jsonl"),
        )
    )
    retrieval_config = Path(
        os.getenv("CHAT_RETRIEVAL_CONFIG", str(ROOT.parent / default_project / "configs" / "default.yaml"))
    )
    return ChatAppConfig(
        api_key=os.getenv("DEEPSEEK_API_KEY", secret_env.get("DEEPSEEK_API_KEY", "")).strip(),
        base_url=os.getenv("DEEPSEEK_BASE_URL", secret_env.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).strip(),
        model=os.getenv("DEEPSEEK_MODEL", secret_env.get("DEEPSEEK_MODEL", "deepseek-chat")).strip(),
        system_prompt=(
            "你是本地记忆助手。优先回答用户问题。"
            "若参考记忆相关，就自然使用；若不相关，不要强行引用。"
            "回答尽量直接，避免编造未提供的事实。"
        ),
        data_dir=data_dir,
        retrieval_mode=os.getenv("CHAT_RETRIEVAL_MODE", "http").strip().lower(),
        retrieval_url=os.getenv("CHAT_RETRIEVAL_URL", "http://127.0.0.1:8891").strip().rstrip("/"),
        retrieval_project=retrieval_project,
        retrieval_config=retrieval_config,
        retrieval_bundle=retrieval_bundle,
        top_k=max(1, int(os.getenv("CHAT_TOP_K", "5"))),
        history_window=max(2, int(os.getenv("CHAT_HISTORY_WINDOW", "8"))),
    )
