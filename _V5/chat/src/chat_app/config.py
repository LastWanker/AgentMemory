from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
V5_ROOT = ROOT.parent
REPO_ROOT = V5_ROOT.parent


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
    feedback_dir: Path
    retrieval_mode: str
    retrieval_url: str
    retrieval_config: Path
    retrieval_bundle: Path
    top_k: int
    history_window: int
    suppressor_enabled: bool
    suppressor_backend: str
    suppressor_artifact_dir: Path
    suppressor_memnet_artifact_dir: Path
    suppressor_memnet_score_threshold: float
    suppressor_memnet_max_drop_per_lane: int
    suppressor_memnet_keep_top_per_lane: int
    suppressor_threshold: float
    suppressor_threshold_coarse: float
    suppressor_threshold_association: float
    suppressor_alpha: float
    suppressor_min_delta: float
    suppressor_min_delta_coarse: float
    suppressor_min_delta_association: float
    suppressor_min_margin: float
    suppressor_min_zscore: float
    suppressor_min_zscore_coarse: float
    suppressor_min_zscore_association: float
    suppressor_min_base_relevance: float
    suppressor_min_type_gap: float
    suppressor_type_conflict_activation: float
    suppressor_max_drop_per_lane: int
    suppressor_keep_top_per_lane: int
    suppressor_extra_candidates: int
    suppressor_apply_to_coarse: bool
    suppressor_apply_to_association: bool
    suppressor_debug: bool
    suppressor_bias_lambda: float
    suppressor_use_calibration: bool


def load_config() -> ChatAppConfig:
    secret_env = _load_env_file(REPO_ROOT / "data" / "_secrets" / "deepseek.env")
    data_dir = ROOT / "data" / "conversations"
    feedback_dir = ROOT / "data"
    return ChatAppConfig(
        api_key=os.getenv("DEEPSEEK_API_KEY", secret_env.get("DEEPSEEK_API_KEY", "")).strip(),
        base_url=os.getenv("DEEPSEEK_BASE_URL", secret_env.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).strip(),
        model=os.getenv("DEEPSEEK_MODEL", secret_env.get("DEEPSEEK_MODEL", "deepseek-chat")).strip(),
        system_prompt=(
            "你是 V5 本地记忆助手。当前参考材料可能来自两路：coarse 粗召回，以及 association 联想召回。"
            "association 内容来自概念图的点亮、上下溯和桥接联想，不等于用户明确说过的话。"
            "优先直接回答用户问题；参考记忆相关就自然使用，不相关就忽略。"
            "不要编造未提供的事实。"
        ),
        data_dir=data_dir,
        feedback_dir=feedback_dir,
        retrieval_mode=os.getenv("CHAT_RETRIEVAL_MODE", "local").strip().lower(),
        retrieval_url=os.getenv("CHAT_RETRIEVAL_URL", "http://127.0.0.1:8891").strip().rstrip("/"),
        retrieval_config=Path(os.getenv("CHAT_RETRIEVAL_CONFIG", str(V5_ROOT / "configs" / "default.yaml"))),
        retrieval_bundle=Path(
            os.getenv("CHAT_RETRIEVAL_BUNDLE", str(REPO_ROOT / "data" / "V5" / "exports" / "chat_memory_bundle.jsonl"))
        ),
        top_k=max(1, int(os.getenv("CHAT_TOP_K", "5"))),
        history_window=max(2, int(os.getenv("CHAT_HISTORY_WINDOW", "8"))),
        suppressor_enabled=os.getenv("SUPPRESSOR_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"},
        suppressor_backend=os.getenv("SUPPRESSOR_BACKEND", "mem_network").strip().lower(),
        suppressor_artifact_dir=Path(
            os.getenv("SUPPRESSOR_ARTIFACT_DIR", str(REPO_ROOT / "data" / "V5" / "suppressor_newfb"))
        ),
        suppressor_memnet_artifact_dir=Path(
            os.getenv(
                "SUPPRESSOR_MEMNET_ARTIFACT_DIR",
                str(REPO_ROOT / "data" / "V5" / "mem_network_suppressor" / "current"),
            )
        ),
        suppressor_memnet_score_threshold=float(os.getenv("MEMNET_SCORE_THRESHOLD", "0.50")),
        suppressor_memnet_max_drop_per_lane=max(
            0,
            int(os.getenv("MEMNET_MAX_DROP_PER_LANE", os.getenv("SUPPRESSOR_MAX_DROP_PER_LANE", "1"))),
        ),
        suppressor_memnet_keep_top_per_lane=max(
            0,
            int(os.getenv("MEMNET_KEEP_TOP_PER_LANE", os.getenv("SUPPRESSOR_KEEP_TOP_PER_LANE", "2"))),
        ),
        suppressor_threshold=float(os.getenv("SUPPRESSOR_THRESHOLD", "0.40")),
        suppressor_threshold_coarse=float(os.getenv("SUPPRESSOR_THRESHOLD_COARSE", os.getenv("SUPPRESSOR_THRESHOLD", "0.40"))),
        suppressor_threshold_association=float(
            os.getenv("SUPPRESSOR_THRESHOLD_ASSOCIATION", os.getenv("SUPPRESSOR_THRESHOLD", "0.40"))
        ),
        suppressor_alpha=float(os.getenv("SUPPRESSOR_ALPHA", "0.35")),
        suppressor_min_delta=float(os.getenv("SUPPRESSOR_MIN_DELTA", "0.08")),
        suppressor_min_delta_coarse=float(os.getenv("SUPPRESSOR_MIN_DELTA_COARSE", os.getenv("SUPPRESSOR_MIN_DELTA", "0.08"))),
        suppressor_min_delta_association=float(
            os.getenv("SUPPRESSOR_MIN_DELTA_ASSOCIATION", os.getenv("SUPPRESSOR_MIN_DELTA", "0.08"))
        ),
        suppressor_min_margin=float(os.getenv("SUPPRESSOR_MIN_MARGIN", "0.08")),
        suppressor_min_zscore=float(os.getenv("SUPPRESSOR_MIN_ZSCORE", "1.0")),
        suppressor_min_zscore_coarse=float(
            os.getenv("SUPPRESSOR_MIN_ZSCORE_COARSE", os.getenv("SUPPRESSOR_MIN_ZSCORE", "1.0"))
        ),
        suppressor_min_zscore_association=float(
            os.getenv("SUPPRESSOR_MIN_ZSCORE_ASSOCIATION", os.getenv("SUPPRESSOR_MIN_ZSCORE", "1.0"))
        ),
        suppressor_min_base_relevance=float(os.getenv("SUPPRESSOR_MIN_BASE_RELEVANCE", "0.0")),
        suppressor_min_type_gap=float(os.getenv("SUPPRESSOR_MIN_TYPE_GAP", "0.03")),
        suppressor_type_conflict_activation=float(os.getenv("SUPPRESSOR_TYPE_CONFLICT_ACTIVATION", "0.14")),
        suppressor_max_drop_per_lane=max(0, int(os.getenv("SUPPRESSOR_MAX_DROP_PER_LANE", "1"))),
        suppressor_keep_top_per_lane=max(0, int(os.getenv("SUPPRESSOR_KEEP_TOP_PER_LANE", "2"))),
        suppressor_extra_candidates=max(0, int(os.getenv("SUPPRESSOR_EXTRA_CANDIDATES", "8"))),
        suppressor_apply_to_coarse=os.getenv("SUPPRESSOR_APPLY_TO_COARSE", "1").strip().lower()
        in {"1", "true", "yes", "on"},
        suppressor_apply_to_association=os.getenv("SUPPRESSOR_APPLY_TO_ASSOCIATION", "1").strip().lower()
        in {"1", "true", "yes", "on"},
        suppressor_debug=os.getenv("SUPPRESSOR_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"},
        suppressor_bias_lambda=float(os.getenv("SUPPRESSOR_BIAS_LAMBDA", "0.0")),
        suppressor_use_calibration=os.getenv("SUPPRESSOR_USE_CALIBRATION", "1").strip().lower()
        in {"1", "true", "yes", "on"},
    )
