from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


V3_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = V3_ROOT.parent


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {cfg_path}")
    return data


def cfg_get(cfg: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    node: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_env_file(path_like: str | Path) -> dict[str, str]:
    path = resolve_path(path_like)
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


def get_secret(key: str, *, env_file: str | Path = "data/_secrets/deepseek.env", default: str = "") -> str:
    value = os.getenv(key, "").strip()
    if value:
        return value
    return load_env_file(env_file).get(key, default)
