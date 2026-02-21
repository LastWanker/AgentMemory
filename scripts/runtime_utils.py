"""Runtime utilities for script config/log/cache conventions."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}

    # YAML is a superset of JSON. Keep parser dependency-free by accepting JSON-formatted YAML.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                f"config file {path} is not valid JSON; install pyyaml for generic YAML parsing."
            ) from exc
        loaded = yaml.safe_load(text)
        return loaded if isinstance(loaded, dict) else {}


def cfg_get(config: Dict[str, Any], dotted_key: str, default: Any) -> Any:
    cur: Any = config
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def sanitize_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
    return slug.strip("_") or "na"


def build_cache_signature(parts: Dict[str, Any]) -> str:
    ordered = [
        f"{key}={parts.get(key)}"
        for key in sorted(parts.keys())
    ]
    return sanitize_slug("__".join(ordered))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def file_sha1_short(path: Path, length: int = 8) -> str:
    hasher = hashlib.sha1()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()[: max(1, length)]


def build_data_fingerprint(memory_path: Path, eval_path: Path) -> str:
    mem_sig = file_sha1_short(memory_path)
    eval_sig = file_sha1_short(eval_path)
    return f"mem_{mem_sig}__eval_{eval_sig}"


def get_git_commit(repo_root: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode == 0:
            return (proc.stdout or "").strip() or "unknown"
    except Exception:
        pass
    return "unknown"


def _normalize_registry_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def upsert_run_registry(registry_path: Path, record: Dict[str, Any]) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = {k: _normalize_registry_value(v) for k, v in record.items() if k}
    run_dir = normalized.get("run_dir", "").strip()
    if not run_dir:
        return

    rows: List[Dict[str, str]] = []
    fieldnames: List[str] = []
    if registry_path.exists():
        with registry_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = [((name or "").replace("\ufeff", "")) for name in (reader.fieldnames or [])]
            for row in reader:
                rows.append({(k or "").replace("\ufeff", ""): (v or "") for k, v in row.items()})
    for key in normalized.keys():
        if key not in fieldnames:
            fieldnames.append(key)
    if "run_dir" not in fieldnames:
        fieldnames.insert(0, "run_dir")
    if "timestamp" in fieldnames:
        fieldnames = ["timestamp"] + [k for k in fieldnames if k != "timestamp"]

    updated = False
    for row in rows:
        if (row.get("run_dir") or "").strip() == run_dir:
            for key, value in normalized.items():
                if value != "":
                    row[key] = value
            updated = True
            break
    if not updated:
        fresh = {key: "" for key in fieldnames}
        fresh.update(normalized)
        rows.append(fresh)

    with registry_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


@dataclass
class Tee:
    stream: Any
    file_handle: Any

    def write(self, data: str) -> int:
        written = self.stream.write(data)
        self.file_handle.write(data)
        self.file_handle.flush()
        return written

    def flush(self) -> None:
        self.stream.flush()
        self.file_handle.flush()

    def isatty(self) -> bool:
        isatty = getattr(self.stream, "isatty", None)
        if callable(isatty):
            try:
                return bool(isatty())
            except Exception:
                return False
        return False


def enable_run_dir_logging(
    run_dir: Optional[str],
    *,
    log_filename: str,
    argv: Iterable[str],
    config_snapshot: Dict[str, Any],
) -> Optional[Path]:
    if not run_dir:
        return None
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    log_path = root / log_filename
    fh = log_path.open("w", encoding="utf-8")
    sys.stdout = Tee(sys.stdout, fh)  # type: ignore[assignment]
    sys.stderr = Tee(sys.stderr, fh)  # type: ignore[assignment]

    meta = {
        "argv": list(argv),
        "config_snapshot": config_snapshot,
    }
    write_json(root / "config.snapshot.json", meta)
    return root
