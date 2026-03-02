"""One-click runtime: build chat supplemental queries (non-preview by default)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "chat_memory.yaml"
DEFAULT_NON_PREVIEW = REPO_ROOT / "data" / "Processed" / "eval_chat_supplemental.api.jsonl"
DEFAULT_PREVIEW = REPO_ROOT / "data" / "Processed" / "eval_chat_supplemental.api_preview.jsonl"
DEFAULT_EVAL_OUT = REPO_ROOT / "data" / "Processed" / "eval_chat.jsonl"
DEFAULT_FOLLOWUP_OUT = REPO_ROOT / "data" / "Processed" / "eval_chat_followup.jsonl"


def resolve_python() -> str:
    venv_py = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_py.exists():
        return str(venv_py)
    return sys.executable


def run(cmd: list[str]) -> None:
    print("[runtime-chat] exec:", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def build_llm_cmd(
    *,
    py: str,
    config: str,
    memory_in: str | None,
    out_path: Path,
    base_url: str | None,
    model: str | None,
    concurrency: int | None,
    max_per_layer: int | None,
    max_clusters: int | None,
    fallback_only: bool,
) -> list[str]:
    cmd = [
        py,
        "-m",
        "scripts.memory_processer.build_chat_supplemental_queries",
        "--config",
        config,
        "--out",
        str(out_path),
    ]
    if memory_in:
        cmd.extend(["--memory-in", memory_in])
    if base_url:
        cmd.extend(["--base-url", base_url])
    if model:
        cmd.extend(["--model", model])
    if concurrency is not None:
        cmd.extend(["--concurrency", str(concurrency)])
    if max_per_layer is not None:
        cmd.extend(["--max-per-layer", str(max_per_layer)])
    if max_clusters is not None:
        cmd.extend(["--max-clusters", str(max_clusters)])
    if fallback_only:
        cmd.append("--fallback-only")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Build chat supplemental (non-preview) and merge eval outputs.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--memory-in", help="Override memory input path.")
    parser.add_argument("--non-preview-out", default=str(DEFAULT_NON_PREVIEW))
    parser.add_argument("--preview-out", default=str(DEFAULT_PREVIEW))
    parser.add_argument("--build-preview", action="store_true", help="Also build preview artifact.")
    parser.add_argument("--preview-max-clusters", type=int, default=1)
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--max-per-layer", type=int)
    parser.add_argument("--fallback-only", action="store_true")
    parser.add_argument("--skip-merge-queries", action="store_true")
    parser.add_argument("--eval-out", default=str(DEFAULT_EVAL_OUT))
    parser.add_argument("--followup-out", default=str(DEFAULT_FOLLOWUP_OUT))
    args = parser.parse_args()

    py = resolve_python()
    non_preview_out = Path(args.non_preview_out)
    preview_out = Path(args.preview_out)
    eval_out = Path(args.eval_out)
    followup_out = Path(args.followup_out)

    run(
        build_llm_cmd(
            py=py,
            config=args.config,
            memory_in=args.memory_in,
            out_path=non_preview_out,
            base_url=args.base_url,
            model=args.model,
            concurrency=args.concurrency,
            max_per_layer=args.max_per_layer,
            max_clusters=None,
            fallback_only=args.fallback_only,
        )
    )

    if args.build_preview:
        run(
            build_llm_cmd(
                py=py,
                config=args.config,
                memory_in=args.memory_in,
                out_path=preview_out,
                base_url=args.base_url,
                model=args.model,
                concurrency=args.concurrency,
                max_per_layer=args.max_per_layer,
                max_clusters=args.preview_max_clusters,
                fallback_only=args.fallback_only,
            )
        )

    if not args.skip_merge_queries:
        merge_cmd = [
            py,
            "-m",
            "scripts.memory_processer.build_chat_queries",
            "--config",
            args.config,
            "--supplemental-queries",
            str(non_preview_out),
            "--eval-out",
            str(eval_out),
            "--followup-out",
            str(followup_out),
        ]
        if args.memory_in:
            merge_cmd.extend(["--memory-in", args.memory_in])
        run(merge_cmd)

    print("\n[runtime-chat] generator=scripts/memory_processer/build_chat_supplemental_queries.py")
    print(f"[runtime-chat] non_preview: {non_preview_out} | rows={count_jsonl(non_preview_out)}")
    if preview_out.exists():
        print(f"[runtime-chat] preview: {preview_out} | rows={count_jsonl(preview_out)}")
    else:
        print(f"[runtime-chat] preview: {preview_out} | rows=0 (missing)")
    if not args.skip_merge_queries:
        print(f"[runtime-chat] merged_eval: {eval_out} | rows={count_jsonl(eval_out)}")
        print(f"[runtime-chat] merged_followup: {followup_out} | rows={count_jsonl(followup_out)}")


if __name__ == "__main__":
    main()
