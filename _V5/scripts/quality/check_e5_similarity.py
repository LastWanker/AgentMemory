from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


V3_ROOT = Path(__file__).resolve().parents[1]
if str(V3_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(V3_ROOT / "src"))

from agentmemory_v3.config import cfg_get, load_yaml_config
from agentmemory_v3.encoder import HFSentenceEncoder, SentenceEncoderConfig


def _pair_rows(terms: list[str], sims: np.ndarray, threshold: float) -> list[tuple[float, str]]:
    rows: list[tuple[float, str]] = []
    for i in range(len(terms)):
        for j in range(i + 1, len(terms)):
            score = float(sims[i, j])
            mark = "YES" if score >= threshold else "NO"
            rows.append((score, f"{terms[i]} <-> {terms[j]} | cosine={score:.6f} | alias@{threshold:.2f}={mark}"))
    rows.sort(key=lambda item: item[0], reverse=True)
    return rows


def _encode_raw_texts(encoder: HFSentenceEncoder, texts: list[str]) -> np.ndarray:
    vectors = encoder.model.encode(
        texts,
        batch_size=max(1, int(encoder.config.batch_size)),
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)


def _prefixed(text: str, mode: str) -> str:
    raw = str(text).strip()
    if mode == "query":
        return f"query: {raw}"
    if mode == "passage":
        return f"passage: {raw}"
    return raw


def _print_matrix(
    encoder: HFSentenceEncoder,
    *,
    left: str,
    right: str,
    left_modes: list[str] | None = None,
    right_modes: list[str] | None = None,
) -> None:
    left_modes = left_modes or ["raw", "query", "passage"]
    right_modes = right_modes or ["raw", "query", "passage"]

    left_texts = [_prefixed(left, mode) for mode in left_modes]
    right_texts = [_prefixed(right, mode) for mode in right_modes]
    all_texts = left_texts + right_texts
    vectors = _encode_raw_texts(encoder, all_texts)
    left_vecs = vectors[: len(left_texts)]
    right_vecs = vectors[len(left_texts) :]
    scores = left_vecs @ right_vecs.T

    print(f"[matrix] left='{left}' right='{right}'")
    print("           " + "  ".join(f"{mode:>8s}" for mode in right_modes))
    for i, l_mode in enumerate(left_modes):
        row_vals = "  ".join(f"{float(scores[i, j]):8.6f}" for j in range(len(right_modes)))
        print(f"{l_mode:>8s}  {row_vals}")


def _run_requested_cases(encoder: HFSentenceEncoder) -> None:
    print("[requested] case 1-4: 用户 / 苹果")
    explicit_pairs = [
        ("passage: 用户", "passage: 苹果"),
        ("query: 用户", "query: 苹果"),
        ("query: 用户", "passage: 苹果"),
        ("passage: 用户", "query: 苹果"),
    ]
    vecs = _encode_raw_texts(encoder, [text for pair in explicit_pairs for text in pair])
    for idx, (left, right) in enumerate(explicit_pairs):
        left_vec = vecs[idx * 2]
        right_vec = vecs[idx * 2 + 1]
        score = float(np.dot(left_vec, right_vec))
        print(f"  - {left}  <->  {right}  | cosine={score:.6f}")

    print("[requested] case 5 single-pair checks")
    extra_pairs = [
        ("计算机", "电脑"),
        ("超级赛亚人", "国际象棋"),
        ("query: 天上掉下个林妹妹", "passage: 三体人ETO组织降临地球"),
    ]
    extra_vecs = _encode_raw_texts(encoder, [text for pair in extra_pairs for text in pair])
    for idx, (left, right) in enumerate(extra_pairs):
        left_vec = extra_vecs[idx * 2]
        right_vec = extra_vecs[idx * 2 + 1]
        score = float(np.dot(left_vec, right_vec))
        print(f"  - {left}  <->  {right}  | cosine={score:.6f}")

    print("[requested] prefix-position matrices")
    _print_matrix(encoder, left="用户", right="苹果")
    _print_matrix(encoder, left="计算机", right="电脑")
    _print_matrix(encoder, left="超级赛亚人", right="国际象棋")
    _print_matrix(encoder, left="天上掉下个林妹妹", right="三体人ETO组织降临地球")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check E5 cosine similarity exactly as V5 association uses it.")
    parser.add_argument("--config", default="_V5/configs/default.yaml")
    parser.add_argument(
        "--terms",
        nargs="+",
        default=["用户", "苹果", "苹果公司", "苹果手机", "Windows", "Linux", "猫", "狗"],
        help="Terms to encode and compare.",
    )
    parser.add_argument("--mode", choices=["passage", "query"], default="passage")
    parser.add_argument("--threshold", type=float, default=-1.0, help="Alias threshold; default uses association.merge_threshold.")
    parser.add_argument(
        "--requested-cases",
        action="store_true",
        help="Run the requested prefix-position tests and pair matrices.",
    )
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    threshold = (
        float(args.threshold)
        if float(args.threshold) >= 0.0
        else float(cfg_get(cfg, "association.merge_threshold", 0.85))
    )

    encoder = HFSentenceEncoder(
        SentenceEncoderConfig(
            model_name=str(cfg_get(cfg, "encoder.model_name", "intfloat/multilingual-e5-small")),
            use_e5_prefix=bool(cfg_get(cfg, "encoder.use_e5_prefix", True)),
            local_files_only=bool(cfg_get(cfg, "encoder.local_files_only", True)),
            offline=bool(cfg_get(cfg, "encoder.offline", True)),
            device=str(cfg_get(cfg, "encoder.device", "auto")),
            batch_size=int(cfg_get(cfg, "encoder.batch_size_cuda", cfg_get(cfg, "encoder.batch_size", 128))),
        )
    )

    terms = [str(item).strip() for item in args.terms if str(item).strip()]
    if len(terms) < 2:
        raise RuntimeError("Need at least two terms.")
    if args.mode == "query":
        vectors = encoder.encode_query_texts(terms)
    else:
        vectors = encoder.encode_passage_texts(terms)
    norms = np.linalg.norm(vectors, axis=1)
    sims = vectors @ vectors.T

    print(f"[e5-check] model={encoder.config.model_name} mode={args.mode} threshold={threshold:.2f}")
    print("[e5-check] vector norms")
    for term, norm in zip(terms, norms):
        print(f"  - {term}: norm={float(norm):.6f}")
    print(f"[e5-check] min_norm={float(np.min(norms)):.6f} max_norm={float(np.max(norms)):.6f}")
    print("[e5-check] pairwise cosine (sorted desc)")
    for _, line in _pair_rows(terms, sims, threshold):
        print(f"  - {line}")
    if bool(args.requested_cases):
        _run_requested_cases(encoder)


if __name__ == "__main__":
    main()
