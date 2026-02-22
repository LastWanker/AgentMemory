"""One-click runtime trainer: pairwise + listwise rerankers."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_WEIGHTS_DIR = REPO_ROOT / "data" / "ModelWeights"


def resolve_python() -> str:
    venv_py = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_py.exists():
        return str(venv_py)
    return sys.executable


def run(cmd: List[str]) -> None:
    print("[runtime-train] exec:", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def default_by_profile(profile: str) -> Dict[str, int]:
    if profile == "quick":
        return {"epochs": 3, "top_n": 20}
    # formal: slower but closer to your intended final run.
    return {"epochs": 20, "top_n": 30}


def maybe_reset_weights(reset_weights: bool) -> None:
    if not reset_weights:
        return
    for name in ("pairwise_reranker.pt", "listwise_reranker.pt"):
        path = MODEL_WEIGHTS_DIR / name
        if path.exists():
            path.unlink()
            print(f"[runtime-train] removed old weight => {path}")


def build_train_cmd(
    *,
    py: str,
    module_name: str,
    config: str,
    dataset: str,
    encoder_backend: str,
    run_dir: Path,
    save_path: Path,
    epochs: int,
    top_n: int,
    seed: int,
    device: str,
    hf_local_only: bool,
) -> List[str]:
    cmd = [
        py,
        "-m",
        module_name,
        "--config",
        config,
        "--dataset",
        dataset,
        "--encoder-backend",
        encoder_backend,
        "--run-dir",
        str(run_dir),
        "--save-path",
        str(save_path),
        "--epochs",
        str(epochs),
        "--top-n",
        str(top_n),
        "--seed",
        str(seed),
        "--device",
        device,
    ]
    if hf_local_only:
        cmd.append("--hf-local-only")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Train pairwise and listwise rerankers in one command.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", default="followup")
    parser.add_argument("--encoder-backend", choices=("hf", "simple"), default="hf")
    parser.add_argument("--profile", choices=("quick", "formal"), default="formal")
    parser.add_argument("--epochs", type=int, help="Override profile epochs.")
    parser.add_argument("--top-n", type=int, help="Override profile top_n.")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--device",
        default="cuda",
        help="Training device. 'cuda' is preferred; train_reranker auto-fallbacks to CPU if unavailable.",
    )
    parser.add_argument("--hf-online", action="store_true", help="Allow HF online mode (default local-only).")
    parser.add_argument(
        "--reset-weights",
        action="store_true",
        help="Delete existing pairwise/listwise .pt before training.",
    )
    args = parser.parse_args()

    defaults = default_by_profile(args.profile)
    epochs = int(args.epochs if args.epochs is not None else defaults["epochs"])
    top_n = int(args.top_n if args.top_n is not None else defaults["top_n"])
    hf_local_only = not args.hf_online

    # Pitfall note:
    # keep paths stable for eval scripts; wrappers will auto-backup when overwriting.
    pair_weight = MODEL_WEIGHTS_DIR / "pairwise_reranker.pt"
    list_weight = MODEL_WEIGHTS_DIR / "listwise_reranker.pt"
    pair_weight.parent.mkdir(parents=True, exist_ok=True)
    list_weight.parent.mkdir(parents=True, exist_ok=True)
    maybe_reset_weights(args.reset_weights)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = REPO_ROOT / "runs" / "rt" / "train_pairwise_and_listwise" / ts
    pair_train_dir = root / "pairwise" / "train"
    list_train_dir = root / "listwise" / "train"

    py = resolve_python()
    run(
        build_train_cmd(
            py=py,
            module_name="scripts.memory_indexer.train_pairwise_reranker",
            config=args.config,
            dataset=args.dataset,
            encoder_backend=args.encoder_backend,
            run_dir=pair_train_dir,
            save_path=pair_weight,
            epochs=epochs,
            top_n=top_n,
            seed=args.seed,
            device=args.device,
            hf_local_only=hf_local_only,
        )
    )
    run(
        build_train_cmd(
            py=py,
            module_name="scripts.memory_indexer.train_listwise_reranker",
            config=args.config,
            dataset=args.dataset,
            encoder_backend=args.encoder_backend,
            run_dir=list_train_dir,
            save_path=list_weight,
            epochs=epochs,
            top_n=top_n,
            seed=args.seed,
            device=args.device,
            hf_local_only=hf_local_only,
        )
    )

    summary = {
        "profile": args.profile,
        "epochs": epochs,
        "top_n": top_n,
        "seed": args.seed,
        "dataset": args.dataset,
        "encoder_backend": args.encoder_backend,
        "device": args.device,
        "hf_local_only": hf_local_only,
        "pairwise_weight": str(pair_weight),
        "listwise_weight": str(list_weight),
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "train.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n[runtime-train] done")
    print(f"[runtime-train] artifacts => {root}")
    print(f"[runtime-train] pairwise => {pair_weight}")
    print(f"[runtime-train] listwise => {list_weight}")


if __name__ == "__main__":
    main()


