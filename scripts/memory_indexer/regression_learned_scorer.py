"""最小回归：TinyReranker/LearnedFieldScorer/eval_router(learned) 可跑通。"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import torch

from src.memory_indexer.scorer import TinyReranker, LearnedFieldScorer, compute_sim_matrix


REPO_ROOT = Path(__file__).resolve().parents[2]


def run(cmd: list[str]) -> None:
    print("[run]", " ".join(cmd))
    env = dict(os.environ)
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + old_pythonpath if old_pythonpath else "")
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)


def main() -> None:
    # 1) TinyReranker 前向 shape
    model = TinyReranker()
    batch = torch.randn(4, 8, 8)
    out = model(batch)
    assert out.shape == (4,), f"unexpected shape: {out.shape}"
    print("[ok] TinyReranker forward shape")

    # 2) LearnedFieldScorer.score 可用
    weight_path = REPO_ROOT / "data" / "ModelWeights" / "tiny_reranker_regression.pt"
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict()}, weight_path)
    scorer = LearnedFieldScorer(reranker_path=str(weight_path))
    q_vecs = torch.randn(8, 16)
    m_vecs = torch.randn(8, 16)
    q_vecs = torch.nn.functional.normalize(q_vecs, dim=1)
    m_vecs = torch.nn.functional.normalize(m_vecs, dim=1)
    score, debug = scorer.score(q_vecs.tolist(), m_vecs.tolist())
    assert isinstance(score, float)
    assert "sigmoid_score" in debug
    sim = compute_sim_matrix(q_vecs.tolist(), m_vecs.tolist())
    assert sim.shape == (8, 8)
    print("[ok] LearnedFieldScorer score")

    # 3) eval_router 启用 learned scorer 跑通（simple backend，避免网络依赖）
    eval_router_script = REPO_ROOT / "scripts" / "eval_router.py"
    run([sys.executable, str(eval_router_script), "--dataset", "easy", "--top-n", "20", "--top-k", "5", "--policies", "soft", "--encoder-backend", "simple", "--use-learned-scorer", "--reranker-path", str(weight_path), "--no-consistency-pass"])
    print("[ok] eval_router learned scorer run")


if __name__ == "__main__":
    main()

