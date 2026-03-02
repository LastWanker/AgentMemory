from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from agentmemory_v3.encoder import HFSentenceEncoder, SentenceEncoderConfig


@dataclass
class DenseArtifact:
    mem_ids: list[str]
    backend: str = "e5_sentence"
    model_name: str = "intfloat/multilingual-e5-small"
    use_e5_prefix: bool = True
    local_files_only: bool = True
    offline: bool = True
    device: str = "auto"
    batch_size: int = 128
    dim: int = 384


class DenseIndex:
    def __init__(self, artifact: DenseArtifact, matrix: np.ndarray) -> None:
        self.artifact = artifact
        self.matrix = np.asarray(matrix, dtype=np.float32)
        self._encoder: HFSentenceEncoder | None = None

    @classmethod
    def build(
        cls,
        mem_ids: list[str],
        matrix: np.ndarray,
        *,
        model_name: str,
        use_e5_prefix: bool = True,
        local_files_only: bool = True,
        offline: bool = True,
        device: str = "auto",
        batch_size: int = 128,
    ) -> "DenseIndex":
        dense = _l2_normalize(np.asarray(matrix, dtype=np.float32))
        dim = int(dense.shape[1]) if dense.ndim == 2 and dense.shape[0] > 0 else 0
        artifact = DenseArtifact(
            mem_ids=list(mem_ids),
            model_name=str(model_name),
            use_e5_prefix=bool(use_e5_prefix),
            local_files_only=bool(local_files_only),
            offline=bool(offline),
            device=str(device),
            batch_size=int(batch_size),
            dim=int(dim),
        )
        return cls(artifact, dense)

    def transform_texts(self, texts: list[str], *, role: str = "passage") -> np.ndarray:
        if not texts:
            return np.zeros((0, self.artifact.dim), dtype=np.float32)
        encoder = self._get_encoder()
        if str(role).strip().lower() == "query":
            return encoder.encode_query_texts(texts)
        return encoder.encode_passage_texts(texts)

    def search(self, query_text: str, top_n: int) -> list[tuple[int, float]]:
        if not query_text.strip() or self.matrix.size == 0:
            return []
        q_vec = self.transform_texts([query_text], role="query")[0]
        return self.search_vector(q_vec, top_n)

    def search_vector(self, query_vec: np.ndarray, top_n: int) -> list[tuple[int, float]]:
        if self.matrix.size == 0:
            return []
        q_vec = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        if q_vec.size == 0:
            return []
        q_norm = float(np.linalg.norm(q_vec))
        if q_norm <= 1e-8:
            return []
        q_vec = q_vec / q_norm
        scores = self.matrix @ q_vec
        limit = min(int(top_n), scores.shape[0])
        if limit <= 0:
            return []
        if limit >= scores.shape[0]:
            top_idx = np.argsort(scores)[::-1]
        else:
            part_idx = np.argpartition(scores, -limit)[-limit:]
            top_idx = part_idx[np.argsort(scores[part_idx])[::-1]]
        return [(int(i), float(scores[int(i)])) for i in top_idx]

    def save(self, artifact_path: Path, matrix_path: Path) -> None:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        matrix_path.parent.mkdir(parents=True, exist_ok=True)
        with artifact_path.open("wb") as f:
            pickle.dump(self.artifact, f)
        np.save(matrix_path, self.matrix)

    @classmethod
    def load(cls, artifact_path: Path, matrix_path: Path) -> "DenseIndex":
        with artifact_path.open("rb") as f:
            artifact = pickle.load(f)
        matrix = np.load(matrix_path)
        return cls(artifact, matrix)

    def _get_encoder(self) -> HFSentenceEncoder:
        if self._encoder is None:
            self._encoder = HFSentenceEncoder(
                SentenceEncoderConfig(
                    model_name=self.artifact.model_name,
                    use_e5_prefix=bool(self.artifact.use_e5_prefix),
                    local_files_only=bool(self.artifact.local_files_only),
                    offline=bool(self.artifact.offline),
                    device=str(self.artifact.device),
                    batch_size=int(self.artifact.batch_size),
                )
            )
        return self._encoder


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.size == 0:
        return matrix.reshape((0, 0)).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms
