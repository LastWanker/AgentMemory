from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class DenseArtifact:
    mem_ids: list[str]
    word_vectorizer: TfidfVectorizer
    char_vectorizer: TfidfVectorizer
    svd: TruncatedSVD


class DenseIndex:
    def __init__(self, artifact: DenseArtifact, matrix: np.ndarray) -> None:
        self.artifact = artifact
        self.matrix = np.asarray(matrix, dtype=np.float32)

    @classmethod
    def build(cls, mem_ids: list[str], texts: list[str], *, n_components: int = 192) -> "DenseIndex":
        word_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            max_features=40000,
            sublinear_tf=True,
        )
        char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            min_df=1,
            max_features=50000,
            sublinear_tf=True,
        )
        word_x = word_vectorizer.fit_transform(texts)
        char_x = char_vectorizer.fit_transform(texts)
        full_x = sparse.hstack([word_x, char_x], format="csr")
        if full_x.shape[0] <= 1 or full_x.shape[1] <= 1:
            dense = full_x.toarray().astype(np.float32)
            dense = _l2_normalize(dense)
            svd = TruncatedSVD(n_components=1, random_state=11)
            svd.fit(np.eye(max(1, dense.shape[1]), dtype=np.float32))
            return cls(DenseArtifact(mem_ids, word_vectorizer, char_vectorizer, svd), dense)
        max_components = max(8, min(n_components, full_x.shape[0] - 1, full_x.shape[1] - 1))
        svd = TruncatedSVD(n_components=max_components, random_state=11)
        dense = svd.fit_transform(full_x)
        dense = _l2_normalize(dense)
        return cls(DenseArtifact(mem_ids, word_vectorizer, char_vectorizer, svd), dense)

    def transform_texts(self, texts: list[str]) -> np.ndarray:
        word_x = self.artifact.word_vectorizer.transform(texts)
        char_x = self.artifact.char_vectorizer.transform(texts)
        full_x = sparse.hstack([word_x, char_x], format="csr")
        dense = self.artifact.svd.transform(full_x)
        return _l2_normalize(dense)

    def search(self, query_text: str, top_n: int) -> list[tuple[int, float]]:
        if not query_text.strip() or self.matrix.size == 0:
            return []
        q_vec = self.transform_texts([query_text])[0]
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


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms
