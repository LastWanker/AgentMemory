"""工具函数：向量计算。"""

from __future__ import annotations

from typing import Iterable, List
import hashlib
import math

Vector = List[float]

def stable_hash(token: str, dims: int) -> Vector:
    """把 token 映射成固定维度向量（确定性）。

    - 用 MD5 作为种子，保证同一 token 结果一致。
    - 向量值落在 [-1, 1]，便于做余弦相似度。
    """

    digest = hashlib.md5(token.encode("utf-8")).digest()
    # 这里用 digest 的字节来拼出伪随机向量
    values = []
    for i in range(dims):
        byte = digest[i % len(digest)]
        values.append((byte / 127.5) - 1.0)
    return values


def dot(a: Vector, b: Vector) -> float:
    """向量点积。"""

    return sum(x * y for x, y in zip(a, b))


def norm(a: Vector) -> float:
    """向量范数。"""

    return math.sqrt(dot(a, a))


def normalize(a: Vector) -> Vector:
    """向量归一化，避免除零。"""

    n = norm(a) or 1e-12
    return [x / n for x in a]


def mean(vectors: Iterable[Vector]) -> Vector:
    """简单平均。"""

    vectors = list(vectors)
    if not vectors:
        return []
    dims = len(vectors[0])
    sums = [0.0] * dims
    for v in vectors:
        for i, x in enumerate(v):
            sums[i] += x
    return [x / len(vectors) for x in sums]
