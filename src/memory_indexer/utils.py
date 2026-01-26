"""工具函数：向量计算与简单分词。"""

from __future__ import annotations

from typing import Iterable, List
import hashlib
import math
import re

Vector = List[float]

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> List[str]:
    """非常轻量的分词器：英文按词，中文按单字。

    这里只是为了让示例能跑通，真实项目请替换为更靠谱的中文分词器。
    """

    return _TOKEN_RE.findall(text)


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
