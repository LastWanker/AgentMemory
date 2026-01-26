"""自监督训练相关数据处理。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple


@dataclass
class PairDataset:
    """(x, y) 对话对，作为自监督样本。"""

    pairs: List[Tuple[str, str]]


def build_pairs(dialogue_turns: Iterable[str]) -> PairDataset:
    """把连续对话切成 (x, y) 对。

    中文注释：这里只做相邻句，真实场景可用更复杂的窗口。
    """

    turns = list(dialogue_turns)
    pairs = []
    for i in range(len(turns) - 1):
        pairs.append((turns[i], turns[i + 1]))
    return PairDataset(pairs=pairs)
