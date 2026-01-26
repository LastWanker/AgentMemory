"""训练器骨架：留好扩展入口。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from .datasets import PairDataset


@dataclass
class TrainingConfig:
    """训练配置（占位版）。"""

    epochs: int = 1
    batch_size: int = 16


class ProjectionHead:
    """小投影头的占位实现。

    中文注释：真实版本可替换为 PyTorch/TF 模块。
    """

    def __call__(self, vectors: List[List[float]]) -> List[List[float]]:
        return vectors


class SelfSupervisedTrainer:
    """自监督训练器。

    这里不实现真实训练，只保留接口，便于后续接入。
    """

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def train_projection_head(self, dataset: PairDataset) -> Tuple[int, int]:
        # 中文注释：返回 (epochs, samples) 作为训练报告
        report = (self.config.epochs, len(dataset.pairs))
        return report
