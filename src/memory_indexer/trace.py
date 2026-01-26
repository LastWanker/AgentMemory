"""运行输出控制。"""

from __future__ import annotations

from dataclasses import dataclass
import os
import sys
from typing import Optional


@dataclass
class TraceConfig:
    """运行输出配置。"""

    enabled: bool = False


TRACE_CONFIG = TraceConfig(enabled=os.getenv("MEMORY_INDEXER_TRACE", "0") == "1")


def set_trace(enabled: bool) -> None:
    """中文注释：手动开启或关闭运行输出。"""

    TRACE_CONFIG.enabled = enabled


def trace(message: str) -> None:
    """中文注释：统一的输出入口，避免到处散落 print。"""

    if TRACE_CONFIG.enabled:
        print(f"[运行] {message}")


class TraceProgress:
    """简易进度条：同一步骤只输出一条滚动进度。"""

    def __init__(self, title: str, total: int, width: int = 20) -> None:
        self.title = title
        self.total = max(total, 1)
        self.width = width
        self.current = 0
        self._enabled = TRACE_CONFIG.enabled
        self._finished = False
        if self._enabled:
            print(f"[运行] {self.title}：开始")

    def update(self, step: Optional[int] = None) -> None:
        if not self._enabled or self._finished:
            return
        if step is None:
            self.current += 1
        else:
            self.current = min(step, self.total)
        ratio = self.current / self.total
        filled = int(ratio * self.width)
        bar = "▬" * filled + "·" * (self.width - filled)
        sys.stdout.write(f"\r[运行] {self.title} [{bar}] {self.current}/{self.total}")
        sys.stdout.flush()
        if self.current >= self.total:
            self.finish()

    def finish(self) -> None:
        if not self._enabled or self._finished:
            return
        self.current = self.total
        ratio = 1.0
        filled = int(ratio * self.width)
        bar = "▬" * filled + "·" * (self.width - filled)
        sys.stdout.write(f"\r[运行] {self.title} [{bar}] {self.current}/{self.total}\n")
        sys.stdout.flush()
        print(f"[运行] {self.title}：完成")
        self._finished = True


def trace_progress(title: str, total: int) -> TraceProgress:
    """中文注释：创建一个简洁进度条。"""

    return TraceProgress(title=title, total=total)
