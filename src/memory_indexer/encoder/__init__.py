"""编码器实现集合。"""

from .base import Encoder
from .simple import SimpleHashEncoder

__all__ = ["Encoder", "SimpleHashEncoder", "HFSentenceEncoder", "E5TokenEncoder"]


def __getattr__(name: str):
    if name == "HFSentenceEncoder":
        from .hf_sentence import HFSentenceEncoder

        return HFSentenceEncoder
    if name == "E5TokenEncoder":
        from .e5_token import E5TokenEncoder

        return E5TokenEncoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
