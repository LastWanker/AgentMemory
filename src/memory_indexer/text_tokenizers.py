"""可替换分词器接口与内置实现。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Union
import re


class Tokenizer(Protocol):
    """分词器接口。"""

    def tokenize(self, text: str) -> List[str]:
        """把文本切分为 token 列表。"""


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]")


@dataclass
class SimpleTokenizer:
    """默认分词器：英文按词，中文按单字。"""

    token_re: re.Pattern = _TOKEN_RE

    def tokenize(self, text: str) -> List[str]:
        return self.token_re.findall(text)


class JiebaTokenizer:
    """可选依赖的中文分词器（需要安装 jieba）。"""

    def __init__(self, *, cut_all: bool = False) -> None:
        try:
            import jieba  # type: ignore
        except ImportError as exc:
            raise ImportError("未安装 jieba，可执行 `pip install jieba`") from exc
        self._jieba = jieba
        self.cut_all = cut_all

    def tokenize(self, text: str) -> List[str]:
        return [token for token in self._jieba.cut(text, cut_all=self.cut_all) if token.strip()]


TokenizerInput = Union[str, Tokenizer, None]


def resolve_tokenizer(tokenizer: TokenizerInput) -> Tokenizer:
    """解析分词器输入，支持名称或自定义实例。"""

    if tokenizer is None:
        return SimpleTokenizer()
    if isinstance(tokenizer, str):
        name = tokenizer.lower()
        if name in {"simple", "default"}:
            return SimpleTokenizer()
        if name == "jieba":
            return JiebaTokenizer()
        raise ValueError(f"未知分词器: {tokenizer}")
    return tokenizer


def tokenize(text: str, tokenizer: TokenizerInput = None) -> List[str]:
    """兼容旧接口的 tokenize 函数。"""

    return resolve_tokenizer(tokenizer).tokenize(text)
