"""记忆唤醒索引器：最小可运行框架。"""

from .models import EmbeddingRecord, MemoryItem, Query, RetrieveResult, RouteOutput
from .encoder.base import Encoder
from .encoder.simple import SimpleHashEncoder
from .encoder.e5_token import E5TokenEncoder
from .vectorizer import Vectorizer
from .index import CoarseIndex, LexicalIndex
from .scorer import FieldScorer
from .store import MemoryStore
from .retriever import Retriever, Router
from .pipeline import build_memory_index, retrieve_top_k
from .ingest import build_memory_items
from .tokenizers import Tokenizer, tokenize
from .trace import set_trace, trace

__all__ = [
    "EmbeddingRecord",
    "MemoryItem",
    "Query",
    "RetrieveResult",
    "RouteOutput",
    "Encoder",
    "SimpleHashEncoder",
    "E5TokenEncoder",
    "Vectorizer",
    "CoarseIndex",
    "LexicalIndex",
    "FieldScorer",
    "MemoryStore",
    "Retriever",
    "Router",
    "build_memory_index",
    "build_memory_items",
    "retrieve_top_k",
    "Tokenizer",
    "tokenize",
    "set_trace",
    "trace",
]
