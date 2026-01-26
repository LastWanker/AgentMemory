"""记忆唤醒索引器：最小可运行框架。"""

from .models import EmbeddingRecord, MemoryItem, Query, RetrieveResult
from .encoder.base import Encoder
from .encoder.simple import SimpleHashEncoder
from .vectorizer import Vectorizer
from .index import CoarseIndex
from .scorer import FieldScorer
from .store import MemoryStore
from .retriever import Retriever
from .pipeline import build_memory_index, retrieve_top_k
from .trace import set_trace, trace

__all__ = [
    "EmbeddingRecord",
    "MemoryItem",
    "Query",
    "RetrieveResult",
    "Encoder",
    "SimpleHashEncoder",
    "Vectorizer",
    "CoarseIndex",
    "FieldScorer",
    "MemoryStore",
    "Retriever",
    "build_memory_index",
    "retrieve_top_k",
    "set_trace",
    "trace",
]
