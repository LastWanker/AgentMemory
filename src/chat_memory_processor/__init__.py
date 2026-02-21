"""Chat memory preprocessing pipeline.

This package is intentionally kept separate from ``src/memory_indexer``:
- ``memory_indexer`` focuses on retrieval/routing/scoring.
- ``chat_memory_processor`` focuses on conversation extraction/cleaning/segmentation.
"""

from .pipeline import (
    ProcessorConfig,
    build_processed_turns,
    write_jsonl,
)

__all__ = [
    "ProcessorConfig",
    "build_processed_turns",
    "write_jsonl",
]

