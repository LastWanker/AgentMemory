"""Chat memory preprocessing pipeline.

This package is intentionally kept separate from ``src/memory_indexer``:
- ``memory_indexer`` focuses on retrieval/routing/scoring.
- ``chat_memory_processor`` focuses on conversation extraction/cleaning/segmentation.
"""

from .pipeline import (
    PipelineOutput,
    ProcessorConfig,
    build_processed_turns,
    write_jsonl,
)
from .querygen import (
    build_query_samples,
    load_supplemental_query_samples,
    merge_query_samples,
    to_eval_rows,
    to_followup_rows,
)
from .llm_querygen import (
    LLMQueryGenConfig,
    build_llm_supplemental_rows,
)

__all__ = [
    "PipelineOutput",
    "ProcessorConfig",
    "build_processed_turns",
    "build_query_samples",
    "load_supplemental_query_samples",
    "merge_query_samples",
    "to_eval_rows",
    "to_followup_rows",
    "LLMQueryGenConfig",
    "build_llm_supplemental_rows",
    "write_jsonl",
]
