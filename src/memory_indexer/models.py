"""数据模型定义。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time

Vector = List[float]
VectorGroup = List[Vector]


@dataclass
class MemoryItem:
    """记忆条目（内容层）。"""

    mem_id: str
    text: str
    created_at: float = field(default_factory=lambda: time.time())
    source: str = "unknown"
    meta: Dict[str, str] = field(default_factory=dict)
    summary: Optional[str] = None
    keywords: List[str] = field(default_factory=list)


@dataclass
class EmbeddingRecord:
    """向量记录（表示层）。"""

    emb_id: str
    mem_id: str
    encoder_id: str
    strategy: str
    dims: int
    n_vecs: int
    vecs: VectorGroup
    coarse_vec: Optional[Vector] = None
    aux: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class Query:
    """查询对象。"""

    query_id: str
    text: str
    context: Optional[str] = None
    encoder_id: str = ""
    strategy: str = ""
    q_vecs: Optional[VectorGroup] = None
    coarse_vec: Optional[Vector] = None
    aux: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class RetrieveResult:
    """检索结果。"""

    mem_id: str
    score: float
    coarse_score: float
    debug: Dict[str, List[float]] = field(default_factory=dict)
    features: Dict[str, float] = field(default_factory=dict)
    route_output: Optional["RouteOutput"] = None
    # 【新增】route_output 用于承接软/半硬/硬路由的可解释输出与评估指标。


@dataclass
class RouteOutput:
    """路由输出（用于后续阶段化路由）。"""

    policy: str
    weights: Dict[str, float] = field(default_factory=dict)
    selected_ids: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    explain: Dict[str, List[str]] = field(default_factory=dict)
    scores: Dict[str, float] = field(default_factory=dict)
    # 【新增】policy=soft/half_hard/hard；weights/selected_ids/metrics/explain 兼容阶段化路由。 
