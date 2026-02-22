from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .models import UserTurn


@dataclass
class ClusterUnit:
    local_cluster_id: str
    session_id: str
    turn_ids: List[str]
    text: str


def build_cluster_units(turns: Sequence[UserTurn]) -> List[ClusterUnit]:
    grouped: Dict[str, List[UserTurn]] = defaultdict(list)
    for turn in turns:
        if turn.is_excluded:
            continue
        grouped[turn.local_cluster_id].append(turn)

    units: List[ClusterUnit] = []
    for cluster_id, rows in grouped.items():
        rows = sorted(rows, key=lambda item: int(item.turn_id))
        head = rows[:3]
        text = " ".join(item.text for item in head if item.text.strip())
        units.append(
            ClusterUnit(
                local_cluster_id=cluster_id,
                session_id=rows[0].session_id,
                turn_ids=[item.turn_id for item in rows],
                text=text,
            )
        )
    return units


def assign_global_clusters(
    turns: List[UserTurn],
    *,
    enable_cross_session: bool = True,
    merge_similarity_threshold: float | None = None,
) -> None:
    units = build_cluster_units(turns)
    if not units:
        return
    if len(units) == 1:
        gid = "g0001"
        for turn in turns:
            if turn.local_cluster_id == units[0].local_cluster_id:
                turn.global_cluster_id = gid
                turn.cluster_id = gid
        return

    texts = [unit.text for unit in units]
    sims = _safe_pairwise_similarity(texts)
    if sims is None:
        # Fallback: no reliable cross-cluster lexical signal, keep local clusters separated.
        _assign_no_merge(turns, units)
        return
    threshold = (
        float(merge_similarity_threshold)
        if merge_similarity_threshold is not None
        else _adaptive_merge_threshold(sims)
    )

    parent = list(range(len(units)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(units)):
        for j in range(i + 1, len(units)):
            if not enable_cross_session and units[i].session_id != units[j].session_id:
                continue
            if sims[i, j] >= threshold:
                union(i, j)

    root_to_gid: Dict[int, str] = {}
    gid_counter = 1
    local_to_gid: Dict[str, str] = {}
    for idx, unit in enumerate(units):
        root = find(idx)
        if root not in root_to_gid:
            root_to_gid[root] = f"g{gid_counter:04d}"
            gid_counter += 1
        local_to_gid[unit.local_cluster_id] = root_to_gid[root]

    for turn in turns:
        gid = local_to_gid.get(turn.local_cluster_id)
        if not gid:
            continue
        turn.global_cluster_id = gid
        turn.cluster_id = gid


def _safe_pairwise_similarity(texts: Sequence[str]):
    if not texts:
        return None
    if sum(1 for text in texts if text.strip()) <= 1:
        return None
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        min_df=1,
        max_features=5000,
    )
    try:
        x = vec.fit_transform(texts)
        return cosine_similarity(x)
    except ValueError:
        return None


def _assign_no_merge(turns: List[UserTurn], units: Sequence[ClusterUnit]) -> None:
    local_to_gid: Dict[str, str] = {}
    for idx, unit in enumerate(units, start=1):
        local_to_gid[unit.local_cluster_id] = f"g{idx:04d}"
    for turn in turns:
        gid = local_to_gid.get(turn.local_cluster_id)
        if not gid:
            continue
        turn.global_cluster_id = gid
        turn.cluster_id = gid


def _adaptive_merge_threshold(sims: np.ndarray) -> float:
    if sims.size <= 1:
        return 0.62
    vals: List[float] = []
    n = sims.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            vals.append(float(sims[i, j]))
    if not vals:
        return 0.62
    vals.sort()
    p90 = vals[int(0.90 * (len(vals) - 1))]
    p75 = vals[int(0.75 * (len(vals) - 1))]
    return max(0.50, min(0.88, max(p75 + 0.08, p90)))
