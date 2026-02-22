from __future__ import annotations

import math
import re
from collections import Counter
from statistics import median
from typing import List, Sequence, Set, Tuple

from .models import UserTurn

_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+")


def lexical_cosine(a: str, b: str) -> float:
    ta = _tokenize(a)
    tb = _tokenize(b)
    if not ta or not tb:
        return 0.0
    ca = Counter(ta)
    cb = Counter(tb)
    shared = set(ca) & set(cb)
    dot = sum(ca[t] * cb[t] for t in shared)
    na = math.sqrt(sum(v * v for v in ca.values()))
    nb = math.sqrt(sum(v * v for v in cb.values()))
    if na <= 0 or nb <= 0:
        return 0.0
    return float(dot / (na * nb))


def assign_local_clusters(
    turns: List[UserTurn],
    *,
    mode: str = "adaptive",
    fixed_sim_threshold: float = 0.35,
    novelty_threshold: float = 0.80,
    min_segment_len: int = 2,
) -> None:
    if not turns:
        return
    if len(turns) == 1:
        turns[0].topic_break_flag = False
        turns[0].sim_to_prev = None
        cluster_id = f"{turns[0].session_id}:c1"
        turns[0].cluster_id = cluster_id
        turns[0].local_cluster_id = cluster_id
        return

    sims: List[float] = [0.0]
    for idx in range(1, len(turns)):
        sims.append(lexical_cosine(turns[idx - 1].text, turns[idx].text))

    if mode == "fixed":
        cut_dist = max(0.0, min(1.0, 1.0 - fixed_sim_threshold))
    else:
        distances = [1.0 - x for x in sims[1:]]
        cut_dist = _adaptive_distance_cut(distances)

    cluster_index = 1
    last_break_idx = 0
    turns[0].topic_break_flag = False
    turns[0].sim_to_prev = None
    seed_cluster_id = f"{turns[0].session_id}:c{cluster_index}"
    turns[0].cluster_id = seed_cluster_id
    turns[0].local_cluster_id = seed_cluster_id

    for idx in range(1, len(turns)):
        prev_text = turns[idx - 1].text
        curr_text = turns[idx].text
        sim = sims[idx]
        dist = 1.0 - sim
        turns[idx].sim_to_prev = sim

        prev_tokens = set(_tokenize(prev_text))
        curr_tokens = set(_tokenize(curr_text))
        novelty = _novelty_ratio(prev_tokens, curr_tokens)
        enough_gap = (idx - last_break_idx) >= max(1, min_segment_len)
        is_break = bool(dist >= cut_dist and novelty >= novelty_threshold and enough_gap)
        turns[idx].topic_break_flag = is_break
        if is_break:
            cluster_index += 1
            last_break_idx = idx

        cluster_id = f"{turns[idx].session_id}:c{cluster_index}"
        turns[idx].cluster_id = cluster_id
        turns[idx].local_cluster_id = cluster_id


def _adaptive_distance_cut(distances: Sequence[float]) -> float:
    if not distances:
        return 0.65
    if len(distances) <= 3:
        return max(0.60, min(0.85, max(distances)))
    sorted_d = sorted(distances)
    q1 = _quantile(sorted_d, 0.25)
    q3 = _quantile(sorted_d, 0.75)
    iqr = max(1e-6, q3 - q1)
    robust_cut = q3 + 1.5 * iqr
    median_cut = median(sorted_d) + 0.5 * iqr
    # Clamp range to avoid over-fragmentation and over-merging.
    return float(max(0.55, min(0.92, max(robust_cut, median_cut))))


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if q <= 0:
        return float(sorted_values[0])
    if q >= 1:
        return float(sorted_values[-1])
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    w = pos - lo
    return float((1.0 - w) * sorted_values[lo] + w * sorted_values[hi])


def _tokenize(text: str) -> List[str]:
    return [x.lower() for x in _TOKEN_RE.findall(text or "")]


def _novelty_ratio(prev_tokens: Set[str], curr_tokens: Set[str]) -> float:
    if not curr_tokens:
        return 0.0
    new_tokens = [token for token in curr_tokens if token not in prev_tokens]
    return len(new_tokens) / max(1, len(curr_tokens))
