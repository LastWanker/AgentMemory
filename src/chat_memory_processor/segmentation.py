from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, List, Set, Tuple

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


def should_break_topic(
    prev_text: str,
    curr_text: str,
    *,
    sim_threshold: float,
    novelty_threshold: float = 0.80,
) -> Tuple[bool, float]:
    sim = lexical_cosine(prev_text, curr_text)
    if sim >= sim_threshold:
        return False, sim

    prev_tokens = set(_tokenize(prev_text))
    curr_tokens = set(_tokenize(curr_text))
    novelty = _novelty_ratio(prev_tokens, curr_tokens)

    # Topic break candidate: low similarity + high novelty.
    is_break = novelty >= novelty_threshold
    return is_break, sim


def assign_clusters(turns: List[UserTurn], *, sim_threshold: float) -> None:
    if not turns:
        return

    cluster_index = 1
    turns[0].cluster_id = f"{turns[0].session_id}:c{cluster_index}"
    turns[0].topic_break_flag = False
    turns[0].sim_to_prev = None

    for idx in range(1, len(turns)):
        prev_turn = turns[idx - 1]
        curr_turn = turns[idx]
        is_break, sim = should_break_topic(
            prev_turn.text,
            curr_turn.text,
            sim_threshold=sim_threshold,
        )
        curr_turn.sim_to_prev = sim
        curr_turn.topic_break_flag = is_break
        if is_break:
            cluster_index += 1
        curr_turn.cluster_id = f"{curr_turn.session_id}:c{cluster_index}"


def _tokenize(text: str) -> List[str]:
    return [x.lower() for x in _TOKEN_RE.findall(text or "")]


def _novelty_ratio(prev_tokens: Set[str], curr_tokens: Set[str]) -> float:
    if not curr_tokens:
        return 0.0
    new_tokens = [token for token in curr_tokens if token not in prev_tokens]
    return len(new_tokens) / max(1, len(curr_tokens))

