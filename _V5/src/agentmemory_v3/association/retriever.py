from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from agentmemory_v3.config import cfg_get, load_yaml_config, resolve_path
from agentmemory_v3.encoder import HFSentenceEncoder, SentenceEncoderConfig
from agentmemory_v3.retrieval.interfaces import RetrieveResult
from agentmemory_v3.utils.io import read_jsonl


@dataclass(frozen=True)
class AssociationRetrieveConfig:
    seed_candidate_pool: int = 40
    seed_top_k: int = 0
    seed_top_k_base: int = 5
    seed_top_k_cap: int = 12
    string_boost_value: float = 0.03
    string_boost_max_extra: int = 1
    packet_ttl: int = 2
    alpha_up: float = 0.8
    alpha_down: float = 0.4
    alpha_bridge: float = 0.55
    descend_sample_ratio: float = 0.5
    descend_max_children_l3: int = 5
    descend_max_children_l2: int = 5
    bridge_max_neighbors: int = 5
    conf_sharpen_tau: float = 35.0
    top_l1_for_memory: int = 500
    memory_noisy_or_lambda: float = 0.85
    seed_bonus: float = 0.03
    bridge_bonus: float = 0.05
    activation_min_score: float = 0.05
    max_queue_steps: int = 20000
    max_reasons_per_node: int = 10
    max_memory_breakdowns: int = 20


@dataclass(frozen=True)
class SeedCandidate:
    node_id: str
    level: str
    name: str
    dense_score: float
    final_score: float
    source_kind: str
    string_boost: float = 0.0


@dataclass(frozen=True)
class ActivationPacket:
    packet_id: str
    node_id: str
    level: str
    score: float
    ttl: int
    source_kind: str
    origin_type: str
    came_from_node_id: str | None
    blocked_node_id: str | None
    hop_down_count: int


@dataclass
class NodeActivationState:
    node_id: str
    level: str
    name: str
    best_score: float = 0.0
    hit_count: int = 0
    first_source_kind: str = ""
    reasons: list[dict] = field(default_factory=list)
    origin_types: set[str] = field(default_factory=set)


class AssociationRetriever:
    def __init__(
        self,
        *,
        memory_rows: list[dict],
        concepts: list[dict],
        concept_ids: list[str],
        concept_matrix: np.ndarray,
        parent_edges: list[dict],
        bridge_edges: list[dict],
        memory_links: dict[str, list[str]],
        encoder: HFSentenceEncoder,
        cfg: AssociationRetrieveConfig,
    ) -> None:
        self.memory_rows = memory_rows
        self.memory_by_id = {str(row.get("memory_id") or ""): row for row in memory_rows}
        self.concepts = concepts
        self.concept_ids = [str(item) for item in concept_ids]
        self.concept_matrix = np.asarray(concept_matrix, dtype=np.float32)
        self.parent_edges = parent_edges
        self.bridge_edges = bridge_edges
        self.memory_links = memory_links
        self.encoder = encoder
        self.cfg = cfg

        self.node_by_id: dict[str, dict] = {}
        self.name_by_node_id: dict[str, str] = {}
        self.level_by_node_id: dict[str, str] = {}
        self.node_ids_by_level: dict[str, list[str]] = {"L1": [], "L2": [], "L3": []}
        self.name_to_node_ids: dict[str, list[str]] = defaultdict(list)
        for item in concepts:
            node_id = str(item.get("node_id") or "").strip()
            level = str(item.get("level") or "").strip().upper()
            name = str(item.get("canonical_name") or "").strip()
            if not node_id or level not in {"L1", "L2", "L3"}:
                continue
            self.node_by_id[node_id] = item
            self.name_by_node_id[node_id] = name
            self.level_by_node_id[node_id] = level
            self.node_ids_by_level[level].append(node_id)
            if name:
                self.name_to_node_ids[name].append(node_id)

        self.concept_index_by_node_id = {node_id: idx for idx, node_id in enumerate(self.concept_ids)}

        self.l1_parent: dict[str, tuple[str, float]] = {}
        self.l2_parent: dict[str, tuple[str, float]] = {}
        self.l2_children: dict[str, list[tuple[str, float]]] = defaultdict(list)
        self.l3_children: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for edge in parent_edges:
            child_id = str(edge.get("child_id") or "").strip()
            parent_id = str(edge.get("parent_id") or "").strip()
            conf = float(edge.get("conf") or 0.0)
            if child_id.startswith("l1_"):
                self.l1_parent[child_id] = (parent_id, conf)
                self.l2_children[parent_id].append((child_id, conf))
            elif child_id.startswith("l2_"):
                self.l2_parent[child_id] = (parent_id, conf)
                self.l3_children[parent_id].append((child_id, conf))
        for parent_id in list(self.l2_children.keys()):
            self.l2_children[parent_id].sort(key=lambda item: (-float(item[1]), self.name_by_node_id.get(item[0], "")))
        for parent_id in list(self.l3_children.keys()):
            self.l3_children[parent_id].sort(key=lambda item: (-float(item[1]), self.name_by_node_id.get(item[0], "")))

        self.bridge_adj: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for edge in bridge_edges:
            src = str(edge.get("source_id") or "").strip()
            tgt = str(edge.get("target_id") or "").strip()
            weight = float(edge.get("weight") or 0.0)
            if not src or not tgt:
                continue
            self.bridge_adj[src].append((tgt, weight))
            self.bridge_adj[tgt].append((src, weight))
        for node_id in list(self.bridge_adj.keys()):
            self.bridge_adj[node_id].sort(key=lambda item: (-float(item[1]), self.name_by_node_id.get(item[0], "")))

        self.l1_to_memory_ids: dict[str, list[str]] = defaultdict(list)
        for memory_id, l1_ids in memory_links.items():
            for node_id in l1_ids:
                self.l1_to_memory_ids[str(node_id)].append(str(memory_id))

    @classmethod
    def from_config(cls, config_path: str | Path) -> "AssociationRetriever":
        cfg = load_yaml_config(config_path)
        root_dir = resolve_path(cfg_get(cfg, "data.root_dir", "data/V5"))
        assoc_dir = resolve_path(cfg_get(cfg, "association.dir", "data/V5/association"))
        memory_rows = list(read_jsonl(root_dir / "processed" / "memory.jsonl"))
        concepts = list(read_jsonl(assoc_dir / "concepts.jsonl"))
        parent_edges = list(read_jsonl(assoc_dir / "parent_edges.jsonl"))
        bridge_edges = list(read_jsonl(assoc_dir / "bridge_edges.jsonl"))
        concept_ids = json.loads((assoc_dir / "concept_ids.json").read_text(encoding="utf-8"))
        concept_matrix = np.load(assoc_dir / "concept_matrix.npy").astype(np.float32)
        memory_links_rows = list(read_jsonl(assoc_dir / "memory_links.jsonl"))
        memory_links = {
            str(row.get("memory_id") or ""): [str(item) for item in row.get("l1_ids") or []] for row in memory_links_rows
        }
        encoder = HFSentenceEncoder(
            SentenceEncoderConfig(
                model_name=str(cfg_get(cfg, "encoder.model_name", "intfloat/multilingual-e5-small")),
                use_e5_prefix=bool(cfg_get(cfg, "encoder.use_e5_prefix", True)),
                local_files_only=bool(cfg_get(cfg, "encoder.local_files_only", True)),
                offline=bool(cfg_get(cfg, "encoder.offline", True)),
                device=str(cfg_get(cfg, "encoder.device", "auto")),
                batch_size=int(cfg_get(cfg, "encoder.batch_size_cuda", cfg_get(cfg, "encoder.batch_size", 128))),
            )
        )
        retrieve_cfg = AssociationRetrieveConfig(
            seed_candidate_pool=int(cfg_get(cfg, "association.retrieve.seed_candidate_pool", cfg_get(cfg, "association.retrieve.seed_top_r", 40))),
            seed_top_k=int(cfg_get(cfg, "association.retrieve.seed_top_k", 0)),
            seed_top_k_base=int(cfg_get(cfg, "association.retrieve.seed_top_k_base", 5)),
            seed_top_k_cap=int(cfg_get(cfg, "association.retrieve.seed_top_k_cap", 12)),
            string_boost_value=float(cfg_get(cfg, "association.retrieve.string_boost_value", 0.03)),
            string_boost_max_extra=int(cfg_get(cfg, "association.retrieve.string_boost_max_extra", 1)),
            packet_ttl=int(cfg_get(cfg, "association.retrieve.packet_ttl", 2)),
            alpha_up=float(cfg_get(cfg, "association.retrieve.alpha_up", 0.8)),
            alpha_down=float(cfg_get(cfg, "association.retrieve.alpha_down", 0.4)),
            alpha_bridge=float(cfg_get(cfg, "association.retrieve.alpha_bridge", 0.55)),
            descend_sample_ratio=float(cfg_get(cfg, "association.retrieve.descend_sample_ratio", 0.5)),
            descend_max_children_l3=int(cfg_get(cfg, "association.retrieve.descend_max_children_l3", 5)),
            descend_max_children_l2=int(cfg_get(cfg, "association.retrieve.descend_max_children_l2", 5)),
            bridge_max_neighbors=int(cfg_get(cfg, "association.retrieve.bridge_max_neighbors", 5)),
            conf_sharpen_tau=float(cfg_get(cfg, "association.retrieve.conf_sharpen_tau", 35.0)),
            top_l1_for_memory=int(cfg_get(cfg, "association.retrieve.top_l1_for_memory", 500)),
            memory_noisy_or_lambda=float(cfg_get(cfg, "association.retrieve.memory_noisy_or_lambda", 0.85)),
            seed_bonus=float(cfg_get(cfg, "association.retrieve.seed_bonus", 0.03)),
            bridge_bonus=float(cfg_get(cfg, "association.retrieve.bridge_bonus", 0.05)),
            activation_min_score=float(cfg_get(cfg, "association.retrieve.activation_min_score", 0.05)),
            max_queue_steps=int(cfg_get(cfg, "association.retrieve.max_queue_steps", 20000)),
            max_reasons_per_node=int(cfg_get(cfg, "association.retrieve.max_reasons_per_node", 10)),
            max_memory_breakdowns=int(cfg_get(cfg, "association.retrieve.max_memory_breakdowns", 20)),
        )
        return cls(
            memory_rows=memory_rows,
            concepts=concepts,
            concept_ids=concept_ids,
            concept_matrix=concept_matrix,
            parent_edges=parent_edges,
            bridge_edges=bridge_edges,
            memory_links=memory_links,
            encoder=encoder,
            cfg=retrieve_cfg,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        *,
        exclude_memory_ids: set[str] | None = None,
    ) -> tuple[list[RetrieveResult], dict]:
        debug = self.retrieve_debug(query, top_k=top_k, exclude_memory_ids=exclude_memory_ids or set())
        return debug["hits"], debug["trace"]

    def retrieve_debug(
        self,
        query: str,
        top_k: int = 20,
        *,
        exclude_memory_ids: set[str] | None = None,
    ) -> dict:
        query_text = str(query or "").strip()
        if not query_text:
            return {"hits": [], "trace": {"query": query_text, "error": "empty_query"}}
        if self.concept_matrix.size == 0:
            return {"hits": [], "trace": {"query": query_text, "error": "empty_concept_matrix"}}

        exclude = exclude_memory_ids or set()
        rng = np.random.default_rng(_stable_seed(query_text))
        seed_candidates, seed_trace = self._build_seed_candidates(query_text)
        packets = deque(self._init_packets(seed_candidates))
        activation_states: dict[str, NodeActivationState] = {}
        packet_best_scores: dict[tuple[str, str, str, str, int], float] = {}
        packet_trace: list[dict] = []
        step_count = 0

        while packets and step_count < max(1, int(self.cfg.max_queue_steps)):
            packet = packets.popleft()
            step_count += 1
            if float(packet.score) < float(self.cfg.activation_min_score):
                continue
            accepted, packet_reason = self._accept_packet(packet, activation_states, packet_best_scores)
            if not accepted:
                continue
            if packet_reason:
                packet_trace.append(packet_reason)
            for nxt in self._expand_packet(packet, rng):
                if float(nxt.score) >= float(self.cfg.activation_min_score):
                    packets.append(nxt)

        hits, memory_breakdown = self._build_hits(
            activation_states=activation_states,
            top_k=max(1, int(top_k)),
            exclude_memory_ids=exclude,
        )
        top_l1 = self._top_activation_nodes(activation_states, level="L1", top_n=12)
        top_l2 = self._top_activation_nodes(activation_states, level="L2", top_n=8)
        top_l3 = self._top_activation_nodes(activation_states, level="L3", top_n=5)
        bridge_hits = [item for item in top_l1 if item["via_bridge"]]
        trace = {
            "query": query_text,
            "seed_resolution": seed_trace,
            "accepted_seed_count": len(seed_candidates),
            "activation_counts": self._activation_counts(activation_states),
            "queue_steps": step_count,
            "top_l1": top_l1,
            "top_l2": top_l2,
            "top_l3": top_l3,
            "bridge_hits": bridge_hits,
            "packet_trace_sample": packet_trace[:200],
            "memory_score_breakdown": memory_breakdown,
        }
        return {
            "hits": hits,
            "trace": trace,
            "activation_summary": {"l1": top_l1, "l2": top_l2, "l3": top_l3},
        }

    def _build_seed_candidates(self, query_text: str) -> tuple[list[SeedCandidate], dict]:
        query_vec = self.encoder.encode_query_texts([query_text])[0]
        scores = self.concept_matrix @ np.asarray(query_vec, dtype=np.float32)
        pool_limit = min(max(1, int(self.cfg.seed_candidate_pool)), int(scores.shape[0]))
        if pool_limit >= scores.shape[0]:
            top_idx = np.argsort(scores)[::-1]
        else:
            part = np.argpartition(scores, -pool_limit)[-pool_limit:]
            top_idx = part[np.argsort(scores[part])[::-1]]

        raw_candidates: list[SeedCandidate] = []
        for idx in top_idx:
            score = float(scores[int(idx)])
            if score <= 0:
                continue
            node_id = self.concept_ids[int(idx)]
            raw_candidates.append(
                SeedCandidate(
                    node_id=node_id,
                    level=self.level_by_node_id.get(node_id, ""),
                    name=self.name_by_node_id.get(node_id, node_id),
                    dense_score=score,
                    final_score=score,
                    source_kind="dense",
                )
            )

        seed_top_k = self._resolve_seed_top_k(query_text)
        accepted = raw_candidates[:seed_top_k]
        boosted = self._apply_string_boost(query_text, scores, accepted)
        accepted_by_id = {item.node_id: item for item in accepted}
        for item in boosted:
            prev = accepted_by_id.get(item.node_id)
            if prev is None or float(item.final_score) > float(prev.final_score):
                accepted_by_id[item.node_id] = item
        accepted_final = sorted(
            accepted_by_id.values(),
            key=lambda item: (float(item.final_score), float(item.dense_score)),
            reverse=True,
        )
        accepted_final = accepted_final[: max(1, seed_top_k + int(self.cfg.string_boost_max_extra))]

        trace = {
            "seed_top_k": seed_top_k,
            "candidate_pool": pool_limit,
            "raw_candidates": [self._seed_payload(item) for item in raw_candidates[:20]],
            "accepted_seeds": [self._seed_payload(item) for item in accepted_final],
            "string_boost_hits": [self._seed_payload(item) for item in accepted_final if float(item.string_boost) > 0.0],
        }
        return accepted_final, trace

    def _resolve_seed_top_k(self, query_text: str) -> int:
        explicit = int(self.cfg.seed_top_k)
        if explicit > 0:
            return explicit
        size = max(1, len(query_text))
        value = int(self.cfg.seed_top_k_base) + int(math.floor(math.log10(size)))
        return max(1, min(int(self.cfg.seed_top_k_cap), value))

    def _apply_string_boost(
        self,
        query_text: str,
        scores: np.ndarray,
        accepted: list[SeedCandidate],
    ) -> list[SeedCandidate]:
        limit = max(0, int(self.cfg.string_boost_max_extra))
        if limit <= 0:
            return []
        current_ids = {item.node_id for item in accepted}
        matches: list[SeedCandidate] = []
        for name, node_ids in self.name_to_node_ids.items():
            if not name or name not in query_text:
                continue
            for node_id in node_ids:
                idx = self.concept_index_by_node_id.get(node_id)
                if idx is None:
                    continue
                dense_score = float(scores[int(idx)])
                matches.append(
                    SeedCandidate(
                        node_id=node_id,
                        level=self.level_by_node_id.get(node_id, ""),
                        name=self.name_by_node_id.get(node_id, node_id),
                        dense_score=dense_score,
                        final_score=dense_score + float(self.cfg.string_boost_value),
                        source_kind="string_boost" if node_id not in current_ids else "dense",
                        string_boost=float(self.cfg.string_boost_value),
                    )
                )
        if not matches:
            return []
        matches.sort(
            key=lambda item: (
                float(item.final_score),
                len(item.name),
                float(item.dense_score),
            ),
            reverse=True,
        )
        out: list[SeedCandidate] = []
        seen: set[str] = set()
        extra_left = limit
        for item in matches:
            if item.node_id in seen:
                continue
            seen.add(item.node_id)
            if item.source_kind == "string_boost":
                if extra_left <= 0:
                    continue
                extra_left -= 1
            out.append(item)
        return out

    def _init_packets(self, seeds: list[SeedCandidate]) -> list[ActivationPacket]:
        out = []
        ttl = max(0, int(self.cfg.packet_ttl))
        for idx, seed in enumerate(seeds, start=1):
            out.append(
                ActivationPacket(
                    packet_id=f"seed-{idx:03d}",
                    node_id=seed.node_id,
                    level=seed.level,
                    score=float(seed.final_score),
                    ttl=ttl,
                    source_kind=seed.source_kind,
                    origin_type="direct",
                    came_from_node_id=None,
                    blocked_node_id=None,
                    hop_down_count=0,
                )
            )
        return out

    def _accept_packet(
        self,
        packet: ActivationPacket,
        activation_states: dict[str, NodeActivationState],
        packet_best_scores: dict[tuple[str, str, str, str, int], float],
    ) -> tuple[bool, dict | None]:
        packet_key = (
            str(packet.node_id),
            str(packet.origin_type),
            str(packet.came_from_node_id or ""),
            str(packet.blocked_node_id or ""),
            int(packet.ttl),
        )
        prev_packet_score = float(packet_best_scores.get(packet_key, -1.0))
        if float(packet.score) <= prev_packet_score + 1e-6:
            return False, None
        packet_best_scores[packet_key] = float(packet.score)

        state = activation_states.get(packet.node_id)
        if state is None:
            state = NodeActivationState(
                node_id=packet.node_id,
                level=packet.level,
                name=self.name_by_node_id.get(packet.node_id, packet.node_id),
                best_score=float(packet.score),
                hit_count=1,
                first_source_kind=str(packet.source_kind),
                reasons=[],
                origin_types={str(packet.origin_type)},
            )
            activation_states[packet.node_id] = state
        else:
            state.best_score = max(float(state.best_score), float(packet.score))
            state.hit_count += 1
            state.origin_types.add(str(packet.origin_type))

        reason = {
            "packet_id": packet.packet_id,
            "node_id": packet.node_id,
            "name": self.name_by_node_id.get(packet.node_id, packet.node_id),
            "level": packet.level,
            "score": float(packet.score),
            "ttl": int(packet.ttl),
            "origin_type": packet.origin_type,
            "source_kind": packet.source_kind,
            "came_from_node_id": packet.came_from_node_id,
            "blocked_node_id": packet.blocked_node_id,
        }
        if len(state.reasons) < max(1, int(self.cfg.max_reasons_per_node)):
            state.reasons.append(reason)
        return True, reason

    def _expand_packet(self, packet: ActivationPacket, rng: np.random.Generator) -> list[ActivationPacket]:
        out: list[ActivationPacket] = []
        out.extend(self._run_ascend(packet))
        out.extend(self._run_descend(packet, rng))
        out.extend(self._run_bridge(packet, rng))
        return out

    def _run_ascend(self, packet: ActivationPacket) -> list[ActivationPacket]:
        if packet.origin_type == "from_parent":
            return []
        if packet.level == "L1":
            parent = self.l1_parent.get(packet.node_id)
        elif packet.level == "L2":
            parent = self.l2_parent.get(packet.node_id)
        else:
            parent = None
        if not parent:
            return []
        parent_id, conf = parent
        return [
            ActivationPacket(
                packet_id=self._child_packet_id(packet, "asc", parent_id),
                node_id=parent_id,
                level=self.level_by_node_id.get(parent_id, ""),
                score=float(packet.score) * float(conf) * float(self.cfg.alpha_up),
                ttl=int(packet.ttl),
                source_kind=packet.source_kind,
                origin_type="from_child",
                came_from_node_id=packet.node_id,
                blocked_node_id=packet.node_id,
                hop_down_count=int(packet.hop_down_count),
            )
        ]

    def _run_descend(self, packet: ActivationPacket, rng: np.random.Generator) -> list[ActivationPacket]:
        if int(packet.ttl) <= 0:
            return []
        if packet.level == "L3":
            children = self.l3_children.get(packet.node_id, [])
            limit = int(self.cfg.descend_max_children_l3)
        elif packet.level == "L2":
            children = self.l2_children.get(packet.node_id, [])
            limit = int(self.cfg.descend_max_children_l2)
        else:
            return []
        candidates = [(child_id, conf) for child_id, conf in children if child_id != str(packet.blocked_node_id or "")]
        chosen = self._sample_children(candidates, rng=rng, limit=limit)
        out = []
        for child_id, conf in chosen:
            out.append(
                ActivationPacket(
                    packet_id=self._child_packet_id(packet, "des", child_id),
                    node_id=child_id,
                    level=self.level_by_node_id.get(child_id, ""),
                    score=float(packet.score) * float(conf) * float(self.cfg.alpha_down),
                    ttl=max(0, int(packet.ttl) - 1),
                    source_kind=packet.source_kind,
                    origin_type="from_parent",
                    came_from_node_id=packet.node_id,
                    blocked_node_id=packet.node_id,
                    hop_down_count=int(packet.hop_down_count) + 1,
                )
            )
        return out

    def _run_bridge(self, packet: ActivationPacket, rng: np.random.Generator) -> list[ActivationPacket]:
        if packet.level != "L1" or int(packet.ttl) <= 0:
            return []
        neighbors = [
            (node_id, weight)
            for node_id, weight in self.bridge_adj.get(packet.node_id, [])
            if node_id != str(packet.blocked_node_id or "")
        ]
        chosen = self._sample_children(neighbors, rng=rng, limit=int(self.cfg.bridge_max_neighbors))
        out = []
        for peer_id, weight in chosen:
            out.append(
                ActivationPacket(
                    packet_id=self._child_packet_id(packet, "brg", peer_id),
                    node_id=peer_id,
                    level="L1",
                    score=float(packet.score) * float(weight) * float(self.cfg.alpha_bridge),
                    ttl=max(0, int(packet.ttl) - 1),
                    source_kind=packet.source_kind,
                    origin_type="from_bridge",
                    came_from_node_id=packet.node_id,
                    blocked_node_id=packet.node_id,
                    hop_down_count=int(packet.hop_down_count) + 1,
                )
            )
        return out

    def _sample_children(
        self,
        candidates: list[tuple[str, float]],
        *,
        rng: np.random.Generator,
        limit: int,
    ) -> list[tuple[str, float]]:
        if not candidates:
            return []
        count = len(candidates)
        take = min(max(1, int(round(count * float(self.cfg.descend_sample_ratio)))), max(1, int(limit)), count)
        weights = np.asarray([max(0.0, float(conf)) for _, conf in candidates], dtype=np.float64)
        max_conf = float(np.max(weights))
        tau = float(self.cfg.conf_sharpen_tau)
        sharpened = np.exp(np.clip(tau * (weights - max_conf), -60.0, 0.0))
        if float(np.sum(sharpened)) <= 1e-12:
            sharpened = np.ones_like(sharpened)
        probs = sharpened / np.sum(sharpened)
        chosen_idx = rng.choice(np.arange(count), size=take, replace=False, p=probs)
        chosen = [candidates[int(idx)] for idx in chosen_idx]
        chosen.sort(key=lambda item: (-float(item[1]), self.name_by_node_id.get(item[0], "")))
        return chosen

    def _build_hits(
        self,
        *,
        activation_states: dict[str, NodeActivationState],
        top_k: int,
        exclude_memory_ids: set[str],
    ) -> tuple[list[RetrieveResult], list[dict]]:
        bright_l1 = [state for state in activation_states.values() if state.level == "L1" and float(state.best_score) > 0]
        bright_l1.sort(key=lambda item: float(item.best_score), reverse=True)
        bright_l1 = bright_l1[: max(1, int(self.cfg.top_l1_for_memory))]
        bright_by_id = {state.node_id: state for state in bright_l1}

        scored: list[tuple[float, str, dict]] = []
        for memory_id, l1_ids in self.memory_links.items():
            if memory_id in exclude_memory_ids:
                continue
            matched_states = [bright_by_id[node_id] for node_id in l1_ids if node_id in bright_by_id]
            if not matched_states:
                continue
            probs = []
            local_hits = []
            direct_hit = False
            bridge_hit = False
            for state in sorted(matched_states, key=lambda item: float(item.best_score), reverse=True):
                p_i = min(0.95, max(0.0, float(self.cfg.memory_noisy_or_lambda) * float(state.best_score)))
                probs.append(p_i)
                if "direct" in state.origin_types:
                    direct_hit = True
                if "from_bridge" in state.origin_types:
                    bridge_hit = True
                local_hits.append(
                    {
                        "node_id": state.node_id,
                        "name": state.name,
                        "score": float(state.best_score),
                        "origins": sorted(state.origin_types),
                        "local_prob": float(p_i),
                    }
                )
            miss_prob = 1.0
            for p_i in probs:
                miss_prob *= max(0.0, 1.0 - float(p_i))
            score = 1.0 - miss_prob
            if direct_hit:
                score += float(self.cfg.seed_bonus)
            if bridge_hit:
                score += float(self.cfg.bridge_bonus)
            row = self.memory_by_id.get(memory_id)
            if row is None:
                continue
            scored.append(
                (
                    float(score),
                    memory_id,
                    {
                        "memory_id": memory_id,
                        "cluster_id": str(row.get("cluster_id") or memory_id),
                        "score": float(score),
                        "matched_l1": local_hits[:10],
                        "direct_hit": bool(direct_hit),
                        "bridge_hit": bool(bridge_hit),
                    },
                )
            )

        scored.sort(key=lambda item: item[0], reverse=True)
        hits: list[RetrieveResult] = []
        breakdowns: list[dict] = []
        for score, memory_id, breakdown in scored:
            row = self.memory_by_id.get(memory_id)
            if row is None:
                continue
            hits.append(
                RetrieveResult(
                    memory_id=memory_id,
                    cluster_id=str(row.get("cluster_id") or memory_id),
                    score=float(score),
                    source="association:activation_v1",
                    display_text=str(row.get("raw_text") or row.get("text") or ""),
                )
            )
            if len(breakdowns) < max(1, int(self.cfg.max_memory_breakdowns)):
                breakdowns.append(breakdown)
            if len(hits) >= max(1, int(top_k)):
                break
        return hits, breakdowns

    def _top_activation_nodes(
        self,
        activation_states: dict[str, NodeActivationState],
        *,
        level: str,
        top_n: int,
    ) -> list[dict]:
        rows = [state for state in activation_states.values() if state.level == level]
        rows.sort(key=lambda item: float(item.best_score), reverse=True)
        out = []
        for state in rows[: max(1, int(top_n))]:
            out.append(
                {
                    "node_id": state.node_id,
                    "name": state.name,
                    "score": float(state.best_score),
                    "hit_count": int(state.hit_count),
                    "origins": sorted(state.origin_types),
                    "via_bridge": "from_bridge" in state.origin_types,
                }
            )
        return out

    @staticmethod
    def _activation_counts(activation_states: dict[str, NodeActivationState]) -> dict:
        counts = {"L1": 0, "L2": 0, "L3": 0}
        for state in activation_states.values():
            if state.level in counts:
                counts[state.level] += 1
        return counts

    @staticmethod
    def _child_packet_id(packet: ActivationPacket, action: str, target_id: str) -> str:
        raw = f"{packet.packet_id}|{action}|{target_id}|{packet.ttl}|{packet.score:.6f}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def _seed_payload(item: SeedCandidate) -> dict:
        return {
            "node_id": item.node_id,
            "level": item.level,
            "name": item.name,
            "dense_score": float(item.dense_score),
            "final_score": float(item.final_score),
            "source_kind": item.source_kind,
            "string_boost": float(item.string_boost),
        }


def _stable_seed(text: str) -> int:
    digest = hashlib.md5(str(text).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)
