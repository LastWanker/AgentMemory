from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from agentmemory_v3.config import cfg_get, load_yaml_config, resolve_path
from agentmemory_v3.utils.io import read_jsonl


GraphLevel = str
_LEVELS = {"L1", "L2", "L3"}


@dataclass(frozen=True)
class GraphLookup:
    level: GraphLevel
    name: str


def parse_graph_lookup(raw: str) -> GraphLookup:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("lookup is empty")
    for sep in ("，", ","):
        if sep in text:
            level_raw, name = text.split(sep, 1)
            level = str(level_raw or "").strip().upper()
            clean_name = str(name or "").strip()
            if level not in _LEVELS:
                raise ValueError(f"unsupported level: {level_raw}")
            if not clean_name:
                raise ValueError("name is empty")
            return GraphLookup(level=level, name=clean_name)
    parts = text.split(maxsplit=1)
    if len(parts) == 2 and str(parts[0]).strip().upper() in _LEVELS and str(parts[1]).strip():
        return GraphLookup(level=str(parts[0]).strip().upper(), name=str(parts[1]).strip())
    raise ValueError("lookup must look like 'L1, 拉丁字母'")


class AssociationGraphView:
    def __init__(
        self,
        *,
        concepts: list[dict],
        parent_edges: list[dict],
        bridge_edges: list[dict],
        assoc_dir: Path,
    ) -> None:
        self.assoc_dir = assoc_dir
        self.concepts = concepts
        self.parent_edges = parent_edges
        self.bridge_edges = bridge_edges
        self.node_by_id: dict[str, dict] = {}
        self.name_to_ids_by_level: dict[str, dict[str, list[str]]] = {level: {} for level in _LEVELS}
        self.parent_by_child: dict[str, dict] = {}
        self.children_by_parent: dict[str, list[dict]] = {}
        self.bridges_by_node: dict[str, list[dict]] = {}
        for concept in concepts:
            node_id = str(concept.get("node_id") or "").strip()
            level = str(concept.get("level") or "").strip().upper()
            name = str(concept.get("canonical_name") or "").strip()
            if not node_id or level not in _LEVELS or not name:
                continue
            node = {
                "id": node_id,
                "level": level,
                "name": name,
            }
            self.node_by_id[node_id] = node
            self.name_to_ids_by_level[level].setdefault(name, []).append(node_id)
        for raw_edge in parent_edges:
            child_id = str(raw_edge.get("child_id") or "").strip()
            parent_id = str(raw_edge.get("parent_id") or "").strip()
            conf = float(raw_edge.get("conf") or 0.0)
            if not child_id or not parent_id:
                continue
            edge = {"child_id": child_id, "parent_id": parent_id, "conf": conf}
            self.parent_by_child[child_id] = edge
            self.children_by_parent.setdefault(parent_id, []).append(edge)
        for parent_id, edges in self.children_by_parent.items():
            del parent_id
            edges.sort(
                key=lambda item: (
                    -float(item["conf"]),
                    self.node_by_id.get(str(item["child_id"]), {}).get("name", ""),
                )
            )
        for raw_edge in bridge_edges:
            source_id = str(raw_edge.get("source_id") or "").strip()
            target_id = str(raw_edge.get("target_id") or "").strip()
            weight = float(raw_edge.get("weight") or 0.0)
            if not source_id or not target_id:
                continue
            edge_st = {"source_id": source_id, "target_id": target_id, "weight": weight}
            edge_ts = {"source_id": target_id, "target_id": source_id, "weight": weight}
            self.bridges_by_node.setdefault(source_id, []).append(edge_st)
            self.bridges_by_node.setdefault(target_id, []).append(edge_ts)
        for node_id, edges in self.bridges_by_node.items():
            del node_id
            edges.sort(
                key=lambda item: (
                    -float(item["weight"]),
                    self.node_by_id.get(str(item["target_id"]), {}).get("name", ""),
                )
            )

    @classmethod
    def from_config(cls, config_path: str | Path) -> "AssociationGraphView":
        cfg = load_yaml_config(config_path)
        assoc_dir = resolve_path(cfg_get(cfg, "association.dir", "data/V5/association"))
        concepts = list(read_jsonl(assoc_dir / "concepts.jsonl"))
        parent_edges = list(read_jsonl(assoc_dir / "parent_edges.jsonl"))
        bridge_edges = list(read_jsonl(assoc_dir / "bridge_edges.jsonl"))
        return cls(concepts=concepts, parent_edges=parent_edges, bridge_edges=bridge_edges, assoc_dir=assoc_dir)

    def lookup(self, level: str, name: str) -> dict:
        clean_level = str(level or "").strip().upper()
        clean_name = str(name or "").strip()
        if clean_level not in _LEVELS:
            return {
                "ok": False,
                "error": f"unsupported level: {level}",
                "query": {"level": clean_level or str(level or ""), "name": clean_name},
            }
        if not clean_name:
            return {"ok": False, "error": "name is empty", "query": {"level": clean_level, "name": clean_name}}
        matched_ids = list(self.name_to_ids_by_level.get(clean_level, {}).get(clean_name, []))
        if not matched_ids:
            return {"ok": False, "error": "not_found", "query": {"level": clean_level, "name": clean_name}}
        if len(matched_ids) > 1:
            return {
                "ok": False,
                "error": "ambiguous_name",
                "query": {"level": clean_level, "name": clean_name},
                "matches": [self._node_payload(self.node_by_id[node_id]) for node_id in matched_ids if node_id in self.node_by_id],
            }
        node = self.node_by_id.get(matched_ids[0])
        if node is None:
            return {"ok": False, "error": "not_found", "query": {"level": clean_level, "name": clean_name}}
        if clean_level == "L1":
            result = self._lookup_l1(node)
        elif clean_level == "L2":
            result = self._lookup_l2(node, child_limit=50)
        else:
            result = self._lookup_l3(node, branch_limit=10, child_limit=10)
        return {"ok": True, "query": {"level": clean_level, "name": clean_name}, "result": result}

    def _lookup_l1(self, node: dict) -> dict:
        parent = self._parent_payload(node["id"])
        grandparent = self._parent_payload(parent["id"]) if parent else None
        return {
            "kind": "L1",
            "node": self._node_payload(node),
            "bridges": self._bridge_payload(node["id"], preview_limit=6),
            "parent": parent,
            "grandparent": grandparent,
        }

    def _lookup_l2(self, node: dict, *, child_limit: int) -> dict:
        parent = self._parent_payload(node["id"])
        child_edges = list(self.children_by_parent.get(node["id"], []))
        children = [self._child_payload(edge) for edge in child_edges[: max(1, int(child_limit))]]
        return {
            "kind": "L2",
            "node": self._node_payload(node),
            "bridges": self._bridge_payload(node["id"], preview_limit=6),
            "parent": parent,
            "children": children,
            "child_total": len(child_edges),
            "children_truncated": len(child_edges) > len(children),
        }

    def _lookup_l3(self, node: dict, *, branch_limit: int, child_limit: int) -> dict:
        branch_edges = list(self.children_by_parent.get(node["id"], []))
        branches = []
        for edge in branch_edges[: max(1, int(branch_limit))]:
            child_node = self.node_by_id.get(str(edge["child_id"]))
            if child_node is None:
                continue
            grand_edges = list(self.children_by_parent.get(child_node["id"], []))
            branches.append(
                {
                    "node": self._node_payload(child_node),
                    "conf": float(edge["conf"]),
                    "children": [self._child_payload(item) for item in grand_edges[: max(1, int(child_limit))]],
                    "child_total": len(grand_edges),
                    "children_truncated": len(grand_edges) > max(1, int(child_limit)),
                }
            )
        return {
            "kind": "L3",
            "node": self._node_payload(node),
            "bridges": self._bridge_payload(node["id"], preview_limit=6),
            "branches": branches,
            "branch_total": len(branch_edges),
            "branches_truncated": len(branch_edges) > len(branches),
        }

    def _parent_payload(self, child_id: str) -> dict | None:
        edge = self.parent_by_child.get(str(child_id))
        if edge is None:
            return None
        node = self.node_by_id.get(str(edge["parent_id"]))
        if node is None:
            return None
        payload = self._node_payload(node)
        payload["conf"] = float(edge["conf"])
        return payload

    def _child_payload(self, edge: dict) -> dict:
        node = self.node_by_id.get(str(edge["child_id"]))
        if node is None:
            return {
                "id": str(edge["child_id"]),
                "level": "",
                "name": str(edge["child_id"]),
                "conf": float(edge["conf"]),
            }
        payload = self._node_payload(node)
        payload["conf"] = float(edge["conf"])
        return payload

    def _bridge_payload(self, node_id: str, *, preview_limit: int) -> dict:
        edges = list(self.bridges_by_node.get(str(node_id), []))
        preview = [self._bridge_tag_payload(edge) for edge in edges[: max(1, int(preview_limit))]]
        return {
            "total": len(edges),
            "preview": preview,
            "preview_truncated": len(edges) > len(preview),
        }

    def _bridge_tag_payload(self, edge: dict) -> dict:
        target_id = str(edge["target_id"])
        node = self.node_by_id.get(target_id)
        if node is None:
            return {
                "id": target_id,
                "level": "",
                "name": target_id,
                "weight": float(edge["weight"]),
            }
        payload = self._node_payload(node)
        payload["weight"] = float(edge["weight"])
        return payload

    @staticmethod
    def _node_payload(node: dict) -> dict:
        return {
            "id": str(node["id"]),
            "level": str(node["level"]),
            "name": str(node["name"]),
        }

    def debug_summary(self) -> dict:
        manifest_path = self.assoc_dir / "manifest.json"
        if not manifest_path.exists():
            return {}
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
