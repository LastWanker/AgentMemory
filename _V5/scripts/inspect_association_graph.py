from __future__ import annotations

import argparse
import sys
from pathlib import Path


V3_ROOT = Path(__file__).resolve().parents[1]
if str(V3_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(V3_ROOT / "src"))

from agentmemory_v3.association import AssociationGraphView, parse_graph_lookup


def _format_conf(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):.3f}"


def _print_bridges(result: dict) -> None:
    bridges = result.get("bridges") or {}
    preview = bridges.get("preview") or []
    total = int(bridges.get("total") or 0)
    print(f"水平桥接: {len(preview)} / {total}")
    if not preview:
        print("  没有")
        return
    for idx, item in enumerate(preview, start=1):
        print(f"  {idx:02d}. {item['name']} ({item['id']}) weight={_format_conf(item.get('weight'))}")
    if bool(bridges.get("preview_truncated")):
        print("  ...")


def _print_l1(result: dict) -> None:
    node = result["node"]
    parent = result.get("parent")
    grandparent = result.get("grandparent")
    print(f"L1: {node['name']} ({node['id']})")
    if parent:
        print(f"爸爸: {parent['name']} ({parent['id']}) conf={_format_conf(parent.get('conf'))}")
    else:
        print("爸爸: 没有")
    if grandparent:
        print(f"爷爷: {grandparent['name']} ({grandparent['id']}) conf={_format_conf(grandparent.get('conf'))}")
    else:
        print("爷爷: 没有")
    _print_bridges(result)


def _print_l2(result: dict) -> None:
    node = result["node"]
    parent = result.get("parent")
    children = result.get("children") or []
    print(f"L2: {node['name']} ({node['id']})")
    if parent:
        print(f"父亲: {parent['name']} ({parent['id']}) conf={_format_conf(parent.get('conf'))}")
    else:
        print("父亲: 没有")
    print(f"儿子: {len(children)} / {int(result.get('child_total') or 0)}")
    if not children:
        print("  没有")
        _print_bridges(result)
        return
    for idx, child in enumerate(children, start=1):
        print(f"  {idx:02d}. {child['name']} ({child['id']}) conf={_format_conf(child.get('conf'))}")
    if bool(result.get("children_truncated")):
        print("  ...")
    _print_bridges(result)


def _print_l3(result: dict) -> None:
    node = result["node"]
    branches = result.get("branches") or []
    print(f"L3: {node['name']} ({node['id']})")
    print(f"L2 分支: {len(branches)} / {int(result.get('branch_total') or 0)}")
    if not branches:
        print("  没有")
        _print_bridges(result)
        return
    for idx, branch in enumerate(branches, start=1):
        branch_node = branch["node"]
        print(
            f"  [{idx:02d}] {branch_node['name']} ({branch_node['id']}) "
            f"conf={_format_conf(branch.get('conf'))} "
            f"L1={len(branch.get('children') or [])}/{int(branch.get('child_total') or 0)}"
        )
        for child in branch.get("children") or []:
            print(f"       - {child['name']} ({child['id']}) conf={_format_conf(child.get('conf'))}")
        if bool(branch.get("children_truncated")):
            print("       - ...")
    if bool(result.get("branches_truncated")):
        print("  ...")
    _print_bridges(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect one exact concept in the V5 association graph.")
    parser.add_argument("--config", default="_V5/configs/default.yaml")
    parser.add_argument("--lookup", default="", help="Exact lookup like: L1, 拉丁字母")
    parser.add_argument("--level", default="", help="L1 / L2 / L3")
    parser.add_argument("--name", default="", help="Exact canonical concept name")
    args = parser.parse_args()

    if str(args.lookup).strip():
        parsed = parse_graph_lookup(str(args.lookup))
        level = parsed.level
        name = parsed.name
    else:
        level = str(args.level or "").strip().upper()
        name = str(args.name or "").strip()
        if not level or not name:
            raise RuntimeError("Provide either --lookup \"L1, 拉丁字母\" or both --level and --name.")

    viewer = AssociationGraphView.from_config(args.config)
    payload = viewer.lookup(level, name)
    if not bool(payload.get("ok")):
        print(f"没有: level={level} name={name} error={payload.get('error')}")
        matches = payload.get("matches") or []
        for idx, item in enumerate(matches, start=1):
            print(f"  {idx:02d}. {item['level']} {item['name']} ({item['id']})")
        return

    result = payload["result"]
    print(f"命中: {payload['query']['level']}, {payload['query']['name']}")
    if result["kind"] == "L1":
        _print_l1(result)
    elif result["kind"] == "L2":
        _print_l2(result)
    else:
        _print_l3(result)


if __name__ == "__main__":
    main()
