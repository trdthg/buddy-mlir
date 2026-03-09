#!/usr/bin/env python3
"""Path-based hierarchy tree construction for profile visualization.

This module owns the hierarchy tree itself:

- provenance path normalization helpers
- stable group keys and labels
- clean module tree construction

It intentionally does not know about scene-local boundary nodes or projected
edges. Those live in ``hierarchy_scene.py``.
"""

import re

from .graph_report import (
    extract_module_entries,
    last_path_segment,
    short_python_class_name,
    split_path_tokens,
    unique_preserve,
)


def natural_sort_key(text):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(text or ""))]


def _join_path_tokens(tokens):
    return ".".join(str(token) for token in tokens if token)


def path_depth(path):
    return len(split_path_tokens(path)) if path else 0


def _path_ancestors(path):
    tokens = split_path_tokens(path)
    return [_join_path_tokens(tokens[:depth]) for depth in range(1, len(tokens) + 1)]


def _build_group_key(family, path=None):
    if family == "external-io":
        return f"module-hierarchy:{path}"
    if family == "ungrouped":
        return "module-hierarchy:ungrouped"
    if family == "module":
        return f"module-hierarchy:module:{path}"
    if family == "module-core":
        return f"module-hierarchy:core:{path}"
    raise ValueError(f"Unsupported hierarchy family: {family}")


def scene_id_for_group(group_key):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(group_key))


def display_family(node):
    return node.get("family", "node")


def _group_label(path, source_class):
    segment = last_path_segment(path) or path or "module"
    class_short = short_python_class_name(source_class)
    if class_short:
        return f"{segment} · {class_short}"
    return segment


def sorted_group_children(group_collection, parent_key, node_order_index):
    groups_by_key = group_collection["groups_by_key"]
    children = group_collection["children_by_group"].get(parent_key, [])

    def sort_key(group_key):
        group = groups_by_key[group_key]
        member_ids = group.get("member_ids") or []
        first_order = min((node_order_index.get(node_id, 10**9) for node_id in member_ids), default=10**9)
        return (first_order, natural_sort_key(group.get("label") or group.get("source_path") or group_key))

    return sorted(children, key=sort_key)


def build_module_hierarchy_groups(nodes):
    """Build a clean, unfolded path hierarchy.

    Tree semantics:
    - PlaceholderOp nodes live in a dedicated 输入 bucket
    - OutputOp nodes live in a dedicated 输出 bucket
    - traced nodes are attached to the deepest provenance path available
    - every provenance path is a single module tree node
    - nodes without usable provenance live in 未归类

    Each group stores two memberships:
    - ``direct_node_ids``: raw nodes owned directly by this tree node
    - ``member_ids``: full descendant subtree membership
    """
    groups_by_key = {}
    children_by_group = {}
    root_keys = []
    path_to_group_key = {}
    path_classes = {}

    def ensure_group(key, *, label, family, parent_key=None, source_path=None, source_class=None, heuristic=None):
        group = groups_by_key.get(key)
        if group is None:
            group = {
                "key": key,
                "label": label,
                "family": family,
                "parent_key": parent_key,
                "source_path": source_path,
                "source_class": source_class,
                "heuristic": heuristic,
                "direct_node_ids": [],
                "member_ids": [],
            }
            groups_by_key[key] = group
            children_by_group.setdefault(key, [])
            if parent_key is None:
                if key not in root_keys:
                    root_keys.append(key)
            else:
                children_by_group.setdefault(parent_key, [])
                if key not in children_by_group[parent_key]:
                    children_by_group[parent_key].append(key)
        return group

    input_ids = [node["id"] for node in nodes if node.get("kind") == "PlaceholderOp"]
    output_ids = [node["id"] for node in nodes if node.get("kind") == "OutputOp"]
    if input_ids:
        ensure_group(
            _build_group_key("external-io", "inputs"),
            label="输入",
            family="external-io",
            heuristic="聚合输入占位节点",
        )["direct_node_ids"].extend(input_ids)
    if output_ids:
        ensure_group(
            _build_group_key("external-io", "outputs"),
            label="输出",
            family="external-io",
            heuristic="聚合输出节点",
        )["direct_node_ids"].extend(output_ids)

    for node in nodes:
        for entry in node.get("module_entries") or []:
            path_short = entry.get("path_short")
            class_name = entry.get("class_name")
            if path_short and class_name:
                path_classes.setdefault(path_short, []).append(class_name)

    def ensure_module_path(path):
        if not path:
            return None
        if path in path_to_group_key:
            return path_to_group_key[path]
        group_key = _build_group_key("module", path)
        parent_key = None
        for ancestor_path in reversed(_path_ancestors(path)[:-1]):
            if ancestor_path in path_to_group_key:
                parent_key = path_to_group_key[ancestor_path]
                break
        group = ensure_group(
            group_key,
            label=_group_label(path, (path_classes.get(path) or [None])[0]),
            family="module",
            parent_key=parent_key,
            source_path=path,
            source_class=(path_classes.get(path) or [None])[0],
            heuristic="按源码模块路径分组",
        )
        path_to_group_key[path] = group_key
        return group["key"]

    ungrouped_group = None
    for node in nodes:
        node_id = node["id"]
        kind = node.get("kind")
        if kind in {"PlaceholderOp", "OutputOp"}:
            continue
        hierarchy = node.get("module_hierarchy")
        if not hierarchy or not hierarchy.get("block_path"):
            if ungrouped_group is None:
                ungrouped_group = ensure_group(
                    _build_group_key("ungrouped"),
                    label="未归类",
                    family="ungrouped",
                    heuristic="无可用模块路径 provenance",
                )
            ungrouped_group["direct_node_ids"].append(node_id)
            continue

        chain_paths = []
        for entry in node.get("module_entries") or []:
            path_short = entry.get("path_short")
            if not path_short:
                continue
            if (
                path_short == hierarchy["block_path"]
                or hierarchy["leaf_path"].startswith(path_short + ".")
                or path_short == hierarchy["leaf_path"]
            ):
                chain_paths.append(path_short)
        chain_paths = unique_preserve(sorted(chain_paths, key=path_depth))
        if hierarchy["block_path"] not in chain_paths:
            chain_paths.insert(0, hierarchy["block_path"])
        if hierarchy["leaf_path"] not in chain_paths:
            chain_paths.append(hierarchy["leaf_path"])
        chain_paths = unique_preserve(chain_paths)

        for path in chain_paths:
            ensure_module_path(path)
        deepest_group_key = ensure_module_path(chain_paths[-1])
        groups_by_key[deepest_group_key]["direct_node_ids"].append(node_id)

    for group in groups_by_key.values():
        group["direct_node_ids"] = unique_preserve(group["direct_node_ids"])

    module_group_keys = [
        group_key
        for group_key, group in groups_by_key.items()
        if group.get("family") == "module"
    ]
    for group_key in module_group_keys:
        group = groups_by_key[group_key]
        if not children_by_group.get(group_key) or not group.get("direct_node_ids"):
            continue
        core_group = ensure_group(
            _build_group_key("module-core", group["source_path"]),
            label=f"{last_path_segment(group['source_path']) or group['label']} core",
            family="module-core",
            parent_key=group_key,
            source_path=f"{group['source_path']}.__core__",
            source_class=group.get("source_class"),
            heuristic="聚合当前模块直属 raw 节点",
        )
        core_group["direct_node_ids"].extend(group["direct_node_ids"])
        core_group["direct_node_ids"] = unique_preserve(core_group["direct_node_ids"])
        group["direct_node_ids"] = []

    def populate_member_ids(group_key):
        group = groups_by_key[group_key]
        member_ids = list(group["direct_node_ids"])
        for child_key in children_by_group.get(group_key, []):
            member_ids.extend(populate_member_ids(child_key))
        group["member_ids"] = unique_preserve(member_ids)
        return group["member_ids"]

    for root_key in list(root_keys):
        populate_member_ids(root_key)

    groups = []

    def append_group(group_key):
        groups.append(groups_by_key[group_key])
        for child_key in children_by_group.get(group_key, []):
            append_group(child_key)

    for root_key in root_keys:
        append_group(root_key)

    return {
        "groups": groups,
        "groups_by_key": groups_by_key,
        "children_by_group": children_by_group,
        "root_keys": root_keys,
    }
