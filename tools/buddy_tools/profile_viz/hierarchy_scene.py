#!/usr/bin/env python3
"""Scene projection for hierarchy profile visualization.

This module turns the path-based hierarchy tree into scene-local display
graphs. It owns:

- raw edge reconstruction
- graph model enrichment
- module / I-O scene projection
- boundary-node handling
- cycle-aware core expansion fallback
"""

from .graph_report import (
    extract_module_entries,
    module_path_hierarchy,
    primary_source_entry,
    short_python_class_name,
    summarize_texts,
    summarize_member_nodes,
    summarize_shape_texts,
    unique_preserve,
)
from .hierarchy_model import (
    build_module_hierarchy_groups,
    display_family,
    natural_sort_key,
    scene_id_for_group,
    sorted_group_children,
)


def _first_order_for_ids(node_ids, node_order_index):
    return min((node_order_index.get(node_id, 10**9) for node_id in node_ids), default=10**9)


def _derive_node_input_shapes(node, node_by_id, predecessor_ids):
    shape_texts = []
    for parent_id in predecessor_ids.get(node["id"], []):
        parent_node = node_by_id[parent_id]
        shape_text = parent_node.get("shape_text")
        if shape_text:
            shape_texts.append(shape_text)
    return shape_texts


def _derive_crossing_shape_texts(raw_edges, inside_ids, node_by_id):
    input_shape_texts = []
    output_shape_texts = []
    inside_ids = set(inside_ids)
    for source_id, target_id in raw_edges:
        if source_id not in inside_ids and target_id in inside_ids:
            source_shape = node_by_id[source_id].get("shape_text")
            if source_shape:
                input_shape_texts.append(source_shape)
        elif source_id in inside_ids and target_id not in inside_ids:
            target_input_shape = node_by_id[source_id].get("shape_text")
            if target_input_shape:
                output_shape_texts.append(target_input_shape)
    return {
        "input_shape_summary": summarize_shape_texts(input_shape_texts),
        "output_shape_summary": summarize_shape_texts(output_shape_texts),
    }


def _topological_order(node_ids, raw_edges):
    indegree = {node_id: 0 for node_id in node_ids}
    successors = {node_id: [] for node_id in node_ids}
    for source_id, target_id in raw_edges:
        if source_id not in indegree or target_id not in indegree:
            continue
        successors[source_id].append(target_id)
        indegree[target_id] += 1
    ready = sorted([node_id for node_id, degree in indegree.items() if degree == 0])
    ordered = []
    while ready:
        current = ready.pop(0)
        ordered.append(current)
        for target_id in successors[current]:
            indegree[target_id] -= 1
            if indegree[target_id] == 0:
                ready.append(target_id)
                ready.sort()
    if len(ordered) != len(node_ids):
        return list(node_ids)
    return ordered


def _cyclic_display_keys(display_nodes, edges):
    node_keys = {display_node["key"] for display_node in display_nodes}
    indegree = {node_key: 0 for node_key in node_keys}
    successors = {node_key: [] for node_key in node_keys}
    for edge in edges:
        source_key = edge["source_key"]
        target_key = edge["target_key"]
        if source_key not in node_keys or target_key not in node_keys:
            continue
        successors[source_key].append(target_key)
        indegree[target_key] += 1
    ready = sorted([node_key for node_key, degree in indegree.items() if degree == 0])
    visited = 0
    while ready:
        current = ready.pop(0)
        visited += 1
        for target_key in successors[current]:
            indegree[target_key] -= 1
            if indegree[target_key] == 0:
                ready.append(target_key)
                ready.sort()
    if visited == len(node_keys):
        return set()
    return {node_key for node_key, degree in indegree.items() if degree > 0}


def _build_raw_edges(graph_payload, node_by_id, node_name_to_ids):
    raw_edges = []
    for edge in graph_payload.get("edges", []):
        source_id = edge.get("source_id")
        target_id = edge.get("target_id")
        if source_id in node_by_id and target_id in node_by_id:
            raw_edges.append((source_id, target_id))
    if raw_edges:
        return unique_preserve(raw_edges)

    # Fallback for synthetic unit tests that only provide parent names.
    for node in graph_payload.get("nodes", []):
        target_id = node.get("id")
        if target_id not in node_by_id:
            continue
        for parent_name in node.get("parents", []):
            parent_ids = node_name_to_ids.get(parent_name) or []
            if len(parent_ids) == 1:
                raw_edges.append((parent_ids[0], target_id))
    return unique_preserve(raw_edges)


def build_module_hierarchy_graph_model(graph_payload):
    raw_nodes = []
    for raw_node in graph_payload.get("nodes", []):
        node = dict(raw_node)
        node["shape_text"] = summarize_shape_texts([node.get("shape") and str(node.get("shape"))])
        if node.get("shape") is not None:
            node["shape_text"] = str(node.get("shape")).replace("'", "")
        node["module_entries"] = extract_module_entries(node.get("provenance"))
        source_entry = primary_source_entry(node["module_entries"])
        node["source_path"] = source_entry.get("path_short") if source_entry else None
        node["source_semantic_class"] = source_entry.get("class_name") if source_entry else None
        node["module_hierarchy"] = module_path_hierarchy(node)
        raw_nodes.append(node)

    node_by_id = {node["id"]: node for node in raw_nodes}
    node_name_to_ids = {}
    for node in raw_nodes:
        node_name_to_ids.setdefault(node["name"], []).append(node["id"])

    raw_edges = _build_raw_edges(graph_payload, node_by_id, node_name_to_ids)
    predecessor_ids = {node_id: [] for node_id in node_by_id}
    successor_ids = {node_id: [] for node_id in node_by_id}
    for source_id, target_id in raw_edges:
        if source_id not in node_by_id or target_id not in node_by_id:
            continue
        predecessor_ids[target_id].append(source_id)
        successor_ids[source_id].append(target_id)
    for node in raw_nodes:
        node["input_shape_items"] = _derive_node_input_shapes(node, node_by_id, predecessor_ids)
        node["output_shape_summary"] = node.get("shape_text")
        if node.get("kind") == "OutputOp":
            node["input_shape_summary"] = summarize_shape_texts(node["input_shape_items"])

    node_order = _topological_order([node["id"] for node in raw_nodes], raw_edges)
    node_order_index = {node_id: index for index, node_id in enumerate(node_order)}

    group_collection = build_module_hierarchy_groups(raw_nodes)
    return {
        "func_name": graph_payload.get("func_name"),
        "source_context": graph_payload.get("source_context") or {},
        "raw_nodes": raw_nodes,
        "node_by_id": node_by_id,
        "raw_edges": raw_edges,
        "predecessor_ids": predecessor_ids,
        "successor_ids": successor_ids,
        "node_order_index": node_order_index,
        "group_collection": group_collection,
    }


def is_group_expandable(group_key, group_collection):
    group = group_collection["groups_by_key"][group_key]
    if not group.get("member_ids"):
        return False
    if group.get("family") == "external-io":
        return bool(group.get("direct_node_ids"))
    return bool(group.get("direct_node_ids") or group_collection["children_by_group"].get(group_key))


def _lineage_keys(group_key, group_collection):
    lineage = []
    current_key = group_key
    while current_key is not None:
        lineage.append(current_key)
        current_key = group_collection["groups_by_key"][current_key].get("parent_key")
    return list(reversed(lineage))


def _make_group_display_node(group, member_nodes, *, expand_target_scene=None):
    summary = summarize_member_nodes(member_nodes)
    return {
        "key": f"g:{group['key']}",
        "group_key": group["key"],
        "name": group["label"],
        "kind": group["family"],
        "family": group["family"],
        "source_path": group.get("source_path"),
        "source_class": group.get("source_class"),
        "heuristic": group.get("heuristic"),
        "node_count": len(group.get("member_ids") or []),
        "profile": summary["profile"],
        "estimates": summary["estimates"],
        "shape_summary": summary.get("shape_summary"),
        "expand_target_scene": expand_target_scene,
        "edge_color_key": (
            group["key"] if group.get("family") in {"module", "module-core"} else None
        ),
    }


def _make_raw_display_node(node):
    return {
        "key": f"n:{node['id']}",
        "name": node["name"],
        "kind": node.get("kind"),
        "family": "node",
        "source_path": node.get("source_path"),
        "source_class": node.get("source_semantic_class"),
        "node_count": 1,
        "profile": node.get("profile") or {},
        "estimates": node.get("estimates") or {},
        "shape_summary": node.get("shape_text"),
        "input_shape_items": node.get("input_shape_items") or [],
        "input_shape_summary": summarize_shape_texts(node.get("input_shape_items") or []),
        "output_shape_summary": node.get("output_shape_summary"),
    }


def _make_io_summary_node(name, *, edge_count, outside_node_count, direction):
    node = {
        "key": f"io-summary:{direction}",
        "name": name,
        "kind": "SummaryPort",
        "family": "external-io",
        "boundary_summary": True,
        "boundary_role": direction,
        "node_count": edge_count,
        "outside_node_count": outside_node_count,
        "profile": {},
        "estimates": {},
    }
    if direction == "input":
        node["color_key"] = "external-input"
    elif direction == "output":
        node["color_key"] = "external-output"
    return node


def _make_boundary_display_node(raw_node, *, input_count, output_count):
    boundary_role = "mixed"
    if input_count and not output_count:
        boundary_role = "input"
    elif output_count and not input_count:
        boundary_role = "output"
    node = {
        "key": f"b:{raw_node['id']}",
        "name": raw_node["name"],
        "kind": raw_node.get("kind"),
        "family": "external-io",
        "boundary_role": boundary_role,
        "raw_node_id": raw_node["id"],
        "input_connection_count": input_count,
        "output_connection_count": output_count,
        "node_count": input_count + output_count,
        "profile": raw_node.get("profile") or {},
        "estimates": raw_node.get("estimates") or {},
        "shape_summary": raw_node.get("shape_text"),
        "input_shape_items": raw_node.get("input_shape_items") or [],
        "input_shape_summary": summarize_shape_texts(raw_node.get("input_shape_items") or []),
        "output_shape_summary": raw_node.get("output_shape_summary"),
        "source_path": raw_node.get("source_path"),
        "source_class": raw_node.get("source_semantic_class"),
    }
    if raw_node.get("kind") == "PlaceholderOp":
        node["color_key"] = "external-input"
    return node

def _source_module_group_key(raw_node, groups_by_key):
    if raw_node.get("kind") == "PlaceholderOp":
        return "external-input"

    hierarchy = raw_node.get("module_hierarchy") or {}
    candidate_paths = unique_preserve(
        [
            hierarchy.get("leaf_path"),
            hierarchy.get("submodule_path"),
            hierarchy.get("block_path"),
            raw_node.get("source_path"),
        ]
    )
    for path in candidate_paths:
        if not path:
            continue
        group_key = f"module-hierarchy:module:{path}"
        if group_key in groups_by_key:
            return group_key
    return None


def _source_color_key(raw_source_ids, node_by_id, groups_by_key):
    source_ids = unique_preserve(
        node_id for node_id in raw_source_ids if node_id in node_by_id
    )
    if not source_ids:
        return None

    source_keys = unique_preserve(
        _source_module_group_key(node_by_id[node_id], groups_by_key)
        or f"source:{node_id}"
        for node_id in source_ids
    )
    if len(source_keys) == 1:
        return source_keys[0]
    return "source-set:" + ",".join(str(source_key) for source_key in source_keys)


def _module_label_for_raw_node(raw_node, groups_by_key):
    if raw_node.get("kind") == "PlaceholderOp":
        return "顶层输入"
    if raw_node.get("kind") == "OutputOp":
        return "输出"

    hierarchy = raw_node.get("module_hierarchy") or {}
    candidate_paths = unique_preserve(
        [
            hierarchy.get("leaf_path"),
            hierarchy.get("submodule_path"),
            hierarchy.get("block_path"),
            raw_node.get("source_path"),
        ]
    )
    for path in candidate_paths:
        if not path:
            continue
        group = groups_by_key.get(f"module-hierarchy:module:{path}")
        if group:
            return group.get("label")
    return raw_node.get("name")


def _summarize_display_targets(display_nodes_by_key, target_keys):
    labels = []
    for target_key in unique_preserve(target_keys):
        target = display_nodes_by_key.get(target_key)
        if target:
            labels.append(target.get("name"))
    return summarize_texts(labels)


def _project_scene_edges(*, raw_edges, scope_ids, owner_by_node_id, input_port_key=None, output_port_key=None, boundary_owner_by_node_id=None):
    edge_pairs = []
    input_outside_ids = set()
    output_outside_ids = set()
    input_edge_count = 0
    output_edge_count = 0
    boundary_owner_by_node_id = boundary_owner_by_node_id or {}
    scope_ids = set(scope_ids)
    for source_id, target_id in raw_edges:
        source_inside = source_id in scope_ids
        target_inside = target_id in scope_ids
        if source_inside and target_inside:
            source_key = owner_by_node_id.get(source_id)
            target_key = owner_by_node_id.get(target_id)
            if source_key and target_key and source_key != target_key:
                edge_pairs.append((source_key, target_key))
        elif not source_inside and target_inside:
            target_key = owner_by_node_id.get(target_id)
            if target_key:
                source_key = boundary_owner_by_node_id.get(source_id, input_port_key)
                if source_key:
                    edge_pairs.append((source_key, target_key))
                input_outside_ids.add(source_id)
                input_edge_count += 1
        elif source_inside and not target_inside:
            source_key = owner_by_node_id.get(source_id)
            if source_key:
                target_key = boundary_owner_by_node_id.get(target_id, output_port_key)
                if target_key:
                    edge_pairs.append((source_key, target_key))
                output_outside_ids.add(target_id)
                output_edge_count += 1
    return {
        "edges": [{"source_key": source_key, "target_key": target_key} for source_key, target_key in unique_preserve(edge_pairs)],
        "input_outside_ids": input_outside_ids,
        "output_outside_ids": output_outside_ids,
        "input_edge_count": input_edge_count,
        "output_edge_count": output_edge_count,
    }


def _build_root_scene(graph_model):
    group_collection = graph_model["group_collection"]
    groups_by_key = group_collection["groups_by_key"]
    node_by_id = graph_model["node_by_id"]
    display_nodes = []

    def root_sort_key(group_key):
        group = groups_by_key[group_key]
        first_order = _first_order_for_ids(group.get("member_ids") or [], graph_model["node_order_index"])
        return (first_order, natural_sort_key(group.get("label") or group_key))

    for group_key in sorted(group_collection["root_keys"], key=root_sort_key):
        group = groups_by_key[group_key]
        member_nodes = [node_by_id[node_id] for node_id in group.get("member_ids") or []]
        display_node = _make_group_display_node(
            group,
            member_nodes,
            expand_target_scene=(scene_id_for_group(group_key) if is_group_expandable(group_key, group_collection) else None),
        )
        display_nodes.append(display_node)

    key_by_group = {node["key"][2:]: node["key"] for node in display_nodes if node["key"].startswith("g:")}
    owner_by_node_id = {}
    for group_key in group_collection["root_keys"]:
        display_key = key_by_group.get(group_key)
        if not display_key:
            continue
        for node_id in groups_by_key[group_key].get("member_ids") or []:
            owner_by_node_id[node_id] = display_key

    edge_pairs = []
    for source_id, target_id in graph_model["raw_edges"]:
        source_key = owner_by_node_id.get(source_id)
        target_key = owner_by_node_id.get(target_id)
        if source_key and target_key and source_key != target_key:
            edge_pairs.append((source_key, target_key))

    source_context = graph_model["source_context"]
    title = source_context.get("entry_module_name") or short_python_class_name(source_context.get("entry_module_class")) or "Hierarchy"
    func_name = graph_model.get("func_name") or "graph"
    return {
        "scene_id": "root",
        "parent_scene_id": None,
        "title": f"{title} · {func_name}",
        "subtitle": "按源码模块路径展开",
        "display_nodes": display_nodes,
        "edges": [{"source_key": source_key, "target_key": target_key} for source_key, target_key in unique_preserve(edge_pairs)],
    }


def _build_io_scene(graph_model, focus_group_key):
    group_collection = graph_model["group_collection"]
    groups_by_key = group_collection["groups_by_key"]
    focus_group = groups_by_key[focus_group_key]
    focus_ids = set(focus_group.get("direct_node_ids") or [])
    node_by_id = graph_model["node_by_id"]
    display_nodes = []
    owner_by_node_id = {}

    raw_ids = sorted(focus_ids, key=lambda node_id: graph_model["node_order_index"].get(node_id, 10**9))
    for node_id in raw_ids:
        raw_node = node_by_id[node_id]
        display_node = _make_raw_display_node(raw_node)
        display_nodes.append(display_node)
        owner_by_node_id[node_id] = display_node["key"]

    peer_group_keys = [
        group_key
        for group_key in group_collection["root_keys"]
        if group_key != focus_group_key and groups_by_key[group_key].get("family") != "external-io"
    ]

    def peer_sort_key(group_key):
        group = groups_by_key[group_key]
        first_order = _first_order_for_ids(group.get("member_ids") or [], graph_model["node_order_index"])
        return (first_order, natural_sort_key(group.get("label") or group_key))

    for group_key in sorted(peer_group_keys, key=peer_sort_key):
        group = groups_by_key[group_key]
        member_nodes = [node_by_id[node_id] for node_id in group.get("member_ids") or []]
        display_node = _make_group_display_node(group, member_nodes, expand_target_scene=scene_id_for_group(group_key))
        display_node["peer_summary"] = "顶层模块摘要"
        display_nodes.append(display_node)
        for node_id in group.get("member_ids") or []:
            owner_by_node_id.setdefault(node_id, display_node["key"])

    edge_pairs = []
    for source_id, target_id in graph_model["raw_edges"]:
        source_key = owner_by_node_id.get(source_id)
        target_key = owner_by_node_id.get(target_id)
        if not source_key or not target_key or source_key == target_key:
            continue
        source_in_focus = source_id in focus_ids
        target_in_focus = target_id in focus_ids
        if source_in_focus != target_in_focus:
            edge_pairs.append((source_key, target_key))

    return {
        "scene_id": scene_id_for_group(focus_group_key),
        "parent_scene_id": "root",
        "title": focus_group["label"],
        "subtitle": "I/O 详情",
        "display_nodes": display_nodes,
        "edges": [{"source_key": source_key, "target_key": target_key} for source_key, target_key in unique_preserve(edge_pairs)],
    }


def _build_module_scene(graph_model, focus_group_key):
    group_collection = graph_model["group_collection"]
    groups_by_key = group_collection["groups_by_key"]
    focus_group = groups_by_key[focus_group_key]
    node_by_id = graph_model["node_by_id"]
    focus_ids = set(focus_group.get("member_ids") or [])
    child_group_keys = sorted_group_children(group_collection, focus_group_key, graph_model["node_order_index"])
    direct_node_ids = sorted(focus_group.get("direct_node_ids") or [], key=lambda node_id: graph_model["node_order_index"].get(node_id, 10**9))
    # Module scenes expose real upstream and downstream boundary nodes.
    # This keeps the subgraph interfaces explicit and easier to inspect than
    # a single summary port, as long as the projected scene remains acyclic.
    show_explicit_input_boundaries = True
    show_explicit_output_boundaries = True

    def build_scene_projection(expanded_core_group_keys):
        display_nodes = []
        owner_by_node_id = {}
        extra_raw_ids = set(direct_node_ids)

        for child_key in child_group_keys:
            child_group = groups_by_key[child_key]
            if child_group.get("family") == "module-core" and child_key in expanded_core_group_keys:
                extra_raw_ids.update(child_group.get("direct_node_ids") or [])
                continue
            member_nodes = [node_by_id[node_id] for node_id in child_group.get("member_ids") or []]
            display_node = _make_group_display_node(
                child_group,
                member_nodes,
                expand_target_scene=(scene_id_for_group(child_key) if is_group_expandable(child_key, group_collection) else None),
            )
            crossing_shapes = _derive_crossing_shape_texts(
                graph_model["raw_edges"], child_group.get("member_ids") or [], node_by_id
            )
            display_node.update(crossing_shapes)
            display_nodes.append(display_node)
            for node_id in child_group.get("member_ids") or []:
                owner_by_node_id[node_id] = display_node["key"]

        for node_id in sorted(extra_raw_ids, key=lambda node_id: graph_model["node_order_index"].get(node_id, 10**9)):
            raw_node = node_by_id[node_id]
            display_node = _make_raw_display_node(raw_node)
            display_nodes.append(display_node)
            owner_by_node_id[node_id] = display_node["key"]

        boundary_owner_by_node_id = {}
        boundary_nodes = []
        if show_explicit_input_boundaries or show_explicit_output_boundaries:
            input_counts = {}
            output_counts = {}
            output_source_ids = {}
            for source_id, target_id in graph_model["raw_edges"]:
                source_inside = source_id in focus_ids
                target_inside = target_id in focus_ids
                if show_explicit_input_boundaries and not source_inside and target_inside:
                    input_counts[source_id] = input_counts.get(source_id, 0) + 1
                if show_explicit_output_boundaries and source_inside and not target_inside:
                    output_counts[target_id] = output_counts.get(target_id, 0) + 1
                    output_source_ids.setdefault(target_id, []).append(source_id)
            boundary_ids = sorted(
                set(input_counts) | set(output_counts),
                key=lambda node_id: graph_model["node_order_index"].get(node_id, 10**9),
            )
            for node_id in boundary_ids:
                raw_node = node_by_id[node_id]
                display_node = _make_boundary_display_node(
                    raw_node,
                    input_count=input_counts.get(node_id, 0),
                    output_count=output_counts.get(node_id, 0),
                )
                if (
                    output_counts.get(node_id, 0)
                    and display_node.get("boundary_role") == "output"
                ):
                    display_node["color_key"] = focus_group_key
                elif output_counts.get(node_id, 0):
                    display_node["color_key"] = _source_color_key(
                        output_source_ids.get(node_id) or [],
                        node_by_id,
                        groups_by_key,
                    ) or display_node.get("color_key")
                boundary_nodes.append(display_node)
                boundary_owner_by_node_id[node_id] = display_node["key"]

        projection = _project_scene_edges(
            raw_edges=graph_model["raw_edges"],
            scope_ids=focus_ids,
            owner_by_node_id=owner_by_node_id,
            input_port_key=None if show_explicit_input_boundaries else "io-summary:input",
            output_port_key=None if show_explicit_output_boundaries else "io-summary:output",
            boundary_owner_by_node_id=boundary_owner_by_node_id,
        )

        if show_explicit_input_boundaries:
            input_boundary_nodes = [
                node for node in boundary_nodes if node["boundary_role"] == "input"
            ]
            display_nodes = input_boundary_nodes + display_nodes
        if show_explicit_output_boundaries:
            output_boundary_nodes = [
                node
                for node in boundary_nodes
                if node["boundary_role"] in {"output", "mixed"}
            ]
            display_nodes = display_nodes + output_boundary_nodes
        elif projection["input_edge_count"]:
            display_nodes.insert(
                0,
                _make_io_summary_node(
                    "输入",
                    edge_count=projection["input_edge_count"],
                    outside_node_count=len(projection["input_outside_ids"]),
                    direction="input",
                ),
            )

        if projection["output_edge_count"] and not show_explicit_output_boundaries:
            display_nodes.append(
                _make_io_summary_node(
                    "输出",
                    edge_count=projection["output_edge_count"],
                    outside_node_count=len(projection["output_outside_ids"]),
                    direction="output",
                )
            )

        display_nodes_by_key = {node["key"]: node for node in display_nodes}
        input_target_keys_by_boundary = {}
        output_source_keys_by_boundary = {}
        for edge in projection["edges"]:
            source_key = edge["source_key"]
            target_key = edge["target_key"]
            if source_key.startswith("b:"):
                input_target_keys_by_boundary.setdefault(source_key, []).append(target_key)
            if target_key.startswith("b:"):
                output_source_keys_by_boundary.setdefault(target_key, []).append(source_key)

        for node in display_nodes:
            if node.get("family") == "external-io" and node.get("boundary_role") == "input":
                raw_node = node_by_id.get(node.get("raw_node_id"))
                if raw_node:
                    node["source_summary"] = _module_label_for_raw_node(raw_node, groups_by_key)
                target_keys = input_target_keys_by_boundary.get(node["key"]) or []
                node["target_summary"] = _summarize_display_targets(display_nodes_by_key, target_keys)
            elif node.get("family") == "external-io" and node.get("boundary_role") == "output":
                source_keys = output_source_keys_by_boundary.get(node["key"]) or []
                node["source_summary"] = _summarize_display_targets(display_nodes_by_key, source_keys)
                raw_node = node_by_id.get(node.get("raw_node_id"))
                if raw_node:
                    node["target_summary"] = _module_label_for_raw_node(raw_node, groups_by_key)
            elif node.get("family") == "external-io" and node.get("boundary_role") == "mixed":
                raw_node = node_by_id.get(node.get("raw_node_id"))
                if raw_node:
                    node["target_summary"] = _module_label_for_raw_node(raw_node, groups_by_key)
                input_target_keys = input_target_keys_by_boundary.get(node["key"]) or []
                output_source_keys = output_source_keys_by_boundary.get(node["key"]) or []
                node["mixed_input_target_summary"] = _summarize_display_targets(display_nodes_by_key, input_target_keys)
                node["mixed_output_source_summary"] = _summarize_display_targets(display_nodes_by_key, output_source_keys)
            elif (
                node.get("family") == "external-io"
                and node.get("boundary_summary")
                and node.get("boundary_role") == "output"
            ):
                downstream_labels = [
                    _module_label_for_raw_node(node_by_id[outside_id], groups_by_key)
                    for outside_id in sorted(
                        projection["output_outside_ids"],
                        key=lambda node_id: graph_model["node_order_index"].get(node_id, 10**9),
                    )
                    if outside_id in node_by_id
                ]
                node["target_summary"] = summarize_texts(downstream_labels)

        indegree = {node["key"]: 0 for node in display_nodes}
        outdegree = {node["key"]: 0 for node in display_nodes}
        for edge in projection["edges"]:
            source_key = edge["source_key"]
            target_key = edge["target_key"]
            if source_key in outdegree:
                outdegree[source_key] += 1
            if target_key in indegree:
                indegree[target_key] += 1

        for node in display_nodes:
            if node.get("family") != "node":
                continue
            if indegree.get(node["key"], 0) != 0 or outdegree.get(node["key"], 0) == 0:
                continue
            raw_node_id = node["key"].split(":", 1)[1]
            node["local_input_source"] = True
            node["color_key"] = f"source:{raw_node_id}"
            node["edge_color_key"] = node["color_key"]
        return display_nodes, projection["edges"]

    expanded_core_group_keys = set()
    while True:
        display_nodes, projected_edges = build_scene_projection(expanded_core_group_keys)
        cyclic_keys = _cyclic_display_keys(display_nodes, projected_edges)
        problematic_core_keys = {
            display_node["group_key"]
            for display_node in display_nodes
            if display_node.get("family") == "module-core"
            and display_node.get("key") in cyclic_keys
            and display_node.get("group_key")
        }
        if not problematic_core_keys - expanded_core_group_keys:
            break
        expanded_core_group_keys.update(problematic_core_keys)

    lineage_labels = []
    for lineage_key in _lineage_keys(focus_group_key, group_collection):
        lineage_group = groups_by_key[lineage_key]
        if lineage_group.get("family") in {"module", "module-core"}:
            lineage_labels.append(lineage_group["label"])
    title = " / ".join(lineage_labels) if lineage_labels else focus_group["label"]
    subtitle = focus_group.get("source_path") or "模块子图"
    parent_group_key = focus_group.get("parent_key")
    return {
        "scene_id": scene_id_for_group(focus_group_key),
        "parent_scene_id": "root" if parent_group_key is None else scene_id_for_group(parent_group_key),
        "title": title,
        "subtitle": subtitle,
        "display_nodes": display_nodes,
        "edges": projected_edges,
    }


def build_hierarchy_scene(graph_model, focus_group_key=None):
    if focus_group_key is None:
        return _build_root_scene(graph_model)

    group_collection = graph_model["group_collection"]
    focus_group = group_collection["groups_by_key"][focus_group_key]
    if focus_group.get("family") == "external-io":
        return _build_io_scene(graph_model, focus_group_key)
    return _build_module_scene(graph_model, focus_group_key)
