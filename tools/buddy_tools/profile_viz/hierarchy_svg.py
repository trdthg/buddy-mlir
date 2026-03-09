#!/usr/bin/env python3
"""SVG rendering for the hierarchy profile view."""

import colorsys
import json
import hashlib
import re
import xml.etree.ElementTree as ET

from .graph_report import (
    SVG_NS,
    XLINK_NS,
    dot_escape,
    format_bytes,
    format_flops,
    format_intensity,
    html_escape,
    interpolate_color,
    is_multi_shape_summary,
    number_or_zero,
    render_svg,
    short_python_class_name,
    summarize_operand_shape_texts,
)
from .hierarchy_model import (
    display_family,
)
from .hierarchy_scene import (
    build_hierarchy_scene,
    build_module_hierarchy_graph_model,
    is_group_expandable,
)

def _hash_color_parameters(color_key):
    digest = hashlib.sha1(str(color_key).encode("utf-8")).digest()
    hue = int.from_bytes(digest[:2], "big") / 65535.0
    hue_degrees = hue * 360.0
    reserved_ranges = ((200.0, 235.0), (15.0, 40.0))
    for _ in range(4):
        if any(lower <= hue_degrees <= upper for lower, upper in reserved_ranges):
            hue_degrees = (hue_degrees + 47.0) % 360.0
        else:
            break
    hue = hue_degrees / 360.0
    saturation = 0.56 + (digest[2] / 255.0) * 0.16
    fill_lightness = 0.86 - (digest[3] / 255.0) * 0.08
    stroke_lightness = 0.38 + (digest[4] / 255.0) * 0.10
    return hue, saturation, fill_lightness, stroke_lightness


def _rgb_to_hex(red, green, blue):
    return "#{:02x}{:02x}{:02x}".format(
        round(max(0.0, min(1.0, red)) * 255),
        round(max(0.0, min(1.0, green)) * 255),
        round(max(0.0, min(1.0, blue)) * 255),
    )


def grouped_fill_for_key(color_key):
    hue, saturation, fill_lightness, _ = _hash_color_parameters(color_key)
    red, green, blue = colorsys.hls_to_rgb(hue, fill_lightness, saturation)
    return _rgb_to_hex(red, green, blue)


def grouped_stroke_for_key(color_key):
    hue, saturation, _, stroke_lightness = _hash_color_parameters(color_key)
    red, green, blue = colorsys.hls_to_rgb(hue, stroke_lightness, saturation)
    return _rgb_to_hex(red, green, blue)


def stroke_color_for_key(color_key):
    if not color_key:
        return "#94a3b8"
    if color_key == "external-input":
        return "#1d4ed8"
    if color_key == "external-output":
        return "#c2410c"
    return grouped_stroke_for_key(color_key)


def io_role_for_display_node(node):
    family = display_family(node)
    if family == "external-io":
        boundary_role = node.get("boundary_role")
        if boundary_role in {"input", "output"}:
            return boundary_role
        if node.get("name") == "输入":
            return "input"
        if node.get("name") == "输出":
            return "output"
        return None
    if family == "node":
        if node.get("kind") == "PlaceholderOp":
            return "input"
        if node.get("kind") == "OutputOp":
            return "output"
    return None


def dot_label_for_display_node(node):
    lines = [node["name"]]
    family = display_family(node)
    io_role = io_role_for_display_node(node)
    external_io_summary = family == "external-io" and not node.get("boundary_role")
    if family == "node":
        if io_role == "input":
            lines.append("输入节点")
        elif io_role == "output":
            lines.append("输出节点")
        else:
            if node.get("local_input_source"):
                lines.append("输入节点")
            lines.append(node["kind"])
    elif family == "external-io":
        if node.get("boundary_role"):
            if node.get("boundary_role") == "mixed":
                lines.append("边界节点")
                if node.get("input_connection_count"):
                    lines.append(f"输入连接={node['input_connection_count']}")
                if node.get("output_connection_count"):
                    lines.append(f"输出连接={node['output_connection_count']}")
            else:
                lines.append("输入节点" if io_role == "input" else "输出节点")
                lines.append(f"连接={node['node_count']}")
            if node.get("source_summary"):
                lines.append(f"来自={node['source_summary']}")
            if node.get("target_summary"):
                relation = "连接到" if io_role == "input" else "流向"
                lines.append(f"{relation}={node['target_summary']}")
        else:
            lines.append("输入占位节点" if io_role == "input" else "输出节点")
            if node.get("target_summary"):
                relation = "连接到" if io_role == "input" else "流向"
                lines.append(f"{relation}={node['target_summary']}")
    else:
        source_class = short_python_class_name(node.get("source_class"))
        if source_class:
            lines.append(source_class)
        lines.append(f"{node['node_count']} nodes")

    shape_summary = node.get("shape_summary")
    input_shape_summary = node.get("input_shape_summary")
    output_shape_summary = node.get("output_shape_summary")
    input_shape_items = node.get("input_shape_items") or []
    if node.get("boundary_summary") or external_io_summary:
        if node.get("outside_node_count") not in (None, 0):
            lines.append(f"外部节点={node['outside_node_count']}")
        if node.get("node_count") not in (None, 0):
            metric_name = "节点" if external_io_summary else "连接"
            lines.append(f"{metric_name}={node['node_count']}")
        shape_count = None
        if is_multi_shape_summary(shape_summary):
            match = re.search(r"\((\d+)\)", str(shape_summary))
            if match:
                shape_count = match.group(1)
        elif shape_summary:
            shape_count = "1"
        if shape_count:
            lines.append(f"形状种类={shape_count}")
    elif io_role == "input":
        if family != "external-io":
            if node.get("source_summary"):
                lines.append(f"来自={node['source_summary']}")
            if node.get("target_summary"):
                lines.append(f"连接到={node['target_summary']}")
        io_shape = output_shape_summary or (
            shape_summary if not is_multi_shape_summary(shape_summary) else None
        )
        if io_shape:
            lines.append(f"输出形状={io_shape}")
    elif io_role == "output":
        if family != "external-io":
            if node.get("source_summary"):
                lines.append(f"来自={node['source_summary']}")
        io_shape = input_shape_summary or (
            shape_summary if not is_multi_shape_summary(shape_summary) else None
        )
        if io_shape:
            lines.append(f"输入形状={io_shape}")
        if family != "external-io" and node.get("target_summary"):
            lines.append(f"流向={node['target_summary']}")
    elif family == "external-io" and node.get("boundary_role") == "mixed":
        if node.get("mixed_output_source_summary"):
            lines.append(f"来自={node['mixed_output_source_summary']}")
        if input_shape_summary:
            lines.append(f"输入形状={input_shape_summary}")
        if output_shape_summary:
            lines.append(f"输出形状={output_shape_summary}")
        if node.get("mixed_input_target_summary"):
            lines.append(f"连接到={node['mixed_input_target_summary']}")
        elif node.get("target_summary"):
            lines.append(f"连接到={node['target_summary']}")
    elif family == "node":
        input_shape_text = summarize_operand_shape_texts(input_shape_items)
        if input_shape_text:
            lines.append(f"输入形状={input_shape_text}")
        if output_shape_summary:
            lines.append(f"输出形状={output_shape_summary}")
        elif shape_summary:
            lines.append(f"shape={shape_summary}")
    elif family == "external-io":
        if (
            shape_summary
            and not is_multi_shape_summary(shape_summary)
            and io_role not in {"input", "output"}
        ):
            lines.append(f"shape={shape_summary}")
    elif family not in {"node", "external-io"}:
        if input_shape_summary and not is_multi_shape_summary(input_shape_summary):
            lines.append(f"输入形状={input_shape_summary}")
        if output_shape_summary and not is_multi_shape_summary(output_shape_summary):
            lines.append(f"输出形状={output_shape_summary}")
        if (
            not input_shape_summary
            and not output_shape_summary
            and shape_summary
            and not is_multi_shape_summary(shape_summary)
        ):
            lines.append(f"shape={shape_summary}")

    profile = node.get("profile") or {}
    if profile.get("avg_ms") not in (None, 0):
        lines.append(f"avg_ms={float(profile['avg_ms']):.4f}")
    if profile.get("percentage") not in (None, 0):
        lines.append(f"pct={float(profile['percentage']):.2f}%")

    estimates = node.get("estimates") or {}
    estimated_flops = estimates.get("estimated_flops")
    if estimated_flops not in (None, 0):
        lines.append(f"flops={format_flops(estimated_flops)}")
    estimated_bytes = estimates.get("estimated_logical_bytes")
    if estimated_bytes not in (None, 0):
        lines.append(f"bytes={format_bytes(estimated_bytes)}")
    estimated_intensity = estimates.get("estimated_intensity")
    if estimated_intensity not in (None, 0):
        lines.append(f"intensity={format_intensity(estimated_intensity)}")
    return "\\n".join(dot_escape(line) for line in lines)


def fill_color_for_display_node(node, max_avg_ms):
    family = display_family(node)
    color_key = node.get("color_key")
    if family == "external-io":
        role = node.get("boundary_role") or node.get("name")
        if role == "输入":
            return "#bfdbfe"
        if role == "output" or role == "输出":
            if color_key and color_key != "external-output":
                return grouped_fill_for_key(color_key)
            return "#fed7aa"
        if role == "input":
            if color_key == "external-input" or node.get("kind") == "PlaceholderOp":
                return "#bfdbfe"
            if color_key:
                return grouped_fill_for_key(color_key)
            return grouped_fill_for_key(node.get("name") or node.get("key"))
        if role == "mixed" and color_key:
            return grouped_fill_for_key(color_key)
        return "#c7d2fe"
    if family == "node" and color_key:
        if color_key == "external-input" or node.get("kind") == "PlaceholderOp":
            return "#bfdbfe"
        return grouped_fill_for_key(color_key)
    avg_ms = number_or_zero((node.get("profile") or {}).get("avg_ms"))
    if max_avg_ms > 0.0 and avg_ms > 0.0:
        return interpolate_color((243, 244, 246), (239, 68, 68), avg_ms / max_avg_ms)
    if family == "node":
        return "#f8fafc"
    if family == "ungrouped":
        return "#fef3c7"
    return "#eff6ff"


def border_color_for_display_node(node):
    family = display_family(node)
    color_key = node.get("color_key")
    if family == "external-io":
        role = node.get("boundary_role") or node.get("name")
        if role == "输入":
            return "#1d4ed8"
        if role == "output" or role == "输出":
            if color_key and color_key != "external-output":
                return grouped_stroke_for_key(color_key)
            return "#c2410c"
        if role == "input":
            if color_key == "external-input" or node.get("kind") == "PlaceholderOp":
                return "#1d4ed8"
            if color_key:
                return grouped_stroke_for_key(color_key)
            return grouped_stroke_for_key(node.get("name") or node.get("key"))
        if role == "mixed" and color_key:
            return grouped_stroke_for_key(color_key)
        return "#4338ca"
    if family == "node" and color_key:
        if color_key == "external-input" or node.get("kind") == "PlaceholderOp":
            return "#1d4ed8"
        return grouped_stroke_for_key(color_key)
    if family == "node":
        return "#334155"
    if family == "ungrouped":
        return "#b45309"
    return "#1d4ed8"


def edge_color_for_display_edge(edge, node_by_key):
    source_node = node_by_key.get(edge["source_key"])
    target_node = node_by_key.get(edge["target_key"])
    if source_node and display_family(source_node) == "external-io":
        return border_color_for_display_node(source_node)
    if source_node and source_node.get("edge_color_key"):
        return stroke_color_for_key(source_node.get("edge_color_key"))
    if target_node and display_family(target_node) == "external-io":
        return border_color_for_display_node(target_node)
    return "#94a3b8"


def render_scene_dot(scene):
    max_avg_ms = max(
        (
            number_or_zero((node.get("profile") or {}).get("avg_ms"))
            for node in scene["display_nodes"]
            if display_family(node) != "external-io"
        ),
        default=0.0,
    )
    lines = [
        "digraph module_hierarchy {",
        '  rankdir="LR";',
        '  graph [bgcolor="transparent", pad="0.35", nodesep="0.55", ranksep="0.90", fontname="Helvetica"];',
        '  node [shape="box", style="filled", fontname="Helvetica", margin="0.18,0.12"];',
        '  edge [fontname="Helvetica", color="#94a3b8", arrowsize="0.7", penwidth="1.4"];',
    ]
    for index, node in enumerate(scene["display_nodes"]):
        node_name = f"n{index}"
        node["dot_name"] = node_name
        node_id = f"{scene['scene_id']}__node__{re.sub(r'[^A-Za-z0-9_.-]+', '_', node['key'])}"
        fill = fill_color_for_display_node(node, max_avg_ms)
        stroke = border_color_for_display_node(node)
        lines.append(
            f'  {node_name} [id="{dot_escape(node_id)}", label="{dot_label_for_display_node(node)}", fillcolor="{fill}", color="{stroke}", penwidth="1.5"];'
        )

    node_name_by_key = {
        node["key"]: node["dot_name"] for node in scene["display_nodes"]
    }
    node_by_key = {node["key"]: node for node in scene["display_nodes"]}
    for edge in scene["edges"]:
        source_name = node_name_by_key.get(edge["source_key"])
        target_name = node_name_by_key.get(edge["target_key"])
        if source_name is None or target_name is None:
            continue
        color = edge_color_for_display_edge(edge, node_by_key)
        lines.append(
            f'  {source_name} -> {target_name} [color="{color}", fontcolor="{color}"];'
        )
    lines.append("}")
    return "\n".join(lines) + "\n"


def local_name(tag):
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def parse_view_box(root):
    view_box = root.get("viewBox")
    if not view_box:
        width = float(str(root.get("width", "0")).replace("pt", ""))
        height = float(str(root.get("height", "0")).replace("pt", ""))
        return 0.0, 0.0, width, height
    parts = [float(part) for part in view_box.replace(",", " ").split()]
    if len(parts) != 4:
        raise ValueError(f"Unexpected SVG viewBox: {view_box}")
    return tuple(parts)


def prefix_scene_ids(scene_root, prefix):
    id_map = {}
    for element in scene_root.iter():
        element_id = element.get("id")
        if not element_id:
            continue
        new_id = f"{prefix}{element_id}"
        id_map[element_id] = new_id
        element.set("id", new_id)

    attrs_to_patch = [
        "href",
        f"{{{XLINK_NS}}}href",
        "clip-path",
        "filter",
        "mask",
        "marker-start",
        "marker-mid",
        "marker-end",
        "fill",
        "stroke",
    ]
    for element in scene_root.iter():
        for attr_name in attrs_to_patch:
            value = element.get(attr_name)
            if not value:
                continue
            if value.startswith("#") and value[1:] in id_map:
                element.set(attr_name, f"#{id_map[value[1:]]}")
                continue
            for original_id, new_id in id_map.items():
                value = value.replace(f"url(#{original_id})", f"url(#{new_id})")
            element.set(attr_name, value)


def find_node_bbox(node_group):
    for element in node_group.iter():
        element_name = local_name(element.tag)
        if element_name == "polygon" and element.get("points"):
            points = []
            for pair in element.get("points").split():
                if "," not in pair:
                    continue
                x_str, y_str = pair.split(",", 1)
                points.append((float(x_str), float(y_str)))
            if points:
                xs = [point[0] for point in points]
                ys = [point[1] for point in points]
                return min(xs), min(ys), max(xs), max(ys)
        if element_name == "rect":
            x = float(element.get("x", "0"))
            y = float(element.get("y", "0"))
            width = float(element.get("width", "0"))
            height = float(element.get("height", "0"))
            return x, y, x + width, y + height
        if element_name == "ellipse":
            cx = float(element.get("cx", "0"))
            cy = float(element.get("cy", "0"))
            rx = float(element.get("rx", "0"))
            ry = float(element.get("ry", "0"))
            return cx - rx, cy - ry, cx + rx, cy + ry
    return None


def scene_node_detail(node):
    profile = node.get("profile") or {}
    estimates = node.get("estimates") or {}
    fragments = [node["name"], node["kind"]]
    io_role = io_role_for_display_node(node)
    external_io_summary = (
        display_family(node) == "external-io" and not node.get("boundary_role")
    )
    if node.get("boundary_summary") or external_io_summary:
        if node.get("outside_node_count") not in (None, 0):
            fragments.append(f"外部节点={node['outside_node_count']}")
        if node.get("node_count") not in (None, 0):
            metric_name = "节点" if external_io_summary else "连接"
            fragments.append(f"{metric_name}={node['node_count']}")
        if node.get("target_summary"):
            relation = "连接到" if io_role == "input" else "流向"
            fragments.append(f"{relation}={node['target_summary']}")
        if node.get("outside_summary"):
            fragments.append(node["outside_summary"])
        if node.get("peer_summary"):
            fragments.append(node["peer_summary"])
        shape_summary = node.get("shape_summary")
        if shape_summary:
            if is_multi_shape_summary(shape_summary):
                fragments.append(f"形状种类={shape_summary}")
            else:
                fragments.append(f"形状={shape_summary}")
    elif io_role == "input":
        if node.get("source_summary"):
            fragments.append(f"来自={node['source_summary']}")
        if node.get("target_summary"):
            fragments.append(f"连接到={node['target_summary']}")
        output_shape = node.get("output_shape_summary")
        if output_shape:
            fragments.append(f"输出形状={output_shape}")
        elif node.get("shape_summary") and not is_multi_shape_summary(
            node.get("shape_summary")
        ):
            fragments.append(f"输出形状={node['shape_summary']}")
    elif io_role == "output":
        if node.get("source_summary"):
            fragments.append(f"来自={node['source_summary']}")
        if node.get("target_summary"):
            fragments.append(f"流向={node['target_summary']}")
        input_shape = node.get("input_shape_summary")
        if input_shape:
            fragments.append(f"输入形状={input_shape}")
        elif node.get("shape_summary") and not is_multi_shape_summary(
            node.get("shape_summary")
        ):
            fragments.append(f"输入形状={node['shape_summary']}")
    elif display_family(node) == "external-io" and node.get("boundary_role") == "mixed":
        fragments.append("边界节点")
        if node.get("input_connection_count"):
            fragments.append(f"输入连接={node['input_connection_count']}")
        if node.get("output_connection_count"):
            fragments.append(f"输出连接={node['output_connection_count']}")
        if node.get("mixed_output_source_summary"):
            fragments.append(f"来自={node['mixed_output_source_summary']}")
        if node.get("mixed_input_target_summary"):
            fragments.append(f"连接到={node['mixed_input_target_summary']}")
        elif node.get("target_summary"):
            fragments.append(f"连接到={node['target_summary']}")
        if node.get("input_shape_summary"):
            fragments.append(f"输入形状={node['input_shape_summary']}")
        if node.get("output_shape_summary"):
            fragments.append(f"输出形状={node['output_shape_summary']}")
    else:
        if node.get("local_input_source"):
            fragments.append("输入节点")
        input_shape_text = summarize_operand_shape_texts(
            node.get("input_shape_items") or []
        )
        if input_shape_text:
            fragments.append(f"输入形状={input_shape_text}")
        elif node.get("input_shape_summary"):
            fragments.append(f"输入形状={node['input_shape_summary']}")
        if node.get("output_shape_summary"):
            fragments.append(f"输出形状={node['output_shape_summary']}")
        if node.get("shape_summary"):
            if display_family(node) == "node":
                fragments.append(f"shape={node['shape_summary']}")
            elif is_multi_shape_summary(node.get("shape_summary")):
                fragments.append(f"内部形状={node['shape_summary']}")
            elif not (
                node.get("input_shape_summary") or node.get("output_shape_summary")
            ):
                fragments.append(f"shape={node['shape_summary']}")
    if profile.get("avg_ms") not in (None, 0):
        fragments.append(f"{float(profile['avg_ms']):.4f} ms")
    if profile.get("percentage") not in (None, 0):
        fragments.append(f"{float(profile['percentage']):.2f}%")
    if estimates.get("estimated_flops") not in (None, 0):
        fragments.append(f"flops={format_flops(estimates['estimated_flops'])}")
    if estimates.get("estimated_logical_bytes") not in (None, 0):
        fragments.append(
            f"bytes={format_bytes(estimates['estimated_logical_bytes'])}"
        )
    if estimates.get("estimated_intensity") not in (None, 0):
        fragments.append(
            f"intensity={format_intensity(estimates['estimated_intensity'])}"
        )
    if node.get("source_class"):
        fragments.append(short_python_class_name(node["source_class"]))
    if node.get("source_path"):
        fragments.append(node["source_path"])
    if node.get("peer_summary") and not node.get("boundary_summary"):
        fragments.append(node["peer_summary"])
    return " · ".join(fragment for fragment in fragments if fragment)


def add_expand_button(node_group, target_scene, bbox):
    if bbox is None:
        return
    _, top, right, _ = bbox
    button_size = 16
    gap = 8
    button_x = right + gap
    button_y = top - button_size - 4
    button = ET.Element(
        f"{{{SVG_NS}}}g",
        {
            "class": "expand-button",
            "data-target-scene": target_scene,
            "onclick": "return buddyHierarchyExpand(evt, this.dataset.targetScene);",
        },
    )
    ET.SubElement(
        button,
        f"{{{SVG_NS}}}rect",
        {
            "x": f"{button_x:.2f}",
            "y": f"{button_y:.2f}",
            "width": str(button_size),
            "height": str(button_size),
            "rx": "4",
            "ry": "4",
        },
    )
    text = ET.SubElement(
        button,
        f"{{{SVG_NS}}}text",
        {
            "x": f"{button_x + button_size / 2:.2f}",
            "y": f"{button_y + button_size / 2 + 3.5:.2f}",
            "text-anchor": "middle",
        },
    )
    text.text = "+"
    node_group.append(button)


def prepare_scene_svg(scene, dot_text):
    scene_root = ET.fromstring(render_svg(dot_text))
    prefix_scene_ids(scene_root, f"{scene['scene_id']}__")

    _, _, width, height = parse_view_box(scene_root)
    padding_left = 14.0
    padding_top = 22.0
    padding_right = 34.0
    padding_bottom = 14.0
    content_group = ET.Element(
        f"{{{SVG_NS}}}g",
        {
            "class": "scene-content",
            "transform": f"translate({padding_left:.2f},{padding_top:.2f})",
        },
    )
    for child in list(scene_root):
        scene_root.remove(child)
        content_group.append(child)
    scene_root.append(content_group)
    width += padding_left + padding_right
    height += padding_top + padding_bottom
    scene_root.set("id", f"scene-{scene['scene_id']}")
    scene_root.set("class", "module-scene")
    scene_root.set("data-scene-id", scene["scene_id"])
    scene_root.set("width", f"{width:.0f}")
    scene_root.set("height", f"{height:.0f}")
    scene_root.set("viewBox", f"0 0 {width:.2f} {height:.2f}")
    scene_root.set("preserveAspectRatio", "xMinYMin meet")

    for node in scene["display_nodes"]:
        node_dom_id = f"{scene['scene_id']}__{scene['scene_id']}__node__{re.sub(r'[^A-Za-z0-9_.-]+', '_', node['key'])}"
        node_group = scene_root.find(f".//*[@id='{node_dom_id}']")
        if node_group is None:
            continue
        node_group.set(
            "class",
            f"{node_group.get('class', '')} hierarchy-node".strip(),
        )
        node_group.set("data-node-key", node["key"])
        node_group.set("data-detail", scene_node_detail(node))
        node_group.set(
            "onclick", "return buddyHierarchySelect(evt, this.id);"
        )
        bbox = find_node_bbox(node_group)
        if node.get("expand_target_scene"):
            node_group.set("data-expandable", "true")
            add_expand_button(node_group, node["expand_target_scene"], bbox)

    return {
        "svg_root": scene_root,
        "width": width,
        "height": height,
    }


def render_module_hierarchy_svg(graph_payload):
    graph_model = build_module_hierarchy_graph_model(graph_payload)
    group_collection = graph_model["group_collection"]
    scenes = [build_hierarchy_scene(graph_model, None)]
    for group in group_collection["groups"]:
        if is_group_expandable(group["key"], group_collection):
            scenes.append(build_hierarchy_scene(graph_model, group["key"]))

    prepared_scenes = []
    scene_meta = {}
    max_scene_width = 0.0
    for scene in scenes:
        dot_text = render_scene_dot(scene)
        prepared = prepare_scene_svg(scene, dot_text)
        prepared_scenes.append(prepared)
        max_scene_width = max(max_scene_width, prepared["width"])
        scene_meta[scene["scene_id"]] = {
            "parent": scene["parent_scene_id"],
            "title": scene["title"],
            "subtitle": scene["subtitle"],
            "width": prepared["width"],
            "height": prepared["height"],
            "status": "点击节点高亮，点击 + 展开模块",
        }

    header_height = 82
    initial_width = scene_meta["root"]["width"]
    initial_height = header_height + scene_meta["root"]["height"]
    svg_style = """
.canvas-bg { fill: #f8fafc; }
.toolbar { font-family: Helvetica, Arial, sans-serif; }
.toolbar-banner { fill: #e2e8f0; stroke: #cbd5e1; stroke-width: 1; }
.toolbar-title { font-size: 20px; font-weight: 700; fill: #0f172a; }
.toolbar-subtitle { font-size: 12px; fill: #475569; }
.toolbar-status { font-size: 12px; fill: #64748b; }
.toolbar-button rect { fill: #ffffff; stroke: #94a3b8; stroke-width: 1; }
.toolbar-button text { font-size: 12px; font-weight: 700; fill: #0f172a; }
.toolbar-button.is-disabled rect { fill: #e2e8f0; stroke: #cbd5e1; }
.toolbar-button.is-disabled text { fill: #94a3b8; }
.module-scene { display: none; overflow: visible; }
.module-scene.is-active { display: inline; }
.hierarchy-node { cursor: pointer; }
.hierarchy-node polygon,
.hierarchy-node path,
.hierarchy-node rect,
.hierarchy-node ellipse { transition: stroke-width 120ms ease, stroke 120ms ease, filter 120ms ease; }
.hierarchy-node.is-selected polygon,
.hierarchy-node.is-selected path,
.hierarchy-node.is-selected rect,
.hierarchy-node.is-selected ellipse { stroke: #0f172a !important; stroke-width: 3 !important; }
.expand-button { cursor: pointer; }
.expand-button rect { fill: #ffffff; stroke: #1d4ed8; stroke-width: 1.2; }
.expand-button text { font-family: Helvetica, Arial, sans-serif; font-size: 14px; font-weight: 700; fill: #1d4ed8; pointer-events: none; }
""".strip()
    svg_script = f"""
const BUDDY_HIERARCHY_SCENES = {json.dumps(scene_meta, ensure_ascii=False)};
let buddyHierarchyCurrentScene = 'root';
let buddyHierarchySelectedNode = null;

function buddyHierarchySceneElement(sceneId) {{
  return document.getElementById(`scene-${{sceneId}}`);
}}

function buddyHierarchyApplyScene(sceneId) {{
  const svg = document.documentElement;
  const meta = BUDDY_HIERARCHY_SCENES[sceneId];
  if (!meta) return false;
  Object.keys(BUDDY_HIERARCHY_SCENES).forEach((candidateId) => {{
    const element = buddyHierarchySceneElement(candidateId);
    if (!element) return;
    element.setAttribute('class', candidateId === sceneId ? 'module-scene is-active' : 'module-scene');
  }});
  svg.setAttribute('width', meta.width);
  svg.setAttribute('height', {header_height} + meta.height);
  svg.setAttribute('viewBox', `0 0 ${{meta.width}} ${{{header_height} + meta.height}}`);
  const background = document.getElementById('canvas-background');
  background.setAttribute('width', meta.width);
  background.setAttribute('height', {header_height} + meta.height);
  document.getElementById('toolbar-title').textContent = meta.title;
  document.getElementById('toolbar-subtitle').textContent = meta.subtitle;
  document.getElementById('toolbar-status').textContent = meta.status;
  const backButton = document.getElementById('toolbar-back');
  if (meta.parent) {{
    backButton.setAttribute('class', 'toolbar-button');
  }} else {{
    backButton.setAttribute('class', 'toolbar-button is-disabled');
  }}
  buddyHierarchyCurrentScene = sceneId;
  buddyHierarchyClearSelection(false);
  return false;
}}

function buddyHierarchyClearSelection(resetStatus = true) {{
  if (buddyHierarchySelectedNode) {{
    buddyHierarchySelectedNode.classList.remove('is-selected');
    buddyHierarchySelectedNode = null;
  }}
  if (resetStatus) {{
    const meta = BUDDY_HIERARCHY_SCENES[buddyHierarchyCurrentScene];
    document.getElementById('toolbar-status').textContent = meta ? meta.status : '';
  }}
  return false;
}}

function buddyHierarchySelect(evt, nodeId) {{
  if (evt) evt.stopPropagation();
  buddyHierarchyClearSelection(false);
  const node = document.getElementById(nodeId);
  if (!node) return false;
  node.classList.add('is-selected');
  buddyHierarchySelectedNode = node;
  document.getElementById('toolbar-status').textContent = node.dataset.detail || '';
  return false;
}}

function buddyHierarchyExpand(evt, targetScene) {{
  if (evt) evt.stopPropagation();
  buddyHierarchyApplyScene(targetScene);
  return false;
}}

function buddyHierarchyBack(evt) {{
  if (evt) evt.stopPropagation();
  const meta = BUDDY_HIERARCHY_SCENES[buddyHierarchyCurrentScene];
  if (meta && meta.parent) {{
    buddyHierarchyApplyScene(meta.parent);
  }}
  return false;
}}

window.addEventListener('load', () => {{
  buddyHierarchyApplyScene('root');
}});
""".strip()
    scene_markup = "\n".join(
        ET.tostring(prepared["svg_root"], encoding="unicode")
        for prepared in prepared_scenes
    )
    return f"""<svg id="buddy-module-hierarchy" xmlns="{SVG_NS}" xmlns:xlink="{XLINK_NS}" width="{initial_width:.0f}" height="{initial_height:.0f}" viewBox="0 0 {initial_width:.0f} {initial_height:.0f}">
  <style><![CDATA[
{svg_style}
  ]]></style>
  <script><![CDATA[
{svg_script}
  ]]></script>
  <rect id="canvas-background" class="canvas-bg" x="0" y="0" width="{initial_width:.0f}" height="{initial_height:.0f}"/>
  <g class="toolbar">
    <rect class="toolbar-banner" x="0" y="0" width="{max_scene_width:.0f}" height="{header_height}"/>
    <g id="toolbar-back" class="toolbar-button is-disabled" transform="translate(18,18)" onclick="return buddyHierarchyBack(evt);">
      <rect x="0" y="0" width="100" height="28" rx="8" ry="8"/>
      <text x="50" y="18" text-anchor="middle">返回上一级</text>
    </g>
    <text id="toolbar-title" class="toolbar-title" x="136" y="34">{html_escape(scene_meta['root']['title'])}</text>
    <text id="toolbar-subtitle" class="toolbar-subtitle" x="136" y="54">{html_escape(scene_meta['root']['subtitle'])}</text>
    <text id="toolbar-status" class="toolbar-status" x="18" y="72">{html_escape(scene_meta['root']['status'])}</text>
  </g>
  <g transform="translate(0,{header_height})" onclick="return buddyHierarchyClearSelection(true);">
{scene_markup}
  </g>
</svg>
"""
