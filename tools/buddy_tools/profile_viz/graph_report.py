#!/usr/bin/env python3
# Shared helpers for profile visualization.

from collections import Counter
import html
import json
from pathlib import Path
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET

DTYPE_SIZES = {
    "bool": 1,
    "int8": 1,
    "uint8": 1,
    "float8": 1,
    "int16": 2,
    "uint16": 2,
    "float16": 2,
    "bfloat16": 2,
    "bf16": 2,
    "int32": 4,
    "uint32": 4,
    "float32": 4,
    "f32": 4,
    "int64": 8,
    "uint64": 8,
    "float64": 8,
    "f64": 8,
}

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)

REDUCTION_KINDS = {"AmaxOp", "MeanOp", "SumDimOp"}
MATMUL_KINDS = {"BatchMatmulOp", "MatmulOp"}
METADATA_ONLY_KINDS = {
    "OutputOp",
    "PlaceholderOp",
    "SqueezeOp",
    "UnsqueezeOp",
    "ViewOp",
}

def interpolate_color(start, end, ratio):
    ratio = max(0.0, min(1.0, ratio))
    channels = [
        round(start[index] + (end[index] - start[index]) * ratio)
        for index in range(3)
    ]
    return "#{:02x}{:02x}{:02x}".format(*channels)


def dot_escape(text):
    return str(text).replace("\\", "\\\\").replace('"', '\\"')


def html_escape(text):
    return html.escape(str(text))


def normalize_shape(shape):
    if shape is None:
        return None
    if not isinstance(shape, (list, tuple)):
        return None
    normalized = []
    for dim in shape:
        if isinstance(dim, bool):
            return None
        try:
            normalized_dim = int(dim)
        except (TypeError, ValueError):
            return None
        if normalized_dim < 0:
            return None
        normalized.append(normalized_dim)
    return normalized


def shape_num_elements(shape):
    normalized = normalize_shape(shape)
    if normalized is None:
        return None
    elements = 1
    for dim in normalized:
        elements *= dim
    return elements


def dtype_size(dtype):
    if dtype is None:
        return None
    return DTYPE_SIZES.get(str(dtype).lower())


def tensor_bytes(shape, dtype):
    elements = shape_num_elements(shape)
    element_size = dtype_size(dtype)
    if elements is None or element_size is None:
        return None
    return elements * element_size


def product(values):
    result = 1
    for value in values:
        result *= value
    return result


def input_nodes_for(node, node_map):
    inputs = []
    seen = set()
    for key in ("parents", "arguments"):
        for item in node.get(key, []):
            if not isinstance(item, str):
                continue
            if item not in node_map or item in seen:
                continue
            seen.add(item)
            inputs.append(node_map[item])
    return inputs


def estimate_matmul_flops(node, input_nodes):
    if len(input_nodes) < 2:
        return None
    lhs_shape = normalize_shape(input_nodes[0].get("shape"))
    rhs_shape = normalize_shape(input_nodes[1].get("shape"))
    out_shape = normalize_shape(node.get("shape"))
    if (
        lhs_shape is None
        or rhs_shape is None
        or out_shape is None
        or len(lhs_shape) < 2
        or len(rhs_shape) < 2
        or len(out_shape) < 2
    ):
        return None

    lhs_k = lhs_shape[-1]
    rhs_k = rhs_shape[-2]
    if lhs_k != rhs_k:
        return None

    batch_shape = out_shape[:-2]
    batch_count = product(batch_shape) if batch_shape else 1
    m = out_shape[-2]
    n = out_shape[-1]
    return 2 * batch_count * m * n * lhs_k


def estimate_reduction_flops(node, input_nodes):
    if not input_nodes:
        return None
    input_elements = shape_num_elements(input_nodes[0].get("shape"))
    output_elements = shape_num_elements(node.get("shape"))
    if input_elements is None:
        return None
    if node.get("kind") == "MeanOp" and output_elements is not None:
        return max(input_elements - (output_elements or 0), 0) + output_elements
    if output_elements is not None:
        return max(input_elements - output_elements, 0)
    return input_elements


def estimate_node_flops(node, input_nodes):
    kind = node.get("kind")
    op_type = node.get("op_type")
    output_elements = shape_num_elements(node.get("shape"))

    if kind in MATMUL_KINDS:
        return estimate_matmul_flops(node, input_nodes), "dense_matmul"
    if kind in REDUCTION_KINDS:
        return estimate_reduction_flops(node, input_nodes), "reduction"
    if kind in METADATA_ONLY_KINDS or op_type in {
        "GetItemType",
        "PlaceholderType",
        "ReshapeType",
    }:
        return 0, "metadata"
    if op_type in {"BroadcastType", "ElementwiseType"}:
        return output_elements, "elementwise_output"
    return None, "unknown"


def estimate_node_bytes(node, input_nodes):
    read_bytes = 0
    known_read = False
    for input_node in input_nodes:
        input_bytes = tensor_bytes(input_node.get("shape"), input_node.get("dtype"))
        if input_bytes is None:
            continue
        read_bytes += input_bytes
        known_read = True

    write_bytes = tensor_bytes(node.get("shape"), node.get("dtype"))
    total_bytes = 0
    has_any = False
    if known_read:
        total_bytes += read_bytes
        has_any = True
    if write_bytes is not None:
        total_bytes += write_bytes
        has_any = True

    if not has_any:
        return None, None, None
    return read_bytes if known_read else None, write_bytes, total_bytes


def enrich_graph_with_estimates(graph_payload):
    node_map = {node["name"]: node for node in graph_payload.get("nodes", [])}
    enriched_nodes = []
    total_estimated_flops = 0
    total_estimated_logical_bytes = 0
    estimated_node_count = 0

    for node in graph_payload.get("nodes", []):
        merged_node = dict(node)
        input_nodes = input_nodes_for(node, node_map)
        estimated_flops, flops_model = estimate_node_flops(node, input_nodes)
        read_bytes, write_bytes, total_bytes = estimate_node_bytes(
            node, input_nodes
        )
        intensity = None
        if estimated_flops is not None and total_bytes not in (None, 0):
            intensity = float(estimated_flops) / float(total_bytes)

        merged_node["estimates"] = {
            "estimated_flops": estimated_flops,
            "estimated_read_bytes": read_bytes,
            "estimated_write_bytes": write_bytes,
            "estimated_logical_bytes": total_bytes,
            "estimated_intensity": intensity,
            "flops_model": flops_model,
            "bytes_model": "logical_tensor_volume",
        }
        if estimated_flops is not None or total_bytes is not None:
            estimated_node_count += 1
        if estimated_flops is not None:
            total_estimated_flops += estimated_flops
        if total_bytes is not None:
            total_estimated_logical_bytes += total_bytes
        enriched_nodes.append(merged_node)

    merged_graph = dict(graph_payload)
    merged_graph["nodes"] = enriched_nodes
    merged_graph["analysis_summary"] = {
        "flop_unit": "scalar_ops",
        "byte_unit": "bytes",
        "estimated_node_count": estimated_node_count,
        "total_estimated_flops": total_estimated_flops,
        "total_estimated_logical_bytes": total_estimated_logical_bytes,
        "notes": [
            "estimated_flops are shape-based semantic estimates, not hardware counters",
            "estimated_logical_bytes sum tensor input bytes and output bytes per node",
            "estimated_logical_bytes are not cache traffic or DRAM bandwidth measurements",
        ],
    }
    return merged_graph


def format_scaled_value(value, base=1000, suffixes=None):
    if value is None:
        return "n/a"
    if suffixes is None:
        suffixes = ["", "K", "M", "G", "T", "P"]
    scaled_value = float(value)
    suffix_index = 0
    while abs(scaled_value) >= base and suffix_index < len(suffixes) - 1:
        scaled_value /= base
        suffix_index += 1
    if suffix_index == 0:
        return f"{scaled_value:.0f}{suffixes[suffix_index]}"
    return f"{scaled_value:.2f}{suffixes[suffix_index]}"


def format_flops(value):
    if value is None:
        return "n/a"
    return f"{format_scaled_value(value)}F"


def format_bytes(value):
    if value is None:
        return "n/a"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    scaled_value = float(value)
    unit_index = 0
    while scaled_value >= 1024.0 and unit_index < len(units) - 1:
        scaled_value /= 1024.0
        unit_index += 1
    if unit_index == 0:
        return f"{scaled_value:.0f}{units[unit_index]}"
    return f"{scaled_value:.2f}{units[unit_index]}"


def format_intensity(value):
    if value is None:
        return "n/a"
    return f"{value:.3f} flop/B"


def format_shape(shape):
    if shape in (None, ""):
        return None
    if isinstance(shape, (list, tuple)):
        return "[" + ", ".join(str(dimension) for dimension in shape) + "]"
    return str(shape)


def summarize_shape_texts(shape_texts):
    shape_texts = unique_preserve(
        [shape_text for shape_text in shape_texts if shape_text]
    )
    if not shape_texts:
        return None
    if len(shape_texts) == 1:
        return shape_texts[0]
    return f"多种({len(shape_texts)})"


def summarize_operand_shape_texts(shape_texts, limit=3):
    shape_texts = [shape_text for shape_text in shape_texts if shape_text]
    if not shape_texts:
        return None
    if len(shape_texts) == 1:
        return shape_texts[0]
    if len(shape_texts) <= limit:
        return " + ".join(shape_texts)
    return " + ".join(shape_texts[:limit]) + f" 等{len(shape_texts)}个输入"


def summarize_shape(member_nodes):
    return summarize_shape_texts(
        format_shape(node.get("shape")) for node in member_nodes
    )


def is_multi_shape_summary(shape_summary):
    return bool(shape_summary) and (
        str(shape_summary).startswith("多种(")
        or str(shape_summary).startswith("mixed(")
    )


def summarize_texts(values, limit=3):
    values = unique_preserve([str(value) for value in values if value])
    if not values:
        return None
    if len(values) <= limit:
        return ", ".join(values)
    return f"{', '.join(values[:limit])} 等 {len(values)} 项"


def load_profile_map(profile_json):
    payload = json.loads(profile_json.read_text())
    records = payload.get("records", [])
    profile_map = {}
    for record in records:
        op_name = record.get("op_name")
        if op_name is not None:
            profile_map[op_name] = record
    return payload, profile_map


def merge_graph_with_profile(graph_payload, profile_map):
    merged_nodes = []
    for node in graph_payload.get("nodes", []):
        merged_node = dict(node)
        profile = profile_map.get(node.get("profile_name"))
        if profile is None:
            profile = profile_map.get(node.get("name"))
        merged_node["profile"] = profile
        merged_nodes.append(merged_node)
    merged_graph = dict(graph_payload)
    merged_graph["nodes"] = merged_nodes
    return merged_graph


def render_dot(graph_payload):
    nodes = graph_payload.get("nodes", [])
    max_avg_ms = max(
        (
            float(node["profile"]["avg_ms"])
            for node in nodes
            if node.get("profile") and node["profile"].get("avg_ms") is not None
        ),
        default=0.0,
    )
    lines = [
        "digraph buddy_graph {",
        '  rankdir="LR";',
        '  graph [fontname="Helvetica"];',
        '  node [shape="box", style="rounded,filled", fontname="Helvetica", fillcolor="#f5f5f4"];',
        '  edge [fontname="Helvetica"];',
    ]
    for node in nodes:
        label_lines = [f'{node["id"]}: {node["name"]}', node["kind"]]
        if node.get("op_type") is not None:
            label_lines.append(f'op_type={node["op_type"]}')
        if node.get("shape") is not None:
            label_lines.append(f'shape={node["shape"]}')
        if node.get("dtype") is not None:
            label_lines.append(f'dtype={node["dtype"]}')
        fillcolor = "#f5f5f4"
        profile = node.get("profile")
        if profile:
            avg_ms = profile.get("avg_ms")
            if avg_ms is not None:
                avg_ms = float(avg_ms)
                label_lines.append(f"avg_ms={avg_ms:.4f}")
                if max_avg_ms > 0.0:
                    fillcolor = interpolate_color(
                        (243, 244, 246),
                        (239, 68, 68),
                        avg_ms / max_avg_ms,
                    )
            percentage = profile.get("percentage")
            if percentage is not None:
                label_lines.append(f"pct={float(percentage):.2f}%")
        estimates = node.get("estimates", {})
        estimated_flops = estimates.get("estimated_flops")
        if estimated_flops not in (None, 0):
            label_lines.append(f"flops={format_flops(estimated_flops)}")
        estimated_bytes = estimates.get("estimated_logical_bytes")
        if estimated_bytes not in (None, 0):
            label_lines.append(f"bytes={format_bytes(estimated_bytes)}")
        estimated_intensity = estimates.get("estimated_intensity")
        if estimated_intensity is not None:
            label_lines.append(
                f"intensity={format_intensity(estimated_intensity)}"
            )
        label = "\\n".join(dot_escape(line) for line in label_lines)
        lines.append(
            f'  n{node["id"]} [label="{label}", fillcolor="{fillcolor}"];'
        )

    node_ids = {node["name"]: node["id"] for node in nodes}
    for edge in graph_payload.get("edges", []):
        src_id = node_ids.get(edge.get("source"))
        dst_id = node_ids.get(edge.get("target"))
        if src_id is None or dst_id is None:
            continue
        lines.append(f"  n{src_id} -> n{dst_id};")

    lines.append("}")
    return "\n".join(lines) + "\n"


def render_svg(dot_text):
    dot_binary = shutil.which("dot")
    if dot_binary is None:
        raise RuntimeError(
            "Graphviz 'dot' was not found in PATH, cannot generate SVG."
        )
    completed = subprocess.run(
        [dot_binary, "-Tsvg"],
        input=dot_text,
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout


def normalize_python_class_name(raw_value):
    if raw_value is None:
        return None
    value = str(raw_value)
    match = re.match(r"<class '([^']+)'>", value)
    return match.group(1) if match else value


def short_python_class_name(class_name):
    if not class_name:
        return None
    return str(class_name).split(".")[-1]


def normalize_module_path(path):
    if not path:
        return None
    value = str(path)
    normalized = re.sub(r"^L\['self'\]\.?", "", value)
    return normalized or value


def path_depth(path):
    if not path:
        return 0
    return len(str(path).split("."))


def last_path_segment(path):
    if not path:
        return None
    return str(path).split(".")[-1]


def is_index_token(token):
    return bool(re.fullmatch(r"\d+", str(token)))


def split_path_tokens(path):
    if not path:
        return []
    return [segment for segment in str(path).split(".") if segment]


def relative_path_tokens(parent_path, child_path):
    parent_tokens = split_path_tokens(parent_path)
    child_tokens = split_path_tokens(child_path)
    if len(child_tokens) <= len(parent_tokens):
        return None
    if child_tokens[: len(parent_tokens)] != parent_tokens:
        return None
    return child_tokens[len(parent_tokens) :]


def join_path_tokens(tokens):
    return ".".join(str(token) for token in tokens if token)


def normalize_indexed_tokens(tokens):
    return ["[*]" if is_index_token(token) else token for token in tokens]


def normalize_indexed_module_path(path):
    return join_path_tokens(normalize_indexed_tokens(split_path_tokens(path)))


def first_indexed_container(parent_path, child_path):
    relative_tokens = relative_path_tokens(parent_path, child_path)
    if not relative_tokens:
        return None

    for index, token in enumerate(relative_tokens):
        if index == 0 or not is_index_token(token):
            continue
        container_prefix = relative_tokens[:index]
        instance_tokens = relative_tokens[: index + 1]
        remainder_tokens = relative_tokens[index + 1 :]
        container_template_tokens = container_prefix + ["[*]"]
        return {
            "container_template_path": join_path_tokens(
                split_path_tokens(parent_path) + container_template_tokens
            ),
            "container_prefix_path": join_path_tokens(
                split_path_tokens(parent_path) + container_prefix
            ),
            "instance_path": join_path_tokens(
                split_path_tokens(parent_path) + instance_tokens
            ),
            "normalized_child_path": join_path_tokens(
                split_path_tokens(parent_path)
                + container_template_tokens
                + normalize_indexed_tokens(remainder_tokens)
            ),
        }
    return None


def extract_module_entries(provenance):
    stack = (provenance or {}).get("nn_module_stack")
    if not isinstance(stack, dict):
        return []

    entries = []
    seen = set()
    for raw_entry in stack.values():
        if not isinstance(raw_entry, (list, tuple)) or len(raw_entry) < 2:
            continue
        path = str(raw_entry[0]) if raw_entry[0] else None
        class_name = normalize_python_class_name(raw_entry[1])
        if path is None and class_name is None:
            continue
        dedupe_key = (path, class_name)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        entries.append(
            {
                "path": path,
                "path_short": normalize_module_path(path),
                "class_name": class_name,
                "class_short": short_python_class_name(class_name),
                "is_torch_module": bool(
                    class_name and str(class_name).startswith("torch.nn.")
                ),
            }
        )

    entries.sort(
        key=lambda entry: (
            path_depth(entry.get("path_short")),
            str(entry.get("path_short") or entry.get("class_name") or ""),
        )
    )
    return entries


def primary_source_entry(module_entries):
    if not module_entries:
        return None
    for entry in module_entries:
        if not entry.get("is_torch_module"):
            return entry
    return module_entries[0]


def module_path_hierarchy(node):
    module_entries = node.get("module_entries") or []
    if not module_entries:
        return None

    non_torch_entries = [
        entry
        for entry in module_entries
        if not entry.get("is_torch_module") and entry.get("path_short")
    ]
    path_entries = [
        entry for entry in module_entries if entry.get("path_short")
    ]
    block_entry = (
        non_torch_entries[0]
        if non_torch_entries
        else (path_entries[0] if path_entries else None)
    )
    if block_entry is None:
        return None
    leaf_entry = (
        sorted(
            path_entries,
            key=lambda entry: path_depth(entry.get("path_short")),
            reverse=True,
        )[0]
        if path_entries
        else block_entry
    )
    deepest_non_torch_entry = (
        sorted(
            non_torch_entries,
            key=lambda entry: path_depth(entry.get("path_short")),
            reverse=True,
        )[0]
        if non_torch_entries
        else None
    )
    submodule_entry = None
    if (
        deepest_non_torch_entry is not None
        and deepest_non_torch_entry.get("path_short")
        not in {
            block_entry.get("path_short"),
            leaf_entry.get("path_short"),
        }
    ):
        submodule_entry = deepest_non_torch_entry

    return {
        "block_path": block_entry.get("path_short"),
        "block_class": block_entry.get("class_name")
        or node.get("source_semantic_class"),
        "submodule_path": (
            submodule_entry.get("path_short")
            if submodule_entry is not None
            else None
        ),
        "submodule_class": (
            submodule_entry.get("class_name")
            if submodule_entry is not None
            else node.get("source_semantic_class")
        ),
        "leaf_path": leaf_entry.get("path_short")
        or block_entry.get("path_short"),
        "leaf_class": leaf_entry.get("class_name")
        or node.get("source_semantic_class"),
    }


def unique_preserve(values):
    seen = set()
    unique_values = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values


def number_or_zero(value):
    if value is None:
        return 0.0
    return float(value)


def intensity_from(flops, byte_count):
    if flops in (None, 0) or byte_count in (None, 0):
        return 0.0
    return float(flops) / float(byte_count)


def summarize_member_nodes(member_nodes):
    total_avg_ms = sum(
        number_or_zero((node.get("profile") or {}).get("avg_ms"))
        for node in member_nodes
    )
    total_percentage = sum(
        number_or_zero((node.get("profile") or {}).get("percentage"))
        for node in member_nodes
    )
    total_flops = sum(
        number_or_zero((node.get("estimates") or {}).get("estimated_flops"))
        for node in member_nodes
    )
    total_bytes = sum(
        number_or_zero(
            (node.get("estimates") or {}).get("estimated_logical_bytes")
        )
        for node in member_nodes
    )
    return {
        "profile": {
            "avg_ms": total_avg_ms,
            "percentage": total_percentage,
        },
        "estimates": {
            "estimated_flops": total_flops,
            "estimated_logical_bytes": total_bytes,
            "estimated_intensity": intensity_from(total_flops, total_bytes),
        },
        "shape_summary": summarize_shape(member_nodes),
    }


def most_common_non_empty(values):
    candidates = [value for value in values if value]
    if not candidates:
        return None
    return Counter(candidates).most_common(1)[0][0]
