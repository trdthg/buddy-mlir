# RUN: %PYTHON %s 2>&1 | FileCheck %s

import importlib.util
import importlib
from collections import defaultdict, deque
from pathlib import Path
import sys


script_path = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "buddy_tools"
    / "profile_viz"
    / "visualize_profile.py"
)
spec = importlib.util.spec_from_file_location(
    "visualize_profile", script_path
)
visualize_profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(visualize_profile)

tools_root = Path(__file__).resolve().parents[2] / "tools"
if str(tools_root) not in sys.path:
    sys.path.insert(0, str(tools_root))
hierarchy_svg = importlib.import_module("buddy_tools.profile_viz.hierarchy_svg")


def scene_is_acyclic(scene):
    node_keys = {
        display_node["key"] for display_node in scene["display_nodes"]
    }
    graph = defaultdict(list)
    indegree = {key: 0 for key in node_keys}
    for edge in scene["edges"]:
        source_key = edge["source_key"]
        target_key = edge["target_key"]
        if source_key not in node_keys or target_key not in node_keys:
            continue
        graph[source_key].append(target_key)
        indegree[target_key] += 1
    queue = deque(key for key, degree in indegree.items() if degree == 0)
    visited = 0
    while queue:
        node_key = queue.popleft()
        visited += 1
        for target_key in graph[node_key]:
            indegree[target_key] -= 1
            if indegree[target_key] == 0:
                queue.append(target_key)
    return visited == len(node_keys)


def all_scenes_are_acyclic(graph_model, visualize_profile_module):
    scenes = [visualize_profile_module.build_hierarchy_scene(graph_model, None)]
    for group in graph_model["group_collection"]["groups"]:
        if group.get("member_ids"):
            scenes.append(
                visualize_profile_module.build_hierarchy_scene(
                    graph_model, group["key"]
                )
            )
    return all(scene_is_acyclic(scene) for scene in scenes)


def layer_stack(layer_index, leaf_path=None, leaf_class=None):
    entries = {
        "model": [
            "L['self'].model",
            "<class 'transformers.models.qwen2.modeling_qwen2.Qwen2Model'>",
        ],
        "layer": [
            f"L['self'].model.layers.{layer_index}",
            "<class 'transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer'>",
        ],
    }
    if leaf_path and leaf_class:
        entries["leaf"] = [leaf_path, leaf_class]
    return {"nn_module_stack": entries}


graph_payload = {
    "func_name": "subgraph0_decode",
    "source_context": {
        "entry_module_class": "transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM",
        "entry_module_name": "Qwen2ForCausalLM",
        "entry_module_qualname": "Qwen2ForCausalLM",
    },
    "nodes": [
        {
            "id": 0,
            "name": "input",
            "profile_name": "op_name_0000_input",
            "kind": "PlaceholderOp",
            "op_type": "PlaceholderType",
            "shape": [1, 16],
            "dtype": "float32",
        },
    ],
    "edges": [],
}

next_id = 1
previous_name = "input"
for layer_index in range(3):
    nodes = [
        {
            "id": next_id,
            "name": f"layer_{layer_index}",
            "profile_name": f"op_name_{next_id:04d}_layer_{layer_index}",
            "kind": "AddOp",
            "op_type": "BroadcastType",
            "shape": [1, 16],
            "dtype": "float32",
            "arguments": [previous_name],
            "parents": [previous_name],
            "provenance": layer_stack(layer_index),
        },
        {
            "id": next_id + 1,
            "name": f"norm_{layer_index}",
            "profile_name": f"op_name_{next_id + 1:04d}_norm_{layer_index}",
            "kind": "MulOp",
            "op_type": "BroadcastType",
            "shape": [1, 16],
            "dtype": "float32",
            "arguments": [f"layer_{layer_index}"],
            "parents": [f"layer_{layer_index}"],
            "provenance": layer_stack(
                layer_index,
                f"L['self'].model.layers.{layer_index}.input_layernorm",
                "<class 'transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm'>",
            ),
        },
        {
            "id": next_id + 2,
            "name": f"attn_{layer_index}",
            "profile_name": f"op_name_{next_id + 2:04d}_attn_{layer_index}",
            "kind": "MatmulOp",
            "op_type": "ReduceType",
            "shape": [1, 16],
            "dtype": "float32",
            "arguments": [f"norm_{layer_index}"],
            "parents": [f"norm_{layer_index}"],
            "provenance": layer_stack(
                layer_index,
                f"L['self'].model.layers.{layer_index}.self_attn",
                "<class 'transformers.models.qwen2.modeling_qwen2.Qwen2Attention'>",
            ),
        },
        {
            "id": next_id + 3,
            "name": f"mlp_{layer_index}",
            "profile_name": f"op_name_{next_id + 3:04d}_mlp_{layer_index}",
            "kind": "MatmulOp",
            "op_type": "ReduceType",
            "shape": [1, 16],
            "dtype": "float32",
            "arguments": [f"attn_{layer_index}"],
            "parents": [f"attn_{layer_index}"],
            "provenance": layer_stack(
                layer_index,
                f"L['self'].model.layers.{layer_index}.mlp",
                "<class 'transformers.models.qwen2.modeling_qwen2.Qwen2MLP'>",
            ),
        },
    ]
    graph_payload["nodes"].extend(nodes)
    graph_payload["edges"].extend(
        [
            {"source": previous_name, "target": f"layer_{layer_index}"},
            {
                "source": f"layer_{layer_index}",
                "target": f"norm_{layer_index}",
            },
            {
                "source": f"norm_{layer_index}",
                "target": f"attn_{layer_index}",
            },
            {
                "source": f"attn_{layer_index}",
                "target": f"mlp_{layer_index}",
            },
        ]
    )
    previous_name = f"mlp_{layer_index}"
    next_id += 4

graph_model = visualize_profile.build_module_hierarchy_graph_model(graph_payload)
group_collection = graph_model["group_collection"]
group_keys = {group["key"] for group in group_collection["groups"]}
root_scene = visualize_profile.build_hierarchy_scene(graph_model, None)
root_scene_labels = [display_node["name"] for display_node in root_scene["display_nodes"]]
model_scene = visualize_profile.build_hierarchy_scene(
    graph_model, "module-hierarchy:module:model"
)
model_scene_labels = [
    display_node["name"] for display_node in model_scene["display_nodes"]
]
inputs_scene = visualize_profile.build_hierarchy_scene(
    graph_model, "module-hierarchy:inputs"
)
inputs_scene_labels = [
    display_node["name"] for display_node in inputs_scene["display_nodes"]
]
svg_text = visualize_profile.render_module_hierarchy_svg(graph_payload)

print("module-hierarchy:template-container:model.layers.[*]" not in group_keys)
print("module-hierarchy:template-family:model.layers.[*].self_attn" not in group_keys)
print("module-hierarchy:module:model" in group_keys)
print("module-hierarchy:inputs" in group_keys)
print("输入" in root_scene_labels)
print("model · Qwen2Model" in root_scene_labels)
print("0 · Qwen2DecoderLayer" in model_scene_labels)
print("1 · Qwen2DecoderLayer" in model_scene_labels)
print("2 · Qwen2DecoderLayer" in model_scene_labels)
print("input_layernorm · Qwen2RMSNorm" not in model_scene_labels)
print("self_attn · Qwen2Attention" not in model_scene_labels)
print("mlp · Qwen2MLP" not in model_scene_labels)
print(inputs_scene_labels == ["input", "model · Qwen2Model"])
print(inputs_scene["edges"] == [{"source_key": "n:0", "target_key": "g:module-hierarchy:module:model"}])
print(scene_is_acyclic(model_scene))
print(all_scenes_are_acyclic(graph_model, visualize_profile))
print("buddy-module-hierarchy" in svg_text)
print("data-target-scene" in svg_text)


recursive_payload = {
    "func_name": "subgraph0_decode",
    "source_context": {
        "entry_module_class": "transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM",
        "entry_module_name": "Qwen2ForCausalLM",
        "entry_module_qualname": "Qwen2ForCausalLM",
    },
    "nodes": [
        {
            "id": 0,
            "name": "input",
            "profile_name": "op_name_0000_input",
            "kind": "PlaceholderOp",
            "op_type": "PlaceholderType",
            "shape": [1, 16],
            "dtype": "float32",
        },
    ],
    "edges": [],
}

next_id = 1
for layer_index in range(3):
    layer_prefix = f"L['self'].model.layers.{layer_index}.self_attn"
    layer_nodes = [
        {
            "id": next_id,
            "name": f"attn_core_{layer_index}",
            "profile_name": f"op_name_{next_id:04d}_attn_core_{layer_index}",
            "kind": "AddOp",
            "op_type": "BroadcastType",
            "shape": [1, 16],
            "dtype": "float32",
            "arguments": ["input"],
            "parents": ["input"],
            "provenance": {
                "nn_module_stack": {
                    "model": [
                        "L['self'].model",
                        "<class 'transformers.models.qwen2.modeling_qwen2.Qwen2Model'>",
                    ],
                    "layer": [
                        f"L['self'].model.layers.{layer_index}",
                        "<class 'transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer'>",
                    ],
                    "attn": [
                        layer_prefix,
                        "<class 'transformers.models.qwen2.modeling_qwen2.Qwen2Attention'>",
                    ],
                }
            },
        },
        {
            "id": next_id + 1,
            "name": f"q_proj_{layer_index}",
            "profile_name": f"op_name_{next_id + 1:04d}_q_proj_{layer_index}",
            "kind": "MatmulOp",
            "op_type": "ReduceType",
            "shape": [1, 16],
            "dtype": "float32",
            "arguments": [f"attn_core_{layer_index}"],
            "parents": [f"attn_core_{layer_index}"],
            "provenance": {
                "nn_module_stack": {
                    "model": [
                        "L['self'].model",
                        "<class 'transformers.models.qwen2.modeling_qwen2.Qwen2Model'>",
                    ],
                    "layer": [
                        f"L['self'].model.layers.{layer_index}",
                        "<class 'transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer'>",
                    ],
                    "attn": [
                        layer_prefix,
                        "<class 'transformers.models.qwen2.modeling_qwen2.Qwen2Attention'>",
                    ],
                    "proj": [
                        f"{layer_prefix}.q_proj",
                        "<class 'torch.nn.modules.linear.Linear'>",
                    ],
                }
            },
        },
        {
            "id": next_id + 2,
            "name": f"k_proj_{layer_index}",
            "profile_name": f"op_name_{next_id + 2:04d}_k_proj_{layer_index}",
            "kind": "MatmulOp",
            "op_type": "ReduceType",
            "shape": [1, 16],
            "dtype": "float32",
            "arguments": [f"attn_core_{layer_index}"],
            "parents": [f"attn_core_{layer_index}"],
            "provenance": {
                "nn_module_stack": {
                    "model": [
                        "L['self'].model",
                        "<class 'transformers.models.qwen2.modeling_qwen2.Qwen2Model'>",
                    ],
                    "layer": [
                        f"L['self'].model.layers.{layer_index}",
                        "<class 'transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer'>",
                    ],
                    "attn": [
                        layer_prefix,
                        "<class 'transformers.models.qwen2.modeling_qwen2.Qwen2Attention'>",
                    ],
                    "proj": [
                        f"{layer_prefix}.k_proj",
                        "<class 'torch.nn.modules.linear.Linear'>",
                    ],
                }
            },
        },
    ]
    recursive_payload["nodes"].extend(layer_nodes)
    recursive_payload["edges"].extend(
        [
            {"source": "input", "target": f"attn_core_{layer_index}"},
            {
                "source": f"attn_core_{layer_index}",
                "target": f"q_proj_{layer_index}",
            },
            {
                "source": f"attn_core_{layer_index}",
                "target": f"k_proj_{layer_index}",
            },
        ]
    )
    next_id += 3

recursive_graph_model = visualize_profile.build_module_hierarchy_graph_model(
    recursive_payload
)
recursive_collection = recursive_graph_model["group_collection"]
self_attn_scene = visualize_profile.build_hierarchy_scene(
    recursive_graph_model, "module-hierarchy:module:model.layers.0.self_attn"
)
self_attn_labels = [
    display_node["name"] for display_node in self_attn_scene["display_nodes"]
]
recursive_keys = {group["key"] for group in recursive_collection["groups"]}

print("module-hierarchy:module:model.layers.0.self_attn" in recursive_keys)
print("module-hierarchy:module:model.layers.0.self_attn.q_proj" in recursive_keys)
print("module-hierarchy:module:model.layers.0.self_attn.k_proj" in recursive_keys)
print("module-hierarchy:core:model.layers.0.self_attn" in recursive_keys)
print(scene_is_acyclic(self_attn_scene))
print("input" in self_attn_labels)
print("self_attn core" in self_attn_labels)
print("q_proj · Linear" in self_attn_labels)
print("k_proj · Linear" in self_attn_labels)
print("self_attn · Qwen2Attention" not in self_attn_labels)
print(all_scenes_are_acyclic(recursive_graph_model, visualize_profile))

color_payload = {
    "func_name": "subgraph0_decode",
    "source_context": {
        "entry_module_class": "Demo",
        "entry_module_name": "Demo",
        "entry_module_qualname": "Demo",
    },
    "nodes": [
        {
            "id": 0,
            "name": "arg0",
            "profile_name": "op_name_0000_arg0",
            "kind": "PlaceholderOp",
            "op_type": "PlaceholderType",
            "shape": [1, 16],
            "dtype": "float32",
        },
        {
            "id": 1,
            "name": "source",
            "profile_name": "op_name_0001_source",
            "kind": "MulOp",
            "op_type": "BroadcastType",
            "shape": [1, 16],
            "dtype": "float32",
            "arguments": ["arg0"],
            "parents": ["arg0"],
            "provenance": {
                "nn_module_stack": {
                    "model": ["L['self'].model", "<class 'DemoModel'>"],
                    "layer": ["L['self'].model.layer0", "<class 'DemoLayer'>"],
                    "norm": ["L['self'].model.layer0.norm", "<class 'Norm'>"],
                }
            },
        },
        {
            "id": 2,
            "name": "consumer_a",
            "profile_name": "op_name_0002_consumer_a",
            "kind": "ViewOp",
            "op_type": "ReshapeType",
            "shape": [1, 16],
            "dtype": "float32",
            "arguments": ["source"],
            "parents": ["source"],
            "provenance": {
                "nn_module_stack": {
                    "model": ["L['self'].model", "<class 'DemoModel'>"],
                    "layer": ["L['self'].model.layer0", "<class 'DemoLayer'>"],
                    "attn": ["L['self'].model.layer0.attn", "<class 'Attention'>"],
                    "q": ["L['self'].model.layer0.attn.q_proj", "<class 'Linear'>"],
                }
            },
        },
        {
            "id": 3,
            "name": "consumer_b",
            "profile_name": "op_name_0003_consumer_b",
            "kind": "ViewOp",
            "op_type": "ReshapeType",
            "shape": [1, 16],
            "dtype": "float32",
            "arguments": ["source"],
            "parents": ["source"],
            "provenance": {
                "nn_module_stack": {
                    "model": ["L['self'].model", "<class 'DemoModel'>"],
                    "layer": ["L['self'].model.layer0", "<class 'DemoLayer'>"],
                    "attn": ["L['self'].model.layer0.attn", "<class 'Attention'>"],
                    "k": ["L['self'].model.layer0.attn.k_proj", "<class 'Linear'>"],
                }
            },
        },
    ],
    "edges": [
        {"source_id": 0, "target_id": 1},
        {"source_id": 1, "target_id": 2},
        {"source_id": 1, "target_id": 3},
    ],
}

color_graph_model = visualize_profile.build_module_hierarchy_graph_model(color_payload)
layer_scene = visualize_profile.build_hierarchy_scene(
    color_graph_model, "module-hierarchy:module:model.layer0.norm"
)
output_boundaries = [
    display_node
    for display_node in layer_scene["display_nodes"]
    if display_node.get("family") == "external-io"
    and display_node.get("boundary_role") == "output"
]
root_scene = visualize_profile.build_hierarchy_scene(color_graph_model, "module-hierarchy:module:model")
input_boundaries = [
    display_node
    for display_node in root_scene["display_nodes"]
    if display_node.get("family") == "external-io"
    and display_node.get("boundary_role") == "input"
]
parent_scene = visualize_profile.build_hierarchy_scene(
    color_graph_model, "module-hierarchy:module:model.layer0"
)
parent_nodes_by_key = {
    display_node["key"]: display_node
    for display_node in parent_scene["display_nodes"]
}
norm_group = next(
    display_node
    for display_node in parent_scene["display_nodes"]
    if display_node.get("source_path") == "model.layer0.norm"
)
attn_group = next(
    display_node
    for display_node in parent_scene["display_nodes"]
    if display_node.get("source_path") == "model.layer0.attn"
)
norm_to_attn_edge = next(
    edge
    for edge in parent_scene["edges"]
    if edge["source_key"] == norm_group["key"]
    and edge["target_key"] == attn_group["key"]
)

print(len({node.get("color_key") for node in output_boundaries}) == 1)
print(
    len(
        {
            hierarchy_svg.border_color_for_display_node(node)
            for node in output_boundaries
        }
    )
    == 1
)
print(all(node.get("color_key") == "external-input" for node in input_boundaries))
print(
    len(
        {
            hierarchy_svg.border_color_for_display_node(node)
            for node in input_boundaries
        }
    )
    == 1
)
print(
    {node.get("color_key") for node in output_boundaries}
    == {norm_group.get("edge_color_key")}
)
print(
    hierarchy_svg.edge_color_for_display_edge(norm_to_attn_edge, parent_nodes_by_key)
    == hierarchy_svg.stroke_color_for_key(norm_group.get("edge_color_key"))
)

local_source_payload = {
    "func_name": "local_source_graph",
    "source_context": {
        "entry_module_class": "DemoModel",
        "entry_module_name": "DemoModel",
        "entry_module_qualname": "DemoModel",
    },
    "nodes": [
        {
            "id": 0,
            "name": "iota",
            "profile_name": "op_name_0000_iota",
            "kind": "IotaOp",
            "op_type": "ElementwiseType",
            "shape": [16],
            "dtype": "float32",
            "provenance": {
                "nn_module_stack": {
                    "model": ["L['self'].model", "<class 'DemoModel'>"],
                    "layer": ["L['self'].model.layer0", "<class 'DemoLayer'>"],
                }
            },
        },
        {
            "id": 1,
            "name": "sink",
            "profile_name": "op_name_0001_sink",
            "kind": "ViewOp",
            "op_type": "ReshapeType",
            "shape": [16],
            "dtype": "float32",
            "arguments": ["iota"],
            "parents": ["iota"],
            "provenance": {
                "nn_module_stack": {
                    "model": ["L['self'].model", "<class 'DemoModel'>"],
                    "layer": ["L['self'].model.layer0", "<class 'DemoLayer'>"],
                }
            },
        },
    ],
    "edges": [{"source_id": 0, "target_id": 1}],
}

local_source_model = visualize_profile.build_module_hierarchy_graph_model(
    local_source_payload
)
local_source_scene = visualize_profile.build_hierarchy_scene(
    local_source_model, "module-hierarchy:module:model.layer0"
)
local_nodes_by_key = {
    display_node["key"]: display_node
    for display_node in local_source_scene["display_nodes"]
}
local_iota = next(
    display_node
    for display_node in local_source_scene["display_nodes"]
    if display_node.get("name") == "iota"
)
local_edge = next(
    edge
    for edge in local_source_scene["edges"]
    if edge["source_key"] == local_iota["key"]
)

print(local_iota.get("color_key") == "source:0")
print(
    hierarchy_svg.edge_color_for_display_edge(local_edge, local_nodes_by_key)
    == hierarchy_svg.stroke_color_for_key("source:0")
)
print(local_iota.get("local_input_source") is True)
print("输入节点" in hierarchy_svg.dot_label_for_display_node(local_iota))
print(
    hierarchy_svg.stroke_color_for_key("module-hierarchy:module:model.rotary_emb")
    != hierarchy_svg.stroke_color_for_key("module-hierarchy:core:model")
)

# CHECK: True
# CHECK: True
# CHECK: True
# CHECK: True
# CHECK: True
# CHECK: True
# CHECK: True
# CHECK: True
# CHECK: True
# CHECK: True
# CHECK: True
# CHECK: True
# CHECK: True
# CHECK: True
# CHECK: True
# CHECK: True
# CHECK: True
# CHECK: True
# CHECK: True
# CHECK: True
# CHECK: True
# CHECK: True
# CHECK: True
# CHECK: True
# CHECK: True
# CHECK: True
# CHECK: True
# CHECK: True
# CHECK: True
# CHECK: True
# CHECK: True
# CHECK: True
# CHECK: True
# CHECK: True
# CHECK: True
# CHECK: True
# CHECK: True
# CHECK: True
