# RUN: %PYTHON %s 2>&1 | FileCheck %s

import json
import os
import sys
import tempfile
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "build" / "python_packages"))

from frontend.Python.graph.graph import Graph, NodeType
from frontend.Python.graph.operation import (
    AddOp,
    OutputOp,
    PlaceholderOp,
)
from frontend.Python.graph.type import TensorDType


graph = Graph({}, "forward")
graph.enable_profile = True
graph.source_context = {
    "entry_module_class": "transformer_model.DeepSeekTransformerBlock",
    "entry_module_name": "DeepSeekTransformerBlock",
}

x = PlaceholderOp()
x.name = "x"
x.tensor_meta = {"shape": (4,), "dtype": TensorDType.Float32}
x.add_children("add")

y = PlaceholderOp()
y.name = "y"
y.tensor_meta = {"shape": (4,), "dtype": TensorDType.Float32}
y.add_children("add")

add = AddOp()
add.name = "add"
add.tensor_meta = {"shape": (4,), "dtype": TensorDType.Float32}
add.provenance = {
    "origin_ids": ["fx:forward:add"],
    "fx_node_name": "add",
    "fx_node_op": "call_function",
}
add.add_argument("x")
add.add_argument("y")
add.add_parent("x")
add.add_parent("y")
add.add_children("output")

output = OutputOp()
output.name = "output"
output.add_argument("add")
output.add_parent("add")

graph.add_node(x, NodeType.InputNode)
graph.add_node(y, NodeType.InputNode)
graph.add_node(add)
graph.add_node(output)

profile_data = {
    graph.get_profile_symbol("add"): {"avg_ms": 1.5, "percentage": 60.0},
    "output": {"avg_ms": 0.25},
}

static_graph = graph.to_static_graph(profile_data)
print(static_graph["profile_name_format"])
print(static_graph["nodes"][2]["profile_name"])
print(static_graph["nodes"][2]["profile"]["avg_ms"])
print(static_graph["source_context"]["entry_module_name"])
print(static_graph["nodes"][2]["provenance"]["origin_ids"])

with tempfile.TemporaryDirectory() as tmp_dir:
    json_path = os.path.join(tmp_dir, "graph.json")
    dot_path = os.path.join(tmp_dir, "graph.dot")
    graph.write_static_graph(json_path, profile_data)
    graph.write_static_graph(dot_path, profile_data)

    payload = json.load(open(json_path))
    dot_text = open(dot_path).read()

    print(payload["nodes"][2]["parents"])
    print(payload["nodes"][2]["provenance"]["fx_node_op"])
    print("avg_ms=1.5000" in dot_text)
    print("n0 -> n2;" in dot_text and "n1 -> n2;" in dot_text)

# CHECK: op_name_{node_index}_{node_name}
# CHECK: op_name_2_add
# CHECK: 1.5
# CHECK: DeepSeekTransformerBlock
# CHECK: ['fx:forward:add']
# CHECK: ['x', 'y']
# CHECK: call_function
# CHECK: True
# CHECK: True
