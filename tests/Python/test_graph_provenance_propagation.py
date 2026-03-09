# RUN: %PYTHON %s 2>&1 | FileCheck %s

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "build" / "python_packages"))

from frontend.Python.graph.graph import Graph
from frontend.Python.graph.operation import AddOp, MulOp, PlaceholderOp


graph = Graph({}, "forward")

x = PlaceholderOp()
x.name = "x"
x.add_children("add")

add = AddOp()
add.name = "add"
add.add_argument("x")
add.add_parent("x")
add.provenance = {
    "origin_ids": ["fx:forward:add"],
    "fx_node_name": "add",
    "nn_module_stack": {
        "attn": [
            "L['self'].self_attn",
            "<class 'model.Attention'>",
        ]
    },
}

x.provenance = {
    "origin_ids": ["fx:forward:x"],
    "fx_node_name": "x",
    "nn_module_stack": {
        "root": [
            "L['self']",
            "<class 'model.Block'>",
        ]
    },
}

graph.add_node(x)
graph.add_node(add)

replacement = MulOp()
replacement.name = "mul"
graph.displace_node(add, replacement)

print(replacement.provenance["origin_ids"])
print(replacement.provenance["fx_node_name"])
print(sorted(replacement.provenance["nn_module_stack"].keys()))

merged = MulOp()
merged.name = "merged"
graph.inherit_provenance(merged, x, replacement)

print(merged.provenance["origin_ids"])
print(sorted(merged.provenance["nn_module_stack"].keys()))
print(merged.provenance["fx_node_name"])

# CHECK: ['fx:forward:add']
# CHECK: add
# CHECK: ['attn']
# CHECK: ['fx:forward:x', 'fx:forward:add']
# CHECK: ['attn', 'root']
# CHECK: x
