# RUN: %PYTHON %s 2>&1 | FileCheck %s

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "build" / "python_packages"))

from frontend.Python.graph.graph import Graph, NodeType
from frontend.Python.graph.graph_driver import GraphDriver
from frontend.Python.graph.operation import AddOp, OutputOp, PlaceholderOp
from frontend.Python.graph.type import TensorDType, DeviceType


graph = Graph({}, "forward")
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

graph.op_groups["subgraph0"] = [add]
graph.group_map_device["subgraph0"] = DeviceType.CPU

driver = GraphDriver(graph)
subgraph = driver.subgraphs[0]

print(subgraph.source_context["entry_module_name"])

# CHECK: DeepSeekTransformerBlock
