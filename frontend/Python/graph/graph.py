# ===- graph.py ----------------------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===---------------------------------------------------------------------------
#
# This is the graph level of the Buddy Compiler frontend.
#
# ===---------------------------------------------------------------------------

from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Dict, List, Optional
from types import FunctionType
from enum import Enum, auto
import ctypes
import functools
import json
import os
import numpy as np

import buddy_mlir.ir as ir
import buddy_mlir.dialects.func as func
from buddy_mlir.dialects import arith, llvm
from buddy_mlir.passmanager import *
from buddy_mlir.execution_engine import *
from buddy_mlir import runtime as rt

from .operation import *
from .type import *


def make_output_memref_descriptor(ranks, dtypes):
    """
    Make an output memref descriptor for the given memref ranks and dtypes.

    Parameters:
    - ranks: List[int]
        A list of integers representing the ranks of each memref.
    - dtypes: List[str]
        A list of strings representing the data types of each memref.

    Returns:
    ctypes.Structure
        An output memref descriptor struct.

    Example:
    ranks = [2, 3, 1]
    dtypes = [np.float32, np.int64, np.bool]
    descriptor = make_output_memref_descriptor(ranks, dtypes)
    # Use the descriptor in your code
    """
    memref_descriptor = []
    for i, rank, dtype in zip(range(len(ranks)), ranks, dtypes):
        memref_descriptor.append(
            (str(i), rt.make_nd_memref_descriptor(rank, dtype))
        )

    class OutputDescriptor(ctypes.Structure):
        """Builds an output struct descriptor for the multi memref."""

        _fields_ = memref_descriptor

    return OutputDescriptor


class NodeType(Enum):
    FakeNode = auto()
    InputNode = auto()
    OtherNode = auto()


class Graph:
    """
    Graph is a graph-level expression for the Buddy Compiler frontends.
    It acts as a model compute graph, which converts a Graph into an equivalent
    MLIR module.

    Attributes:
    - _body: List[Op]
        The sequence of operation nodes in the graph.
    - _inputs: List[TensorMeta]
        The model inputs represented as TensorMeta objects.
    - _fake_params: List[TensorMeta]
        The fake parameters represented as TensorMeta objects.
    - device: str
        The hardware for graph runtime.
    - _imported_module: Union[None, ImportedModuleType]
        The imported MLIR module after compilation, if set.
    - _ops_registry: dict
        The ops lower strategy for the graph.
    - _func_name: str
        The function name for the MLIR module.
    - _ctx: ir.Context
        The context of the MLIR module.
    - _output_memref: Union[None, ctypes.POINTER]
        The memref pointer in the MLIR function output, if set.
    - _output_descriptor: Union[None, OutputDescriptorType]
        The output descriptor for the MLIR function, if set.
    - ee_: Union[None, ExecutionEngineType]
        The execution engine for the graph, if set.
    """

    def __init__(
        self,
        ops_registry: dict,
        func_name: str,
        device: DeviceType = DeviceType.CPU,
        verbose=False,
        enable_profile: bool = False,
        enable_external_calls: bool = False,
    ) -> None:
        """
        Initializes the Graph.

        Args:
            inputs: List[TensorMeta]
                The model inputs represented as TensorMeta objects.
            fake_params: List[TensorMeta]
                The fake parameters represented as TensorMeta objects.
            ops_registry: dict
                The ops lower strategy for the graph.
            func_name: str
                The function name for the MLIR module.
            enable_external_calls: bool
                Enable external function call support (for oneDNN, etc.)
        """
        self._body = []
        self._inputs = []
        self.node_table: Dict[str, Op] = {}
        self._fake_params = []
        self.device = device
        self._imported_module = None
        self._params_ref = None
        self._verbose = verbose
        self._enable_profile = enable_profile
        self._ops_registry = ops_registry
        self._func_name = func_name
        self._ctx = ir.Context()
        self._output_memref = None
        self._output_descriptor = None
        self.execution_engine = None
        self.op_groups: Dict[str, List[Op]] = {}
        self.group_map_device: Dict[str, DeviceType] = {}
        self._enable_external_calls = enable_external_calls
        self._source_context: Dict[str, Any] = {}

    @property
    def body(self):
        return self._body

    @body.setter
    def body(self, new_body):
        self._body = new_body

    @property
    def source_context(self):
        return self._source_context

    @source_context.setter
    def source_context(self, new_source_context):
        self._source_context = new_source_context or {}

    @property
    def enable_profile(self) -> bool:
        return self._enable_profile

    @enable_profile.setter
    def enable_profile(self, value: bool):
        self._enable_profile = bool(value)

    def add_node(self, node: Op, node_type: NodeType = NodeType.OtherNode):
        """
        Adds an operation node to the graph's body.

        Parameters:
        - node: Op
            The operation node to be added to the graph.

        Returns:
        None

        Example:
        graph_instance = Graph(inputs, fake_params, ops_registry, func_name)
        op_node = Op()
        graph_instance.add_node(op_node)
        # The op_node is now part of the graph's body
        """
        node_idx = len(self._body)
        self._body.append(node)
        self.node_table[node.name] = node
        if node_type == NodeType.FakeNode:
            self._fake_params.append(node_idx)
        elif node_type == NodeType.InputNode:
            self._inputs.append(node_idx)

    def get_input(self, i):
        return self._body[self._inputs[i]]

    @property
    def inputs(self) -> list[Op]:
        return [self.get_input(i) for i in range(len(self._inputs))]

    @property
    def inputs_shapes(self) -> list[TensorMeta]:
        tm_list = []
        for input in self.inputs:
            input_tm_dict = input.tensor_meta
            # FIXME: for backwards compatibility reasons we have to
            # check the actual type of the tm stored in the node
            if isinstance(input_tm_dict, TensorMeta):
                tm_list.append(input_tm_dict)
                continue
            tm_list.append(
                TensorMeta(
                    shape=input_tm_dict["shape"],
                    dtype=input_tm_dict["dtype"],
                )
            )

        return tm_list

    def get_fake_params(self, i):
        return self._body[self._fake_params[i]]

    @property
    def params(self) -> list[Op]:
        return [self.get_fake_params(i) for i in range(len(self._fake_params))]

    @property
    def params_shapes(self) -> list[TensorMeta]:
        tm_list = []
        for param in self.params:
            param_tm_dict = param.tensor_meta
            # FIXME: for backwards compatibility reasons we have to
            # check the actual type of the tm stored in the node
            if isinstance(param_tm_dict, TensorMeta):
                tm_list.append(param_tm_dict)
                continue
            tm_list.append(
                TensorMeta(
                    shape=param_tm_dict["shape"],
                    dtype=param_tm_dict["dtype"],
                )
            )

        return tm_list

    def check_delete_node(self, node: Op) -> bool:
        """
        Determines if a node exists in the graph and has no child nodes.

        Args:
            node (Op): The operation node to check for deletion eligibility.

        Returns:
            bool: True if the node exists in the graph and has no children.
        """
        if not (node.name in self.node_table):
            raise KeyError("node{0} not in graph".format(node.name))

        if len(node._children) == 0:
            return True
        return False

    def delete_node(self, node: Op, parents: List[Op]):
        """
        Removes a node from the graph and updates its parent nodes accordingly.

        Args:
            node (Op): The operation node to be deleted from the graph.
            parents (List[Op]): A list of parent operation nodes that reference the node to be deleted.

        Returns:
            None
        """

        node_idx = self._body.index(node)

        if node_idx in self._inputs:
            self._inputs.remove(node_idx)

        for i, ref_idx in enumerate(self._inputs):
            if ref_idx > node_idx:
                self._inputs[i] -= 1

        if node_idx in self._fake_params:
            self._fake_params.remove(node_idx)

        for i, ref_idx in enumerate(self._fake_params):
            if ref_idx > node_idx:
                self._fake_params[i] -= 1

        for i in parents:
            i._children.remove(node.name)
        node.args.clear()
        node.kwargs.clear()
        node._children.clear()
        self._body.remove(node)
        self.node_table.pop(node.name)

    @staticmethod
    def _merge_provenance_value(left, right):
        if left in (None, {}, [], ()):
            return deepcopy(right)
        if right in (None, {}, [], ()):
            return deepcopy(left)
        if left == right:
            return deepcopy(left)
        if isinstance(left, dict) and isinstance(right, dict):
            merged = deepcopy(left)
            for key, value in right.items():
                if key not in merged:
                    merged[key] = deepcopy(value)
            return merged
        if isinstance(left, list) and isinstance(right, list):
            merged = []
            for item in left + right:
                if item not in merged:
                    merged.append(deepcopy(item))
            return merged
        return deepcopy(left)

    def inherit_provenance(self, target: Op, *sources: Op):
        """
        Copy or merge provenance from source nodes into a rewritten target node.

        This is the graph-level provenance propagation hook for transform
        passes. It keeps provenance handling close to node replacement instead
        of re-implementing it in every pass.
        """
        merged = {}
        saw_any = False
        for source in sources:
            if source is None:
                continue
            provenance = getattr(source, "provenance", None) or {}
            if not provenance:
                continue
            saw_any = True
            for key, value in provenance.items():
                if key not in merged:
                    merged[key] = deepcopy(value)
                else:
                    merged[key] = self._merge_provenance_value(
                        merged[key], value
                    )

        if not saw_any:
            return
        target.provenance = merged

    def displace_node(self, node: Op, newnode: Op):
        """
        Replaces an existing node with a new node in the graph.

        Args:
            node (Op): The operation node to be replaced.
            newnode (Op): The new operation node that will replace the existing node.

        Returns:
            None
        """
        newnode._arguments = node.args
        newnode._keyword_arguments = node.kwargs
        newnode._tensor_meta = node.tensor_meta
        newnode._op_type = node._op_type
        self.inherit_provenance(newnode, node)

        for i in node._children:
            newnode.add_children(i)
        users = [self.node_table[i] for i in node._children]
        for user in users:
            if node.name in user._parents:
                user._parents[user._parents.index(node.name)] = newnode.name
            user.args[user.args.index(node.name)] = newnode.name
        node._children.clear()
        # deal with parents+args
        for i in node._parents:
            newnode.add_parent(i)
        parents = [self.node_table[i] for i in node._parents]
        for parent in parents:
            parent._children[parent._children.index(node.name)] = newnode.name
        node._parents.clear()
        # update node table
        self._body[self._body.index(node)] = newnode
        self.node_table.pop(node.name)
        self.node_table[newnode.name] = newnode

    def displace_node_with_chain(self, node: Op, chain: list[Op]):
        """
        Replaces an existing node with a chain of new nodes.
        - The first node is taken to be the "head" of the chain, and all parents of the
            current node will have this node as their child instead of `node`
        - The last node is taken to be the "tail" of the chain, and all children of `node`
            will have this node as their parent instead.
        Args:
            node (Op): The operation to be replaced.
            chain (list[Op]): The a list of nodes to be inserted instead of Op
        """

        for chain_node in chain:
            self.inherit_provenance(chain_node, node)

        # chain[0] is to be head of the chain:
        chain[0]._arguments = node.args
        chain[0]._keyword_arguments = node.kwargs
        # we do not set the op type, because it might have changed.

        for i in node._parents:
            chain[0].add_parent(i)
        parents = [self.node_table[i] for i in node._parents]
        for parent in parents:
            parent._children[parent._children.index(node.name)] = chain[0].name
        node._parents.clear()

        # chain[-1] is to be the tail of the chain
        chain[-1].tensor_meta = node.tensor_meta

        for i in node._children:
            chain[-1].add_children(i)
        users = [self.node_table[i] for i in node._children]
        for user in users:
            if node.name in user._parents:
                user._parents[user._parents.index(node.name)] = chain[-1].name
            user.args[user.args.index(node.name)] = chain[-1].name
        node._children.clear()

        node_idx = self._body.index(node)
        self._body = self.body[:node_idx] + chain + self.body[node_idx + 1 :]

    def replace_as_child(
        self, parent_ops: list[Op] | Op, child_op: Op, new_op: Op
    ):
        """
        Replace `child_op`, a child of the `parent_ops` with `new_op`.

        Args:
            parent_ops (list[Op]): parents op `child_op` to replace `child_op` with `new_op` among the children
            child_op (Op): See above
            new_op (Op): See above
        """

        if not isinstance(parent_ops, list):
            parent_ops = [parent_ops]

        child_name = child_op._name
        new_child_name = new_op._name

        for parent_name in parent_ops:
            parent_op = self.node_table[parent_name]
            parent_op._children[parent_op._children.index(child_name)] = (
                new_child_name
            )

    def replace_as_parent(
        self, parent_op: Op, child_ops: list[Op] | Op, new_op: Op
    ):
        """
        Replace `parent_op` with `new_op` as the the parent node of the `child_ops` list.

        Args:
            parent_op (Op): Parent to replace
            child_ops (list[Op]): Child ops for which replace `parent_op` as their
            new_op (Op): op to replace `parent_op` with
        """

        if not isinstance(child_ops, list):
            child_ops = [child_ops]

        parent_name = parent_op._name
        new_parent_name = new_op._name

        for child_name in child_ops:
            child_op = self.node_table[child_name]

            if parent_name in child_op._parents:
                child_op._parents[child_op._parents.index(parent_name)] = (
                    new_parent_name
                )

            if parent_name in child_op._arguments:
                child_op._arguments[child_op._arguments.index(parent_name)] = (
                    new_parent_name
                )

    def init_op_group(self):
        """
        Initializes operation groups within the graph.

        Returns:
        - None
        """
        for i, op in enumerate(self._body):
            if isinstance(op, PlaceholderOp) or isinstance(op, OutputOp):
                continue
            group = [op]
            subgraph_name = "subgraph{}".format(i)
            self.group_map_device[subgraph_name] = DeviceType.CPU
            self.op_groups[subgraph_name] = group

    def fuse_ops(self, pattern_list: List[FunctionType]):
        """
        Fuse operations in the graph based on provided fusion patterns.

        Args:
        - pattern_list (List[FunctionType]): A list of functions representing
        fusion patterns.

        Returns:
        - None
        """
        # TODO: discuss two fuse strategy
        # 1. fuse ops adapt for DSA(hardware dependent)
        # 2. common fuse strategy(hardware independent)

        # Apply fusion patterns
        for pattern_func in pattern_list:
            pattern_func(self)

    def perform(self, func_list: List[FunctionType]):
        """
        Perform a series of transformations on the graph using the provided list
        of functions.

        Args:
        - func_list (List[FunctionType]): A list of functions representing
        transformations to be applied to the graph.

        Returns:
        - None
        """
        for transform_func in func_list:
            transform_func(self)

    @staticmethod
    def _jsonable_value(value: Any):
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {
                str(key): Graph._jsonable_value(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [Graph._jsonable_value(item) for item in value]
        return str(value)

    @staticmethod
    def _normalize_profile_record(record: Any) -> Optional[Dict[str, Any]]:
        if record is None:
            return None
        if isinstance(record, Mapping):
            normalized = {
                str(key): Graph._jsonable_value(value)
                for key, value in record.items()
            }
            if "avg_ms" not in normalized:
                if "time_ms" in normalized:
                    normalized["avg_ms"] = normalized["time_ms"]
                elif "avg" in normalized:
                    normalized["avg_ms"] = normalized["avg"]
                elif "duration_sec" in normalized:
                    normalized["avg_ms"] = normalized["duration_sec"] * 1000.0
            if "min_ms" not in normalized and "min" in normalized:
                normalized["min_ms"] = normalized["min"]
            if "max_ms" not in normalized and "max" in normalized:
                normalized["max_ms"] = normalized["max"]
            return normalized
        return {"avg_ms": Graph._jsonable_value(record)}

    def _profile_label_width(self) -> int:
        return max(1, len(str(len(self._body))))

    def get_profile_symbol(self, node: int | str | Op) -> str:
        """
        Returns the profiling symbol name injected by `enable_profile`.

        The format matches commit `51cef718918432dcdc49be5e7949e80e2aa71002`:
        `op_name_{node_index}_{node_name}`.
        """
        if isinstance(node, int):
            node_index = node
            graph_node = self._body[node_index]
        else:
            node_name = node.name if isinstance(node, Op) else str(node)
            graph_node = None
            for index, candidate in enumerate(self._body):
                if candidate.name == node_name:
                    node_index = index
                    graph_node = candidate
                    break
            if graph_node is None:
                raise KeyError(f"Node {node_name} not found in graph")
        width = self._profile_label_width()
        return f"op_name_{node_index:0{width}d}_{graph_node.name}"

    def _lookup_profile_record(
        self,
        profile_data: Optional[Mapping[str, Any]],
        node: Op,
        profile_symbol: str,
    ) -> Optional[Dict[str, Any]]:
        if profile_data is None:
            return None
        record = profile_data.get(profile_symbol)
        if record is None:
            record = profile_data.get(node.name)
        return self._normalize_profile_record(record)

    def to_static_graph(
        self, profile_data: Optional[Mapping[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Export the Buddy graph to a stable JSON-serializable DAG structure.

        When `profile_data` is provided, records are matched first by the
        injected profiling symbol and then by raw node name.
        """
        nodes = []
        edges = []
        latest_node_id_by_name = {}
        for index, node in enumerate(self._body):
            profile_symbol = self.get_profile_symbol(index)
            node_record = self._lookup_profile_record(
                profile_data, node, profile_symbol
            )
            tensor_meta = node.tensor_meta
            if isinstance(tensor_meta, TensorMeta):
                shape = tensor_meta.shape
                dtype = tensor_meta.dtype
            else:
                shape = tensor_meta.get("shape")
                dtype = tensor_meta.get("dtype")
            nodes.append(
                {
                    "id": index,
                    "name": node.name,
                    "profile_name": profile_symbol,
                    "kind": type(node).__name__,
                    "op_type": (
                        node._op_type.name if node._op_type is not None else None
                    ),
                    "shape": self._jsonable_value(shape),
                    "dtype": self._jsonable_value(dtype),
                    "arguments": self._jsonable_value(node.args),
                    "kwargs": self._jsonable_value(node.kwargs),
                    "parents": self._jsonable_value(node._parents),
                    "children": self._jsonable_value(node._children),
                    "provenance": self._jsonable_value(
                        getattr(node, "provenance", None)
                    ),
                    "profile": node_record,
                }
            )
            for parent_name in dict.fromkeys(node._parents):
                edge = {"source": parent_name, "target": node.name}
                parent_id = latest_node_id_by_name.get(parent_name)
                if parent_id is not None:
                    edge["source_id"] = parent_id
                    edge["target_id"] = index
                edges.append(edge)
            latest_node_id_by_name[node.name] = index

        return {
            "func_name": self._func_name,
            "device": self.device.value,
            "profile_enabled": self._enable_profile,
            "source_context": self._jsonable_value(self._source_context),
            "profile_name_format": "op_name_{node_index}_{node_name}",
            "node_count": len(nodes),
            "nodes": nodes,
            "edges": edges,
        }

    @staticmethod
    def _dot_escape(text: Any) -> str:
        return str(text).replace("\\", "\\\\").replace('"', '\\"')

    @staticmethod
    def _interpolate_color(
        start: tuple[int, int, int],
        end: tuple[int, int, int],
        ratio: float,
    ) -> str:
        ratio = max(0.0, min(1.0, ratio))
        channels = [
            round(start[index] + (end[index] - start[index]) * ratio)
            for index in range(3)
        ]
        return "#{:02x}{:02x}{:02x}".format(*channels)

    def to_dot(self, profile_data: Optional[Mapping[str, Any]] = None) -> str:
        """
        Render the static graph as Graphviz DOT.
        """
        static_graph = self.to_static_graph(profile_data)
        profiled_nodes = [
            node
            for node in static_graph["nodes"]
            if (
                node["profile"] is not None
                and node["profile"].get("avg_ms") is not None
            )
        ]
        max_avg_ms = max(
            (float(node["profile"]["avg_ms"]) for node in profiled_nodes),
            default=0.0,
        )
        lines = [
            "digraph buddy_graph {",
            '  rankdir="LR";',
            '  graph [fontname="Helvetica"];',
            '  node [shape="box", style="rounded,filled", fontname="Helvetica", fillcolor="#f5f5f4"];',
            '  edge [fontname="Helvetica"];',
        ]

        for node in static_graph["nodes"]:
            label_lines = [
                f'{node["id"]}: {node["name"]}',
                node["kind"],
            ]
            if node["op_type"] is not None:
                label_lines.append(f'op_type={node["op_type"]}')
            if node["shape"] is not None:
                label_lines.append(f'shape={node["shape"]}')
            if node["dtype"] is not None:
                label_lines.append(f'dtype={node["dtype"]}')
            if node["profile"] is not None:
                avg_ms = node["profile"].get("avg_ms")
                if avg_ms is not None:
                    label_lines.append(f"avg_ms={float(avg_ms):.4f}")
                percentage = node["profile"].get("percentage")
                if percentage is not None:
                    label_lines.append(f"pct={float(percentage):.2f}%")
            fillcolor = "#f5f5f4"
            if (
                node["profile"] is not None
                and node["profile"].get("avg_ms") is not None
                and max_avg_ms > 0.0
            ):
                fillcolor = self._interpolate_color(
                    (243, 244, 246),
                    (239, 68, 68),
                    float(node["profile"]["avg_ms"]) / max_avg_ms,
                )
            label = "\\n".join(self._dot_escape(line) for line in label_lines)
            lines.append(
                f'  n{node["id"]} [label="{label}", fillcolor="{fillcolor}"];'
            )

        node_ids = {
            node["name"]: node["id"] for node in static_graph["nodes"]
        }
        for edge in static_graph["edges"]:
            src_id = node_ids.get(edge["source"])
            dst_id = node_ids.get(edge["target"])
            if src_id is None or dst_id is None:
                continue
            lines.append(f"  n{src_id} -> n{dst_id};")

        lines.append("}")
        return "\n".join(lines) + "\n"

    def write_static_graph(
        self,
        path: str | os.PathLike[str],
        profile_data: Optional[Mapping[str, Any]] = None,
        fmt: Optional[str] = None,
    ) -> None:
        """
        Write the static graph as JSON or DOT.

        The output format defaults to the file suffix.
        """
        output_path = os.fspath(path)
        if fmt is None:
            _, suffix = os.path.splitext(output_path)
            fmt = suffix.lstrip(".").lower() or "json"
        with open(output_path, "w") as output_file:
            if fmt == "json":
                json.dump(
                    self.to_static_graph(profile_data),
                    output_file,
                    indent=2,
                )
                output_file.write("\n")
            elif fmt == "dot":
                output_file.write(self.to_dot(profile_data))
            else:
                raise ValueError(f"Unsupported static graph format: {fmt}")

    def lower_to_top_level_ir(self):
        """
        Lowers the graph to top-level MLIR dialects.

        Parameters:
        - do_params_pack: bool, optional (default=False)
            Flag indicating whether to perform parameters packing to one memref.

        Returns:
        None

        Example:
        graph_instance = Graph(inputs, fake_params, ops_registry, func_name)
        graph_instance.lower_to_top_level_ir(do_params_pack=True)
        # The graph is now lowered to top-level MLIR dialects
        """
        with ir.Location.unknown(self._ctx):
            fx_importer = GraphImporter(
                self._body,
                self.params_shapes,
                self.inputs_shapes,
                self._func_name,
                self._ops_registry,
                False,
                self.device,
                verbose=self._verbose,
                enable_profile=self._enable_profile,
                enable_external_calls=self._enable_external_calls,
            )
            self._imported_module = fx_importer.import_graph()
            outputs = fx_importer.get_output_nodes()
        self._output_memref = []
        output_ranks = []
        output_dtypes = []
        for out_node in outputs:
            out_type = ir.RankedTensorType(out_node.type)
            shape = list(out_type.shape)
            dtype = out_type.element_type
            match str(dtype):
                case "i1":
                    np_type = np.dtype(np.bool_)
                case "i8":
                    np_type = np.dtype(np.int8)
                case "i32":
                    np_type = np.dtype(np.int32)
                case "i64":
                    np_type = np.dtype(np.int64)
                case "f16":
                    np_type = np.dtype(np.float16)
                case "bf16":
                    # BF16 is stored as uint16, similar to F16
                    np_type = np.dtype(np.uint16)
                case "f32":
                    np_type = np.dtype(np.float32)
                case "f64":
                    np_type = np.dtype(np.float64)
                case "complex<f32>":
                    np_type = np.dtype(np.complex64)
                case "complex<f64>":
                    np_type = np.dtype(np.complex128)
                case _:
                    raise NotImplementedError(f"Unsupported dtype {dtype}")
            self._output_memref.append(
                ctypes.pointer(
                    ctypes.pointer(
                        rt.make_nd_memref_descriptor(
                            len(shape), rt.as_ctype(np_type)
                        )()
                    )
                )
            )
            output_ranks.append(len(shape))
            output_dtypes.append(rt.as_ctype(np_type))
        self._output_descriptor = make_output_memref_descriptor(
            output_ranks, output_dtypes
        )

    def lower_to_llvm_ir(self):
        """
        Lower graph to llvm ir.
        """
        if self._imported_module is None:
            self.lower_to_top_level_ir()

        with ir.Location.unknown(self._ctx):
            pm = PassManager("builtin.module")
            pm.add("func.func(tosa-to-linalg-named)")
            pm.add("func.func(tosa-to-linalg)")
            pm.add("func.func(tosa-to-tensor)")
            pm.add("func.func(tosa-to-arith)")
            pm.run(self._imported_module.operation)
            pm.add("arith-expand")
            pm.add("eliminate-empty-tensors")
            pm.add("empty-tensor-to-alloc-tensor")
            pm.add("convert-elementwise-to-linalg")
            pm.add("one-shot-bufferize{bufferize-function-boundaries}")
            pm.add("func.func(linalg-generalize-named-ops)")
            pm.add("func.func(convert-linalg-to-loops)")
            pm.add("affine-loop-fusion")
            pm.add("func.func(affine-parallelize)")
            pm.add("convert-scf-to-openmp")
            pm.add("expand-strided-metadata")
            pm.add("lower-affine")
            pm.add("convert-vector-to-llvm")
            pm.add("memref-expand")
            pm.add("arith-expand")
            pm.add("convert-complex-to-llvm")
            pm.add("convert-arith-to-llvm")
            pm.add("finalize-memref-to-llvm")
            pm.add("convert-scf-to-cf")
            pm.add("convert-cf-to-llvm")
            pm.add("func.func(llvm-request-c-wrappers)")
            pm.add("convert-openmp-to-llvm")
            pm.add("convert-math-to-llvm")
            pm.add("convert-math-to-libm")
            pm.add("convert-func-to-llvm")
            pm.add("reconcile-unrealized-casts")
            pm.run(self._imported_module.operation)

    def compile(self):
        """
        Compile graph from Buddy Graph to LLVM IR.
        """
        self.lower_to_top_level_ir()
        self.lower_to_llvm_ir()


class GraphImporter:
    """
    Imports an buddy graph and generates an MLIR module in high-level dialects.

    Attributes:
        _symbol_table (dict): A dictionary to keep track of the symbols.
        _body (List[Op]): The FX graph module to be imported.
        _func_name (str): Name of the generated MLIR function.
        _inputs (List[TensorMeta]): Input tensor(s) of the FX graph.
        _num_input_visited (int): Number of input nodes that have been visited.
        _module (buddy_mlir.ir.Module): The generated MLIR module.
        _ops_registry (dict): Registry for the candidate operations.
    """

    def __init__(
        self,
        body: List[Op],
        params_shapes: List[TensorMeta],
        inputs_shapes: List[TensorMeta],
        func_name: str,
        ops_registry: dict,
        do_param_pack: bool = False,
        device: DeviceType = DeviceType.CPU,
        verbose=False,
        enable_profile=True,
        enable_external_calls: bool = False,
    ):
        """
        Initializes the buddy Graph importer.

        Args:
            gm (Graph): The buddy graph that will be imported.
            inputs (List[TensorMeta]): Input tensor(s) of the buddy graph.
            func_name (str): Name of the generated MLIR function.
            ops_registry (dict): Registry for the candidate operations.
            enable_external_calls (bool): Enable external function call support (for oneDNN, etc.)
        """
        if ops_registry is None:
            ops_registry = {}
        self._symbol_table = {}
        self._body = body
        self._device = device
        self._func_name = func_name
        self._params_shapes = params_shapes
        self._inputs_shapes = inputs_shapes
        self._verbose = verbose
        self._enable_profile = enable_profile
        self._do_param_pack = do_param_pack
        self._param_packs = []
        self._num_input_visited = 0
        self._module = ir.Module.create()
        self._ops_registry = ops_registry
        self._current_param_pack_offset = None
        self._enable_external_calls = enable_external_calls

    def _str_to_mlir_dtype(self, dtype: str) -> ir.Type:
        """
        Converts a str to the corresponding MLIR dtype.

        Args:
            dtype (str): The tensor type.

        Returns:
            buddy_mlir.ir.Type: The corresponding MLIR data type.

        Raises:
            NotImplementedError: If the given dtype is not supported.
        """
        match dtype:
            case TensorDType.Int8:
                return ir.IntegerType.get_signless(8)
            case TensorDType.Int32:
                return ir.IntegerType.get_signless(32)
            case TensorDType.Int64:
                return ir.IntegerType.get_signless(64)
            case TensorDType.Float16:
                return ir.F16Type.get()
            case TensorDType.BFloat16:
                return ir.BF16Type.get()
            case TensorDType.Float32:
                return ir.F32Type.get()
            case TensorDType.Float64:
                return ir.F64Type.get()
            case TensorDType.Bool:
                return ir.IntegerType.get_signless(1)
            case TensorDType.Complex64:
                return ir.ComplexType.get(ir.F32Type.get())
            case TensorDType.Complex128:
                return ir.ComplexType.get(ir.F64Type.get())
            case _:
                raise NotImplementedError(f"Unsupported dtype {dtype}")

    def _pack_params(self) -> None:
        """
        Packs parameters of the graph to one memref.

        Returns:
        None

        Example:
        graph_instance = Graph(inputs, fake_params, ops_registry, func_name)
        graph_instance._pack_params()
        # The parameters of the graph are now packed to one memref.
        """
        dtypes = list(set([param.dtype for param in self._params_shapes]))
        dtypes.sort(key=str)
        self._current_param_pack_offset = {dtype: 0 for dtype in dtypes}
        for dtype in dtypes:
            params_of_dtype = [
                param for param in self._params_shapes if param.dtype == dtype
            ]
            param_total_size = 0
            for param in params_of_dtype:
                param_total_size += functools.reduce(
                    lambda x, y: x * y, list(param.shape), 1
                )
            mlir_dtype = self._str_to_mlir_dtype(dtype)
            self._param_packs.append(
                ir.MemRefType.get([param_total_size], mlir_dtype)
            )

    def import_graph(self) -> ir.Module:
        """
        Imports buddy graph and generates an MLIR module in high-level dialects.

        Returns:
            buddy_mlir.ir.Module: An MLIR module in high-level dialects.
        """
        assert self._do_param_pack == False
        node_names = [node.name for node in self._body]
        with ir.InsertionPoint(self._module.body):
            if self._enable_profile:
                # Add clock func signature
                f64 = ir.F64Type.get()
                ptr = llvm.PointerType.get()
                # func.func private @rtclock() -> f64
                func.FuncOp(
                    name="rtclock",
                    type=ir.FunctionType.get(inputs=[], results=[f64]),
                    visibility="private",
                )
                # func.func private @record_timing(!llvm.ptr, f64)
                func.FuncOp(
                    name="record_timing",
                    type=ir.FunctionType.get(inputs=[ptr, f64], results=[]),
                    visibility="private",
                )

                width = len(str(len(node_names)))
                for i, name in enumerate(node_names):
                    sym_name = f"op_name_{i:0{width}d}_{name}"
                    content = sym_name + "\00"
                    content_type = ir.Type.parse(
                        f"!llvm.array<{len(content)} x i8>"
                    )
                    linkage_attr = ir.Attribute.parse("#llvm.linkage<private>")
                    llvm.GlobalOp(
                        content_type,
                        ir.StringAttr.get(sym_name),
                        linkage_attr,
                        constant=True,
                        value=ir.StringAttr.get(content),
                        addr_space=ir.IntegerAttr.get(
                            ir.IntegerType.get_signless(32), 0
                        ),  # 关键字参数
                    )

            arguments = []
            inputs = self._params_shapes + self._inputs_shapes
            for arg in inputs:
                shape_list = list(arg.shape)
                dtype = arg.dtype
                mlir_dtype = self._str_to_mlir_dtype(dtype)
                tensor_arg = ir.RankedTensorType.get(shape_list, mlir_dtype)
                arguments.append(tensor_arg)
            extern_func = []
            for node in self._body:
                if isinstance(node, FuncOp):
                    extern_func.append(node)
                    self._import_op(node)

            @func.FuncOp.from_py_func(*arguments, name=self._func_name)
            def generated_func(*args):
                args_list = list(args)
                func_op_index = (
                    len(node_names) + 2 if self._enable_profile else 0
                )
                func_op = self._module.body.operations[func_op_index]
                for i, node in enumerate(self._body):
                    if node in extern_func:
                        continue

                    if self._enable_profile:
                        # %t_start = call @rtclock() : () -> f64
                        f64_type = ir.F64Type.get()
                        t_start = func.CallOp(
                            [f64_type], ir.FlatSymbolRefAttr.get("rtclock"), []
                        ).results[0]

                    old_ops = [op for op in func_op.body.blocks[0].operations]
                    if isinstance(node, OutputOp):
                        output_node_args = node.args
                        returns = [
                            self._symbol_table.get((str(output_arg), 0))
                            for output_arg in output_node_args
                        ]
                        self._symbol_table[("output", 0)] = returns
                    elif isinstance(node, PlaceholderOp):
                        self._import_placeholder(node, args_list)
                    elif isinstance(node, GetItemOp):
                        self._symbol_table[(str(node.name), 0)] = (
                            self._symbol_table[
                                (str(node.args[0]), node.args[1])
                            ]
                        )
                    else:
                        self._import_op(node)
                    new_ops = [op for op in func_op.body.blocks[0].operations]
                    if self._verbose:
                        print("=" * 20 + "Graph Node" + "=" * 20)
                        print("Node: " + node.name)
                        print("Type: " + str(node._op_type))
                        print("Arguments: " + str(node.args))
                        print("Parents: " + str(node._parents))
                        print("Children: " + str(node._children))
                        print("-" * 20 + "MLIR OPS" + "-" * 20)
                        for op in new_ops:
                            if op not in old_ops:
                                print(op)
                        print("")

                    if self._enable_profile:
                        # %t_end = call @rtclock() : () -> f64
                        t_end = func.CallOp(
                            [f64_type], ir.FlatSymbolRefAttr.get("rtclock"), []
                        ).results[0]

                        # %duration = arith.subf %t_end, %t_start : f64
                        duration = arith.SubFOp(t_end, t_start).result

                        # %name_ptr = llvm.mlir.addressof @op_name_post_attn_layernorm : !llvm.ptr
                        ptr_type = llvm.PointerType.get()
                        name_ptr = llvm.AddressOfOp(
                            ptr_type,
                            ir.FlatSymbolRefAttr.get(
                                f"op_name_{i:0{width}d}_{node.name}"
                            ),
                        ).result

                        # call @record_timing(%name_ptr, %duration) : (!llvm.ptr, f64) -> ()
                        func.CallOp(
                            [],
                            ir.FlatSymbolRefAttr.get("record_timing"),
                            [name_ptr, duration],
                        )

                return self._symbol_table.get(("output", 0))

            # Generate external function declarations for CallExternalOp nodes (only if enabled)
            if self._enable_external_calls:
                from .operation import CallExternalOp

                for node in self._body:
                    if isinstance(node, CallExternalOp):
                        self._generate_external_func_decl(node)

        return self._module

    def import_main_graph(self) -> ir.Module:
        """
        Imports buddy main graph to organize all subgraphs and generates an MLIR
        module in high-level dialects with memref.

        Returns:
            buddy_mlir.ir.Module: An MLIR module in high-level dialects.
        """
        with ir.InsertionPoint(self._module.body):
            arguments = []
            if self._do_param_pack:
                self._pack_params()
                arguments.extend(self._param_packs)
                inputs = self._inputs_shapes
            else:
                inputs = self._params_shapes + self._inputs_shapes
            for arg in inputs:
                shape_list = list(arg.shape)
                dtype = arg.dtype
                mlir_dtype = self._str_to_mlir_dtype(dtype)
                tensor_arg = ir.MemRefType.get(shape_list, mlir_dtype)
                arguments.append(tensor_arg)
            extern_func = []
            for node in self._body:
                if isinstance(node, FuncOp):
                    extern_func.append(node)
                    self._import_op(node)

            @func.FuncOp.from_py_func(*arguments, name=self._func_name)
            def generated_func(*args):
                args_list = list(args)
                for node in self._body:
                    if node in extern_func:
                        continue
                    if isinstance(node, OutputOp):
                        output_node_args = node.args
                        returns = [
                            self._symbol_table.get((str(output_arg), 0))
                            for output_arg in output_node_args
                        ]
                        self._symbol_table[("output", 0)] = returns
                    elif isinstance(node, PlaceholderOp):
                        self._import_placeholder(node, args_list)
                    elif isinstance(node, GetItemOp):
                        self._symbol_table[(str(node.name), 0)] = (
                            self._symbol_table[
                                (str(node.args[0]), node.args[1])
                            ]
                        )
                    else:
                        self._import_op(node)

                return self._symbol_table.get(("output", 0))

        return self._module

    def _import_placeholder(
        self, node: PlaceholderOp, args_list: List[ir.BlockArgument]
    ):
        """
        Imports a placeholder node from the Buddy graph.

        Parameters:
        - node (PlaceholderOp): The PlaceholderOp node representing the
        placeholder.
        - args_list (List[buddy_mlir.ir.BlockArgument]): List of input memrefs.

        Returns:
        None
        """
        if (
            self._num_input_visited < len(self._params_shapes)
            and self._do_param_pack
        ):
            dtype = node.tensor_meta["dtype"]
            pack_of_dtype = None
            for pack in args_list:
                if ir.MemRefType(
                    pack.type
                ).element_type == self._str_to_mlir_dtype(dtype):
                    pack_of_dtype = pack
                    break
            placeholder_name = self._ops_registry["param.extract"](
                node, self._current_param_pack_offset[dtype], pack_of_dtype
            ).result
            self._current_param_pack_offset[dtype] += functools.reduce(
                lambda x, y: x * y, list(node.tensor_meta["shape"]), 1
            )
        elif self._do_param_pack:
            if len(self._params_shapes) > 0:
                placeholder_name = args_list[
                    self._num_input_visited
                    - len(self._params_shapes)
                    + len(self._param_packs)
                ]
            else:
                placeholder_name = args_list[self._num_input_visited]
        else:
            placeholder_name = args_list[self._num_input_visited]

        self._symbol_table[(str(node.name), 0)] = placeholder_name
        self._num_input_visited += 1

    def _generate_external_func_decl(self, call_node):
        """
        Generate external function declaration for CallExternalOp.

        Args:
            call_node: CallExternalOp node that calls an external function
        """
        from .operation import CallExternalOp
        from ..ops.utils import mlir_element_type_get

        if not isinstance(call_node, CallExternalOp):
            return

        # Get function name
        func_name = call_node.call_func_name

        # Build argument types from CallOp's arguments
        arg_types = []
        for i, arg in enumerate(call_node.args):
            # Get the node that produces this argument
            arg_node = None
            for node in self._body:
                if node.name == arg:
                    arg_node = node
                    break

            if arg_node and hasattr(arg_node, "tensor_meta"):
                # Handle both dict and TensorMeta object
                if isinstance(arg_node.tensor_meta, dict):
                    shape = arg_node.tensor_meta.get("shape", [])
                    dtype = arg_node.tensor_meta.get("dtype", "float32")
                else:
                    shape = arg_node.tensor_meta.shape
                    dtype = arg_node.tensor_meta.dtype
                mlir_dtype = mlir_element_type_get(dtype)
                arg_types.append(
                    ir.RankedTensorType.get(list(shape), mlir_dtype)
                )

        # Build result types from CallOp's tensor_meta
        result_types = []
        if (
            hasattr(call_node, "tensor_meta")
            and "shape" in call_node.tensor_meta
        ):
            shape = call_node.tensor_meta["shape"]
            dtype = call_node.tensor_meta["dtype"]

            if (
                isinstance(shape, (list, tuple))
                and len(shape) > 0
                and isinstance(shape[0], (list, tuple))
            ):
                # Multiple outputs
                for i, s in enumerate(shape):
                    mlir_dtype = mlir_element_type_get(dtype[i])
                    result_types.append(ir.RankedTensorType.get(s, mlir_dtype))
            else:
                # Single output
                mlir_dtype = mlir_element_type_get(dtype)
                result_types.append(
                    ir.RankedTensorType.get(list(shape), mlir_dtype)
                )

        # Create function type
        function_type = ir.FunctionType.get(
            inputs=arg_types, results=result_types
        )

        # Create private function declaration
        with ir.InsertionPoint(self._module.body):
            func_decl = func.FuncOp(
                name=func_name, type=function_type, visibility="private"
            )
            # Add llvm.emit_c_interface attribute for C ABI compatibility
            func_decl.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()

    def _import_op(self, node: Op):
        """
        Imports an operation node from the buddy graph.

        Args:
            node (Op): The buddy node representing the operation.

        """
        op_name = node.__class__.__name__
        op_ret: ir.Operation | ir.Value | tuple | List | ir.OpResult = (
            self._ops_registry[op_name](node, self._symbol_table)
        )
        if isinstance(op_ret, tuple | List | ir.OpResultList):
            for i, operation in enumerate(op_ret):
                if isinstance(operation, ir.Operation) or isinstance(
                    operation, ir.OpView
                ):
                    self._symbol_table[(str(node.name), i)] = operation.result
                elif isinstance(operation, ir.OpResult):
                    self._symbol_table[(str(node.name), i)] = operation
                else:
                    raise NotImplementedError
        elif isinstance(op_ret, ir.OpResult):
            self._symbol_table[(str(node.name), 0)] = op_ret
        elif isinstance(op_ret, ir.BlockArgument):
            self._symbol_table[(str(node.name), 0)] = op_ret
        else:
            for i, result in enumerate(op_ret.results):
                self._symbol_table[(str(node.name), i)] = result

    def get_output_nodes(self):
        """
        Get output nodes from the lowered mlir func.
        """
        return self._symbol_table.get(("output", 0))
