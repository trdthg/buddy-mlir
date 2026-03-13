# ===- fuse_ops.py -------------------------------------------------------------
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
# Construct op fusion pattern.
#
# ===---------------------------------------------------------------------------

from .. import Graph
from ..operation import *
from .. import DeviceType
from torch.fx.immutable_collections import immutable_list

classicfuse_register = {
    "transpose_matmul_fusion": TransposeMatmulFusedOp,
    "flash_attention_prefill_fusion": FlashAttentionForCpuPrefillOp,
    "gqa_attention_fusion": GQAAttentionFusedOp,
    "rotate_half_fusion": RotateHalfOp,
    "rotary_embedding_fusion": RotaryEmbeddingOp,
}

# TODO: classify op type for op fusion
# OP_TYPE_FUSABLE = [OpType.BroadcastType, OpType.ElementwiseType, OpType.ReshapeType]
# OP_TYPE_UNFUSABLE = [OpType.Unfusable, OpType.ConcatType]
# OP_TYPE_FUSABLE_BY_SPECIFIC_PASS = []
# ANCHOR_OP_TYPE = []


def classic_fuse_check(graph: Graph):
    """
    Function to identifies and fuses PermuteOp operations with preceding
    MatmulOp operations in a computation graph to optimize performance.

    Args:
        graph (Graph): The computation graph to analyze and optimize.

    Returns:
        None
    """
    for op in graph.body:
        pattern = None
        if isinstance(op, MatmulOp):
            parentop = [graph.node_table[str(i)] for i in op._parents]
            for target in parentop:
                if isinstance(target, PermuteOp) and target.args[
                    1
                ] == immutable_list([1, 0]):
                    pattern = target, parentop, "transpose_matmul_fusion"
        if pattern:
            transpose_matmul_fusion(
                graph, op, pattern[0], pattern[1], pattern[2]
            )


def transpose_matmul_fusion(
    graph: Graph, node, target: Op, parents: List[Op], pattern: str
):
    """
    Function to fuse some typical operations into one operation.
    Such as transpose + matmul
    Args:
    - graph (Graph): The input graph to be simplified.
    - node (Op): The operation to be fused.
    - target (Op): The target operation to be fused.
    - parents (List[Op]): The parents of the node to be fused.
    - pattern (str): The pattern of the fusion.
    Returns:
    - None: Modifies the input graph in place.
    """
    fused_op = classicfuse_register.get(pattern)()
    # matmulop -> fusedmatmulopnode
    fused_op.name = "fused" + node.name
    graph.displace_node(node, fused_op)
    fused_op.args.pop(fused_op.args.index(target.name))
    fused_op._parents.pop(fused_op._parents.index(target.name))
    fused_op.args.extend(target.args)

    fused_op._parents.extend(target._parents)
    targets_parent = [graph.node_table[i] for i in target._parents]
    for i in targets_parent:
        i.add_children(fused_op.name)
    target._children.pop(target._children.index(fused_op.name))

    if graph.check_delete_node(target):
        graph.delete_node(target, targets_parent)


def apply_classic_fusion(graph: Graph):
    """
    Function to fuse some typical operations into one operation and fuse
    all operations into one graph.

    Args:
    - graph (Graph): The input graph to be simplified.

    Returns:
    - None: Modifies the input graph in place.
    """
    new_op_group = []
    device = DeviceType.CPU
    # Run the first round of op fusion
    classic_fuse_check(graph)
    for op in graph.body:
        if isinstance(op, PlaceholderOp):
            continue
        new_op_group.append(op)
    graph.op_groups = {}
    graph.op_groups["subgraph0"] = new_op_group
    graph.group_map_device = {"subgraph0": device}


def simply_fuse(graph: Graph):
    """
    Function to fuse all operations into one graph.

    Args:
    - graph (Graph): The input graph to be simplified.

    Returns:
    - None: Modifies the input graph in place.
    """
    new_op_group = []
    device = DeviceType.CPU
    for op in graph.body:
        if isinstance(op, PlaceholderOp):
            continue
        new_op_group.append(op)
    graph.op_groups = {}
    graph.op_groups["subgraph0"] = new_op_group
    graph.group_map_device = {"subgraph0": device}


def rotate_half_fusion(graph: Graph):
    """
    Fuse slice + negate + cat patterns that implement RoPE rotate_half.
    """
    for op in list(graph.body):
        match = _match_rotate_half_cat(graph, op)
        if match is None:
            continue
        replace_rotate_half_with_fused_op(graph, op, *match)


def rotary_embedding_fusion(graph: Graph):
    """
    Fuse q*cos + rotate_half(q)*sin into one RotaryEmbeddingOp.
    """
    for op in list(graph.body):
        match = _match_rotary_embedding_add(graph, op)
        if match is None:
            continue
        replace_rotary_embedding_with_fused_op(graph, op, *match)


def _match_rotate_half_cat(graph: Graph, node: Op):
    if not isinstance(node, CatOp):
        return None

    inputs = node.args[0]
    if len(inputs) != 2:
        return None

    output_shape = list(node.tensor_meta["shape"])
    rank = len(output_shape)
    dim = int(node.args[1]) if len(node.args) > 1 else 0
    if dim < 0:
        dim += rank
    if dim != rank - 1:
        return None

    neg_node = graph.node_table.get(str(inputs[0]), None)
    lower_slice = graph.node_table.get(str(inputs[1]), None)
    if not isinstance(neg_node, NegOp) or not isinstance(lower_slice, SliceOp):
        return None
    if len(neg_node._parents) != 1 or len(lower_slice._parents) != 1:
        return None

    upper_slice = graph.node_table.get(neg_node._parents[0], None)
    if not isinstance(upper_slice, SliceOp):
        return None
    if len(upper_slice._parents) != 1:
        return None
    if upper_slice._parents[0] != lower_slice._parents[0]:
        return None

    base = graph.node_table.get(lower_slice._parents[0], None)
    if base is None:
        return None

    def _slice_signature(slice_node: SliceOp):
        if len(slice_node.args) < 4:
            return None
        slice_dim = int(slice_node.args[1])
        if slice_dim < 0:
            slice_dim += rank
        start = int(slice_node.args[2])
        end = int(slice_node.args[3])
        base_shape = list(base.tensor_meta["shape"])
        if end > base_shape[slice_dim]:
            end = base_shape[slice_dim]
        step = int(slice_node.args[4]) if len(slice_node.args) > 4 else 1
        return slice_dim, start, end, step

    lower_sig = _slice_signature(lower_slice)
    upper_sig = _slice_signature(upper_slice)
    if lower_sig is None or upper_sig is None:
        return None

    slice_dim, lower_start, lower_end, lower_step = lower_sig
    upper_dim, upper_start, upper_end, upper_step = upper_sig
    if slice_dim != dim or upper_dim != dim:
        return None
    if lower_step != 1 or upper_step != 1:
        return None

    full_extent = output_shape[dim]
    if full_extent <= 0 or full_extent % 2 != 0:
        return None
    half_extent = full_extent // 2
    if (lower_start, lower_end) != (0, half_extent):
        return None
    if (upper_start, upper_end) != (half_extent, full_extent):
        return None

    lower_shape = list(lower_slice.tensor_meta["shape"])
    upper_shape = list(upper_slice.tensor_meta["shape"])
    if lower_shape != upper_shape:
        return None
    if lower_shape[dim] != half_extent:
        return None

    return base, lower_slice, upper_slice, neg_node, dim


def replace_rotate_half_with_fused_op(
    graph: Graph,
    cat_node: CatOp,
    base: Op,
    lower_slice: SliceOp,
    upper_slice: SliceOp,
    neg_node: NegOp,
    dim: int,
):
    fused_op = classicfuse_register.get("rotate_half_fusion")()
    fused_op.name = "RotateHalf_" + cat_node.name
    graph.displace_node(cat_node, fused_op)
    fused_op._op_type = OpType.ElementwiseType

    for old_parent_name in list(fused_op._parents):
        old_parent = graph.node_table.get(old_parent_name, None)
        if old_parent is None:
            continue
        if fused_op.name in old_parent._children:
            old_parent._children.remove(fused_op.name)

    fused_op.args.clear()
    fused_op._parents.clear()
    fused_op.args.extend([base.name, dim])
    fused_op._parents.append(base.name)
    base.add_children(fused_op.name)

    if graph.check_delete_node(lower_slice):
        graph.delete_node(lower_slice, [base])
    if graph.check_delete_node(neg_node):
        graph.delete_node(neg_node, [upper_slice])
    if graph.check_delete_node(upper_slice):
        graph.delete_node(upper_slice, [base])


def _match_rotary_embedding_add(graph: Graph, node: Op):
    if not isinstance(node, AddOp):
        return None
    if len(node._parents) != 2:
        return None

    lhs = graph.node_table.get(node._parents[0], None)
    rhs = graph.node_table.get(node._parents[1], None)
    if not isinstance(lhs, MulOp) or not isinstance(rhs, MulOp):
        return None

    def _classify_mul(mul_node: MulOp):
        if len(mul_node._parents) != 2:
            return None
        parent0 = graph.node_table.get(mul_node._parents[0], None)
        parent1 = graph.node_table.get(mul_node._parents[1], None)
        if isinstance(parent0, RotateHalfOp):
            return ("rot", parent0, parent1)
        if isinstance(parent1, RotateHalfOp):
            return ("rot", parent1, parent0)
        return ("plain", parent0, parent1)

    lhs_kind, lhs_data, lhs_other = _classify_mul(lhs)
    rhs_kind, rhs_data, rhs_other = _classify_mul(rhs)
    if {lhs_kind, rhs_kind} != {"plain", "rot"}:
        return None

    plain_mul = lhs if lhs_kind == "plain" else rhs
    plain_data = lhs_data if lhs_kind == "plain" else rhs_data
    plain_other = lhs_other if lhs_kind == "plain" else rhs_other
    rot_mul = lhs if lhs_kind == "rot" else rhs
    rotate_half = lhs_data if lhs_kind == "rot" else rhs_data
    sin_input = lhs_other if lhs_kind == "rot" else rhs_other

    if not isinstance(rotate_half, RotateHalfOp):
        return None
    base = graph.node_table.get(rotate_half._parents[0], None)
    if base is None:
        return None

    if plain_data == base:
        cos_input = plain_other
    elif plain_other == base:
        cos_input = plain_data
    else:
        return None

    if cos_input is None or sin_input is None:
        return None

    dim = int(rotate_half.args[1]) if len(rotate_half.args) > 1 else -1
    return base, cos_input, sin_input, rotate_half, plain_mul, rot_mul, dim


def replace_rotary_embedding_with_fused_op(
    graph: Graph,
    add_node: AddOp,
    base: Op,
    cos_input: Op,
    sin_input: Op,
    rotate_half: RotateHalfOp,
    plain_mul: MulOp,
    rot_mul: MulOp,
    dim: int,
):
    fused_op = classicfuse_register.get("rotary_embedding_fusion")()
    fused_op.name = "RotaryEmbedding_" + add_node.name
    graph.displace_node(add_node, fused_op)
    fused_op._op_type = OpType.ElementwiseType

    for old_parent_name in list(fused_op._parents):
        old_parent = graph.node_table.get(old_parent_name, None)
        if old_parent is None:
            continue
        if fused_op.name in old_parent._children:
            old_parent._children.remove(fused_op.name)

    fused_op.args.clear()
    fused_op._parents.clear()
    fused_op.args.extend([base.name, cos_input.name, sin_input.name, dim])
    fused_op._parents.extend([base.name, cos_input.name, sin_input.name])
    base.add_children(fused_op.name)
    cos_input.add_children(fused_op.name)
    sin_input.add_children(fused_op.name)

    if graph.check_delete_node(plain_mul):
        graph.delete_node(
            plain_mul,
            [graph.node_table[p] for p in plain_mul._parents],
        )
    if graph.check_delete_node(rot_mul):
        graph.delete_node(rot_mul, [graph.node_table[p] for p in rot_mul._parents])
    if graph.check_delete_node(rotate_half):
        graph.delete_node(rotate_half, [base])


def flash_attention_prefill(graph: Graph):
    """
    Replace ScaledDotProductFlashAttentionForCpuOp with FlashAttentionForCpuPrefillOp.
    """
    new_op_group = []
    device = DeviceType.CPU
    replace_attention_op(graph)

    for op in graph.body:
        if isinstance(op, PlaceholderOp):
            continue
        new_op_group.append(op)

    graph.op_groups = {"subgraph0": new_op_group}
    graph.group_map_device = {"subgraph0": device}


def replace_attention_op(graph: Graph):
    """
    replace ScaledDotProductFlashAttentionForCpuOp with FlashAttentionForCpuPrefillOp.
    """
    for op in list(graph.body):
        if isinstance(op, ScaledDotProductFlashAttentionForCpuOp):
            new_op = classicfuse_register.get(
                "flash_attention_prefill_fusion"
            )()
            new_op.name = "FlashAttentionForCpuPrefillOp"
            graph.displace_node(op, new_op)


def gqa_attention_fusion(graph: Graph):
    """
    Function to fuse GQA Attention operations into one operation and fuse
    all operations into one graph.

    Args:
    - graph (Graph): The input graph to be simplified.

    Returns:
    - None: Modifies the input graph in place.
    """
    new_op_group = []
    device = DeviceType.CPU
    gqa_attention_fusion_check(graph)
    for op in graph.body:
        if isinstance(op, PlaceholderOp):
            continue
        new_op_group.append(op)
    graph.op_groups = {}
    graph.op_groups["subgraph0"] = new_op_group
    graph.group_map_device = {"subgraph0": device}


def gqa_attention_fusion_check(graph: Graph):
    for op in graph.body:
        # === GQA Attention pattern ===
        if isinstance(op, ScaledDotProductFlashAttentionForCpuOp):

            # get KV and View nodes
            k_view_node = graph.node_table.get(op._parents[1], None)
            v_view_node = graph.node_table.get(op._parents[2], None)

            if not (
                isinstance(k_view_node, ViewOp)
                and isinstance(v_view_node, ViewOp)
            ):
                continue

            # trace Key branch for torch2.10:
            # View <- Clone <- Expand <- Unsqueeze <- IndexPut
            k_clone = graph.node_table.get(k_view_node._parents[0], None)
            if not isinstance(k_clone, CloneOp):
                continue
            k_expand = graph.node_table.get(k_clone._parents[0], None)
            if not isinstance(k_expand, ExpandOp):
                continue
            k_cache_unsqueeze = graph.node_table.get(k_expand._parents[0], None)
            if not isinstance(k_cache_unsqueeze, UnsqueezeOp):
                continue
            k_index_put = graph.node_table.get(
                k_cache_unsqueeze._parents[0], None
            )
            if not isinstance(k_index_put, IndexPutOp):
                continue

            # trace Value branch for torch2.10:
            # View <- Clone <- Expand <- Unsqueeze <- IndexPut
            v_clone = graph.node_table.get(v_view_node._parents[0], None)
            if not isinstance(v_clone, CloneOp):
                continue
            v_expand = graph.node_table.get(v_clone._parents[0], None)
            if not isinstance(v_expand, ExpandOp):
                continue
            v_cache_unsqueeze = graph.node_table.get(v_expand._parents[0], None)
            if not isinstance(v_cache_unsqueeze, UnsqueezeOp):
                continue
            v_index_put = graph.node_table.get(
                v_cache_unsqueeze._parents[0], None
            )
            if not isinstance(v_index_put, IndexPutOp):
                continue
            replace_gqa_attention_with_fused_op(
                graph,
                op,
                k_view_node,
                k_clone,
                k_expand,
                k_cache_unsqueeze,
                v_view_node,
                v_clone,
                v_expand,
                v_cache_unsqueeze,
                "gqa_attention_fusion",
            )


def replace_gqa_attention_with_fused_op(
    graph: Graph,
    sdpa_node: Op,
    k_view: Op,
    k_clone: Op,
    k_expand: Op,
    k_cache_unsqueeze: Op,
    v_view: Op,
    v_clone: Op,
    v_expand: Op,
    v_cache_unsqueeze: Op,
    pattern: str,
):
    """
    Fuse GQA subgraph
    into one GQAAttentionFusedOp.
    """
    fused_cls = classicfuse_register.get(pattern)
    fused_op = fused_cls()
    fused_op.name = "GQAAttentionFusedOp"

    # replace SDPA node with GQAAttentionFusedOp
    graph.displace_node(sdpa_node, fused_op)

    # clear old KV View input inherited by SDPA
    # assume sdpa_node.args[0] is Query, keep unchanged
    # args[1] and args[2] are k_view and v_view, need to pop
    fused_op.args.pop(fused_op.args.index(k_view.name))
    fused_op._parents.pop(fused_op._parents.index(k_view.name))
    fused_op.args.pop(fused_op.args.index(v_view.name))
    fused_op._parents.pop(fused_op._parents.index(v_view.name))

    for k_parent in k_cache_unsqueeze._parents:
        fused_op._parents.append(k_parent)
        fused_op.args.append(k_parent)
    for v_parent in v_cache_unsqueeze._parents:
        fused_op._parents.append(v_parent)
        fused_op.args.append(v_parent)

    k_view._children.clear()
    if graph.check_delete_node(k_view):
        graph.delete_node(k_view, [k_clone])
    if graph.check_delete_node(k_clone):
        graph.delete_node(k_clone, [k_expand])
    if graph.check_delete_node(k_expand):
        graph.delete_node(k_expand, [k_cache_unsqueeze])
    if graph.check_delete_node(k_cache_unsqueeze):
        k_orig_parents = [
            graph.node_table.get(p, None) for p in k_cache_unsqueeze._parents
        ]
        graph.delete_node(k_cache_unsqueeze, k_orig_parents)

    v_view._children.clear()
    if graph.check_delete_node(v_view):
        graph.delete_node(v_view, [v_clone])
    if graph.check_delete_node(v_clone):
        graph.delete_node(v_clone, [v_expand])
    if graph.check_delete_node(v_expand):
        graph.delete_node(v_expand, [v_cache_unsqueeze])
    if graph.check_delete_node(v_cache_unsqueeze):
        v_orig_parents = [
            graph.node_table.get(p, None) for p in v_cache_unsqueeze._parents
        ]
        graph.delete_node(v_cache_unsqueeze, v_orig_parents)
