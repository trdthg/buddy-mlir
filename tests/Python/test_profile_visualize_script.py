# RUN: %PYTHON %s 2>&1 | FileCheck %s

import json
import subprocess
import sys
import tempfile
from pathlib import Path


script_path = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "buddy_tools"
    / "profile_viz"
    / "visualize_profile.py"
)

graph_payload = {
    "func_name": "forward",
    "source_context": {
        "entry_module_class": "transformer_model.DeepSeekTransformerBlock",
        "entry_module_name": "DeepSeekTransformerBlock",
        "entry_module_qualname": "DeepSeekTransformerBlock",
    },
    "nodes": [
        {
            "id": 0,
            "name": "x",
            "profile_name": "op_name_0_x",
            "kind": "PlaceholderOp",
            "op_type": "PlaceholderType",
            "shape": [2, 3],
            "dtype": "float32",
            "provenance": {
                "nn_module_stack": {
                    "arg": [
                        "L['self'].mlp",
                        "<class 'transformer_model.DeepSeekFFN'>",
                    ]
                }
            },
        },
        {
            "id": 1,
            "name": "w",
            "profile_name": "op_name_1_w",
            "kind": "PlaceholderOp",
            "op_type": "PlaceholderType",
            "shape": [3, 4],
            "dtype": "float32",
            "provenance": {
                "nn_module_stack": {
                    "weight": [
                        "L['self'].mlp.gate_proj",
                        "<class 'torch.nn.modules.linear.Linear'>",
                    ],
                    "mlp": [
                        "L['self'].mlp",
                        "<class 'transformer_model.DeepSeekFFN'>",
                    ],
                }
            },
        },
        {
            "id": 2,
            "name": "mm",
            "profile_name": "op_name_2_mm",
            "kind": "MatmulOp",
            "op_type": "ReduceType",
            "shape": [2, 4],
            "dtype": "float32",
            "arguments": ["x", "w"],
            "parents": ["x", "w"],
            "provenance": {
                "nn_module_stack": {
                    "weight": [
                        "L['self'].mlp.gate_proj",
                        "<class 'torch.nn.modules.linear.Linear'>",
                    ],
                    "mlp": [
                        "L['self'].mlp",
                        "<class 'transformer_model.DeepSeekFFN'>",
                    ],
                }
            },
        },
    ],
    "edges": [
        {"source": "x", "target": "mm"},
        {"source": "w", "target": "mm"},
    ],
}

profile_payload = {
    "unit": "ms",
    "total_avg_ms": 2.25,
    "records": [
        {
            "op_name": "op_name_2_mm",
            "avg_ms": 2.25,
            "percentage": 100.0,
        }
    ],
}

with tempfile.TemporaryDirectory() as tmp_dir:
    tmp_path = Path(tmp_dir)
    graph_json = tmp_path / "graph.json"
    profile_json = tmp_path / "profile.json"
    output_dot = tmp_path / "profile.dot"
    output_svg = tmp_path / "profile.svg"
    output_hierarchy_svg = tmp_path / "module_hierarchy.svg"
    output_json = tmp_path / "profile_merged.json"

    graph_json.write_text(json.dumps(graph_payload) + "\n")
    profile_json.write_text(json.dumps(profile_payload) + "\n")

    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--graph-json",
            str(graph_json),
            "--profile-json",
            str(profile_json),
            "--output-dot",
            str(output_dot),
            "--output-svg",
            str(output_svg),
            "--output-hierarchy-svg",
            str(output_hierarchy_svg),
            "--output-json",
            str(output_json),
        ],
        check=True,
    )

    merged_payload = json.loads(output_json.read_text())
    dot_text = output_dot.read_text()
    svg_text = output_svg.read_text()
    hierarchy_svg_text = output_hierarchy_svg.read_text()

    print(merged_payload["nodes"][2]["profile"]["avg_ms"])
    print(merged_payload["nodes"][2]["estimates"]["estimated_flops"])
    print(merged_payload["nodes"][2]["estimates"]["estimated_logical_bytes"])
    print("n0 -> n2;" in dot_text and "n1 -> n2;" in dot_text)
    print("flops=48F" in dot_text)
    print("<svg" in svg_text)
    print("buddy-module-hierarchy" in hierarchy_svg_text)
    print("返回上一级" in hierarchy_svg_text)
    print("DeepSeekTransformerBlock · forward" in hierarchy_svg_text)
    print("flops=48F" in hierarchy_svg_text)
    print("bytes=104B" in hierarchy_svg_text)
    print("shape=[2, 4]" in hierarchy_svg_text)
    print("输入形状=[2, 3] + [3, 4]" in hierarchy_svg_text)
    print("输出形状=[2, 3]" in hierarchy_svg_text)
    print("输出形状=[3, 4]" in hierarchy_svg_text)
    print("data-target-scene" in hierarchy_svg_text)
    print("patterns" not in merged_payload and "pattern_summary" not in merged_payload)

# CHECK: 2.25
# CHECK: 48
# CHECK: 104
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
