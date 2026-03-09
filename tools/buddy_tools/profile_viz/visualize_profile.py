#!/usr/bin/env python3
# ===- visualize_profile.py -------------------------------------------------
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
# ===-----------------------------------------------------------------------
#
# Merge Buddy static graph JSON with timing JSON and emit annotated reports.
#
# The heavy lifting lives in `graph_report.py`, `hierarchy_model.py`,
# `hierarchy_svg.py`, and `api.py`. This file stays as the stable CLI
# entrypoint and re-exports the hierarchy helpers used by the Python tests.
#
# ===-----------------------------------------------------------------------

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    _TOOLS_DIR = Path(__file__).resolve().parents[2]
    if str(_TOOLS_DIR) not in sys.path:
        sys.path.insert(0, str(_TOOLS_DIR))

from buddy_tools.profile_viz.graph_report import (
    enrich_graph_with_estimates,
    load_profile_map,
    merge_graph_with_profile,
    render_dot,
    render_svg,
)
from buddy_tools.profile_viz.hierarchy_scene import (
    build_hierarchy_scene,
    build_module_hierarchy_graph_model,
)
from buddy_tools.profile_viz.hierarchy_svg import (
    render_module_hierarchy_svg,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render a profiled Buddy static graph as Graphviz DOT."
    )
    parser.add_argument(
        "--graph-json",
        type=Path,
        required=True,
        help="Path to subgraph0_graph.json.",
    )
    parser.add_argument(
        "--profile-json",
        type=Path,
        required=True,
        help="Path to subgraph0_profile.json.",
    )
    parser.add_argument(
        "--output-dot",
        type=Path,
        default=None,
        help="Output DOT path. Defaults to subgraph0_profile.dot.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional merged JSON output. Defaults to subgraph0_profile_merged.json.",
    )
    parser.add_argument(
        "--output-svg",
        type=Path,
        default=None,
        help="Optional SVG output path. Defaults to subgraph0_profile.svg.",
    )

    parser.add_argument(
        "--output-hierarchy-svg",
        type=Path,
        default=None,
        help=(
            "Optional single-file module hierarchy SVG path. "
            "Defaults to subgraph0_module_hierarchy.svg."
        ),
    )
    parser.add_argument(
        "--hierarchy-only",
        action="store_true",
        help=(
            "Only emit the module hierarchy SVG and merged JSON, skipping flat "
            "SVG rendering."
        ),
    )
    return parser.parse_args()




def main():
    args = parse_args()
    output_dot = args.output_dot or args.graph_json.with_name("subgraph0_profile.dot")
    output_json = args.output_json or args.graph_json.with_name(
        "subgraph0_profile_merged.json"
    )
    output_svg = args.output_svg or args.graph_json.with_name(
        "subgraph0_profile.svg"
    )

    output_hierarchy_svg = args.output_hierarchy_svg or args.graph_json.with_name(
        "subgraph0_module_hierarchy.svg"
    )

    graph_payload = json.loads(args.graph_json.read_text())
    profile_payload, profile_map = load_profile_map(args.profile_json)
    merged_graph = merge_graph_with_profile(graph_payload, profile_map)
    merged_graph["profile_summary"] = {
        "unit": profile_payload.get("unit", "ms"),
        "total_avg_ms": profile_payload.get("total_avg_ms", 0.0),
        "record_count": len(profile_payload.get("records", [])),
    }
    merged_graph = enrich_graph_with_estimates(merged_graph)

    dot_text = render_dot(merged_graph)
    hierarchy_svg_text = render_module_hierarchy_svg(merged_graph)

    output_json.write_text(json.dumps(merged_graph, indent=2) + "\n")
    output_hierarchy_svg.write_text(hierarchy_svg_text)
    output_dot.write_text(dot_text)

    if not args.hierarchy_only:
        svg_text = render_svg(dot_text)
        output_svg.write_text(svg_text)

    print(f"Merged profile JSON: {output_json}")
    print(f"Profile DOT: {output_dot}")
    print(f"Module hierarchy SVG: {output_hierarchy_svg}")
    if not args.hierarchy_only:
        print(f"Profile SVG: {output_svg}")


if __name__ == "__main__":
    main()
