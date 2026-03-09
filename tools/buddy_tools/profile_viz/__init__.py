"""Buddy static-graph profile visualization package."""

from .hierarchy_model import build_module_hierarchy_groups
from .hierarchy_scene import (
    build_hierarchy_scene,
    build_module_hierarchy_graph_model,
)
from .hierarchy_svg import (
    border_color_for_display_node,
    dot_label_for_display_node,
    fill_color_for_display_node,
    render_module_hierarchy_svg,
    render_scene_dot,
    scene_node_detail,
)
from .graph_report import (
    enrich_graph_with_estimates,
    load_profile_map,
    merge_graph_with_profile,
    render_dot,
    render_svg,
)

__all__ = [
    "build_hierarchy_scene",
    "build_module_hierarchy_graph_model",
    "build_module_hierarchy_groups",
    "border_color_for_display_node",
    "dot_label_for_display_node",
    "fill_color_for_display_node",
    "render_module_hierarchy_svg",
    "render_scene_dot",
    "scene_node_detail",
    "enrich_graph_with_estimates",
    "load_profile_map",
    "merge_graph_with_profile",
    "render_dot",
    "render_svg",
]
