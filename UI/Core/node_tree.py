"""Core node tree definitions and add-menu categories for Continuum Flow."""

import bpy
from nodeitems_utils import NodeCategory, NodeItem, NodeItemCustom


NODE_TREE_ID = "BLENDERCFD_NODE_TREE"
NODE_CATEGORIES_ID = "BLENDERCFD_NODE_CATEGORIES"


class BlenderCFDNodeTree(bpy.types.NodeTree):
    """Custom node tree used as the main editor space for Continuum Flow nodes."""

    bl_idname = NODE_TREE_ID
    bl_label = "Continuum Flow Nodes"
    bl_icon = "FORCE_TURBULENCE"


class BlenderCFDNodeCategory(NodeCategory):
    """Node category shown in the add menu of the Continuum Flow node tree."""

    @classmethod
    def poll(cls, context):
        """Return whether the category should be visible in the current editor."""
        space_data = getattr(context, "space_data", None)
        return space_data is not None and space_data.tree_type == NODE_TREE_ID


def build_node_categories():
    """Build the add menu categories for all registered Continuum Flow nodes."""
    return [
        BlenderCFDNodeCategory(
            "BLENDERCFD_NODES",
            "Continuum Flow",
            items=[
                NodeItem("BLENDERCFD_DOMAIN_NODE"),
                NodeItem("BLENDERCFD_GEOMETRY_NODE"),
                NodeItem("BLENDERCFD_OUTPUT_NODE"),
                NodeItem("BLENDERCFD_PHYSICS_NODE"),
                NodeItem("BLENDERCFD_SIMULATION_NODE"),
                NodeItem("BLENDERCFD_SOURCE_NODE"),
                NodeItem("BLENDERCFD_OBSTACLE_NODE"),
                NodeItem("BLENDERCFD_VIEWER_NODE"),
            ],
        ),
        BlenderCFDNodeCategory(
            "BLENDERCFD_FORCES",
            "Forces",
            items=[
                NodeItem("BLENDERCFD_FORCE_CONSTANT_NODE"),
                NodeItem("BLENDERCFD_FORCE_SWIRL_NODE"),
                NodeItem("BLENDERCFD_FORCE_POINT_NODE"),
                NodeItem("BLENDERCFD_FORCE_TURBULENCE_NODE"),
            ],
        ),
        BlenderCFDNodeCategory(
            "BLENDERCFD_PRESETS",
            "Presets",
            items=[
                NodeItemCustom(
                    draw=lambda _item, layout, _context: layout.operator(
                        "blendercfd.add_basic_setup",
                        text="Basic Continuum Flow Setup",
                        icon="FORCE_TURBULENCE",
                    ),
                ),
            ],
        ),
    ]


classes = (
    BlenderCFDNodeTree,
)
