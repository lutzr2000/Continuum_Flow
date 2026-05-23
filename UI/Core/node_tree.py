import bpy
from nodeitems_utils import NodeCategory, NodeItem, NodeItemCustom

NODE_TREE_ID = "CONTINUUM_FLOW_NODE_TREE"
NODE_CATEGORIES_ID = "CONTINUUM_FLOW_NODE_CATEGORIES"


class ContinuumFlowNodeTree(bpy.types.NodeTree):
    """
    Custom node tree used as the main editor space for Continuum Flow nodes.
    """

    bl_idname = NODE_TREE_ID
    bl_label = "Continuum Flow Nodes"
    bl_icon = "FORCE_TURBULENCE"


class ContinuumFlowNodeCategory(NodeCategory):
    """
    Node category shown in the add menu of the Continuum Flow node tree.
    """

    @classmethod
    def poll(cls, context):
        """
        Return whether the category should be visible in the current editor.
        """
        space_data = getattr(context, "space_data", None)
        return space_data is not None and space_data.tree_type == NODE_TREE_ID


def build_node_categories():
    """
    Build the add menu categories for all registered Continuum Flow nodes.
    """
    return [
        ContinuumFlowNodeCategory(
            "CONTINUUM_FLOW_NODES",
            "Continuum Flow",
            items=[
                NodeItem("CONTINUUM_FLOW_DOMAIN_NODE"),
                NodeItem("CONTINUUM_FLOW_GEOMETRY_NODE"),
                NodeItem("CONTINUUM_FLOW_OUTPUT_NODE"),
                NodeItem("CONTINUUM_FLOW_PHYSICS_NODE"),
                NodeItem("CONTINUUM_FLOW_SIMULATION_NODE"),
                NodeItem("CONTINUUM_FLOW_SOURCE_NODE"),
                NodeItem("CONTINUUM_FLOW_OBSTACLE_NODE"),
                NodeItem("CONTINUUM_FLOW_VIEWER_NODE"),
            ],
        ),
        ContinuumFlowNodeCategory(
            "CONTINUUM_FLOW_FORCES",
            "Forces",
            items=[
                NodeItem("CONTINUUM_FLOW_FORCE_CONSTANT_NODE"),
                NodeItem("CONTINUUM_FLOW_FORCE_SWIRL_NODE"),
                NodeItem("CONTINUUM_FLOW_FORCE_POINT_NODE"),
                NodeItem("CONTINUUM_FLOW_FORCE_TURBULENCE_NODE"),
            ],
        ),
        ContinuumFlowNodeCategory(
            "CONTINUUM_FLOW_PRESETS",
            "Presets",
            items=[
                NodeItemCustom(
                    draw=lambda _item, layout, _context: layout.operator(
                        "continuum_flow.add_basic_setup",
                        text="Basic Continuum Flow Setup",
                        icon="FORCE_TURBULENCE",
                    ),
                ),
            ],
        ),
    ]


classes = (ContinuumFlowNodeTree,)
