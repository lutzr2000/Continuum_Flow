import bpy 
from .node_tree import ContinuumFlowNodeTree

def _node_cursor_location(context):
    """
    Return the preferred placement location for newly created nodes.
    """
    space_data = getattr(context, "space_data", None)
    cursor = getattr(space_data, "cursor_location", None)
    if cursor is None:
        return (0.0, 0.0)
    return (float(cursor[0]), float(cursor[1]))

def _active_continuum_flow_tree(context):
    """
    Return the active Continuum Flow node tree from the current editor context.
    """
    space_data = getattr(context, "space_data", None)
    if (
        space_data is None
        or getattr(space_data, "tree_type", "") != ContinuumFlowNodeTree.bl_idname
    ):
        return None
    return getattr(space_data, "edit_tree", None) or getattr(
        space_data, "node_tree", None
    )

class ContinuumFlow_OT_add_basic_setup(bpy.types.Operator):
    """
    Add a ready-to-use Continuum Flow starter setup with the core nodes linked.
    """

    bl_idname = "continuum_flow.add_basic_setup"
    bl_label = "Add Basic Continuum Flow Setup"
    bl_description = "Create a starter setup with Domain, Physics, Simulation, Viewer, and Output already connected"

    @classmethod
    def poll(cls, context):
        return _active_continuum_flow_tree(context) is not None
    
    def execute(self, context):
        node_tree = _active_continuum_flow_tree(context)
        if node_tree is None:
            self.report({"ERROR"}, "Open a Continuum Flow node tree first.")
            return {"CANCELLED"}

        cursor_x, cursor_y = _node_cursor_location(context)
        node_specs = (
            ("domain", "CONTINUUM_FLOW_DOMAIN_NODE", (cursor_x - 520.0, cursor_y + 250.0)),
            (
                "physics",
                "CONTINUUM_FLOW_PHYSICS_NODE",
                (cursor_x - 520.0, cursor_y - 140.0),
            ),
            ("simulation", "CONTINUUM_FLOW_SIMULATION_NODE", (cursor_x - 120.0, cursor_y)),
            ("viewer", "CONTINUUM_FLOW_VIEWER_NODE", (cursor_x + 280.0, cursor_y + 120.0)),
            ("output", "CONTINUUM_FLOW_OUTPUT_NODE", (cursor_x + 280.0, cursor_y - 120.0)),
        )

        for node in node_tree.nodes:
            node.select = False

        created_nodes = {}
        for key, node_type, location in node_specs:
            created_node = node_tree.nodes.new(node_type)
            created_node.location = location
            created_node.select = True
            created_nodes[key] = created_node

        node_tree.links.new(
            created_nodes["domain"].outputs["Domain"],
            created_nodes["simulation"].inputs["Domain"],
        )
        node_tree.links.new(
            created_nodes["physics"].outputs["Physics"],
            created_nodes["simulation"].inputs["Physics"],
        )
        node_tree.links.new(
            created_nodes["simulation"].outputs["Result"],
            created_nodes["viewer"].inputs["Result"],
        )
        node_tree.links.new(
            created_nodes["simulation"].outputs["Result"],
            created_nodes["output"].inputs["Result"],
        )

        node_tree.nodes.active = created_nodes["simulation"]
        return {"FINISHED"}