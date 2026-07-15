from . import sockets
from . import node_base
import bpy
from bpy.props import PointerProperty


class ContinuumFlowGeometryNode(node_base.ContinuumFlowBaseNode):
    """
    Node used to reference a Blender object as geometry inside the CFD graph.
    """

    bl_idname = "CONTINUUM_FLOW_GEOMETRY_NODE"
    bl_label = "Geometry"
    bl_icon = "OUTLINER_OB_MESH"
    bl_width_default = 220.0
    bl_width_min = 200.0
    bl_width_max = 360.0

    source_object: PointerProperty(name="Object", type=bpy.types.Object)  # type: ignore

    def _sync_node(self):
        self._ensure_named_output("NodeSocketGeometry", "Geometry")

    def draw_buttons(self, context, layout):
        self._set_layout_enabled(context, layout)
        layout.prop(self, "source_object", text="Object")