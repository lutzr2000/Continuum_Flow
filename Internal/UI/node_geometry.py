from . import sockets
from . import node_base
import bpy
from bpy.props import BoolProperty, FloatProperty, PointerProperty


class ContinuumFlowGeometryNode(node_base.ContinuumFlowBaseNode):
    """
    Node used to select a mesh for obstacles or sources.
    """

    bl_idname = "CONTINUUM_FLOW_GEOMETRY_NODE"
    bl_label = "Geometry"
    bl_icon = "OUTLINER_OB_MESH"
    bl_width_default = 220.0
    bl_width_min = 200.0
    bl_width_max = 360.0

    source_object: PointerProperty(name="Object", type=bpy.types.Object)  # type: ignore
    deforming: BoolProperty(
        name="Deforming",
        description="If a mesh is deforming (Armature, shape keys) this should be selected. With dense meshes this will degrade performance",
        default=False,
    )  # type: ignore
    velocity_transfer: FloatProperty(
        name="Velocity Transfer",
        default=0.0,
        soft_min=-100.0,
        soft_max=100.0,
        subtype="PERCENTAGE",
        description="Percentage of object motion transferred to the fluid when used as a source; values beyond +/-100% can be entered manually",
        options=set(),
    )  # type: ignore

    def _sync_node(self):
        self._ensure_named_output(
            sockets.ContinuumFlowGeometrySocket.bl_idname, "Geometry"
        )

    def draw_buttons(self, context, layout):
        self._set_layout_enabled(context, layout)
        layout.prop(self, "source_object", text="Object")
        layout.prop(self, "deforming")
        layout.prop(self, "velocity_transfer", slider=True)
