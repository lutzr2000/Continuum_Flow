import bpy
from bpy.props import BoolProperty, FloatProperty, PointerProperty

from . import node_base
from . import sockets


class ContinuumFlowCollectionNode(node_base.ContinuumFlowBaseNode):
    """Every mesh in the collection is used."""

    bl_idname = "CONTINUUM_FLOW_COLLECTION_NODE"
    bl_label = "Collection"
    bl_icon = "OUTLINER_COLLECTION"
    bl_width_default = 220.0
    bl_width_min = 200.0
    bl_width_max = 360.0

    source_collection: PointerProperty(
        name="Collection",
        type=bpy.types.Collection,
        description="Collection whose mesh objects are used as geometry",
    )  # type: ignore
    deforming: BoolProperty(
        name="Deforming",
        description="If meshes in the collection are deforming (Armature, shape keys) this should be selected. With dense meshes this will degrade performance",
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
            sockets.ContinuumFlowMultiGeometrySocket.bl_idname,
            "Geometry",
        )

    def draw_buttons(self, context, layout):
        self._set_layout_enabled(context, layout)
        layout.prop(self, "source_collection", text="Collection")
        layout.prop(self, "deforming")
        layout.prop(self, "velocity_transfer", slider=True)
