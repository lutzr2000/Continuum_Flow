import bpy
from bpy.props import FloatProperty, PointerProperty

from . import node_base
from . import sockets


def reference_frame_object_poll(_self, source_object):
    """Allow meshes and empties to define a simulation reference frame."""
    return getattr(source_object, "type", None) in {"MESH", "EMPTY"}


class ContinuumFlowReferenceFrameNode(node_base.ContinuumFlowBaseNode):
    """Node used to select an object as a simulation reference frame."""

    bl_idname = "CONTINUUM_FLOW_REFERENCE_FRAME_NODE"
    bl_label = "Reference Frame"
    bl_icon = "ORIENTATION_LOCAL"
    bl_width_default = 220.0
    bl_width_min = 200.0
    bl_width_max = 360.0

    source_object: PointerProperty(
        name="Object",
        type=bpy.types.Object,
        poll=reference_frame_object_poll,
    )  # type: ignore
    velocity_transfer: FloatProperty(
        name="Velocity Transfer",
        default=0.0,
        min=-1.0,
        max=1.0,
        description="Amount of reference-frame velocity transferred to the fluid",
    )  # type: ignore

    def _sync_node(self):
        self._ensure_named_output(
            sockets.ContinuumFlowReferenceFrameSocket.bl_idname,
            "Reference Frame",
        )

    def draw_buttons(self, context, layout):
        self._set_layout_enabled(context, layout)
        layout.prop(self, "source_object", text="Object")
        layout.prop(self, "velocity_transfer", slider=True)
