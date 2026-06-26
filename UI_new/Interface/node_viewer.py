from . import helper_functions
from . import sockets
from . import node_base
from ..Core import viewer
from bpy.props import BoolProperty


class ContinuumFlowViewerNode(node_base.ContinuumFlowBaseNode):
    """
    Node used as a lightweight endpoint for inspecting simulation results.
    """

    bl_idname = "CONTINUUM_FLOW_VIEWER_NODE"
    bl_label = "Viewer"
    bl_icon = "HIDE_OFF"
    bl_width_default = 180.0
    bl_width_min = 160.0
    bl_width_max = 260.0
    domain_preview_active: BoolProperty(default=False, options={"HIDDEN", "SKIP_SAVE"})  # type: ignore
    live_preview: BoolProperty(name="Live Preview", default=True, description="Show newly written VDB frames in Blender while the bake is still running", options=set())  # type: ignore
    debug: BoolProperty(name="Debug", default=False, description="Enable solver debug timing output during the bake", options=set())  # type: ignore

    def _sync_node(self):
        helper_functions.ensure_socket(self.inputs, sockets.ContinuumFlowResultSocket.bl_idname, "Result")

    def free(self):
        self.domain_preview_active = False
        viewer.disable_domain_preview()

    def draw_buttons(self, context, layout):
        self._set_layout_enabled(context, layout)
        is_domain_preview_active = viewer.viewer_domain_preview_enabled(self)
        col = layout.column(align=True)
        col.operator(
            "continuum_flow.viewer_toggle_domain",
            text="Hide Domain" if is_domain_preview_active else "Show Domain",
            icon="HIDE_ON" if is_domain_preview_active else "HIDE_OFF",
        )
        col.prop(self, "live_preview")
        col.prop(self, "debug")