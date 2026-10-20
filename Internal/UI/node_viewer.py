from . import sockets
from . import node_base
from ..Core import viewer
from ..Core import preview
from ..Core.solver import solver_status
from ..Core.solver.solver_manager import solver_manager
from bpy.props import BoolProperty, FloatProperty, FloatVectorProperty


class ContinuumFlowViewerNode(node_base.ContinuumFlowBaseNode):
    """
    Node used as a lightweight endpoint for inspecting simulation results.
    """

    bl_idname = "CONTINUUM_FLOW_VIEWER_NODE"
    bl_label = "Viewer"
    bl_icon = "HIDE_OFF"
    bl_width_default = 260.0
    bl_width_min = 240.0
    bl_width_max = 520.0
    domain_preview_active: BoolProperty(default=False, options={"HIDDEN", "SKIP_SAVE"})  # type: ignore
    preview_smoke_density: FloatProperty(name="Smoke Density", default=2.0, min=0.0, options=set())  # type: ignore
    preview_smoke_color: FloatVectorProperty(name="Smoke Color", default=(0.32, 0.34, 0.38), size=3, min=0.0, max=1.0, subtype="COLOR", options=set())  # type: ignore
    preview_flame_density: FloatProperty(name="Flame Density", default=5.0, min=0.0, options=set())  # type: ignore
    preview_flame_color: FloatVectorProperty(name="Flame Color", default=(1.0, 0.12, 0.01), size=3, min=0.0, max=1.0, subtype="COLOR", options=set())  # type: ignore

    def _sync_node(self):
        self._ensure_socket(
            self.inputs, sockets.ContinuumFlowResultSocket.bl_idname, "Result"
        )

    def free(self):
        self.domain_preview_active = False
        viewer.disable_domain_preview()
        if preview.preview_running_for(self):
            preview.request_preview_stop()

    def draw_buttons(self, context, layout):
        is_domain_preview_active = viewer.viewer_domain_preview_enabled(self)
        is_live_preview_active = preview.preview_running_for(self)

        col = layout.column(align=True)

        col.operator(
            "continuum_flow.viewer_toggle_domain",
            text="Hide Domain" if is_domain_preview_active else "Show Domain",
            icon="HIDE_ON" if is_domain_preview_active else "HIDE_OFF",
        )

        col.separator()
        col.label(text="Preview")
        preview_row = col.row()
        preview_row.enabled = not solver_status.bake_running and (
            not solver_status.preview_running or is_live_preview_active
        )
        preview_row.operator(
            "continuum_flow.viewer_preview",
            text="Stop" if is_live_preview_active else "Play",
            icon="PAUSE" if is_live_preview_active else "PLAY",
        )

        col.separator()
        settings = col.column(align=True)
        settings.enabled = not (
            solver_status.bake_running or solver_status.preview_running
        )
        settings.prop(self, "preview_smoke_density")
        settings.prop(self, "preview_flame_density")

        settings.separator()
        settings.prop(self, "preview_smoke_color")
        settings.prop(self, "preview_flame_color")

        stats = solver_manager.get_stats()

        if stats:
            col.separator()

            active_cells = int(stats["active_tiles"]) * 64
            total_cells = int(stats["total_tiles"]) * 64
            active_cells_text = f"{active_cells:,}".replace(",", ".")
            total_cells_text = f"{total_cells:,}".replace(",", ".")

            col.label(text=f"Cells: {active_cells_text} / {total_cells_text}")
