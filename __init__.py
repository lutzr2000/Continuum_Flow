bl_info = {
    "name": "Continuum Flow",
    "author": "...",
    "version": (0, 1, 0),
    "blender": (5, 0, 0),
    "category": "Node",
}

import bpy

from .Internal.Core import environment
from .Internal.UI import register as registry


def _tag_ui_redraw():
    window_manager = getattr(bpy.context, "window_manager", None)
    if window_manager is None:
        return

    for window in window_manager.windows:
        screen = getattr(window, "screen", None)
        if screen is None:
            continue

        for area in screen.areas:
            if area.type in {"PREFERENCES", "NODE_EDITOR", "PROPERTIES"}:
                area.tag_redraw()


class ContinuumFlow_OT_install_solver_environment(bpy.types.Operator):
    bl_idname = "continuum_flow.install_solver_environment"
    bl_label = "Install Solver Environment"
    bl_description = "Create the Continuum Flow solver environment with uv"

    def execute(self, context):
        try:
            result = environment.install_solver_environment(__file__)
        except (FileNotFoundError, RuntimeError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        registry.check_solver_status()
        _tag_ui_redraw()
        self.report(
            {"INFO"},
            f"Solver environment is ready: {result['python_executable']}",
        )
        return {"FINISHED"}


class ContinuumFlowPreferences(bpy.types.AddonPreferences):
    bl_idname = __name__

    def draw(self, context):
        layout = self.layout
        status = environment.solver_environment_status(__file__)

        box = layout.box()
        box.label(text="Solver Environment")
        box.label(
            text="Status: Ready" if status["ready"] else "Status: Missing",
            icon="CHECKMARK" if status["ready"] else "ERROR",
        )
        box.operator(
            ContinuumFlow_OT_install_solver_environment.bl_idname,
            icon="CONSOLE",
        )


_ROOT_CLASSES = (
    ContinuumFlow_OT_install_solver_environment,
    ContinuumFlowPreferences,
)


def register():
    for cls in _ROOT_CLASSES:
        bpy.utils.register_class(cls)
    registry.register()


def unregister():
    registry.unregister()
    for cls in reversed(_ROOT_CLASSES):
        bpy.utils.unregister_class(cls)
