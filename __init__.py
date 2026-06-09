bl_info = {
    "name": "Continuum Flow",
    "author": "Lutz Rohlfing",
    "version": (0, 1, 0),
    "blender": (5, 0, 0),
    "location": "Node Editor > Add > Continuum Flow",
    "description": "Node-based CFD baking for Blender with an external solver environment",
    "category": "Node",
}

import importlib.util
from pathlib import Path

import bpy


_PACKAGE_ROOT = Path(__file__).resolve().parent


def _load_module(module_name, relative_path):
    module_path = _PACKAGE_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module '{module_name}' from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


Environment = _load_module(
    "continuum_flow_environment",
    Path("UI") / "Core" / "environment.py",
)
CoreRegister = _load_module(
    "continuum_flow_core_register",
    Path("UI") / "Core" / "register.py",
)


class ContinuumFlow_OT_install_solver_environment(bpy.types.Operator):
    bl_idname = "continuum_flow.install_solver_environment"
    bl_label = "Install Solver Environment"
    bl_description = "Create the Continuum Flow solver environment with uv"

    def execute(self, context):
        try:
            result = Environment.install_solver_environment(__file__)
        except (FileNotFoundError, RuntimeError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        self.report(
            {"INFO"},
            f"Solver environment is ready: {result['python_executable']}",
        )
        return {"FINISHED"}


class ContinuumFlowPreferences(bpy.types.AddonPreferences):
    bl_idname = __name__

    def draw(self, context):
        layout = self.layout
        status = Environment.solver_environment_status(__file__)

        box = layout.box()
        box.label(text="Solver Environment")
        box.label(text=f"Project Root: {status['project_root']}")
        box.label(text=f"Environment Path: {status['environment_path']}")
        box.label(text=f"Python Path: {status['python_path']}")
        box.label(
            text="Status: Ready" if status["ready"] else "Status: Missing",
            icon="CHECKMARK" if status["ready"] else "ERROR",
        )
        box.operator(
            ContinuumFlow_OT_install_solver_environment.bl_idname,
            icon="CONSOLE",
        )

        info_box = layout.box()
        info_box.label(text="Install Flow")
        info_box.label(
            text="1. Install the add-on from the Git ZIP in Blender Preferences."
        )
        info_box.label(
            text="2. Click 'Install Solver Environment' once to create .venv with uv."
        )
        info_box.label(
            text="3. Start the bake; Blender keeps VDB writing in its own Python."
        )


_ROOT_CLASSES = (
    ContinuumFlow_OT_install_solver_environment,
    ContinuumFlowPreferences,
)


def register():
    for cls in _ROOT_CLASSES:
        bpy.utils.register_class(cls)
    CoreRegister.register()


def unregister():
    CoreRegister.unregister()
    for cls in reversed(_ROOT_CLASSES):
        bpy.utils.unregister_class(cls)
