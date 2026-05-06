"""UI node definition for BlenderCFD output settings and the bake/free-bake button."""

import importlib.util
import sys
from pathlib import Path

import bpy
import blendercfd_general_nodes as GeneralNodes
from bpy.props import BoolProperty, EnumProperty, IntProperty, StringProperty


def _load_sibling_module(module_name, file_name):
    module = sys.modules.get(module_name)
    if module is not None:
        return module

    if "__file__" in globals():
        module_path = Path(__file__).resolve().with_name(file_name)
    else:
        module_path = (Path.cwd() / "UI" / "Nodes" / file_name).resolve()

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module '{module_name}' from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


BakeRuntime = _load_sibling_module("blendercfd_bake_runtime", "bake_runtime.py")
BakeStorage = _load_sibling_module("blendercfd_bake_storage", "bake_storage.py")

BlenderCFDNodeTree = GeneralNodes.BlenderCFDNodeTree
BlenderCFDResultSocket = GeneralNodes.BlenderCFDResultSocket
tree_has_invalid_links = GeneralNodes._sockets_module.tree_has_invalid_links
is_bake_running = GeneralNodes.is_bake_running


class BlenderCFDOutputNode(bpy.types.Node):
    """Node used to configure which simulation results should be written to disk."""

    bl_idname = "BLENDERCFD_OUTPUT_NODE"
    bl_label = "Output"
    bl_icon = "OUTPUT"
    bl_width_default = 260.0
    bl_width_min = 240.0
    bl_width_max = 420.0
    field_rows = (
        ("export_velocity", "sparse_velocity", "velocity"),
        ("export_p", "sparse_p", "pressure"),
        ("export_t", "sparse_t", "temperature"),
        ("export_smoke", "sparse_smoke", "density"),
        ("export_fuel", "sparse_fuel", "fuel"),
        ("export_flame", "sparse_flame", "flame"),
    )

    fps: IntProperty(name="FPS", default=24, min=1, soft_min=1, description="Output frame rate", options=set())  # type: ignore
    writer_processes: IntProperty(name="Writers", default=4, min=1, max=16, soft_min=1, description="How many writer processes are launched, usually four give the best performance", options=set())  # type: ignore
    output_precision: EnumProperty(  # type: ignore
        name="Precision",
        items=(
            ("float16", "Half (float16)", "Write VDB grids with 16-bit floating point values"),
            ("float32", "Full (float32)", "Write VDB grids with 32-bit floating point values"),
        ),
        default="float16",
    )
    export_velocity: BoolProperty(name="velocity", default=False)  # type: ignore
    sparse_velocity: BoolProperty(name="sparse", default=True)  # type: ignore
    export_p: BoolProperty(name="pressure", default=False)  # type: ignore
    sparse_p: BoolProperty(name="sparse", default=True)  # type: ignore
    export_t: BoolProperty(name="temperature", default=False)  # type: ignore
    sparse_t: BoolProperty(name="sparse", default=True)  # type: ignore
    export_smoke: BoolProperty(name="density", default=True)  # type: ignore
    sparse_smoke: BoolProperty(name="sparse", default=True)  # type: ignore
    export_fuel: BoolProperty(name="fuel", default=False)  # type: ignore
    sparse_fuel: BoolProperty(name="sparse", default=True)  # type: ignore
    export_flame: BoolProperty(name="flame", default=True)  # type: ignore
    sparse_flame: BoolProperty(name="sparse", default=True)  # type: ignore
    output_path: StringProperty(name="Path", default="", subtype="DIR_PATH")  # type: ignore

    @classmethod
    def poll(cls, ntree):
        return ntree.bl_idname == BlenderCFDNodeTree.bl_idname

    def _ensure_input_socket(self):
        socket = self.inputs.get("Result")
        if socket is None:
            socket = self.inputs.new(BlenderCFDResultSocket.bl_idname, "Result")
        return socket

    def _sync_input_socket(self):
        self._ensure_input_socket()

    def _sync_defaults_from_scene(self, context):
        scene = getattr(context, "scene", None) or getattr(bpy.context, "scene", None)
        if scene is None:
            return
        render = getattr(scene, "render", None)
        if render is None:
            return
        fps = getattr(render, "fps", None)
        if fps is not None and self.fps == 24:
            self.fps = fps

    def init(self, context):
        self._sync_input_socket()
        self._sync_defaults_from_scene(context)

    def copy(self, node):
        self._sync_input_socket()

    def update(self):
        self._sync_input_socket()

    def _draw_field_row(self, layout, export_attr, sparse_attr, label=None):
        row = layout.row(align=True)
        row.prop(self, export_attr, text=label)
        sparse_row = row.row(align=True)
        sparse_row.enabled = bool(getattr(self, export_attr))
        sparse_row.prop(self, sparse_attr, text="sparse")

    def draw_buttons(self, context, layout):
        layout.enabled = not is_bake_running(context)
        layout.prop(self, "fps")
        layout.prop(self, "writer_processes")
        layout.prop(self, "output_precision")
        fields_box = layout.box()
        fields_box.label(text="Fields")
        fields_col = fields_box.column(align=True)
        for export_attr, sparse_attr, label in self.field_rows:
            self._draw_field_row(fields_col, export_attr, sparse_attr, label=label)
        layout.prop(self, "output_path")
        layout.separator()
        resolved_output_path = bpy.path.abspath(self.output_path) if self.output_path else ""
        button_is_free_bake = bool(resolved_output_path) and BakeStorage._output_directory_has_vdbs(resolved_output_path)
        has_invalid_links = tree_has_invalid_links(getattr(self, "id_data", None))
        if has_invalid_links:
            layout.label(text="Bake disabled: invalid socket connection", icon="ERROR")
        button_row = layout.row()
        button_row.enabled = (not is_bake_running(context)) and not has_invalid_links
        operator = button_row.operator(
            BakeRuntime.BlenderCFD_OT_bake.bl_idname,
            text="Free Bake" if button_is_free_bake else "Bake",
            icon="TRASH" if button_is_free_bake else "RENDER_STILL",
        )
        operator.output_path_hint = self.output_path


classes = (
    BlenderCFDOutputNode,
)
