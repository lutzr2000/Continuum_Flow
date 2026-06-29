import bpy
from . import helper_functions
from . import sockets
from . import node_base
from ..Core import solver_status
from ..Core import main as bake_main
from bpy.props import IntProperty
from bpy.props import EnumProperty
from bpy.props import BoolProperty
from bpy.props import StringProperty

class CONTINUUM_FLOW_OT_output_bake_button(bpy.types.Operator):
    bl_idname = "continuum_flow.output_bake_button"
    bl_label = "Bake"
    bl_description = "Run the solver and write a new baked VDB sequence for this output"

    def execute(self, context):
        return bpy.ops.continuum_flow.bake()


class CONTINUUM_FLOW_OT_output_free_bake_button(bpy.types.Operator):
    bl_idname = "continuum_flow.output_free_bake_button"
    bl_label = "Free Bake"
    bl_description = "Delete the last baked VDB result folder for this output"

    def execute(self, context):
        return bpy.ops.continuum_flow.free_bake()


class ContinuumFlowOutputNode(node_base.ContinuumFlowBaseNode):
    """
    Node used to configure which simulation results should be written to disk.
    """

    bl_idname = "CONTINUUM_FLOW_OUTPUT_NODE"
    bl_label = "Output"
    bl_icon = "OUTPUT"
    bl_width_default = 260.0
    bl_width_min = 240.0
    bl_width_max = 420.0
    field_rows = (
        ("export_velocity", "velocity"),
        ("export_p", "pressure"),
        ("export_t", "temperature"),
        ("export_smoke", "density"),
        ("export_fuel", "fuel"),
        ("export_flame", "flame"),
    )

    fps: IntProperty(name="FPS", default=24, min=1, soft_min=1, description="Output frame rate", options=set())  # type: ignore
    writer_processes: IntProperty(name="Writers", default=4, min=1, max=16, soft_min=1, description="How many writer processes are launched, usually four give the best performance", options=set())  # type: ignore
    output_precision: EnumProperty(  # type: ignore
        name="Precision",
        items=(
            (
                "float16",
                "Half (float16)",
                "Write VDB grids with 16-bit floating point values",
            ),
            (
                "float32",
                "Full (float32)",
                "Write VDB grids with 32-bit floating point values",
            ),
        ),
        default="float16",
        options=set(),
    )
    export_velocity: BoolProperty(name="velocity", default=False, options=set())  # type: ignore
    export_p: BoolProperty(name="pressure", default=False, options=set())  # type: ignore
    export_t: BoolProperty(name="temperature", default=False, options=set())  # type: ignore
    export_smoke: BoolProperty(name="density", default=True, options=set())  # type: ignore
    export_fuel: BoolProperty(name="fuel", default=False, options=set())  # type: ignore
    export_flame: BoolProperty(name="flame", default=True, options=set())  # type: ignore
    output_path: StringProperty(name="Path", default="", subtype="DIR_PATH", options=set())  # type: ignore
    last_bake_directory: StringProperty(default="", options={"HIDDEN"})  # type: ignore

    def _ensure_input_socket(self):
        socket = self.inputs.get("Result")
        if socket is None:
            socket = self.inputs.new(
                sockets.ContinuumFlowResultSocket.bl_idname,
                "Result",
            )
        return socket

    def _sync_node(self):
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
        self._sync_node()
        self._sync_defaults_from_scene(context)

    def copy(self, node):
        self._sync_node()

    def update(self):
        self._sync_node()

    def _draw_field_row(self, layout, export_attr, label=None):
        row = layout.row(align=True)
        row.prop(self, export_attr, text=label)

    def _linked_simulation_node(self):
        socket = self.inputs.get("Result")
        if socket is None or not socket.is_linked:
            return None

        for link in socket.links:
            simulation_node = getattr(link, "from_node", None)
            if simulation_node is None:
                continue
            if (
                getattr(simulation_node, "bl_idname", "")
                == "CONTINUUM_FLOW_SIMULATION_NODE"
            ):
                return simulation_node
        return None

    def _bake_disable_reason(self):
        simulation_node = self._linked_simulation_node()
        if simulation_node is None:
            return "Bake disabled: output is not connected to a simulation"

        domain_socket = simulation_node.inputs.get("Domain")
        if domain_socket is None or not domain_socket.is_linked:
            return "Bake disabled: simulation has no domain node"

        physics_socket = simulation_node.inputs.get("Physics")
        if physics_socket is None or not physics_socket.is_linked:
            return "Bake disabled: simulation has no physics node"

        return None

    def draw_buttons(self, context, layout):
        self._set_layout_enabled(context, layout)

        layout.prop(self, "fps")
        layout.prop(self, "writer_processes")
        layout.prop(self, "output_precision")

        layout.separator()

        for export_attr, label in self.field_rows:
            self._draw_field_row(layout, export_attr, label)

        layout.separator()

        layout.prop(self, "output_path")

        layout.separator()

        disable_reason = self._bake_disable_reason()
        is_free_bake = bake_main.output_node_has_baked_data(self) and not solver_status.bake_running

        button_row = layout.row()
        button_row.enabled = disable_reason is None and not solver_status.bake_running
        if is_free_bake:
            button_row.operator("continuum_flow.output_free_bake_button", text="Free Bake", icon='TRASH')
        else:
            button_row.operator("continuum_flow.output_bake_button", text="Bake", icon='RENDER_STILL')

        if disable_reason is not None:
            layout.label(text=disable_reason, icon='INFO')
        elif solver_status.bake_running:
            layout.label(text="Bake is running", icon='INFO')
