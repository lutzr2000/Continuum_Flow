import bpy
import sockets

class ContinuumFlowOutputNode(bpy.types.Node):
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
    def _ensure_input_socket(self):
        socket = self.inputs.get("Result")
        if socket is None:
            socket = self.inputs.new(sockets.ContinuumFlowResultSocket.bl_idname, "Result")
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


