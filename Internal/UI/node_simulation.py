import bpy
from . import sockets
from . import node_base
from ..Core.solver import solver_status
from bpy.props import FloatProperty
from bpy.props import IntProperty
from bpy.props import EnumProperty
from bpy.props import BoolProperty
from bpy.props import BoolVectorProperty


def _default_device_selection():
    return (True,) + (False,) * (solver_status.MAX_SELECTABLE_DEVICES - 1)


def _update_simulation_start_frame(self, _context):
    self._clamp_frame_range()


def _update_simulation_end_frame(self, _context):
    self._clamp_frame_range()


class ContinuumFlowSimulationNode(node_base.ContinuumFlowBaseNode):
    """
    Node used to collect all simulation-wide settings and input dependencies.
    """

    cpu_available = False
    gpu_available = False

    bl_idname = "CONTINUUM_FLOW_SIMULATION_NODE"
    bl_label = "Simulation"
    bl_icon = "TIME"
    bl_width_default = 260.0
    bl_width_min = 240.0
    bl_width_max = 420.0
    property_groups = (
        ("Time", ("start_frame", "end_frame", "cfl")),
        (
            "Solver",
            (
                "iterations",
                "advection_substeps",
                "simulate_sparsely",
                "adaptive_domain_threshold",
            ),
        ),
    )

    solver_backend: bpy.props.EnumProperty(
        name="Solver",
        items=(
            ("GPU", "GPU", "Use the GPU solver"),
            ("CPU", "CPU", "Use the CPU solver"),
        ),
        default="CPU",
        options=set(),
    )  # type: ignore
    start_frame: IntProperty(name="Start Frame", default=1, min=0, description="Starting frame of the simulation", options=set(), update=_update_simulation_start_frame)  # type: ignore
    end_frame: IntProperty(name="End Frame", default=250, min=2, description="End frame of the simulation", options=set(), update=_update_simulation_end_frame)  # type: ignore
    cfl: FloatProperty(name="CFL", default=10.0, min=0.1, soft_min=0.1, soft_max=30.0, precision=3, description="Maximum CFL number used for adaptive timesteps", options=set())  # type: ignore
    iterations: IntProperty(name="Iterations", default=1, min=1, max=10, soft_min=1, soft_max=10, description="Number of solver itterations", options=set())  # type: ignore
    advection_substeps: IntProperty(name="Advection Substeps", default=1, min=1, soft_min=1, soft_max=10, description="Number of integration substeps used for advection tracing", options=set())  # type: ignore
    simulate_sparsely: BoolProperty(name="Adaptive Domain", default=True, description="Domain adapts to the smoke and flame field to save computational cost", options=set())  # type: ignore
    adaptive_domain_threshold: FloatProperty(name="Threshold", default=0.01, min=0.0, precision=6, description="Cells containing more smoke, fuel or flame than this are considered active", options=set())  # type: ignore
    cpu_devices: BoolVectorProperty(name="CPU Devices", size=solver_status.MAX_SELECTABLE_DEVICES, default=_default_device_selection(), options=set())  # type: ignore
    gpu_devices: BoolVectorProperty(name="GPU Devices", size=solver_status.MAX_SELECTABLE_DEVICES, default=_default_device_selection(), options=set())  # type: ignore

    def selected_opencl_devices(self, backend=None):
        backend = str(backend or self.solver_backend).upper()
        devices = solver_status.opencl_devices.get(backend, ())
        selection = self.gpu_devices if backend == "GPU" else self.cpu_devices
        return [
            dict(device) for index, device in enumerate(devices) if selection[index]
        ]

    def has_selected_opencl_device(self):
        return bool(self.selected_opencl_devices())

    def _ensure_input_socket(self, name, *, multi_input=False):
        socket_type = (
            sockets.ContinuumFlowReferenceFrameSocket.bl_idname
            if name == "Reference Frame"
            else (
                sockets.ContinuumFlowForceSocket.bl_idname
                if name == "Forces"
                else sockets.ContinuumFlowLinkSocket.bl_idname
            )
        )
        return self._ensure_socket(
            self.inputs, socket_type, name, multi_input=multi_input
        )

    def _sync_node(self):
        reference_frame = self._ensure_input_socket("Reference Frame")
        reference_frame_index = list(self.inputs).index(reference_frame)
        if reference_frame_index != 0:
            self.inputs.move(reference_frame_index, 0)
        self._ensure_input_socket("Domain")
        self._ensure_input_socket("Physics")
        self._ensure_input_socket("Obstacles")
        self._ensure_input_socket("Source", multi_input=True)
        self._ensure_input_socket("Forces", multi_input=True)
        self._ensure_named_output(sockets.ContinuumFlowResultSocket.bl_idname, "Result")

    def init(self, context):
        scene = getattr(context, "scene", None) or getattr(bpy.context, "scene", None)
        if scene is not None:
            self.start_frame = int(getattr(scene, "frame_start", self.start_frame))
            self.end_frame = int(getattr(scene, "frame_end", self.end_frame))
        self._clamp_frame_range()
        self._sync_node()

    def _clamp_frame_range(self):
        minimum_end_frame = max(1, int(self.start_frame) + 1)
        if int(self.end_frame) < minimum_end_frame:
            self.end_frame = minimum_end_frame

        maximum_start_frame = max(0, int(self.end_frame) - 1)
        if int(self.start_frame) > maximum_start_frame:
            self.start_frame = maximum_start_frame

    def draw_buttons(self, context, layout):
        self._set_layout_enabled(context, layout)

        solver_row = layout.row(align=True)

        solver_row.prop_enum(self, "solver_backend", "CPU")

        gpu_row = solver_row.row(align=True)
        gpu_row.prop_enum(self, "solver_backend", "GPU")

        backend = str(self.solver_backend).upper()
        devices = solver_status.opencl_devices.get(backend, ())
        selection_property = "gpu_devices" if backend == "GPU" else "cpu_devices"
        device_box = layout.box()
        device_box.label(text=f"{backend} Devices")
        if not devices:
            device_box.label(text="No device found", icon="ERROR")
        else:
            for index, device in enumerate(devices):
                device_box.prop(
                    self,
                    selection_property,
                    index=index,
                    text=device["device_name"],
                )

        for title, property_names in self.property_groups:
            self._draw_group(layout, title, property_names)
