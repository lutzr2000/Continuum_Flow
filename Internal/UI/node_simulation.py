import bpy
from . import helper_functions
from . import sockets
from . import node_base
from ..Core import solver_status
from bpy.props import FloatProperty
from bpy.props import IntProperty
from bpy.props import EnumProperty
from bpy.props import BoolProperty


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
    start_frame: IntProperty(name="Start Frame", default=1, min=0, description="Starting frame of the simulation", options=set())  # type: ignore
    end_frame: IntProperty(name="End Frame", default=250, min=2, description="End frame of the simulation", options=set())  # type: ignore
    cfl: FloatProperty(name="CFL", default=10.0, min=0.0, soft_min=0.0, soft_max=10.0, precision=3, description="Maximum CFL number used for adaptive timesteps", options=set())  # type: ignore
    iterations: IntProperty(name="Iterations", default=2, min=1, max=500, soft_min=1, soft_max=500, description="Number of solver itterations", options=set())  # type: ignore
    simulate_sparsely: BoolProperty(name="Adaptive Domain", default=True, description="Domain adapts to the smoke and flame field to save computational cost", options=set())  # type: ignore
    adaptive_domain_threshold: FloatProperty(name="Threshold", default=0.001, min=0.0, precision=6, description="Cells containing more smoke, fuel or flame than this are considered active", options=set())  # type: ignore

    def _ensure_input_socket(self, name, *, multi_input=False):
        socket_type = (
            sockets.ContinuumFlowForceSocket.bl_idname
            if name == "Forces"
            else sockets.ContinuumFlowLinkSocket.bl_idname
        )
        return helper_functions.ensure_socket(self.inputs, socket_type, name, multi_input=multi_input)

    def _sync_node(self):
        self._ensure_input_socket("Domain")
        self._ensure_input_socket("Physics")
        self._ensure_input_socket("Obstacles")
        self._ensure_input_socket("Source", multi_input=True)
        self._ensure_input_socket("Forces", multi_input=True)
        helper_functions.ensure_named_output(self, sockets.ContinuumFlowResultSocket.bl_idname, "Result")

    def init(self, context):
        scene = getattr(context, "scene", None) or getattr(bpy.context, "scene", None)
        if scene is not None:
            self.start_frame = int(getattr(scene, "frame_start", self.start_frame))
            self.end_frame = int(getattr(scene, "frame_end", self.end_frame))
        self._sync_node()
    
    def set_solver_status(self, cpu_available, gpu_available):
        self.cpu_available = cpu_available
        self.gpu_available = gpu_available 

    def draw_buttons(self, context, layout):
        self._set_layout_enabled(context, layout)

        solver_row = layout.row(align=True)

        solver_row.prop_enum(self, "solver_backend", "CPU")

        gpu_row = solver_row.row(align=True)
        gpu_row.enabled = solver_status.gpu_available
        gpu_row.prop_enum(self, "solver_backend", "GPU")

        if self.solver_backend == "GPU" and not solver_status.gpu_available:
            self.solver_backend = "CPU"

        for title, property_names in self.property_groups:
            self._draw_group(layout, title, property_names)