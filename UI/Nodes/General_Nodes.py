import bpy
import importlib
import importlib.util
import sys
from pathlib import Path
from bpy.props import FloatProperty, FloatVectorProperty, IntProperty, PointerProperty


def _ui_directory():
    """Return the root UI directory."""
    if "__file__" in globals():
        return Path(__file__).resolve().parents[1]
    return (Path.cwd() / "UI").resolve()


def _load_ui_module(module_name, file_names, package_names=()):
    """Load a sibling UI module across package, absolute, and file-based contexts."""
    if isinstance(file_names, str):
        file_names = (file_names,)
    if isinstance(package_names, str):
        package_names = (package_names,)

    package_context = globals().get("__package__")
    for package_name in package_names:
        if package_name.startswith(".") and not package_context:
            continue
        try:
            if package_name.startswith("."):
                return importlib.import_module(package_name, package=package_context)
            return importlib.import_module(package_name)
        except (ImportError, TypeError):
            continue

    for file_name in file_names:
        module_path = (_ui_directory() / file_name).resolve()
        if not module_path.exists():
            continue

        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            continue

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    readable_names = ", ".join(file_names)
    raise ImportError(f"Could not load any of: {readable_names}")


_node_tree_module = _load_ui_module(
    "blendercfd_nodetree",
    ("Node_Tree.py", "NodeTree.py"),
    (".Node_Tree", "UI.Node_Tree", "Node_Tree", "UI.NodeTree", "NodeTree"),
)
BlenderCFDNodeTree = _node_tree_module.BlenderCFDNodeTree

_sockets_module = _load_ui_module(
    "blendercfd_sockets",
    "Sockets.py",
    (".Sockets", "UI.Sockets", "Sockets"),
)

BlenderCFDViewerModule = _load_ui_module(
    "blendercfd_viewer",
    "Viewer.py",
    (".Viewer", "UI.Viewer", "Viewer"),
)

BlenderCFDForceSocket = _sockets_module.BlenderCFDForceSocket
BlenderCFDIntSocket = _sockets_module.BlenderCFDIntSocket
BlenderCFDLinkSocket = _sockets_module.BlenderCFDLinkSocket
BlenderCFDReferenceFrameSocket = _sockets_module.BlenderCFDReferenceFrameSocket
BlenderCFDResultSocket = _sockets_module.BlenderCFDResultSocket
BLENDERCFD_BAKE_RUNNING_KEY = "blendercfd_bake_running"


def _window_manager_from_context(context=None):
    """Return the current Blender window manager if available."""
    return getattr(context, "window_manager", None) or getattr(bpy.context, "window_manager", None)


def set_bake_running(is_running, context=None):
    """Store whether BlenderCFD currently has a bake running."""
    window_manager = _window_manager_from_context(context)
    if window_manager is None:
        return

    if is_running:
        window_manager[BLENDERCFD_BAKE_RUNNING_KEY] = True
    else:
        window_manager.pop(BLENDERCFD_BAKE_RUNNING_KEY, None)


def is_bake_running(context=None):
    """Return whether BlenderCFD currently has a bake running."""
    window_manager = _window_manager_from_context(context)
    if window_manager is None:
        return False
    return bool(window_manager.get(BLENDERCFD_BAKE_RUNNING_KEY, False))


class BlenderCFDDomainNode(bpy.types.Node):
    """Node used to define the CFD domain resolution and boundary conditions."""

    bl_idname = "BLENDERCFD_DOMAIN_NODE"
    bl_label = "Domain"
    bl_icon = "MESH_GRID"
    bl_width_default = 220.0
    bl_width_min = 200.0
    bl_width_max = 360.0

    boundary_condition_items = (
        ("WALL", "Wall", "No-slip wall boundary"),
        ("SLIP_WALL", "Slip Wall", "Slip wall boundary"),
        ("OUTFLOW", "Outflow", "Outflow boundary"),
        ("INFLOW", "Inflow", "Inflow boundary with prescribed velocity"),
    )
    resolution: FloatProperty(name="Resolution", default=0.1, min=0.000001, soft_min=0.001, unit="LENGTH", description="Grid resolution")  # type: ignore
    nx: IntProperty(name="NX", default=128, min=32, max=8192, soft_min=32, soft_max=8192, description="Grid cells in x")  # type: ignore
    ny: IntProperty(name="NY", default=128, min=32, max=8192, soft_min=32, soft_max=8192, description="Grid cells in y")  # type: ignore
    nz: IntProperty(name="NZ", default=128, min=32, max=8192, soft_min=32, soft_max=8192, description="Grid cells in z") # type: ignore
    x_low_bc: bpy.props.EnumProperty(name="X Low", items=boundary_condition_items, default="OUTFLOW")  # type: ignore
    x_high_bc: bpy.props.EnumProperty(name="X High", items=boundary_condition_items, default="OUTFLOW")  # type: ignore
    y_low_bc: bpy.props.EnumProperty(name="Y Low", items=boundary_condition_items, default="OUTFLOW")  # type: ignore
    y_high_bc: bpy.props.EnumProperty(name="Y High", items=boundary_condition_items, default="OUTFLOW")  # type: ignore
    z_low_bc: bpy.props.EnumProperty(name="Z Low", items=boundary_condition_items, default="OUTFLOW")  # type: ignore
    z_high_bc: bpy.props.EnumProperty(name="Z High", items=boundary_condition_items, default="OUTFLOW")  # type: ignore
    x_low_velocity: FloatVectorProperty(name="Velocity", size=3, subtype="XYZ", default=(0.0, 0.0, 0.0), unit="VELOCITY")  # type: ignore
    x_high_velocity: FloatVectorProperty(name="Velocity", size=3, subtype="XYZ", default=(0.0, 0.0, 0.0), unit="VELOCITY")  # type: ignore
    y_low_velocity: FloatVectorProperty(name="Velocity", size=3, subtype="XYZ", default=(0.0, 0.0, 0.0), unit="VELOCITY")  # type: ignore
    y_high_velocity: FloatVectorProperty(name="Velocity", size=3, subtype="XYZ", default=(0.0, 0.0, 0.0), unit="VELOCITY")  # type: ignore
    z_low_velocity: FloatVectorProperty(name="Velocity", size=3, subtype="XYZ", default=(0.0, 0.0, 0.0), unit="VELOCITY")  # type: ignore
    z_high_velocity: FloatVectorProperty(name="Velocity", size=3, subtype="XYZ", default=(0.0, 0.0, 0.0), unit="VELOCITY")  # type: ignore

    @classmethod
    def poll(cls, ntree):
        return ntree.bl_idname == BlenderCFDNodeTree.bl_idname

    def _ensure_output_socket(self, name):
        socket = self.outputs.get(name)
        if socket is None:
            socket = self.outputs.new(BlenderCFDIntSocket.bl_idname, name)
        return socket

    def _sync_output_socket(self):
        self._ensure_output_socket("Domain")

    def init(self, context):
        self._sync_output_socket()

    def copy(self, node):
        self._sync_output_socket()

    def update(self):
        self._sync_output_socket()

    def _draw_boundary_controls(self, layout, label, condition_attr, velocity_attr):
        box = layout.box()
        row = box.row(align=True)
        row.label(text=label)
        row.prop(self, condition_attr, text="")
        if getattr(self, condition_attr) == "INFLOW":
            box.prop(self, velocity_attr, text="Velocity")

    def draw_buttons(self, context, layout):
        layout.enabled = not is_bake_running(context)
        col = layout.column(align=True)
        col.prop(self, "resolution")
        col.prop(self, "nx")
        col.prop(self, "ny")
        col.prop(self, "nz")
        layout.separator()
        layout.label(text="Boundary Conditions")
        self._draw_boundary_controls(layout, "X Low", "x_low_bc", "x_low_velocity")
        self._draw_boundary_controls(layout, "X High", "x_high_bc", "x_high_velocity")
        self._draw_boundary_controls(layout, "Y Low", "y_low_bc", "y_low_velocity")
        self._draw_boundary_controls(layout, "Y High", "y_high_bc", "y_high_velocity")
        self._draw_boundary_controls(layout, "Z Low", "z_low_bc", "z_low_velocity")
        self._draw_boundary_controls(layout, "Z High", "z_high_bc", "z_high_velocity")


class BlenderCFDPhysicsNode(bpy.types.Node):
    """Node used to store the physical coefficients of the CFD simulation."""

    bl_idname = "BLENDERCFD_PHYSICS_NODE"
    bl_label = "Physics"
    bl_icon = "MOD_PHYSICS"
    bl_width_default = 220.0
    bl_width_min = 200.0
    bl_width_max = 360.0

    fluid_density: FloatProperty(name="Fluid Density", default=1.225, min=0.1, max=10, precision=4, description="Density of the fluid, default is air")  # type: ignore
    fluid_viscosity: FloatProperty(name="Fluid Viscosity", default=1.81e-5, min=0.0, max=0.1, precision=6, description="Viscosity of the fluid, default is air")  # type: ignore
    temperature_dissipation: FloatProperty(name="Temperature Dissipation", default=0.1, min=0.0, max=10, description="Rate of temperature dissipation, lower means slower dissipation")  # type: ignore
    reference_temperature: FloatProperty(name="Reference Temperature", default=300.0, min=0.0, max=2000, unit="TEMPERATURE", description="Air cooler than this goes down, warmer than this goes up")  # type: ignore
    buoyancy: FloatProperty(name="Buoyancy", default=0.0033, min=0.0, max=0.1, precision=4, description="Higher values result in quicker rising of air")  # type: ignore
    expansion_rate: FloatProperty(name="Expansion Rate", default=0.003, min=0.0, max=0.1, precision=4, description="Higher values result in more expansion of warm air")  # type: ignore
    smoke_dissipation: FloatProperty(name="Smoke Dissipation", default=0.1, min=0.0, max=100.0, precision=4, description="Rate of smoke dissipation, lower means slower dissipation")  # type: ignore
    smoke_production_rate: FloatProperty(name="Smoke Production Rate", default=1.0, min=0.0, max=100.0, precision=4, description="Rate of smoke production due to burning, higher means more production")   # type: ignore
    fuel_dissipation: FloatProperty(name="Fuel Dissipation", default=0.001, min=0.0, max=100.0, precision=4, description="Rate of fuel dissipation, lower means slower dissipation")   # type: ignore
    fuel_burn_rate: FloatProperty(name="Fuel Burn Rate", default=0.1, min=0.0, max=100.0, precision=4, description="How quickly fuel is burned, lower means slower burning")   # type: ignore
    fuel_ignition_temperature: FloatProperty(name="Fuel Ignition Temperature", default=500.0, min=0.0, max=2000.0, unit="TEMPERATURE", description="If the air is warmer than this and contains fuel, the fuel will ignite")  # type: ignore
    vorticity: FloatProperty(name="Vorticity", default=1.0, min=0.0, max=5.0, precision=4, description="Amount of additional vorticity in the flow, zero is physically accurate, higher values produce more swirl")  # type: ignore

    @classmethod
    def poll(cls, ntree):
        return ntree.bl_idname == BlenderCFDNodeTree.bl_idname

    def _ensure_output_socket(self):
        socket = self.outputs.get("Physics")
        if socket is None:
            socket = self.outputs.new(BlenderCFDIntSocket.bl_idname, "Physics")
        return socket

    def _sync_output_socket(self):
        self._ensure_output_socket()

    def init(self, context):
        self._sync_output_socket()

    def copy(self, node):
        self._sync_output_socket()

    def update(self):
        self._sync_output_socket()

    def _draw_group(self, layout, title, property_names):
        box = layout.box()
        box.label(text=title)
        col = box.column(align=True)
        for property_name in property_names:
            col.prop(self, property_name)

    def draw_buttons(self, context, layout):
        layout.enabled = not is_bake_running(context)
        self._draw_group(layout, "Fluid", ("fluid_density", "fluid_viscosity"))
        self._draw_group(layout, "Temperature", ("temperature_dissipation", "reference_temperature", "buoyancy", "expansion_rate"))
        self._draw_group(layout, "Smoke", ("smoke_dissipation", "smoke_production_rate"))
        self._draw_group(layout, "Fuel", ("fuel_dissipation", "fuel_burn_rate", "fuel_ignition_temperature"))
        self._draw_group(layout, "Extras", ("vorticity",))


class BlenderCFDReferenceFrameNode(bpy.types.Node):
    """Node used to define an object-based reference frame for the simulation."""

    bl_idname = "BLENDERCFD_REFERENCE_FRAME_NODE"
    bl_label = "Reference Frame"
    bl_icon = "EMPTY_AXIS"
    bl_width_default = 220.0
    bl_width_min = 200.0
    bl_width_max = 360.0

    source_object: PointerProperty(name="Object", type=bpy.types.Object)  # type: ignore

    @classmethod
    def poll(cls, ntree):
        return ntree.bl_idname == BlenderCFDNodeTree.bl_idname

    def _ensure_output_socket(self):
        socket = self.outputs.get("Reference Frame")
        if socket is None:
            socket = self.outputs.new(BlenderCFDReferenceFrameSocket.bl_idname, "Reference Frame")
        return socket

    def _sync_sockets(self):
        self._ensure_output_socket()

    def init(self, context):
        self._sync_sockets()

    def copy(self, node):
        self._sync_sockets()

    def update(self):
        self._sync_sockets()

    def draw_buttons(self, context, layout):
        layout.enabled = not is_bake_running(context)
        layout.prop(self, "source_object", text="Object")


class BlenderCFDSimulationNode(bpy.types.Node):
    """Node used to collect all simulation-wide settings and input dependencies."""

    bl_idname = "BLENDERCFD_SIMULATION_NODE"
    bl_label = "Simulation"
    bl_icon = "TIME"
    bl_width_default = 260.0
    bl_width_min = 240.0
    bl_width_max = 420.0

    start_frame: IntProperty(name="Start Frame", default=1, min=0, description="Starting frame of the simulation")  # type: ignore
    end_frame: IntProperty(name="End Frame", default=250, min=2, description="End frame of the simulation")  # type: ignore
    cfl: FloatProperty(name="CFL", default=0.9, min=0.000001, max=1.0, soft_max=1, description="CFL condition for the solver")  # type: ignore
    iterations: IntProperty(name="Iterations", default=4, min=1, max=500, soft_min=1, soft_max=500, description="Number of pressure itterations")  # type: ignore

    @classmethod
    def poll(cls, ntree):
        return ntree.bl_idname == BlenderCFDNodeTree.bl_idname

    def _ensure_input_socket(self, name, *, multi_input=False):
        socket = self.inputs.get(name)
        if socket is None:
            socket = self.inputs.new(
                BlenderCFDReferenceFrameSocket.bl_idname if name == "Reference Frame" else BlenderCFDForceSocket.bl_idname if name == "Forces" else BlenderCFDLinkSocket.bl_idname,
                name,
                use_multi_input=multi_input,
            )
        if multi_input and hasattr(socket, "link_limit"):
            socket.link_limit = 0
        return socket

    def _ensure_output_socket(self):
        socket = self.outputs.get("Result")
        if socket is None:
            socket = self.outputs.new(BlenderCFDResultSocket.bl_idname, "Result")
        return socket

    def _sync_sockets(self):
        self._ensure_input_socket("Reference Frame")
        self._ensure_input_socket("Domain")
        self._ensure_input_socket("Physics")
        self._ensure_input_socket("Obstacles")
        self._ensure_input_socket("Source", multi_input=True)
        self._ensure_input_socket("Forces", multi_input=True)
        self._ensure_output_socket()

    def init(self, context):
        scene = getattr(context, "scene", None) or getattr(bpy.context, "scene", None)
        if scene is not None:
            self.start_frame = int(getattr(scene, "frame_start", self.start_frame))
            self.end_frame = int(getattr(scene, "frame_end", self.end_frame))
        self._sync_sockets()

    def copy(self, node):
        self._sync_sockets()

    def update(self):
        self._sync_sockets()

    def _draw_group(self, layout, title, property_names):
        box = layout.box()
        box.label(text=title)
        col = box.column(align=True)
        for property_name in property_names:
            col.prop(self, property_name)

    def draw_buttons(self, context, layout):
        layout.enabled = not is_bake_running(context)
        self._draw_group(layout, "Time", ("start_frame", "end_frame", "cfl"))
        self._draw_group(layout, "Solver", ("iterations",))


class BlenderCFDViewerNode(bpy.types.Node):
    """Node used as a lightweight endpoint for inspecting simulation results."""

    bl_idname = "BLENDERCFD_VIEWER_NODE"
    bl_label = "Viewer"
    bl_icon = "HIDE_OFF"
    bl_width_default = 180.0
    bl_width_min = 160.0
    bl_width_max = 260.0

    @classmethod
    def poll(cls, ntree):
        return ntree.bl_idname == BlenderCFDNodeTree.bl_idname

    def _ensure_result_input_socket(self):
        socket = self.inputs.get("Result")
        if socket is None:
            socket = self.inputs.new(BlenderCFDResultSocket.bl_idname, "Result")
        return socket

    def _sync_input_socket(self):
        self._ensure_result_input_socket()

    def init(self, context):
        self._sync_input_socket()

    def copy(self, node):
        self._sync_input_socket()

    def update(self):
        self._sync_input_socket()

    def free(self):
        BlenderCFDViewerModule.disable_domain_preview()

    def draw_buttons(self, context, layout):
        layout.enabled = not is_bake_running(context)
        col = layout.column(align=True)
        col.operator("blendercfd.viewer_show_domain", text="Show Domain", icon="HIDE_OFF")
        col.operator("blendercfd.viewer_hide_domain", text="Hide Domain", icon="HIDE_ON")


classes = (
    BlenderCFDDomainNode,
    BlenderCFDPhysicsNode,
    BlenderCFDReferenceFrameNode,
    BlenderCFDSimulationNode,
    BlenderCFDViewerNode,
)
