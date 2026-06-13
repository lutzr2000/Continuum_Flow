import bpy
import importlib
import importlib.util
import subprocess
import sys
from pathlib import Path
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    PointerProperty,
)
from bpy.app.handlers import persistent

MODULE_NAME_ALIASES = {}
CONTINUUM_FLOW_BAKE_RUNNING_KEY = "continuum_flow_bake_running"


def _ui_directory():
    """
    Return the root UI directory.
    """
    if "__file__" in globals():
        return Path(__file__).resolve().parents[1]
    return (Path.cwd() / "UI").resolve()


def _project_root_directory():
    """
    Return the Continuum Flow project root directory.
    """
    return _ui_directory().parent


def _solver_python_executable():
    """
    Return the dedicated Continuum Flow solver Python executable.
    """
    return Path(
        ContinuumFlowEnvironmentModule.solver_python_executable(__file__)
    ).resolve()


def _solver_environment_ready():
    """
    Return whether the external solver environment exists.
    """
    return _solver_python_executable().exists()


def _register_module_aliases(module, module_name):
    """
    Expose a loaded module under its canonical import name.
    """
    sys.modules[module_name] = module
    for alias in MODULE_NAME_ALIASES.get(module_name, ()):
        sys.modules[alias] = module


def _load_ui_module(module_name, file_names, package_names=()):
    """
    Load a sibling UI module across package, absolute, and file-based contexts.
    """
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
        _register_module_aliases(module, module_name)
        spec.loader.exec_module(module)
        return module

    readable_names = ", ".join(file_names)
    raise ImportError(f"Could not load any of: {readable_names}")


_node_tree_module = _load_ui_module(
    "continuum_flow_nodetree",
    "Core/node_tree.py",
    (".Core.node_tree", "UI.Core.node_tree", "Core.node_tree"),
)
ContinuumFlowNodeTree = _node_tree_module.ContinuumFlowNodeTree

_sockets_module = _load_ui_module(
    "continuum_flow_sockets",
    "Core/sockets.py",
    (".Core.sockets", "UI.Core.sockets", "Core.sockets"),
)

ContinuumFlowViewerModule = _load_ui_module(
    "continuum_flow_viewer",
    "Core/viewer.py",
    (".Core.viewer", "UI.Core.viewer", "Core.viewer"),
)
ContinuumFlowEnvironmentModule = _load_ui_module(
    "continuum_flow_environment",
    "Core/environment.py",
    (".Core.environment", "UI.Core.environment", "Core.environment"),
)

ContinuumFlowForceSocket = _sockets_module.ContinuumFlowForceSocket
ContinuumFlowIntSocket = _sockets_module.ContinuumFlowIntSocket
ContinuumFlowLinkSocket = _sockets_module.ContinuumFlowLinkSocket
ContinuumFlowResultSocket = _sockets_module.ContinuumFlowResultSocket
_SYNC_DEBUGGED_ACTIONS = set()
_GPU_SOLVER_AVAILABLE_CACHE = None


def _gpu_solver_available():
    """
    Return whether a usable CUDA backend is available in the solver environment.
    """
    global _GPU_SOLVER_AVAILABLE_CACHE
    if _GPU_SOLVER_AVAILABLE_CACHE is not None:
        return _GPU_SOLVER_AVAILABLE_CACHE

    try:
        python_executable = _solver_python_executable()
        if not python_executable.exists():
            _GPU_SOLVER_AVAILABLE_CACHE = False
            return _GPU_SOLVER_AVAILABLE_CACHE

        result = subprocess.run(
            [
                str(python_executable),
                "-c",
                "from numba import cuda; print(int(cuda.is_available()))",
            ],
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        _GPU_SOLVER_AVAILABLE_CACHE = (
            result.returncode == 0 and result.stdout.strip() == "1"
        )
    except Exception:
        _GPU_SOLVER_AVAILABLE_CACHE = False
    return _GPU_SOLVER_AVAILABLE_CACHE


def _window_manager_from_context(context=None):
    """
    Return the current Blender window manager if available.
    """
    return getattr(context, "window_manager", None) or getattr(
        bpy.context, "window_manager", None
    )


def set_bake_running(is_running, context=None):
    """
    Store whether Continuum Flow currently has a bake running.
    """
    window_manager = _window_manager_from_context(context)
    if window_manager is None:
        return

    if is_running:
        window_manager[CONTINUUM_FLOW_BAKE_RUNNING_KEY] = True
    else:
        window_manager.pop(CONTINUUM_FLOW_BAKE_RUNNING_KEY, None)


def is_bake_running(context=None):
    """
    Return whether Continuum Flow currently has a bake running.
    """
    window_manager = _window_manager_from_context(context)
    if window_manager is None:
        return False
    return bool(window_manager.get(CONTINUUM_FLOW_BAKE_RUNNING_KEY, False))


class ContinuumFlowBaseNode(bpy.types.Node):
    """
    Shared poll, lifecycle, and small UI helpers for Continuum Flow nodes.
    """

    @classmethod
    def poll(cls, ntree):
        return ntree.bl_idname == ContinuumFlowNodeTree.bl_idname

    def _sync_node(self):
        """
        Keep sockets or other lightweight node state in sync.
        """

    def init(self, context):
        self._sync_node()

    def copy(self, node):
        self._sync_node()

    def update(self):
        self._sync_node()

    def _set_layout_enabled(self, context, layout):
        layout.enabled = not is_bake_running(context)

    def _draw_group(self, layout, title, property_names):
        box = layout.box()
        box.label(text=title)
        col = box.column(align=True)
        for property_name in property_names:
            col.prop(self, property_name)


def ensure_socket(collection, socket_type, name, multi_input=False):
    """
    Return a socket, creating it on demand with optional multi-input support.
    """
    socket = collection.get(name)
    if socket is None:
        socket = collection.new(socket_type, name, use_multi_input=multi_input)
    if multi_input and hasattr(socket, "link_limit"):
        socket.link_limit = 0
    return socket


def ensure_geometry_input(node):
    """
    Ensure that a node exposes the standard multi-input geometry socket.
    """
    return ensure_socket(
        node.inputs, "NodeSocketGeometry", "Geometry", multi_input=True
    )


def ensure_named_output(node, socket_type, name):
    """
    Ensure that a node exposes one named output socket of the given type.
    """
    return ensure_socket(node.outputs, socket_type, name)


def _active_continuum_flow_tree(context):
    """
    Return the active Continuum Flow node tree from the current editor context.
    """
    space_data = getattr(context, "space_data", None)
    if (
        space_data is None
        or getattr(space_data, "tree_type", "") != ContinuumFlowNodeTree.bl_idname
    ):
        return None
    return getattr(space_data, "edit_tree", None) or getattr(
        space_data, "node_tree", None
    )


def _node_cursor_location(context):
    """
    Return the preferred placement location for newly created nodes.
    """
    space_data = getattr(context, "space_data", None)
    cursor = getattr(space_data, "cursor_location", None)
    if cursor is None:
        return (0.0, 0.0)
    return (float(cursor[0]), float(cursor[1]))


def _iter_continuum_flow_node_trees():
    """
    Yield all Continuum Flow node trees in the current file.
    """
    for node_tree in bpy.data.node_groups:
        if getattr(node_tree, "bl_idname", "") == ContinuumFlowNodeTree.bl_idname:
            yield node_tree


def _set_node_property_component(node, property_name, value, array_index):
    """
    Assign one scalar F-curve value to a scalar or vector node property.
    """
    current_value = getattr(node, property_name)
    is_vector_like = hasattr(current_value, "__len__") and not isinstance(
        current_value, (str, bytes)
    )

    if array_index < 0 or not is_vector_like:
        setattr(node, property_name, value)
        return

    current_value = list(current_value)
    if array_index >= len(current_value):
        return
    current_value[array_index] = value
    setattr(node, property_name, current_value)


def _iter_keyframeable_node_properties(node_tree):
    """
    Yield writable Continuum Flow node properties that expose a valid RNA path.
    """
    for node in getattr(node_tree, "nodes", ()):
        for prop in node.bl_rna.properties:
            if prop.identifier == "rna_type" or prop.is_readonly:
                continue
            yield node, prop.identifier


def _iter_action_fcurves(action):
    """
    Yield F-curves from legacy or layered Blender actions.
    """
    if action is None:
        return

    legacy_fcurves = getattr(action, "fcurves", None)
    if legacy_fcurves is not None:
        for fcurve in legacy_fcurves:
            yield fcurve
        return

    layers = getattr(action, "layers", None)
    if layers is None:
        return

    for layer in layers:
        for strip in getattr(layer, "strips", ()):
            channelbags = getattr(strip, "channelbags", None)
            if channelbags is None:
                continue
            for channelbag in channelbags:
                for fcurve in getattr(channelbag, "fcurves", ()):
                    yield fcurve


def _sync_node_tree_animation(node_tree, frame_value):
    """
    Evaluate one Continuum Flow node-tree action and push values onto node properties.
    """
    animation_data = getattr(node_tree, "animation_data", None)
    action = getattr(animation_data, "action", None)
    if action is None:
        return

    fcurves = list(_iter_action_fcurves(action))
    if not fcurves:
        return

    property_path_map = {}
    for node, property_name in _iter_keyframeable_node_properties(node_tree):
        try:
            property_path_map[node.path_from_id(property_name)] = (node, property_name)
        except Exception:
            continue

    matched_any_curve = False
    for fcurve in fcurves:
        property_target = property_path_map.get(getattr(fcurve, "data_path", ""))
        if property_target is None:
            continue

        node, property_name = property_target
        evaluated_value = fcurve.evaluate(frame_value)
        _set_node_property_component(
            node,
            property_name,
            evaluated_value,
            int(getattr(fcurve, "array_index", -1)),
        )
        matched_any_curve = True

    action_key = str(getattr(action, "name_full", getattr(action, "name", "")))
    if not matched_any_curve and action_key not in _SYNC_DEBUGGED_ACTIONS:
        _SYNC_DEBUGGED_ACTIONS.add(action_key)
        print("Continuum Flow animation sync: no matching node property paths found.")
        print(f"  Node tree: {getattr(node_tree, 'name', '<unnamed>')}")
        print("  Known animatable paths:")
        for known_path in sorted(property_path_map.keys()):
            print(f"    {known_path}")
        print("  Action F-Curve paths:")
        for fcurve in fcurves:
            print(
                f"    {getattr(fcurve, 'data_path', '')} [{int(getattr(fcurve, 'array_index', -1))}]"
            )


def sync_all_continuum_flow_node_animations(scene=None):
    """
    Evaluate all Continuum Flow node-tree animations for the current frame.
    """
    if scene is None:
        scene = getattr(bpy.context, "scene", None)
    if scene is None:
        return

    frame_value = float(getattr(scene, "frame_current", 0))
    for node_tree in _iter_continuum_flow_node_trees():
        _sync_node_tree_animation(node_tree, frame_value)


def _tag_animation_editors_redraw():
    """
    Refresh node editors after animation values were applied.
    """
    window_manager = _window_manager_from_context()
    if window_manager is None:
        return

    for window in window_manager.windows:
        screen = getattr(window, "screen", None)
        if screen is None:
            continue
        for area in screen.areas:
            if area.type in {"NODE_EDITOR", "PROPERTIES"}:
                area.tag_redraw()


@persistent
def continuum_flow_frame_change_post(scene, _depsgraph):
    """
    Force Continuum Flow custom node properties to follow their F-curves per frame.
    """
    sync_all_continuum_flow_node_animations(scene)
    _tag_animation_editors_redraw()


class ContinuumFlowDomainNode(ContinuumFlowBaseNode):
    """
    Node used to define the CFD domain resolution and boundary conditions.
    """

    bl_idname = "CONTINUUM_FLOW_DOMAIN_NODE"
    bl_label = "Domain"
    bl_icon = "MESH_GRID"
    bl_width_default = 220.0
    bl_width_min = 200.0
    bl_width_max = 360.0

    boundary_axes = (
        ("X Low", "x_low_bc", "x_low_velocity"),
        ("X High", "x_high_bc", "x_high_velocity"),
        ("Y Low", "y_low_bc", "y_low_velocity"),
        ("Y High", "y_high_bc", "y_high_velocity"),
        ("Z Low", "z_low_bc", "z_low_velocity"),
        ("Z High", "z_high_bc", "z_high_velocity"),
    )
    boundary_condition_items = (
        ("WALL", "Wall", "No-slip wall boundary"),
        ("SLIP_WALL", "Slip Wall", "Slip wall boundary"),
        ("OUTFLOW", "Outflow", "Outflow boundary"),
        ("INFLOW", "Inflow", "Inflow boundary with prescribed velocity"),
    )
    resolution: FloatProperty(name="Resolution", default=0.1, min=0.000001, soft_min=0.000001, unit="LENGTH", description="Grid resolution", options=set())  # type: ignore
    nx: IntProperty(name="NX", default=128, min=32, max=8192, soft_min=32, description="Grid cells in x", options=set())  # type: ignore
    ny: IntProperty(name="NY", default=128, min=32, max=8192, soft_min=32, description="Grid cells in y", options=set())  # type: ignore
    nz: IntProperty(name="NZ", default=128, min=32, max=8192, soft_min=32, description="Grid cells in z", options=set())  # type: ignore
    x_low_bc: bpy.props.EnumProperty(name="X Low", items=boundary_condition_items, default="OUTFLOW", options=set())  # type: ignore
    x_high_bc: bpy.props.EnumProperty(name="X High", items=boundary_condition_items, default="OUTFLOW", options=set())  # type: ignore
    y_low_bc: bpy.props.EnumProperty(name="Y Low", items=boundary_condition_items, default="OUTFLOW", options=set())  # type: ignore
    y_high_bc: bpy.props.EnumProperty(name="Y High", items=boundary_condition_items, default="OUTFLOW", options=set())  # type: ignore
    z_low_bc: bpy.props.EnumProperty(name="Z Low", items=boundary_condition_items, default="OUTFLOW", options=set())  # type: ignore
    z_high_bc: bpy.props.EnumProperty(name="Z High", items=boundary_condition_items, default="OUTFLOW", options=set())  # type: ignore
    x_low_velocity: FloatVectorProperty(name="Velocity", size=3, subtype="XYZ", default=(0.0, 0.0, 0.0), unit="VELOCITY", options=set())  # type: ignore
    x_high_velocity: FloatVectorProperty(name="Velocity", size=3, subtype="XYZ", default=(0.0, 0.0, 0.0), unit="VELOCITY", options=set())  # type: ignore
    y_low_velocity: FloatVectorProperty(name="Velocity", size=3, subtype="XYZ", default=(0.0, 0.0, 0.0), unit="VELOCITY", options=set())  # type: ignore
    y_high_velocity: FloatVectorProperty(name="Velocity", size=3, subtype="XYZ", default=(0.0, 0.0, 0.0), unit="VELOCITY", options=set())  # type: ignore
    z_low_velocity: FloatVectorProperty(name="Velocity", size=3, subtype="XYZ", default=(0.0, 0.0, 0.0), unit="VELOCITY", options=set())  # type: ignore
    z_high_velocity: FloatVectorProperty(name="Velocity", size=3, subtype="XYZ", default=(0.0, 0.0, 0.0), unit="VELOCITY", options=set())  # type: ignore

    def _sync_node(self):
        ensure_named_output(self, ContinuumFlowIntSocket.bl_idname, "Domain")

    def _draw_boundary_controls(self, layout, label, condition_attr, velocity_attr):
        box = layout.box()
        row = box.row(align=True)
        row.label(text=label)
        row.prop(self, condition_attr, text="")
        if getattr(self, condition_attr) == "INFLOW":
            box.prop(self, velocity_attr, text="Velocity")

    def draw_buttons(self, context, layout):
        self._set_layout_enabled(context, layout)
        col = layout.column(align=True)
        for property_name in ("resolution", "nx", "ny", "nz"):
            col.prop(self, property_name)
        layout.separator()
        layout.label(text="Boundary Conditions")
        for boundary_args in self.boundary_axes:
            self._draw_boundary_controls(layout, *boundary_args)


class ContinuumFlowPhysicsNode(ContinuumFlowBaseNode):
    """
    Node used to store the physical coefficients of the CFD simulation.
    """

    bl_idname = "CONTINUUM_FLOW_PHYSICS_NODE"
    bl_label = "Physics"
    bl_icon = "MOD_PHYSICS"
    bl_width_default = 220.0
    bl_width_min = 200.0
    bl_width_max = 360.0
    property_groups = (
        ("Fluid", ("fluid_density", "fluid_viscosity")),
        (
            "Temperature",
            (
                "temperature_dissipation",
                "temperature_production_rate",
                "reference_temperature",
                "buoyancy",
                "expansion_rate",
            ),
        ),
        ("Smoke", ("smoke_dissipation", "smoke_production_rate")),
        (
            "Fuel",
            (
                "fuel_dissipation",
                "fuel_burn_rate",
                "fuel_ignition_temperature",
                "minimum_oxygen_concentration",
            ),
        ),
        ("Extras", ("vorticity",)),
    )

    fluid_density: FloatProperty(name="Fluid Density", default=1.225, min=0.0001, precision=4, description="Density of the fluid, default is air", options=set())  # type: ignore
    fluid_viscosity: FloatProperty(name="Fluid Viscosity", default=1.81e-5, min=0.0, precision=6, description="Viscosity of the fluid, default is air", options=set())  # type: ignore
    temperature_dissipation: FloatProperty(name="Temperature Dissipation", default=0.01, min=0.0, description="Rate of temperature dissipation, lower means slower dissipation", options={"ANIMATABLE"})  # type: ignore
    temperature_production_rate: FloatProperty(name="Temperature Production Rate", default=1.0, min=0.0, precision=4, description="Rate of temperature production due to burning, higher means more heat released", options={"ANIMATABLE"})  # type: ignore
    reference_temperature: FloatProperty(name="Reference Temperature", default=300.0, min=0.0, unit="TEMPERATURE", description="Air cooler than this goes down, warmer than this goes up", options={"ANIMATABLE"})  # type: ignore
    buoyancy: FloatProperty(name="Buoyancy", default=0.0033, min=0.0, precision=4, description="Higher values result in quicker rising of air", options={"ANIMATABLE"})  # type: ignore
    expansion_rate: FloatProperty(name="Expansion Rate", default=0.003, min=0.0, precision=4, description="Higher values result in more expansion of warm air", options={"ANIMATABLE"})  # type: ignore
    smoke_dissipation: FloatProperty(name="Smoke Dissipation", default=0.01, min=0.0, precision=4, description="Rate of smoke dissipation, lower means slower dissipation", options={"ANIMATABLE"})  # type: ignore
    smoke_production_rate: FloatProperty(name="Smoke Production Rate", default=1.0, min=0.0, precision=4, description="Rate of smoke production due to burning, higher means more production", options={"ANIMATABLE"})  # type: ignore
    fuel_dissipation: FloatProperty(name="Fuel Dissipation", default=0.0, min=0.0, precision=4, description="Rate of fuel dissipation, lower means slower dissipation", options={"ANIMATABLE"})  # type: ignore
    fuel_burn_rate: FloatProperty(name="Fuel Burn Rate", default=0.1, min=0.0, precision=4, description="How quickly fuel is burned, lower means slower burning", options={"ANIMATABLE"})  # type: ignore
    fuel_ignition_temperature: FloatProperty(name="Fuel Ignition Temperature", default=500.0, min=0.0, unit="TEMPERATURE", description="If the air is warmer than this and contains fuel, the fuel will ignite", options={"ANIMATABLE"})  # type: ignore
    minimum_oxygen_concentration: FloatProperty(name="Minimum Oxygen Concentration", default=0.0, min=0.0, max=100.0, soft_min=0.0, soft_max=100.0, subtype="PERCENTAGE", description="Minimum oxygen concentration required for fuel to burn", options={"ANIMATABLE"})  # type: ignore
    vorticity: FloatProperty(name="Vorticity", default=0.1, min=0.0, precision=4, description="Amount of additional vorticity in the flow, higher values produce more swirl", options={"ANIMATABLE"})  # type: ignore

    def _sync_node(self):
        ensure_named_output(self, ContinuumFlowIntSocket.bl_idname, "Physics")

    def draw_buttons(self, context, layout):
        self._set_layout_enabled(context, layout)
        for title, property_names in self.property_groups:
            self._draw_group(layout, title, property_names)


class ContinuumFlowSimulationNode(ContinuumFlowBaseNode):
    """
    Node used to collect all simulation-wide settings and input dependencies.
    """

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
                "maccormack_factor",
                "simulate_sparsely",
                "adaptive_domain_threshold",
            ),
        ),
    )

    solver_backend: bpy.props.EnumProperty(
        name="Solver",
        items=(
            ("GPU", "GPU", "Use the CUDA GPU solver"),
            ("CPU", "CPU", "Use the Numba CPU solver"),
        ),
        default="CPU",
        options=set(),
    )  # type: ignore
    start_frame: IntProperty(name="Start Frame", default=1, min=0, description="Starting frame of the simulation", options=set())  # type: ignore
    end_frame: IntProperty(name="End Frame", default=250, min=2, description="End frame of the simulation", options=set())  # type: ignore
    cfl: FloatProperty(name="CFL", default=10, min=0.000001, max=10.0, soft_max=100, description="CFL condition for the solver", options=set())  # type: ignore
    iterations: IntProperty(name="Iterations", default=10, min=1, max=500, soft_min=1, soft_max=500, description="Number of pressure itterations", options=set())  # type: ignore
    maccormack_factor: FloatProperty(name="MacCormack Factor", default=0.25, min=0.0, max=0.5, soft_min=0.0, soft_max=0.5, precision=3, description="Higher values make the flow more swirly, but can produce artefacts", options=set())  # type: ignore
    simulate_sparsely: BoolProperty(name="Adaptive Domain", default=True, description="Domain adapts to the smoke and flame field to save computational cost", options=set())  # type: ignore
    adaptive_domain_threshold: FloatProperty(name="Threshold", default=0.001, min=0.0, precision=6, description="Cells containing more smoke, fuel or flame than this are considered active", options=set())  # type: ignore

    def _ensure_input_socket(self, name, *, multi_input=False):
        socket_type = (
            ContinuumFlowForceSocket.bl_idname
            if name == "Forces"
            else ContinuumFlowLinkSocket.bl_idname
        )
        return ensure_socket(self.inputs, socket_type, name, multi_input=multi_input)

    def _sync_node(self):
        self._ensure_input_socket("Domain")
        self._ensure_input_socket("Physics")
        self._ensure_input_socket("Obstacles")
        self._ensure_input_socket("Source", multi_input=True)
        self._ensure_input_socket("Forces", multi_input=True)
        ensure_named_output(self, ContinuumFlowResultSocket.bl_idname, "Result")

    def init(self, context):
        scene = getattr(context, "scene", None) or getattr(bpy.context, "scene", None)
        if scene is not None:
            self.start_frame = int(getattr(scene, "frame_start", self.start_frame))
            self.end_frame = int(getattr(scene, "frame_end", self.end_frame))
        self._sync_node()

    def draw_buttons(self, context, layout):
        self._set_layout_enabled(context, layout)
        if not _solver_environment_ready():
            warning_box = layout.box()
            warning_box.label(text="Solver environment missing", icon="ERROR")
            warning_box.label(text="Use the install button to create .venv with uv.")
            warning_box.operator(
                "continuum_flow.install_solver_environment",
                icon="CONSOLE",
            )
        solver_row = layout.row(align=True)
        cpu_row = solver_row.row(align=True)
        cpu_row.prop_enum(self, "solver_backend", "CPU")
        gpu_row = solver_row.row(align=True)
        gpu_row.enabled = _gpu_solver_available()
        gpu_row.prop_enum(self, "solver_backend", "GPU")
        if not _gpu_solver_available() and self.solver_backend == "GPU":
            self.solver_backend = "CPU"
        for title, property_names in self.property_groups:
            self._draw_group(layout, title, property_names)


class ContinuumFlowSourceNode(ContinuumFlowBaseNode):
    """
    Node used to define a generic CFD source region and its scalar and velocity targets.
    """

    bl_idname = "CONTINUUM_FLOW_SOURCE_NODE"
    bl_label = "Source"
    bl_icon = "LIGHT_SUN"
    bl_width_default = 220.0
    bl_width_min = 200.0
    bl_width_max = 360.0
    scalar_property_names = ("fuel", "smoke", "temperature", "extra_pressure")
    velocity_space_items = (
        ("WORLD", "World Space", "Apply the source velocity in world coordinates"),
        (
            "LOCAL",
            "Local Space",
            "Apply the source velocity in each linked object's local coordinates",
        ),
    )

    fuel: FloatProperty(name="Fuel Emission", default=0.0, min=0.0, max=100.0, soft_min=0.0, soft_max=100.0, subtype="PERCENTAGE", description="Fuel emission rate in percent per second", options={"ANIMATABLE"})  # type: ignore
    smoke: FloatProperty(name="Smoke Emission", default=0.0, min=0.0, max=100.0, soft_min=0.0, soft_max=100.0, subtype="PERCENTAGE", description="Smoke emission rate in percent per second", options={"ANIMATABLE"})  # type: ignore
    temperature: FloatProperty(name="Temperature", default=300.0, min=0.0, soft_min=0.0, unit="TEMPERATURE", description="Amount of temperature to spawn", options={"ANIMATABLE"})  # type: ignore
    extra_pressure: FloatProperty(name="Extra Pressure", default=0.0, precision=4, description="Additional pressure added in the source", options={"ANIMATABLE"})  # type: ignore
    velocity_space: EnumProperty(name="Space", items=velocity_space_items, default="WORLD", options=set())  # type: ignore
    velocity: FloatVectorProperty(name="Velocity", size=3, default=(0.0, 0.0, 0.0), subtype="VELOCITY", description="Source velocity", options={"ANIMATABLE"})  # type: ignore

    def _sync_node(self):
        ensure_geometry_input(self)
        ensure_named_output(self, ContinuumFlowIntSocket.bl_idname, "Source")

    def draw_buttons(self, context, layout):
        self._set_layout_enabled(context, layout)
        col = layout.column(align=True)
        for property_name in self.scalar_property_names:
            col.prop(self, property_name)

        velocity_col = layout.column(align=True)
        velocity_col.label(text="Velocity")
        velocity_col.prop(self, "velocity_space", text="Space")
        velocity_col.prop(self, "velocity", text="")


class ContinuumFlowGeometryNode(ContinuumFlowBaseNode):
    """
    Node used to reference a Blender object as geometry inside the CFD graph.
    """

    bl_idname = "CONTINUUM_FLOW_GEOMETRY_NODE"
    bl_label = "Geometry"
    bl_icon = "OUTLINER_OB_MESH"
    bl_width_default = 220.0
    bl_width_min = 200.0
    bl_width_max = 360.0

    source_object: PointerProperty(name="Object", type=bpy.types.Object)  # type: ignore

    def _sync_node(self):
        ensure_named_output(self, "NodeSocketGeometry", "Geometry")

    def draw_buttons(self, context, layout):
        self._set_layout_enabled(context, layout)
        layout.prop(self, "source_object", text="Object")


class ContinuumFlowObstacleNode(ContinuumFlowBaseNode):
    """
    Node used to define obstacle geometry inside the CFD domain.
    """

    bl_idname = "CONTINUUM_FLOW_OBSTACLE_NODE"
    bl_label = "Obstacle"
    bl_icon = "CUBE"
    bl_width_default = 220.0
    bl_width_min = 200.0
    bl_width_max = 360.0

    def _sync_node(self):
        ensure_geometry_input(self)
        ensure_named_output(self, ContinuumFlowIntSocket.bl_idname, "Obstacle")


class ContinuumFlowViewerNode(ContinuumFlowBaseNode):
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

    def _sync_node(self):
        ensure_socket(self.inputs, ContinuumFlowResultSocket.bl_idname, "Result")

    def free(self):
        self.domain_preview_active = False
        ContinuumFlowViewerModule.disable_domain_preview()

    def draw_buttons(self, context, layout):
        self._set_layout_enabled(context, layout)
        col = layout.column(align=True)
        col.operator(
            "continuum_flow.viewer_toggle_domain",
            text="Hide Domain" if bool(self.domain_preview_active) else "Show Domain",
            icon="HIDE_ON" if bool(self.domain_preview_active) else "HIDE_OFF",
        )
        col.prop(self, "live_preview")


class ContinuumFlow_OT_add_basic_setup(bpy.types.Operator):
    """
    Add a ready-to-use Continuum Flow starter setup with the core nodes linked.
    """

    bl_idname = "continuum_flow.add_basic_setup"
    bl_label = "Add Basic Continuum Flow Setup"
    bl_description = "Create a starter setup with Domain, Physics, Simulation, Viewer, and Output already connected"

    @classmethod
    def poll(cls, context):
        return _active_continuum_flow_tree(context) is not None

    def execute(self, context):
        node_tree = _active_continuum_flow_tree(context)
        if node_tree is None:
            self.report({"ERROR"}, "Open a Continuum Flow node tree first.")
            return {"CANCELLED"}

        cursor_x, cursor_y = _node_cursor_location(context)
        node_specs = (
            ("domain", "CONTINUUM_FLOW_DOMAIN_NODE", (cursor_x - 520.0, cursor_y + 140.0)),
            (
                "physics",
                "CONTINUUM_FLOW_PHYSICS_NODE",
                (cursor_x - 520.0, cursor_y - 140.0),
            ),
            ("simulation", "CONTINUUM_FLOW_SIMULATION_NODE", (cursor_x - 120.0, cursor_y)),
            ("viewer", "CONTINUUM_FLOW_VIEWER_NODE", (cursor_x + 280.0, cursor_y + 120.0)),
            ("output", "CONTINUUM_FLOW_OUTPUT_NODE", (cursor_x + 280.0, cursor_y - 120.0)),
        )

        for node in node_tree.nodes:
            node.select = False

        created_nodes = {}
        for key, node_type, location in node_specs:
            created_node = node_tree.nodes.new(node_type)
            created_node.location = location
            created_node.select = True
            created_nodes[key] = created_node

        node_tree.links.new(
            created_nodes["domain"].outputs["Domain"],
            created_nodes["simulation"].inputs["Domain"],
        )
        node_tree.links.new(
            created_nodes["physics"].outputs["Physics"],
            created_nodes["simulation"].inputs["Physics"],
        )
        node_tree.links.new(
            created_nodes["simulation"].outputs["Result"],
            created_nodes["viewer"].inputs["Result"],
        )
        node_tree.links.new(
            created_nodes["simulation"].outputs["Result"],
            created_nodes["output"].inputs["Result"],
        )

        node_tree.nodes.active = created_nodes["simulation"]
        return {"FINISHED"}


classes = (
    ContinuumFlowDomainNode,
    ContinuumFlowPhysicsNode,
    ContinuumFlowSimulationNode,
    ContinuumFlowSourceNode,
    ContinuumFlowGeometryNode,
    ContinuumFlowObstacleNode,
    ContinuumFlowViewerNode,
    ContinuumFlow_OT_add_basic_setup,
)
