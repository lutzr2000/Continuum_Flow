import bpy
import json
import importlib.util
import subprocess
import sys
from pathlib import Path
from bpy.props import BoolProperty, EnumProperty, FloatProperty, FloatVectorProperty, IntProperty, PointerProperty, StringProperty

try:
    from .NodeTree import BlenderCFDNodeTree
except ImportError:
    try:
        from NodeTree import BlenderCFDNodeTree
    except ImportError:
        # Fallback for direct execution in Blender's text editor without package context.
        if "__file__" in globals():
            node_tree_path = Path(__file__).resolve().with_name("NodeTree.py")
        else:
            node_tree_path = (Path.cwd() / "UI" / "NodeTree.py").resolve()

        spec = importlib.util.spec_from_file_location("blendercfd_nodetree", node_tree_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["blendercfd_nodetree"] = module
        spec.loader.exec_module(module)
        BlenderCFDNodeTree = module.BlenderCFDNodeTree

try:
    from .Sockets import (
        BlenderCFDForceSocket,
        BlenderCFDIntSocket,
        BlenderCFDLinkSocket,
        BlenderCFDReferenceFrameSocket,
        BlenderCFDResultSocket,
    )
except ImportError:
    try:
        from Sockets import (
            BlenderCFDForceSocket,
            BlenderCFDIntSocket,
            BlenderCFDLinkSocket,
            BlenderCFDReferenceFrameSocket,
            BlenderCFDResultSocket,
        )
    except ImportError:
        # Fallback for direct execution in Blender's text editor without package context.
        if "__file__" in globals():
            sockets_path = Path(__file__).resolve().with_name("Sockets.py")
        else:
            sockets_path = (Path.cwd() / "UI" / "Sockets.py").resolve()

        spec = importlib.util.spec_from_file_location("blendercfd_sockets", sockets_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["blendercfd_sockets"] = module
        spec.loader.exec_module(module)
        BlenderCFDForceSocket = module.BlenderCFDForceSocket
        BlenderCFDIntSocket = module.BlenderCFDIntSocket
        BlenderCFDLinkSocket = module.BlenderCFDLinkSocket
        BlenderCFDReferenceFrameSocket = module.BlenderCFDReferenceFrameSocket
        BlenderCFDResultSocket = module.BlenderCFDResultSocket

try:
    from . import Viewer as BlenderCFDViewerModule
except ImportError:
    try:
        import Viewer as BlenderCFDViewerModule
    except ImportError:
        if "__file__" in globals():
            viewer_path = Path(__file__).resolve().with_name("Viewer.py")
        else:
            viewer_path = (Path.cwd() / "UI" / "Viewer.py").resolve()

        spec = importlib.util.spec_from_file_location("blendercfd_viewer", viewer_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["blendercfd_viewer"] = module
        spec.loader.exec_module(module)
        BlenderCFDViewerModule = module

try:
    from . import Create_Config_Dict as BlenderCFDConfigModule
except ImportError:
    try:
        import Create_Config_Dict as BlenderCFDConfigModule
    except ImportError:
        if "__file__" in globals():
            config_path = Path(__file__).resolve().with_name("Create_Config_Dict.py")
        else:
            config_path = (Path.cwd() / "UI" / "Create_Config_Dict.py").resolve()

        spec = importlib.util.spec_from_file_location("blendercfd_create_config_dict", config_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["blendercfd_create_config_dict"] = module
        spec.loader.exec_module(module)
        BlenderCFDConfigModule = module


def _kernel_directory():
    """Return the local Kernel directory path."""
    if "__file__" in globals():
        return Path(__file__).resolve().parents[1] / "Kernel"
    return (Path.cwd() / "Kernel").resolve()


def _resolve_python_executable():
    """Use the project-local virtual environment for baking."""
    python_executable = Path(__file__).resolve().parents[1] / "BlenderCFD_env" / "Scripts" / "python.exe"
    if not python_executable.exists():
        raise FileNotFoundError(f"Python executable not found: {python_executable}")
    return str(python_executable)


def _run_kernel(config_dict):
    """Run the CFD kernel in the project venv and pass the config from memory via stdin."""
    python_executable = _resolve_python_executable()
    kernel_dir = _kernel_directory()
    bootstrap_code = (
        "import json, sys; "
        "sys.path.insert(0, sys.argv[1]); "
        "import Kernel_GPU; "
        "Kernel_GPU.main(json.load(sys.stdin))"
    )
    process = subprocess.Popen(
        [python_executable, "-c", bootstrap_code, str(kernel_dir)],
        cwd=str(kernel_dir),
        stdin=subprocess.PIPE,
        text=True,
    )
    process.stdin.write(json.dumps(config_dict))
    process.stdin.close()
    return python_executable


class BlenderCFDDomainNode(bpy.types.Node):
    """
    Node used to define the CFD domain resolution and boundary conditions.

    The node stores the cell counts in x, y, and z direction and lets the user
    choose boundary conditions for all six domain faces. It currently exposes
    one placeholder output socket for future graph connections.
    """

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

    nx: IntProperty(  # type: ignore
        name="NX",
        default=128,
        min=32,
        max=4096,
        soft_min=32,
        soft_max=4096,
    )

    resolution: FloatProperty(  # type: ignore
        name="Resolution",
        default=0.1,
        min=0.000001,
        soft_min=0.01,
        unit="LENGTH",
    )

    ny: IntProperty(  # type: ignore
        name="NY",
        default=128,
        min=32,
        max=4096,
        soft_min=32,
        soft_max=4096,
    )

    nz: IntProperty(  # type: ignore
        name="NZ",
        default=128,
        min=32,
        max=4096,
        soft_min=32,
        soft_max=4096,
    )

    x_low_bc: EnumProperty(  # type: ignore
        name="X Low",
        items=boundary_condition_items,
        default="OUTFLOW",
    )

    x_high_bc: EnumProperty(  # type: ignore
        name="X High",
        items=boundary_condition_items,
        default="OUTFLOW",
    )

    y_low_bc: EnumProperty(  # type: ignore
        name="Y Low",
        items=boundary_condition_items,
        default="OUTFLOW",
    )

    y_high_bc: EnumProperty(  # type: ignore
        name="Y High",
        items=boundary_condition_items,
        default="OUTFLOW",
    )

    z_low_bc: EnumProperty(  # type: ignore
        name="Z Low",
        items=boundary_condition_items,
        default="OUTFLOW",
    )

    z_high_bc: EnumProperty(  # type: ignore
        name="Z High",
        items=boundary_condition_items,
        default="OUTFLOW",
    )

    x_low_velocity: FloatVectorProperty(  # type: ignore
        name="Velocity",
        size=3,
        subtype="XYZ",
        default=(0.0, 0.0, 0.0),
        unit="VELOCITY",
    )

    x_high_velocity: FloatVectorProperty(  # type: ignore
        name="Velocity",
        size=3,
        subtype="XYZ",
        default=(0.0, 0.0, 0.0),
        unit="VELOCITY",
    )

    y_low_velocity: FloatVectorProperty(  # type: ignore
        name="Velocity",
        size=3,
        subtype="XYZ",
        default=(0.0, 0.0, 0.0),
        unit="VELOCITY",
    )

    y_high_velocity: FloatVectorProperty(  # type: ignore
        name="Velocity",
        size=3,
        subtype="XYZ",
        default=(0.0, 0.0, 0.0),
        unit="VELOCITY",
    )

    z_low_velocity: FloatVectorProperty(  # type: ignore
        name="Velocity",
        size=3,
        subtype="XYZ",
        default=(0.0, 0.0, 0.0),
        unit="VELOCITY",
    )

    z_high_velocity: FloatVectorProperty(  # type: ignore
        name="Velocity",
        size=3,
        subtype="XYZ",
        default=(0.0, 0.0, 0.0),
        unit="VELOCITY",
    )

    @classmethod
    def poll(cls, ntree):
        """Return whether the node can be added to the given node tree."""
        return ntree.bl_idname == BlenderCFDNodeTree.bl_idname

    def _ensure_output_socket(self, name):
        """Ensure that the placeholder output socket exists."""
        socket = self.outputs.get(name)
        if socket is None:
            socket = self.outputs.new(BlenderCFDIntSocket.bl_idname, name)
        return socket

    def _sync_output_socket(self):
        """Refresh the node output layout."""
        self._ensure_output_socket("Domain")

    def init(self, context):
        """Initialize sockets when the node is created."""
        self._sync_output_socket()

    def copy(self, node):
        """Restore sockets after the node has been copied."""
        self._sync_output_socket()

    def update(self):
        """Keep the node socket layout in sync."""
        self._sync_output_socket()

    def _draw_boundary_controls(self, layout, label, condition_attr, velocity_attr):
        """Draw one boundary-condition row and its optional inflow velocity."""
        box = layout.box()
        row = box.row(align=True)
        row.label(text=label)
        row.prop(self, condition_attr, text="")
        if getattr(self, condition_attr) == "INFLOW":
            box.prop(self, velocity_attr, text="Velocity")

    def draw_buttons(self, context, layout):
        """Draw the editable controls shown inside the node body."""
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


class BlenderCFDSourceNode(bpy.types.Node):
    """
    Node used to define a generic CFD source region and its scalar strengths.

    The node accepts geometry as an input selection and stores source values for
    fuel, smoke, and temperature. It currently exposes a placeholder output
    socket for future graph connections.
    """

    bl_idname = "BLENDERCFD_SOURCE_NODE"
    bl_label = "Source"
    bl_icon = "LIGHT_SUN"
    bl_width_default = 220.0
    bl_width_min = 200.0
    bl_width_max = 360.0

    fuel: FloatProperty(  # type: ignore
        name="Fuel",
        default=0.0,
        min=0.0,
        max=10000.0,
        soft_min=0.0,
        soft_max=10000.0,
    )

    smoke: FloatProperty(  # type: ignore
        name="Smoke",
        default=0.0,
        min=0.0,
        max=10000.0,
        soft_min=0.0,
        soft_max=10000.0,
    )

    temperature: FloatProperty(  # type: ignore
        name="Temperature",
        default=0.0,
        min=0.0,
        max=10000.0,
        soft_min=0.0,
        soft_max=10000.0,
        unit="TEMPERATURE",
    )

    @classmethod
    def poll(cls, ntree):
        """Return whether the node can be added to the given node tree."""
        return ntree.bl_idname == BlenderCFDNodeTree.bl_idname

    def _ensure_geometry_input(self):
        """Ensure that the geometry input socket exists."""
        socket = self.inputs.get("Geometry")
        if socket is None:
            socket = self.inputs.new("NodeSocketGeometry", "Geometry", use_multi_input=True)
        if hasattr(socket, "link_limit"):
            socket.link_limit = 0
        return socket

    def _ensure_output_socket(self, name):
        """Ensure that the placeholder output socket exists."""
        socket = self.outputs.get(name)
        if socket is None:
            socket = self.outputs.new(BlenderCFDIntSocket.bl_idname, name)
        return socket

    def _sync_sockets(self):
        """Refresh the node socket layout."""
        self._ensure_geometry_input()
        self._ensure_output_socket("Source")

    def init(self, context):
        """Initialize sockets when the node is created."""
        self._sync_sockets()

    def copy(self, node):
        """Restore sockets after the node has been copied."""
        self._sync_sockets()

    def update(self):
        """Keep the node socket layout in sync."""
        self._sync_sockets()

    def draw_buttons(self, context, layout):
        """Draw the editable controls shown inside the node body."""
        col = layout.column(align=True)
        col.prop(self, "fuel")
        col.prop(self, "smoke")
        col.prop(self, "temperature")


class BlenderCFDGeometryNode(bpy.types.Node):
    """
    Node used to reference a Blender object as geometry inside the CFD graph.

    The node currently stores only an object reference and exposes a geometry
    output socket as a placeholder for future evaluation logic.
    """

    bl_idname = "BLENDERCFD_GEOMETRY_NODE"
    bl_label = "Geometry"
    bl_icon = "OUTLINER_OB_MESH"
    bl_width_default = 220.0
    bl_width_min = 200.0
    bl_width_max = 360.0

    source_object: PointerProperty(  # type: ignore
        name="Object",
        type=bpy.types.Object,
    )

    @classmethod
    def poll(cls, ntree):
        """Return whether the node can be added to the given node tree."""
        return ntree.bl_idname == BlenderCFDNodeTree.bl_idname

    def _ensure_output_socket(self):
        """Ensure that the geometry output socket exists."""
        socket = self.outputs.get("Geometry")
        if socket is None:
            socket = self.outputs.new("NodeSocketGeometry", "Geometry")
        return socket

    def _sync_sockets(self):
        """Refresh the node socket layout."""
        self._ensure_output_socket()

    def init(self, context):
        """Initialize sockets when the node is created."""
        self._sync_sockets()

    def copy(self, node):
        """Restore sockets after the node has been copied."""
        self._sync_sockets()

    def update(self):
        """Keep the node socket layout in sync."""
        self._sync_sockets()

    def draw_buttons(self, context, layout):
        """Draw the editable controls shown inside the node body."""
        layout.prop(self, "source_object", text="Object")


class BlenderCFDPhysicsNode(bpy.types.Node):
    """
    Node used to store the physical coefficients of the CFD simulation.

    The node groups fluid, temperature, smoke, and fuel parameters into
    separate UI sections and exposes a placeholder output socket for future
    graph connections.
    """

    bl_idname = "BLENDERCFD_PHYSICS_NODE"
    bl_label = "Physics"
    bl_icon = "MOD_PHYSICS"
    bl_width_default = 220.0
    bl_width_min = 200.0
    bl_width_max = 360.0

    fluid_density: FloatProperty(  # type: ignore
        name="Fluid Density",
        default=1.225,
        min=0.001,
        max=2000,
        precision=3,
    )

    fluid_viscosity: FloatProperty(  # type: ignore
        name="Fluid Viscosity",
        default=1.81e-5,
        min=0.0,
        max=1,
        precision=6,
    )

    temperature_diffusion: FloatProperty(  # type: ignore
        name="Temperature Diffusion",
        default=0.01,
        min=0.0,
        max=1,
        precision=4,
    )

    temperature_dissipation: FloatProperty(  # type: ignore
        name="Temperature Dissipation",
        default=0.1,
        min=0.0,
        max=100
    )

    reference_temperature: FloatProperty(  # type: ignore
        name="Reference Temperature",
        default=300.0,
        min=0.0,
        max=10000.0,
        unit="TEMPERATURE",
    )

    buoyancy: FloatProperty(  # type: ignore
        name="Buoyancy",
        default=0.0033,
        min=0.0,
        max=1.0,
        precision=4,
    )

    expansion_rate: FloatProperty(  # type: ignore
        name="Expansion Rate",
        default=0.003,
        min=0.0,
        max=1.0,
        precision=4,
    )

    smoke_diffusion: FloatProperty(  # type: ignore
        name="Smoke Diffusion",
        default=0.001,
        min=0.0,
        max=1.0,
        precision=4,
    )

    smoke_dissipation: FloatProperty(  # type: ignore
        name="Smoke Dissipation",
        default=0.1,
        min=0.0,
        max=100.0
    )

    fuel_diffusion: FloatProperty(  # type: ignore
        name="Fuel Diffusion",
        default=0.001,
        min=0.0,
        max=1.0,
        precision=4,
    )

    fuel_dissipation: FloatProperty(  # type: ignore
        name="Fuel Dissipation",
        default=0.001,
        min=0.0,
        max=100.0
    )

    fuel_burn_rate: FloatProperty(  # type: ignore
        name="Fuel Burn Rate",
        default=0.1,
        min=0.0,
        max=100.0
    )

    fuel_ignition_temperature: FloatProperty(  # type: ignore
        name="Fuel Ignition Temperature",
        default=500.0,
        min=0.0, 
        max=10000.0,
        unit="TEMPERATURE",
    )

    @classmethod
    def poll(cls, ntree):
        """Return whether the node can be added to the given node tree."""
        return ntree.bl_idname == BlenderCFDNodeTree.bl_idname

    def _ensure_output_socket(self):
        """Ensure that the placeholder output socket exists."""
        socket = self.outputs.get("Physics")
        if socket is None:
            socket = self.outputs.new(BlenderCFDIntSocket.bl_idname, "Physics")
        return socket

    def _sync_output_socket(self):
        """Refresh the node socket layout."""
        self._ensure_output_socket()

    def init(self, context):
        """Initialize sockets when the node is created."""
        self._sync_output_socket()

    def copy(self, node):
        """Restore sockets after the node has been copied."""
        self._sync_output_socket()

    def update(self):
        """Keep the node socket layout in sync."""
        self._sync_output_socket()

    def _draw_group(self, layout, title, property_names):
        """Draw one grouped section of physics properties."""
        box = layout.box()
        box.label(text=title)
        col = box.column(align=True)
        for property_name in property_names:
            col.prop(self, property_name)

    def draw_buttons(self, context, layout):
        """Draw the editable controls shown inside the node body."""
        self._draw_group(
            layout,
            "Fluid",
            ("fluid_density", "fluid_viscosity"),
        )
        self._draw_group(
            layout,
            "Temperature",
            (
                "temperature_diffusion",
                "temperature_dissipation",
                "reference_temperature",
                "buoyancy",
                "expansion_rate",
            ),
        )
        self._draw_group(
            layout,
            "Smoke",
            ("smoke_diffusion", "smoke_dissipation"),
        )
        self._draw_group(
            layout,
            "Fuel",
            (
                "fuel_diffusion",
                "fuel_dissipation",
                "fuel_burn_rate",
                "fuel_ignition_temperature",
            ),
        )


class BlenderCFDReferenceFrameNode(bpy.types.Node):
    """
    Node used to define an object-based reference frame for the simulation.

    The node stores a Blender object reference and exposes a dedicated output
    socket so the simulation node can consume the chosen frame source.
    """

    bl_idname = "BLENDERCFD_REFERENCE_FRAME_NODE"
    bl_label = "Reference Frame"
    bl_icon = "EMPTY_AXIS"
    bl_width_default = 220.0
    bl_width_min = 200.0
    bl_width_max = 360.0

    source_object: PointerProperty(  # type: ignore
        name="Object",
        type=bpy.types.Object,
    )

    @classmethod
    def poll(cls, ntree):
        """Return whether the node can be added to the given node tree."""
        return ntree.bl_idname == BlenderCFDNodeTree.bl_idname

    def _ensure_output_socket(self):
        """Ensure that the reference-frame output socket exists."""
        socket = self.outputs.get("Reference Frame")
        if socket is None:
            socket = self.outputs.new(BlenderCFDReferenceFrameSocket.bl_idname, "Reference Frame")
        return socket

    def _sync_sockets(self):
        """Refresh the node socket layout."""
        self._ensure_output_socket()

    def init(self, context):
        """Initialize sockets when the node is created."""
        self._sync_sockets()

    def copy(self, node):
        """Restore sockets after the node has been copied."""
        self._sync_sockets()

    def update(self):
        """Keep the node socket layout in sync."""
        self._sync_sockets()

    def draw_buttons(self, context, layout):
        """Draw the editable controls shown inside the node body."""
        layout.prop(self, "source_object", text="Object")


class BlenderCFDSimulationNode(bpy.types.Node):
    """
    Node used to collect all simulation-wide settings and input dependencies.

    The node stores time and solver parameters, accepts the upstream CFD setup
    nodes as logical inputs, and exposes a purple result socket for future
    execution output.
    """

    bl_idname = "BLENDERCFD_SIMULATION_NODE"
    bl_label = "Simulation"
    bl_icon = "TIME"
    bl_width_default = 260.0
    bl_width_min = 240.0
    bl_width_max = 420.0

    simulation_length: FloatProperty(  # type: ignore
        name="Simulation Length",
        default=10.0,
        min=0.001,
        unit="TIME",
    )

    cfl: FloatProperty(  # type: ignore
        name="CFL",
        default=0.8,
        min=0.000001,
        max=1.0
    )

    iterations: IntProperty(  # type: ignore
        name="Iterations",
        default=4,
        min=0,
        max=500,
        soft_min=0,
        soft_max=500,
    )

    @classmethod
    def poll(cls, ntree):
        """Return whether the node can be added to the given node tree."""
        return ntree.bl_idname == BlenderCFDNodeTree.bl_idname

    def _ensure_input_socket(self, name, *, multi_input=False):
        """Ensure that a logical input socket exists."""
        socket = self.inputs.get(name)
        if socket is None:
            socket = self.inputs.new(
                BlenderCFDReferenceFrameSocket.bl_idname
                if name == "Reference Frame"
                else BlenderCFDForceSocket.bl_idname
                if name == "Forces"
                else BlenderCFDLinkSocket.bl_idname,
                name,
                use_multi_input=multi_input,
            )
        if multi_input and hasattr(socket, "link_limit"):
            socket.link_limit = 0
        return socket

    def _ensure_output_socket(self):
        """Ensure that the result output socket exists."""
        socket = self.outputs.get("Result")
        if socket is None:
            socket = self.outputs.new(BlenderCFDResultSocket.bl_idname, "Result")
        return socket

    def _sync_sockets(self):
        """Refresh the node socket layout."""
        self._ensure_input_socket("Reference Frame")
        self._ensure_input_socket("Domain")
        self._ensure_input_socket("Physics")
        self._ensure_input_socket("Obstacles")
        self._ensure_input_socket("Source", multi_input=True)
        self._ensure_input_socket("Forces", multi_input=True)
        self._ensure_output_socket()

    def init(self, context):
        """Initialize sockets when the node is created."""
        self._sync_sockets()

    def copy(self, node):
        """Restore sockets after the node has been copied."""
        self._sync_sockets()

    def update(self):
        """Keep the node socket layout in sync."""
        self._sync_sockets()

    def _draw_group(self, layout, title, property_names):
        """Draw one grouped section of simulation properties."""
        box = layout.box()
        box.label(text=title)
        col = box.column(align=True)
        for property_name in property_names:
            col.prop(self, property_name)

    def draw_buttons(self, context, layout):
        """Draw the editable controls shown inside the node body."""
        self._draw_group(
            layout,
            "Time",
            ("simulation_length", "cfl"),
        )
        self._draw_group(
            layout,
            "Solver",
            ("iterations",),
        )


class BlenderCFDOutputNode(bpy.types.Node):
    """
    Node used to configure which simulation results should be written to disk.

    The node accepts the simulation result link, stores output timing and field
    selection settings, and keeps the export path in one place for later use.
    """

    bl_idname = "BLENDERCFD_OUTPUT_NODE"
    bl_label = "Output"
    bl_icon = "OUTPUT"
    bl_width_default = 260.0
    bl_width_min = 240.0
    bl_width_max = 420.0

    fps: IntProperty(  # type: ignore
        name="FPS",
        default=24,
        min=1,
        max=240,
        soft_min=1,
        soft_max=120,
    )

    export_u: BoolProperty(  # type: ignore
        name="Velocity x",
        default=True,
    )

    export_v: BoolProperty(  # type: ignore
        name="Velocity y",
        default=True,
    )

    export_w: BoolProperty(  # type: ignore
        name="Velocity z",
        default=True,
    )

    export_p: BoolProperty(  # type: ignore
        name="Pressure",
        default=True,
    )

    export_t: BoolProperty(  # type: ignore
        name="Temperature",
        default=True,
    )

    export_smoke: BoolProperty(  # type: ignore
        name="Smoke",
        default=True,
    )

    export_fuel: BoolProperty(  # type: ignore
        name="Fuel",
        default=True,
    )

    export_flame: BoolProperty(  # type: ignore
        name="Flame",
        default=True,
    )

    output_path: StringProperty(  # type: ignore
        name="Path",
        default="",
        subtype="DIR_PATH",
    )

    @classmethod
    def poll(cls, ntree):
        """Return whether the node can be added to the given node tree."""
        return ntree.bl_idname == BlenderCFDNodeTree.bl_idname

    def _ensure_input_socket(self):
        """Ensure that the result input socket exists."""
        socket = self.inputs.get("Result")
        if socket is None:
            socket = self.inputs.new(BlenderCFDResultSocket.bl_idname, "Result")
        return socket

    def _sync_input_socket(self):
        """Refresh the node socket layout."""
        self._ensure_input_socket()

    def _sync_defaults_from_scene(self, context):
        """Initialize node defaults from the active Blender scene."""
        scene = getattr(context, "scene", None)
        if scene is None:
            scene = getattr(bpy.context, "scene", None)
        if scene is None:
            return

        render = getattr(scene, "render", None)
        if render is None:
            return

        fps = getattr(render, "fps", None)
        if fps is not None and self.fps == 24:
            self.fps = fps

    def init(self, context):
        """Initialize sockets when the node is created."""
        self._sync_input_socket()
        self._sync_defaults_from_scene(context)

    def copy(self, node):
        """Restore sockets after the node has been copied."""
        self._sync_input_socket()

    def update(self):
        """Keep the node socket layout in sync."""
        self._sync_input_socket()

    def draw_buttons(self, context, layout):
        """Draw the editable controls shown inside the node body."""
        layout.prop(self, "fps")

        fields_box = layout.box()
        fields_box.label(text="Fields")
        fields_col = fields_box.column(align=True)
        fields_col.prop(self, "export_u")
        fields_col.prop(self, "export_v")
        fields_col.prop(self, "export_w")
        fields_col.prop(self, "export_p")
        fields_col.prop(self, "export_t")
        fields_col.prop(self, "export_smoke")
        fields_col.prop(self, "export_fuel")
        fields_col.prop(self, "export_flame")

        layout.prop(self, "output_path")
        layout.separator()
        layout.operator("blendercfd.bake", text="Bake", icon="RENDER_STILL")


class BlenderCFD_OT_bake(bpy.types.Operator):
    """
    Operator that exports the active node tree config and starts the kernel.
    """

    bl_idname = "blendercfd.bake"
    bl_label = "Bake BlenderCFD"
    bl_description = "Start the BlenderCFD bake process"

    def execute(self, context):
        """Build the current config dict and run the CFD kernel."""
        try:
            config_dict = BlenderCFDConfigModule.build_config_dict(context)
            python_executable = _run_kernel(config_dict)
        except ModuleNotFoundError as exc:
            missing_module = getattr(exc, "name", None) or str(exc)
            self.report(
                {"ERROR"},
                f"Bake failed: missing Python module '{missing_module}' in {sys.executable}",
            )
            return {"CANCELLED"}
        except Exception as exc:
            self.report({"ERROR"}, f"Bake failed: {exc}")
            return {"CANCELLED"}

        simulation_count = len(config_dict.get("simulations", ()))
        self.report({"INFO"}, f"Started BlenderCFD bake for {simulation_count} simulation(s) via {python_executable}.")
        return {"FINISHED"}


class BlenderCFDViewerNode(bpy.types.Node):
    """
    Node used as a lightweight endpoint for inspecting simulation results.

    The node accepts a simulation result link and can trigger a simple viewport
    preview of the upstream domain bounds.
    """

    bl_idname = "BLENDERCFD_VIEWER_NODE"
    bl_label = "Viewer"
    bl_icon = "HIDE_OFF"
    bl_width_default = 180.0
    bl_width_min = 160.0
    bl_width_max = 260.0

    @classmethod
    def poll(cls, ntree):
        """Return whether the node can be added to the given node tree."""
        return ntree.bl_idname == BlenderCFDNodeTree.bl_idname

    def _ensure_result_input_socket(self):
        """Ensure that the result input socket exists."""
        socket = self.inputs.get("Result")
        if socket is None:
            socket = self.inputs.new(BlenderCFDResultSocket.bl_idname, "Result")
        return socket

    def _sync_input_socket(self):
        """Refresh the node socket layout."""
        self._ensure_result_input_socket()

    def init(self, context):
        """Initialize sockets when the node is created."""
        self._sync_input_socket()

    def copy(self, node):
        """Restore sockets after the node has been copied."""
        self._sync_input_socket()

    def update(self):
        """Keep the node socket layout in sync."""
        self._sync_input_socket()

    def free(self):
        """Hide the preview if this viewer node is removed."""
        BlenderCFDViewerModule.disable_domain_preview()

    def draw_buttons(self, context, layout):
        """Draw simple viewport preview controls."""
        col = layout.column(align=True)
        col.operator("blendercfd.viewer_show_domain", text="Show Domain", icon="HIDE_OFF")
        col.operator("blendercfd.viewer_hide_domain", text="Hide Domain", icon="HIDE_ON")


class BlenderCFDForceConstantNode(bpy.types.Node):
    """
    Node used to define a constant force vector for the CFD simulation.

    The node stores constant force components in x, y, and z direction and
    exposes a placeholder output socket for future graph connections.
    """

    bl_idname = "BLENDERCFD_FORCE_CONSTANT_NODE"
    bl_label = "Force Constant"
    bl_icon = "FORCE_FORCE"
    bl_width_default = 220.0
    bl_width_min = 200.0
    bl_width_max = 320.0

    fx: FloatProperty(  # type: ignore
        name="Fx",
        default=0.0,
        min=-10000.0,
        max=10000.0,
        soft_min=-10000.0,
        soft_max=10000.0,
    )

    fy: FloatProperty(  # type: ignore
        name="Fy",
        default=0.0,
        min=-10000.0,
        max=10000.0,
        soft_min=-10000.0,
        soft_max=10000.0,
    )

    fz: FloatProperty(  # type: ignore
        name="Fz",
        default=0.0,
        min=-10000.0,
        max=10000.0,
        soft_min=-10000.0,
        soft_max=10000.0,
    )

    @classmethod
    def poll(cls, ntree):
        """Return whether the node can be added to the given node tree."""
        return ntree.bl_idname == BlenderCFDNodeTree.bl_idname

    def _ensure_output_socket(self):
        """Ensure that the placeholder output socket exists."""
        socket = self.outputs.get("Force")
        if socket is None:
            socket = self.outputs.new(BlenderCFDForceSocket.bl_idname, "Force")
        return socket

    def _sync_output_socket(self):
        """Refresh the node socket layout."""
        self._ensure_output_socket()

    def init(self, context):
        """Initialize sockets when the node is created."""
        self._sync_output_socket()

    def copy(self, node):
        """Restore sockets after the node has been copied."""
        self._sync_output_socket()

    def update(self):
        """Keep the node socket layout in sync."""
        self._sync_output_socket()

    def draw_buttons(self, context, layout):
        """Draw the editable controls shown inside the node body."""
        col = layout.column(align=True)
        col.prop(self, "fx")
        col.prop(self, "fy")
        col.prop(self, "fz")


class BlenderCFDForcePointNode(bpy.types.Node):
    """
    Node used to define a point force at a given position in space.

    The node stores one signed force strength and a 3D position and exposes a
    placeholder output socket for future graph connections.
    """

    bl_idname = "BLENDERCFD_FORCE_POINT_NODE"
    bl_label = "Force Point"
    bl_icon = "EMPTY_ARROWS"
    bl_width_default = 220.0
    bl_width_min = 200.0
    bl_width_max = 340.0

    strength: FloatProperty(  # type: ignore
        name="Strength",
        default=0.0,
        min=-10000.0,
        max=10000.0,
        soft_min=-10000.0,
        soft_max=10000.0,
    )

    position: FloatVectorProperty(  # type: ignore
        name="Position",
        size=3,
        subtype="XYZ",
        default=(0.0, 0.0, 0.0),
        unit="LENGTH",
    )

    @classmethod
    def poll(cls, ntree):
        """Return whether the node can be added to the given node tree."""
        return ntree.bl_idname == BlenderCFDNodeTree.bl_idname

    def _ensure_output_socket(self):
        """Ensure that the placeholder output socket exists."""
        socket = self.outputs.get("Force")
        if socket is None:
            socket = self.outputs.new(BlenderCFDForceSocket.bl_idname, "Force")
        return socket

    def _sync_output_socket(self):
        """Refresh the node socket layout."""
        self._ensure_output_socket()

    def init(self, context):
        """Initialize sockets when the node is created."""
        self._sync_output_socket()

    def copy(self, node):
        """Restore sockets after the node has been copied."""
        self._sync_output_socket()

    def update(self):
        """Keep the node socket layout in sync."""
        self._sync_output_socket()

    def draw_buttons(self, context, layout):
        """Draw the editable controls shown inside the node body."""
        col = layout.column(align=True)
        col.prop(self, "strength")
        col.prop(self, "position")


class BlenderCFDForceTurbulenceNode(bpy.types.Node):
    """
    Node used to define a procedural turbulence force field.

    The node stores scale, frequency, and amplitude parameters and exposes a
    placeholder output socket for future graph connections.
    """

    bl_idname = "BLENDERCFD_FORCE_TURBULENCE_NODE"
    bl_label = "Force Turbulence"
    bl_icon = "FORCE_TURBULENCE"
    bl_width_default = 220.0
    bl_width_min = 200.0
    bl_width_max = 340.0

    scale: FloatProperty(  # type: ignore
        name="Scale",
        default=1.0,
        min=0.0,
    )

    frequency: FloatProperty(  # type: ignore
        name="Frequency",
        default=1.0,
        min=0.0,
    )

    amplitude: FloatProperty(  # type: ignore
        name="Amplitude",
        default=1.0,
        min=0.0,
    )

    @classmethod
    def poll(cls, ntree):
        """Return whether the node can be added to the given node tree."""
        return ntree.bl_idname == BlenderCFDNodeTree.bl_idname

    def _ensure_output_socket(self):
        """Ensure that the placeholder output socket exists."""
        socket = self.outputs.get("Force")
        if socket is None:
            socket = self.outputs.new(BlenderCFDForceSocket.bl_idname, "Force")
        return socket

    def _sync_output_socket(self):
        """Refresh the node socket layout."""
        self._ensure_output_socket()

    def init(self, context):
        """Initialize sockets when the node is created."""
        self._sync_output_socket()

    def copy(self, node):
        """Restore sockets after the node has been copied."""
        self._sync_output_socket()

    def update(self):
        """Keep the node socket layout in sync."""
        self._sync_output_socket()

    def draw_buttons(self, context, layout):
        """Draw the editable controls shown inside the node body."""
        col = layout.column(align=True)
        col.prop(self, "scale")
        col.prop(self, "frequency")
        col.prop(self, "amplitude")


class BlenderCFDObstacleNode(bpy.types.Node):
    """
    Node used to define obstacle geometry inside the CFD domain.

    The node accepts geometry as an input selection and currently exposes a
    placeholder output socket for future graph connections.
    """

    bl_idname = "BLENDERCFD_OBSTACLE_NODE"
    bl_label = "Obstacle"
    bl_icon = "CUBE"
    bl_width_default = 220.0
    bl_width_min = 200.0
    bl_width_max = 360.0

    @classmethod
    def poll(cls, ntree):
        """Return whether the node can be added to the given node tree."""
        return ntree.bl_idname == BlenderCFDNodeTree.bl_idname

    def _ensure_geometry_input(self):
        """Ensure that the geometry input socket exists."""
        socket = self.inputs.get("Geometry")
        if socket is None:
            socket = self.inputs.new("NodeSocketGeometry", "Geometry", use_multi_input=True)
        if hasattr(socket, "link_limit"):
            socket.link_limit = 0
        return socket

    def _ensure_output_socket(self, name):
        """Ensure that the placeholder output socket exists."""
        socket = self.outputs.get(name)
        if socket is None:
            socket = self.outputs.new(BlenderCFDIntSocket.bl_idname, name)
        return socket

    def _sync_sockets(self):
        """Refresh the node socket layout."""
        self._ensure_geometry_input()
        self._ensure_output_socket("Obstacle")

    def init(self, context):
        """Initialize sockets when the node is created."""
        self._sync_sockets()

    def copy(self, node):
        """Restore sockets after the node has been copied."""
        self._sync_sockets()

    def update(self):
        """Keep the node socket layout in sync."""
        self._sync_sockets()


classes = (
    BlenderCFDIntSocket,
    BlenderCFDLinkSocket,
    BlenderCFDForceSocket,
    BlenderCFDReferenceFrameSocket,
    BlenderCFDResultSocket,
    BlenderCFD_OT_bake,
    BlenderCFDDomainNode,
    BlenderCFDGeometryNode,
    BlenderCFDForceConstantNode,
    BlenderCFDForcePointNode,
    BlenderCFDForceTurbulenceNode,
    BlenderCFDOutputNode,
    BlenderCFDPhysicsNode,
    BlenderCFDReferenceFrameNode,
    BlenderCFDSimulationNode,
    BlenderCFDSourceNode,
    BlenderCFDObstacleNode,
    BlenderCFDViewerNode,
)
