import bpy
import importlib.util
import sys
from pathlib import Path
from bpy.props import EnumProperty, FloatProperty, FloatVectorProperty, IntProperty, PointerProperty

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


class BlenderCFDIntSocket(bpy.types.NodeSocket):
    """
    Integer socket used by BlenderCFD nodes to expose bounded scalar values.
    """

    bl_idname = "BLENDERCFD_INT_SOCKET"
    bl_label = "BlenderCFD Integer"

    value: IntProperty(  # type: ignore
        name="Value",
        default=128,
        min=32,
        max=4096,
        soft_min=32,
        soft_max=4096,
    )

    def draw(self, context, layout, node, text):
        """Draw the socket UI in the node editor."""
        if self.is_output or self.is_linked:
            layout.label(text=text)
        else:
            layout.prop(self, "value", text=text)

    def draw_color(self, context, node):
        """Return the display color of the socket."""
        return (0.45, 0.65, 0.95, 1.0)


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
    )

    x_high_velocity: FloatVectorProperty(  # type: ignore
        name="Velocity",
        size=3,
        subtype="XYZ",
        default=(0.0, 0.0, 0.0),
    )

    y_low_velocity: FloatVectorProperty(  # type: ignore
        name="Velocity",
        size=3,
        subtype="XYZ",
        default=(0.0, 0.0, 0.0),
    )

    y_high_velocity: FloatVectorProperty(  # type: ignore
        name="Velocity",
        size=3,
        subtype="XYZ",
        default=(0.0, 0.0, 0.0),
    )

    z_low_velocity: FloatVectorProperty(  # type: ignore
        name="Velocity",
        size=3,
        subtype="XYZ",
        default=(0.0, 0.0, 0.0),
    )

    z_high_velocity: FloatVectorProperty(  # type: ignore
        name="Velocity",
        size=3,
        subtype="XYZ",
        default=(0.0, 0.0, 0.0),
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
        max=2000
    )

    fluid_viscosity: FloatProperty(  # type: ignore
        name="Fluid Viscosity",
        default=1.81e-5,
        min=0.0,
        max=1
    )

    temperature_diffusion: FloatProperty(  # type: ignore
        name="Temperature Diffusion",
        default=0.1,
        min=0.0,
        max=1
    )

    temperature_dissipation: FloatProperty(  # type: ignore
        name="Temperature Dissipation",
        default=1.0,
        min=0.0,
        max=100
    )

    reference_temperature: FloatProperty(  # type: ignore
        name="Reference Temperature",
        default=300.0,
        min=0.0,
        max=10000.0
    )

    buoyancy: FloatProperty(  # type: ignore
        name="Buoyancy",
        default=0.005,
        min=0.0,
        max=1.0
    )

    expansion_rate: FloatProperty(  # type: ignore
        name="Expansion Rate",
        default=0.001,
        min=0.0,
        max=1.0
    )

    smoke_diffusion: FloatProperty(  # type: ignore
        name="Smoke Diffusion",
        default=0.1,
        min=0.0,
        max=1.0
    )

    smoke_dissipation: FloatProperty(  # type: ignore
        name="Smoke Dissipation",
        default=1.0,
        min=0.0,
        max=100.0
    )

    fuel_diffusion: FloatProperty(  # type: ignore
        name="Fuel Diffusion",
        default=0.1,
        min=0.0,
        max=1.0
    )

    fuel_dissipation: FloatProperty(  # type: ignore
        name="Fuel Dissipation",
        default=1.0,
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
        max=10000.0
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
            socket = self.outputs.new(BlenderCFDIntSocket.bl_idname, "Force")
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
    )

    @classmethod
    def poll(cls, ntree):
        """Return whether the node can be added to the given node tree."""
        return ntree.bl_idname == BlenderCFDNodeTree.bl_idname

    def _ensure_output_socket(self):
        """Ensure that the placeholder output socket exists."""
        socket = self.outputs.get("Force")
        if socket is None:
            socket = self.outputs.new(BlenderCFDIntSocket.bl_idname, "Force")
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
            socket = self.outputs.new(BlenderCFDIntSocket.bl_idname, "Force")
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
    BlenderCFDDomainNode,
    BlenderCFDGeometryNode,
    BlenderCFDForceConstantNode,
    BlenderCFDForcePointNode,
    BlenderCFDForceTurbulenceNode,
    BlenderCFDPhysicsNode,
    BlenderCFDSourceNode,
    BlenderCFDObstacleNode,
)
