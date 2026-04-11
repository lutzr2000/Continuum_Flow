import bpy
import importlib.util
import sys
from pathlib import Path
from bpy.props import EnumProperty, FloatVectorProperty, IntProperty

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
    bl_width_default = 260.0
    bl_width_min = 220.0
    bl_width_max = 420.0

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


classes = (
    BlenderCFDIntSocket,
    BlenderCFDDomainNode,
)
