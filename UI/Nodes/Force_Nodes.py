import bpy
import blendercfd_general_nodes as GeneralNodes
from bpy.props import FloatProperty, FloatVectorProperty

BlenderCFDForceSocket = GeneralNodes.BlenderCFDForceSocket
BlenderCFDNodeTree = GeneralNodes.BlenderCFDNodeTree
is_bake_running = GeneralNodes.is_bake_running


class BlenderCFDForceConstantNode(bpy.types.Node):
    """Node used to define a constant force vector for the CFD simulation."""

    bl_idname = "BLENDERCFD_FORCE_CONSTANT_NODE"
    bl_label = "Force Constant"
    bl_icon = "FORCE_FORCE"
    bl_width_default = 220.0
    bl_width_min = 200.0
    bl_width_max = 320.0

    fx: FloatProperty(name="Fx", default=0.0, min=-10000.0, max=10000.0, soft_min=-10000.0, soft_max=10000.0)  # type: ignore
    fy: FloatProperty(name="Fy", default=0.0, min=-10000.0, max=10000.0, soft_min=-10000.0, soft_max=10000.0)  # type: ignore
    fz: FloatProperty(name="Fz", default=0.0, min=-10000.0, max=10000.0, soft_min=-10000.0, soft_max=10000.0)  # type: ignore

    @classmethod
    def poll(cls, ntree):
        return ntree.bl_idname == BlenderCFDNodeTree.bl_idname

    def _ensure_output_socket(self):
        socket = self.outputs.get("Force")
        if socket is None:
            socket = self.outputs.new(BlenderCFDForceSocket.bl_idname, "Force")
        return socket

    def _sync_output_socket(self):
        self._ensure_output_socket()

    def init(self, context):
        self._sync_output_socket()

    def copy(self, node):
        self._sync_output_socket()

    def update(self):
        self._sync_output_socket()

    def draw_buttons(self, context, layout):
        layout.enabled = not is_bake_running(context)
        col = layout.column(align=True)
        col.prop(self, "fx")
        col.prop(self, "fy")
        col.prop(self, "fz")


class BlenderCFDForceSwirlNode(bpy.types.Node):
    """Node used to define a swirl force around an origin and axis."""

    bl_idname = "BLENDERCFD_FORCE_SWIRL_NODE"
    bl_label = "Force Swirl"
    bl_icon = "EMPTY_ARROWS"
    bl_width_default = 220.0
    bl_width_min = 200.0
    bl_width_max = 340.0

    strength: FloatProperty(name="Strength", default=0.0, min=-10000.0, max=10000.0, soft_min=-10000.0, soft_max=10000.0)  # type: ignore
    origin: FloatVectorProperty(name="Origin", size=3, subtype="XYZ", default=(0.0, 0.0, 0.0), unit="LENGTH")  # type: ignore
    axis: FloatVectorProperty(name="Axis", size=3, subtype="XYZ", default=(0.0, 0.0, 1.0))  # type: ignore
    radius: FloatProperty(name="Radius", default=1.0, min=0.0, unit="LENGTH")  # type: ignore

    @classmethod
    def poll(cls, ntree):
        return ntree.bl_idname == BlenderCFDNodeTree.bl_idname

    def _ensure_output_socket(self):
        socket = self.outputs.get("Force")
        if socket is None:
            socket = self.outputs.new(BlenderCFDForceSocket.bl_idname, "Force")
        return socket

    def _sync_output_socket(self):
        self._ensure_output_socket()

    def init(self, context):
        self._sync_output_socket()

    def copy(self, node):
        self._sync_output_socket()

    def update(self):
        self._sync_output_socket()

    def draw_buttons(self, context, layout):
        layout.enabled = not is_bake_running(context)
        col = layout.column(align=True)
        col.prop(self, "strength")
        col.prop(self, "origin")
        col.prop(self, "axis")
        col.prop(self, "radius")


class BlenderCFDForceTurbulenceNode(bpy.types.Node):
    """Node used to define a procedural turbulence force field."""

    bl_idname = "BLENDERCFD_FORCE_TURBULENCE_NODE"
    bl_label = "Force Turbulence"
    bl_icon = "FORCE_TURBULENCE"
    bl_width_default = 220.0
    bl_width_min = 200.0
    bl_width_max = 340.0

    scale: FloatProperty(name="Scale", default=1.0, min=0.0)  # type: ignore
    frequency: FloatProperty(name="Frequency", default=1.0, min=0.0)  # type: ignore
    amplitude: FloatProperty(name="Amplitude", default=1.0, min=0.0)  # type: ignore
    seed: bpy.props.IntProperty(name="Seed", default=0)  # type: ignore

    @classmethod
    def poll(cls, ntree):
        return ntree.bl_idname == BlenderCFDNodeTree.bl_idname

    def _ensure_output_socket(self):
        socket = self.outputs.get("Force")
        if socket is None:
            socket = self.outputs.new(BlenderCFDForceSocket.bl_idname, "Force")
        return socket

    def _sync_output_socket(self):
        self._ensure_output_socket()

    def init(self, context):
        self._sync_output_socket()

    def copy(self, node):
        self._sync_output_socket()

    def update(self):
        self._sync_output_socket()

    def draw_buttons(self, context, layout):
        layout.enabled = not is_bake_running(context)
        col = layout.column(align=True)
        col.prop(self, "scale")
        col.prop(self, "frequency")
        col.prop(self, "amplitude")
        col.prop(self, "seed")


classes = (
    BlenderCFDForceConstantNode,
    BlenderCFDForceSwirlNode,
    BlenderCFDForceTurbulenceNode,
)
