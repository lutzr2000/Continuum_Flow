import bpy
import blendercfd_general_nodes as GeneralNodes
from bpy.props import FloatProperty, PointerProperty

BlenderCFDIntSocket = GeneralNodes.BlenderCFDIntSocket
BlenderCFDNodeTree = GeneralNodes.BlenderCFDNodeTree


class BlenderCFDSourceNode(bpy.types.Node):
    """Node used to define a generic CFD source region and its scalar strengths."""

    bl_idname = "BLENDERCFD_SOURCE_NODE"
    bl_label = "Source"
    bl_icon = "LIGHT_SUN"
    bl_width_default = 220.0
    bl_width_min = 200.0
    bl_width_max = 360.0

    fuel: FloatProperty(name="Fuel", default=0.0, min=0.0, max=10000.0, soft_min=0.0, soft_max=10000.0)  # type: ignore
    smoke: FloatProperty(name="Smoke", default=0.0, min=0.0, max=10000.0, soft_min=0.0, soft_max=10000.0)  # type: ignore
    temperature: FloatProperty(name="Temperature", default=0.0, min=0.0, max=10000.0, soft_min=0.0, soft_max=10000.0, unit="TEMPERATURE")  # type: ignore

    @classmethod
    def poll(cls, ntree):
        return ntree.bl_idname == BlenderCFDNodeTree.bl_idname

    def _ensure_geometry_input(self):
        socket = self.inputs.get("Geometry")
        if socket is None:
            socket = self.inputs.new("NodeSocketGeometry", "Geometry", use_multi_input=True)
        if hasattr(socket, "link_limit"):
            socket.link_limit = 0
        return socket

    def _ensure_output_socket(self, name):
        socket = self.outputs.get(name)
        if socket is None:
            socket = self.outputs.new(BlenderCFDIntSocket.bl_idname, name)
        return socket

    def _sync_sockets(self):
        self._ensure_geometry_input()
        self._ensure_output_socket("Source")

    def init(self, context):
        self._sync_sockets()

    def copy(self, node):
        self._sync_sockets()

    def update(self):
        self._sync_sockets()

    def draw_buttons(self, context, layout):
        col = layout.column(align=True)
        col.prop(self, "fuel")
        col.prop(self, "smoke")
        col.prop(self, "temperature")


class BlenderCFDGeometryNode(bpy.types.Node):
    """Node used to reference a Blender object as geometry inside the CFD graph."""

    bl_idname = "BLENDERCFD_GEOMETRY_NODE"
    bl_label = "Geometry"
    bl_icon = "OUTLINER_OB_MESH"
    bl_width_default = 220.0
    bl_width_min = 200.0
    bl_width_max = 360.0

    source_object: PointerProperty(name="Object", type=bpy.types.Object)  # type: ignore

    @classmethod
    def poll(cls, ntree):
        return ntree.bl_idname == BlenderCFDNodeTree.bl_idname

    def _ensure_output_socket(self):
        socket = self.outputs.get("Geometry")
        if socket is None:
            socket = self.outputs.new("NodeSocketGeometry", "Geometry")
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
        layout.prop(self, "source_object", text="Object")


class BlenderCFDObstacleNode(bpy.types.Node):
    """Node used to define obstacle geometry inside the CFD domain."""

    bl_idname = "BLENDERCFD_OBSTACLE_NODE"
    bl_label = "Obstacle"
    bl_icon = "CUBE"
    bl_width_default = 220.0
    bl_width_min = 200.0
    bl_width_max = 360.0

    @classmethod
    def poll(cls, ntree):
        return ntree.bl_idname == BlenderCFDNodeTree.bl_idname

    def _ensure_geometry_input(self):
        socket = self.inputs.get("Geometry")
        if socket is None:
            socket = self.inputs.new("NodeSocketGeometry", "Geometry", use_multi_input=True)
        if hasattr(socket, "link_limit"):
            socket.link_limit = 0
        return socket

    def _ensure_output_socket(self, name):
        socket = self.outputs.get(name)
        if socket is None:
            socket = self.outputs.new(BlenderCFDIntSocket.bl_idname, name)
        return socket

    def _sync_sockets(self):
        self._ensure_geometry_input()
        self._ensure_output_socket("Obstacle")

    def init(self, context):
        self._sync_sockets()

    def copy(self, node):
        self._sync_sockets()

    def update(self):
        self._sync_sockets()


classes = (
    BlenderCFDSourceNode,
    BlenderCFDGeometryNode,
    BlenderCFDObstacleNode,
)
