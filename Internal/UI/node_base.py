import bpy
from .node_tree import ContinuumFlowNodeTree
from ..Core.solver import solver_status

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
        layout.enabled = not solver_status.bake_running

    def _draw_group(self, layout, title, property_names):
        box = layout.box()
        box.label(text=title)
        col = box.column(align=True)
        for property_name in property_names:
            col.prop(self, property_name)

    def _ensure_socket(self, collection, socket_type, name, multi_input=False):
        """
        Return a socket, creating it on demand with optional multi-input support.
        """
        socket = collection.get(name)
        if socket is None:
            socket = collection.new(socket_type, name, use_multi_input=multi_input)
        if multi_input and hasattr(socket, "link_limit"):
            socket.link_limit = 0
        return socket

    def _ensure_named_output(self, socket_type, name):
        """
        Ensure that this node exposes one named output socket of the given type.
        """
        return self._ensure_socket(self.outputs, socket_type, name)

    def _ensure_geometry_input(self):
        """
        Ensure that this node exposes the standard multi-input geometry socket.
        """
        return self._ensure_socket(
            self.inputs, "NodeSocketGeometry", "Geometry", multi_input=True
        )
