import bpy
from .node_tree import ContinuumFlowNodeTree

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
        layout.enabled = True

    def _draw_group(self, layout, title, property_names):
        box = layout.box()
        box.label(text=title)
        col = box.column(align=True)
        for property_name in property_names:
            col.prop(self, property_name)