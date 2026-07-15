from . import sockets
from . import node_base

class ContinuumFlowObstacleNode(node_base.ContinuumFlowBaseNode):
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
        self._ensure_geometry_input()
        self._ensure_named_output(sockets.ContinuumFlowIntSocket.bl_idname, "Obstacle")