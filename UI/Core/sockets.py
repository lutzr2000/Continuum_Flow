import bpy
from bpy.props import IntProperty

_INVALID_SOCKET_COLOR = (0.95, 0.20, 0.20, 1.0)

_SOCKET_ROLE_BY_NODE_AND_NAME = {
    ("CONTINUUM_FLOW_DOMAIN_NODE", "Domain"): "domain",
    ("CONTINUUM_FLOW_PHYSICS_NODE", "Physics"): "physics",
    ("CONTINUUM_FLOW_OBSTACLE_NODE", "Obstacle"): "obstacles",
    ("CONTINUUM_FLOW_SOURCE_NODE", "Source"): "source",
    ("CONTINUUM_FLOW_SIMULATION_NODE", "Domain"): "domain",
    ("CONTINUUM_FLOW_SIMULATION_NODE", "Physics"): "physics",
    ("CONTINUUM_FLOW_SIMULATION_NODE", "Obstacles"): "obstacles",
    ("CONTINUUM_FLOW_SIMULATION_NODE", "Source"): "source",
    ("CONTINUUM_FLOW_SIMULATION_NODE", "Forces"): "forces",
    ("CONTINUUM_FLOW_SIMULATION_NODE", "Result"): "result",
    ("CONTINUUM_FLOW_OUTPUT_NODE", "Result"): "result",
    ("CONTINUUM_FLOW_VIEWER_NODE", "Result"): "result",
    ("CONTINUUM_FLOW_GEOMETRY_NODE", "Geometry"): "geometry",
    ("CONTINUUM_FLOW_SOURCE_NODE", "Geometry"): "geometry",
    ("CONTINUUM_FLOW_OBSTACLE_NODE", "Geometry"): "geometry",
}

_SOCKET_ROLE_BY_SOCKET_IDNAME = {
    "CONTINUUM_FLOW_FORCE_SOCKET": "forces",
    "CONTINUUM_FLOW_RESULT_SOCKET": "result",
    "NodeSocketGeometry": "geometry",
}


def _socket_role(socket):
    """
    Return the semantic Continuum Flow role associated with a socket.
    """
    node = getattr(socket, "node", None)
    node_idname = getattr(node, "bl_idname", "")
    role = _SOCKET_ROLE_BY_NODE_AND_NAME.get((node_idname, getattr(socket, "name", "")))
    if role is not None:
        return role
    return _SOCKET_ROLE_BY_SOCKET_IDNAME.get(getattr(socket, "bl_idname", ""))


def _is_valid_link(link):
    """
    Return whether a node link connects sockets that share the same role.
    """
    from_socket = getattr(link, "from_socket", None)
    to_socket = getattr(link, "to_socket", None)
    if from_socket is None or to_socket is None:
        return True

    from_role = _socket_role(from_socket)
    to_role = _socket_role(to_socket)
    if from_role is None or to_role is None:
        return True
    return from_role == to_role


def _socket_has_invalid_links(socket):
    """
    Return whether a socket participates in any invalid Continuum Flow links.
    """
    try:
        return any(not _is_valid_link(link) for link in socket.links)
    except Exception:
        return False


def tree_has_invalid_links(node_tree):
    """
    Return whether a node tree contains any invalid Continuum Flow links.
    """
    if node_tree is None:
        return False
    try:
        return any(not _is_valid_link(link) for link in node_tree.links)
    except Exception:
        return False


class ContinuumFlowIntSocket(bpy.types.NodeSocket):
    """
    Integer socket used by Continuum Flow nodes to expose bounded scalar values.
    """

    bl_idname = "CONTINUUM_FLOW_INT_SOCKET"
    bl_label = "Continuum Flow Integer"

    value: IntProperty(  # type: ignore
        name="Value",
        default=128,
        min=32,
        max=4096,
        soft_min=32,
        soft_max=4096,
    )

    def draw(self, context, layout, node, text):
        """
        Draw the socket UI in the node editor.
        """
        if self.is_output or self.is_linked:
            layout.label(text=text)
        else:
            layout.prop(self, "value", text=text)

    def draw_color(self, context, node):
        """
        Return the display color of the socket.
        """
        if _socket_has_invalid_links(self):
            return _INVALID_SOCKET_COLOR
        return (0.90, 0.55, 0.20, 1.0)


class ContinuumFlowLinkSocket(bpy.types.NodeSocket):
    """
    Generic link socket used to connect logical Continuum Flow node outputs.
    """

    bl_idname = "CONTINUUM_FLOW_LINK_SOCKET"
    bl_label = "Continuum Flow Link"

    def draw(self, context, layout, node, text):
        """
        Draw the socket label in the node editor.
        """
        layout.label(text=text)

    def draw_color(self, context, node):
        """
        Return the display color of the socket.
        """
        if _socket_has_invalid_links(self):
            return _INVALID_SOCKET_COLOR
        return (0.90, 0.55, 0.20, 1.0)


class ContinuumFlowForceSocket(bpy.types.NodeSocket):
    """
    Dedicated socket used for force-related links in the Continuum Flow graph.
    """

    bl_idname = "CONTINUUM_FLOW_FORCE_SOCKET"
    bl_label = "Continuum Flow Force"

    def draw(self, context, layout, node, text):
        """
        Draw the socket label in the node editor.
        """
        layout.label(text=text)

    def draw_color(self, context, node):
        """
        Return the display color of the socket.
        """
        if _socket_has_invalid_links(self):
            return _INVALID_SOCKET_COLOR
        return (0.45, 0.65, 0.95, 1.0)


class ContinuumFlowResultSocket(bpy.types.NodeSocket):
    """
    Result socket used by the simulation node to expose the final output link.
    """

    bl_idname = "CONTINUUM_FLOW_RESULT_SOCKET"
    bl_label = "Continuum Flow Result"

    def draw(self, context, layout, node, text):
        """
        Draw the socket label in the node editor.
        """
        layout.label(text=text)

    def draw_color(self, context, node):
        """
        Return the display color of the socket.
        """
        if _socket_has_invalid_links(self):
            return _INVALID_SOCKET_COLOR
        return (0.65, 0.35, 0.85, 1.0)


classes = (
    ContinuumFlowIntSocket,
    ContinuumFlowLinkSocket,
    ContinuumFlowForceSocket,
    ContinuumFlowResultSocket,
)
