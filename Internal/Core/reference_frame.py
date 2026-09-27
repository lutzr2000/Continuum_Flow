"""Resolve and apply the UI-only simulation reference-frame transform."""

from mathutils import Matrix, Vector


def linked_reference_frame_node(simulation_node):
    if simulation_node is None:
        return None
    socket = simulation_node.inputs.get("Reference Frame")
    if socket is None or not socket.is_linked:
        return None
    for link in socket.links:
        node = getattr(link, "from_node", None)
        if getattr(node, "bl_idname", "") == "CONTINUUM_FLOW_REFERENCE_FRAME_NODE":
            return node
    return None


def linked_reference_object(simulation_node):
    node = linked_reference_frame_node(simulation_node)
    source_object = getattr(node, "source_object", None)
    if getattr(source_object, "type", None) in {"MESH", "EMPTY"}:
        return source_object
    return None


def display_matrix(simulation_node):
    source_object = linked_reference_object(simulation_node)
    if source_object is None:
        return Matrix.Identity(4)
    return source_object.matrix_world.copy()


def transform_positions(positions, simulation_node):
    matrix = display_matrix(simulation_node)
    return [tuple(matrix @ Vector(position)) for position in positions]
