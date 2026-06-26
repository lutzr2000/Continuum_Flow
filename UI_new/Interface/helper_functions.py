

def ensure_socket(collection, socket_type, name, multi_input=False):
    """
    Return a socket, creating it on demand with optional multi-input support.
    """
    socket = collection.get(name)
    if socket is None:
        socket = collection.new(socket_type, name, use_multi_input=multi_input)
    if multi_input and hasattr(socket, "link_limit"):
        socket.link_limit = 0
    return socket


def ensure_named_output(node, socket_type, name):
    """
    Ensure that a node exposes one named output socket of the given type.
    """
    return ensure_socket(node.outputs, socket_type, name)


def ensure_geometry_input(node):
    """
    Ensure that a node exposes the standard multi-input geometry socket.
    """
    return ensure_socket(
        node.inputs, "NodeSocketGeometry", "Geometry", multi_input=True
    )