import bpy
import gpu
from gpu_extras.batch import batch_for_shader

draw_handler = None
current_drawn_domain = None


class ContinuumFlow_OT_viewer_toggle_domain(bpy.types.Operator):
    """
    Toggle the linked domain preview in the 3D viewport.
    """

    bl_idname = "continuum_flow.viewer_toggle_domain"
    bl_label = "Toggle Domain Preview"
    bl_description = "Toggle the linked Continuum Flow domain viewport preview"

    def execute(self, context):
        node = getattr(context, "node")

        domain_node = get_linked_domain_node(node)
        node_tree = getattr(domain_node, "id_data", None)
        if domain_node is None:
            self.report({"ERROR"}, "Connect Domain -> Simulation -> Viewer first.")
            return {"CANCELLED"}

        if domain_preview_enabled(domain_node):
            disable_domain_preview()
            self.report({"INFO"}, "Domain preview disabled.")
        else:
            enable_domain_preview(domain_node,node_tree)
            self.report({"INFO"}, "Domain preview enabled.")
        return {"FINISHED"}


classes = (ContinuumFlow_OT_viewer_toggle_domain,)


def get_linked_domain_node(node):
    """
    Resolve the domain node by traversing the viewer's Result input
    through the connected simulation node.
    """
    result_socket = node.inputs.get("Result")

    for link in result_socket.links:
        simulation_node = getattr(link, "from_node", None)
        domain_socket = simulation_node.inputs.get("Domain")
        for domain_link in domain_socket.links:
            domain_node = getattr(domain_link, "from_node", None)
            if getattr(domain_node, "bl_idname", "") == "CONTINUUM_FLOW_DOMAIN_NODE":
                return domain_node


def enable_domain_preview(domain_node, node_tree):
    """
    Enable the preview for the given domain node.
    """
    global draw_handler, current_drawn_domain

    current_drawn_domain = {
        "node_tree_name": str(getattr(node_tree, "name", "")),
        "node_name": str(getattr(domain_node, "name", "")),
    }

    if draw_handler is None:
        draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            draw_preview, (), "WINDOW", "POST_VIEW"
        )

    redraw_viewport()
    redraw_UI()


def disable_domain_preview():
    """
    Disable any active domain preview.
    """
    global draw_handler, current_drawn_domain

    current_drawn_domain = None
    if draw_handler is not None:
        bpy.types.SpaceView3D.draw_handler_remove(draw_handler, "WINDOW")
        draw_handler = None
    redraw_viewport()
    redraw_UI()


def redraw_viewport():
    """
    Request a redraw for all visible 3D viewports.
    """
    window_manager = getattr(bpy.context, "window_manager")

    for window in window_manager.windows:
        screen = getattr(window, "screen")
        for area in screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def redraw_UI():
    """
    Redraws the UI fields
    """
    window_manager = getattr(bpy.context, "window_manager", None)

    for window in window_manager.windows:
        screen = getattr(window, "screen")
        for area in screen.areas:
            if area.type in {"NODE_EDITOR", "PROPERTIES"}:
                area.tag_redraw()


def draw_preview():
    """
    Draw the active domain preview in the 3D viewport.
    """
    node_tree = bpy.data.node_groups.get(current_drawn_domain.get("node_tree_name", ""))
    active_domain = node_tree.nodes.get(current_drawn_domain.get("node_name", ""))

    if not overlay_enabeld():
        return

    try:
        domain_lines, cell_lines = build_segments(active_domain)
    except Exception:
        disable_domain_preview()
        return

    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    gpu.state.blend_set("ALPHA")
    gpu.state.line_width_set(2.0)

    shader.bind()
    shader.uniform_float("color", (1.0, 0.65, 0.20, 1.0))
    batch_for_shader(shader, "LINES", {"pos": domain_lines}).draw(shader)

    shader.uniform_float("color", (0.20, 0.85, 1.0, 1.0))
    batch_for_shader(shader, "LINES", {"pos": cell_lines}).draw(shader)

    gpu.state.line_width_set(1.0)
    gpu.state.blend_set("NONE")


def overlay_enabeld():
    """
    Return whether overlays are enabled in the currently drawing 3D view.
    """
    space_data = getattr(bpy.context, "space_data", None)
    if space_data is None or getattr(space_data, "type", "") != "VIEW_3D":
        return True
    overlay = getattr(space_data, "overlay", None)
    if overlay is None:
        return True
    return bool(getattr(overlay, "show_overlays", True))


def build_segments(domain_node):
    """
    Build the preview line data for the full domain and one sample cell.
    """
    resolution = float(domain_node.resolution)

    width = float(domain_node.nx) * resolution
    depth = float(domain_node.ny) * resolution
    height = float(domain_node.nz) * resolution

    min_corner = (
        -(width * 0.5),
        -(depth * 0.5),
        0.0,
    )
    max_corner = (
        min_corner[0] + width,
        min_corner[1] + depth,
        min_corner[2] + height,
    )

    domain_lines = box_segments(min_corner, max_corner)
    cell_lines = box_segments(
        min_corner,
        (
            min_corner[0] + min(resolution, width),
            min_corner[1] + min(resolution, depth),
            min_corner[2] + min(resolution, height),
        ),
    )

    return domain_lines, cell_lines


def box_segments(min_corner, max_corner):
    """
    Build the line segments for a wireframe box.
    """
    x0, y0, z0 = min_corner
    x1, y1, z1 = max_corner

    corners = (
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    )
    edges = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )

    segments = []
    for start_index, end_index in edges:
        segments.extend((corners[start_index], corners[end_index]))
    return segments


def domain_preview_enabled(domain_node):
    """
    Return whether the preview is active for the given domain node.
    """
    if not current_drawn_domain or domain_node is None:
        return False

    node_tree = bpy.data.node_groups.get(
        current_drawn_domain.get("node_tree_name", "")
    )
    if node_tree is None:
        return False

    active_domain = node_tree.nodes.get(
        current_drawn_domain.get("node_name", "")
    )
    if active_domain is None:
        return False

    active_node_tree = getattr(active_domain, "id_data", None)
    target_node_tree = getattr(domain_node, "id_data", None)

    if active_node_tree is None or target_node_tree is None:
        return False

    active_reference = {
        "node_tree_name": str(getattr(active_node_tree, "name", "")),
        "node_name": str(getattr(active_domain, "name", "")),
    }
    target_reference = {
        "node_tree_name": str(getattr(target_node_tree, "name", "")),
        "node_name": str(getattr(domain_node, "name", "")),
    }

    return active_reference == target_reference


def viewer_domain_preview_enabled(viewer_node):
    """
    Return whether the given viewer node currently controls the active domain preview.
    """
    domain_node = get_linked_domain_node(viewer_node)
    return domain_preview_enabled(domain_node)
