import bpy
import gpu
from gpu_extras.batch import batch_for_shader


_DRAW_HANDLER = None
_ACTIVE_DOMAIN = None
_DOMAIN_FLOOR_CENTER = (0.0, 9.0, 9.0)


def _tag_view3d_redraw():
    """Request a redraw for all visible 3D viewports."""
    window_manager = getattr(bpy.context, "window_manager", None)
    if window_manager is None:
        return

    for window in window_manager.windows:
        screen = getattr(window, "screen", None)
        if screen is None:
            continue
        for area in screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def _domain_dimensions(domain_node):
    """Return the world-space domain dimensions derived from the node settings."""
    return (
        float(domain_node.nx) * float(domain_node.resolution),
        float(domain_node.ny) * float(domain_node.resolution),
        float(domain_node.nz) * float(domain_node.resolution),
    )


def _box_segments(min_corner, max_corner):
    """Build the line segments for a wireframe box."""
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
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )

    segments = []
    for start_index, end_index in edges:
        segments.extend((corners[start_index], corners[end_index]))
    return segments


def _build_preview_segments(domain_node):
    """Build the preview line data for the full domain and one sample cell."""
    width, depth, height = _domain_dimensions(domain_node)
    resolution = float(domain_node.resolution)
    center_x, center_y, floor_z = _DOMAIN_FLOOR_CENTER

    min_corner = (
        center_x - (width * 0.5),
        center_y - (depth * 0.5),
        floor_z,
    )
    max_corner = (
        center_x + (width * 0.5),
        center_y + (depth * 0.5),
        floor_z + height,
    )

    domain_lines = _box_segments(min_corner, max_corner)
    cell_lines = _box_segments(
        min_corner,
        (
            min_corner[0] + min(resolution, width),
            min_corner[1] + min(resolution, depth),
            min_corner[2] + min(resolution, height),
        ),
    )
    return domain_lines, cell_lines


def _draw_preview():
    """Draw the active domain preview in the 3D viewport."""
    if _ACTIVE_DOMAIN is None or _ACTIVE_DOMAIN.id_data is None:
        return

    try:
        domain_lines, cell_lines = _build_preview_segments(_ACTIVE_DOMAIN)
    except Exception:
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


def enable_domain_preview(domain_node):
    """Enable the preview for the given domain node."""
    global _DRAW_HANDLER, _ACTIVE_DOMAIN

    _ACTIVE_DOMAIN = domain_node
    if _DRAW_HANDLER is None:
        _DRAW_HANDLER = bpy.types.SpaceView3D.draw_handler_add(
            _draw_preview,
            (),
            "WINDOW",
            "POST_VIEW",
        )
    _tag_view3d_redraw()


def disable_domain_preview():
    """Disable any active domain preview."""
    global _DRAW_HANDLER, _ACTIVE_DOMAIN

    _ACTIVE_DOMAIN = None
    if _DRAW_HANDLER is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_DRAW_HANDLER, "WINDOW")
        _DRAW_HANDLER = None
    _tag_view3d_redraw()


def _linked_domain_from_simulation(simulation_node):
    """Resolve the linked domain node connected to the simulation node."""
    socket = simulation_node.inputs.get("Domain")
    if socket is None or not socket.is_linked:
        return None

    for link in socket.links:
        from_node = getattr(link, "from_node", None)
        if from_node is not None and getattr(from_node, "bl_idname", "") == "BLENDERCFD_DOMAIN_NODE":
            return from_node
    return None


def _linked_domain_from_viewer(node):
    """Resolve the domain node by traversing the viewer's result input."""
    socket = node.inputs.get("Result")
    if socket is None or not socket.is_linked:
        return None

    for link in socket.links:
        simulation_node = getattr(link, "from_node", None)
        if simulation_node is None:
            continue
        if getattr(simulation_node, "bl_idname", "") != "BLENDERCFD_SIMULATION_NODE":
            continue

        domain_node = _linked_domain_from_simulation(simulation_node)
        if domain_node is not None:
            return domain_node
    return None


class BlenderCFD_OT_viewer_show_domain(bpy.types.Operator):
    """Show the linked domain in the 3D viewport."""

    bl_idname = "blendercfd.viewer_show_domain"
    bl_label = "Show Domain Preview"
    bl_description = "Draw the linked CFD domain as a viewport preview"

    def execute(self, context):
        node = getattr(context, "node", None)
        if node is None:
            self.report({"ERROR"}, "No active viewer node found.")
            return {"CANCELLED"}

        domain_node = _linked_domain_from_viewer(node)
        if domain_node is None:
            self.report({"ERROR"}, "Connect Domain -> Simulation -> Viewer first.")
            return {"CANCELLED"}

        enable_domain_preview(domain_node)
        self.report({"INFO"}, "Domain preview enabled.")
        return {"FINISHED"}


class BlenderCFD_OT_viewer_hide_domain(bpy.types.Operator):
    """Hide the active domain preview."""

    bl_idname = "blendercfd.viewer_hide_domain"
    bl_label = "Hide Domain Preview"
    bl_description = "Remove the CFD domain preview from the viewport"

    def execute(self, context):
        disable_domain_preview()
        self.report({"INFO"}, "Domain preview disabled.")
        return {"FINISHED"}


classes = (
    BlenderCFD_OT_viewer_show_domain,
    BlenderCFD_OT_viewer_hide_domain,
)
