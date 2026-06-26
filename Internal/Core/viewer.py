import bpy
import gpu
from gpu_extras.batch import batch_for_shader

_DRAW_HANDLER = None
_ACTIVE_DOMAIN_REF = None


def _tag_view3d_redraw():
    """
    Request a redraw for all visible 3D viewports.
    """
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


def _tag_viewer_ui_redraw():
    """
    Request a redraw for node editors and properties so viewer buttons update immediately.
    """
    window_manager = getattr(bpy.context, "window_manager", None)
    if window_manager is None:
        return

    for window in window_manager.windows:
        screen = getattr(window, "screen", None)
        if screen is None:
            continue
        for area in screen.areas:
            if area.type in {"NODE_EDITOR", "PROPERTIES"}:
                area.tag_redraw()


def _view3d_overlays_visible():
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


def _domain_node_reference(domain_node):
    """
    Return a stable reference for one domain node across redraws and undo.
    """
    node_tree = getattr(domain_node, "id_data", None)
    if node_tree is None:
        return None
    return {
        "node_tree_name": str(getattr(node_tree, "name", "")),
        "node_name": str(getattr(domain_node, "name", "")),
    }


def _resolve_domain_reference(domain_reference):
    """
    Resolve one stored domain-node reference back to a live node object.
    """
    if not domain_reference:
        return None
    node_tree = bpy.data.node_groups.get(domain_reference.get("node_tree_name", ""))
    if node_tree is None:
        return None
    return node_tree.nodes.get(domain_reference.get("node_name", ""))


def _domain_dimensions(domain_node):
    """
    Return the world-space domain dimensions derived from the node settings.
    """
    if not _is_valid_domain_node(domain_node):
        raise ValueError("Domain preview node is no longer valid.")
    return (
        float(domain_node.nx) * float(domain_node.resolution),
        float(domain_node.ny) * float(domain_node.resolution),
        float(domain_node.nz) * float(domain_node.resolution),
    )


def _is_valid_domain_node(domain_node):
    """
    Return whether one Blender domain node can still be accessed safely.
    """
    if domain_node is None:
        return False
    try:
        if getattr(domain_node, "bl_idname", "") != "CONTINUUM_FLOW_DOMAIN_NODE":
            return False
        if getattr(domain_node, "id_data", None) is None:
            return False
        float(domain_node.resolution)
        int(domain_node.nx)
        int(domain_node.ny)
        int(domain_node.nz)
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return False
    return True


def _domain_origin(domain_node):
    """
    Return the world-space coordinate of grid index (0, 0, 0).
    """
    width, depth, _height = _domain_dimensions(domain_node)
    return (-(width * 0.5), -(depth * 0.5), 0.0)


def _box_segments(min_corner, max_corner):
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


def _build_preview_segments(domain_node):
    """
    Build the preview line data for the full domain and one sample cell.
    """
    width, depth, height = _domain_dimensions(domain_node)
    resolution = float(domain_node.resolution)
    origin_x, origin_y, origin_z = _domain_origin(domain_node)

    min_corner = (origin_x, origin_y, origin_z)
    max_corner = (origin_x + width, origin_y + depth, origin_z + height)

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
    """
    Draw the active domain preview in the 3D viewport.
    """
    active_domain = _resolve_domain_reference(_ACTIVE_DOMAIN_REF)
    if not _is_valid_domain_node(active_domain):
        disable_domain_preview()
        return
    if not _view3d_overlays_visible():
        return

    try:
        domain_lines, cell_lines = _build_preview_segments(active_domain)
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


def enable_domain_preview(domain_node):
    """
    Enable the preview for the given domain node.
    """
    global _DRAW_HANDLER, _ACTIVE_DOMAIN_REF

    _ACTIVE_DOMAIN_REF = _domain_node_reference(domain_node)
    if _DRAW_HANDLER is None:
        _DRAW_HANDLER = bpy.types.SpaceView3D.draw_handler_add(
            _draw_preview, (), "WINDOW", "POST_VIEW"
        )
    _set_all_viewer_preview_flags(active=True)
    _tag_view3d_redraw()
    _tag_viewer_ui_redraw()


def disable_domain_preview():
    """
    Disable any active domain preview.
    """
    global _DRAW_HANDLER, _ACTIVE_DOMAIN_REF

    _ACTIVE_DOMAIN_REF = None
    if _DRAW_HANDLER is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_DRAW_HANDLER, "WINDOW")
        _DRAW_HANDLER = None
    _set_all_viewer_preview_flags(active=False)
    _tag_view3d_redraw()
    _tag_viewer_ui_redraw()


def domain_preview_enabled(domain_node=None):
    """
    Return whether a domain preview is active, optionally for one specific domain node.
    """
    active_domain = _resolve_domain_reference(_ACTIVE_DOMAIN_REF)
    if active_domain is None:
        return False
    if domain_node is None:
        return True
    active_reference = _domain_node_reference(active_domain)
    target_reference = _domain_node_reference(domain_node)
    return bool(
        active_reference and target_reference and active_reference == target_reference
    )


def viewer_domain_preview_enabled(viewer_node):
    """
    Return whether the given viewer node currently controls the active domain preview.
    """
    if viewer_node is None:
        return False
    domain_node = _linked_domain_from_viewer(viewer_node)
    if domain_node is None:
        return False
    return domain_preview_enabled(domain_node)


def _set_all_viewer_preview_flags(active=False):
    """
    Synchronise the domain-preview toggle text across all viewer nodes.
    """
    for node_tree in bpy.data.node_groups:
        for node in getattr(node_tree, "nodes", ()):
            if getattr(node, "bl_idname", "") != "CONTINUUM_FLOW_VIEWER_NODE":
                continue
            try:
                node.domain_preview_active = bool(
                    active and viewer_domain_preview_enabled(node)
                )
            except Exception:
                pass


def _linked_domain_from_simulation(simulation_node):
    """
    Resolve the linked domain node connected to the simulation node.
    """
    socket = simulation_node.inputs.get("Domain")
    if socket is None or not socket.is_linked:
        return None

    for link in socket.links:
        from_node = getattr(link, "from_node", None)
        if (
            from_node is not None
            and getattr(from_node, "bl_idname", "") == "CONTINUUM_FLOW_DOMAIN_NODE"
        ):
            return from_node
    return None


def _linked_domain_from_viewer(node):
    """
    Resolve the domain node by traversing the viewer's result input.
    """
    socket = node.inputs.get("Result")
    if socket is None or not socket.is_linked:
        return None

    for link in socket.links:
        simulation_node = getattr(link, "from_node", None)
        if simulation_node is None:
            continue
        if getattr(simulation_node, "bl_idname", "") != "CONTINUUM_FLOW_SIMULATION_NODE":
            continue

        domain_node = _linked_domain_from_simulation(simulation_node)
        if domain_node is not None:
            return domain_node
    return None


class ContinuumFlow_OT_viewer_toggle_domain(bpy.types.Operator):
    """
    Toggle the linked domain preview in the 3D viewport.
    """

    bl_idname = "continuum_flow.viewer_toggle_domain"
    bl_label = "Toggle Domain Preview"
    bl_description = "Toggle the linked Continuum Flow domain viewport preview"

    def execute(self, context):
        node = getattr(context, "node", None)
        if node is None:
            self.report({"ERROR"}, "No active viewer node found.")
            return {"CANCELLED"}

        domain_node = _linked_domain_from_viewer(node)
        if domain_node is None:
            self.report({"ERROR"}, "Connect Domain -> Simulation -> Viewer first.")
            return {"CANCELLED"}

        if domain_preview_enabled(domain_node):
            disable_domain_preview()
            self.report({"INFO"}, "Domain preview disabled.")
        else:
            enable_domain_preview(domain_node)
            self.report({"INFO"}, "Domain preview enabled.")
        return {"FINISHED"}


classes = (ContinuumFlow_OT_viewer_toggle_domain,)
