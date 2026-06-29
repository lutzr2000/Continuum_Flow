import math

import bpy
import gpu
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

_DRAW_HANDLER = None
_ACTIVE_FORCE_REF = None

###########################
# constant force
###########################

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


def _force_node_reference(force_node):
    """
    Return a stable reference for one force node across redraws and undo.
    """
    node_tree = getattr(force_node, "id_data", None)
    if node_tree is None:
        return None
    return {
        "node_tree_name": str(getattr(node_tree, "name", "")),
        "node_name": str(getattr(force_node, "name", "")),
    }


def _resolve_force_reference(force_reference):
    """
    Resolve one stored force-node reference back to a live node object.
    """
    if not force_reference:
        return None
    node_tree = bpy.data.node_groups.get(force_reference.get("node_tree_name", ""))
    if node_tree is None:
        return None
    return node_tree.nodes.get(force_reference.get("node_name", ""))


def _is_valid_constant_force_node(force_node):
    """
    Return whether one constant force node can still be accessed safely.
    """
    if force_node is None:
        return False
    try:
        if getattr(force_node, "bl_idname", "") != "CONTINUUM_FLOW_FORCE_CONSTANT_NODE":
            return False
        if getattr(force_node, "id_data", None) is None:
            return False
        float(force_node.fx)
        float(force_node.fy)
        float(force_node.fz)
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return False
    return True


def _is_valid_swirl_force_node(force_node):
    """
    Return whether one swirl force node can still be accessed safely.
    """
    if force_node is None:
        return False
    try:
        if getattr(force_node, "bl_idname", "") != "CONTINUUM_FLOW_FORCE_SWIRL_NODE":
            return False
        if getattr(force_node, "id_data", None) is None:
            return False
        float(force_node.strength)
        float(force_node.radius)
        for component in force_node.origin:
            float(component)
        for component in force_node.axis:
            float(component)
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return False
    return True


def _is_valid_turbulence_force_node(force_node):
    """
    Return whether one turbulence force node can still be accessed safely.
    """
    if force_node is None:
        return False
    try:
        if getattr(force_node, "bl_idname", "") != "CONTINUUM_FLOW_FORCE_TURBULENCE_NODE":
            return False
        if getattr(force_node, "id_data", None) is None:
            return False
        float(force_node.scale)
        float(force_node.frequency)
        float(force_node.amplitude)
        int(force_node.seed)
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return False
    return True


def _is_valid_force_preview_node(force_node):
    return (
        _is_valid_constant_force_node(force_node)
        or _is_valid_swirl_force_node(force_node)
        or _is_valid_turbulence_force_node(force_node)
    )


def _linked_simulation_from_force(force_node):
    """
    Resolve the downstream simulation node connected to the force node.
    """
    socket = force_node.outputs.get("Force")
    if socket is None or not socket.is_linked:
        return None

    for link in socket.links:
        simulation_node = getattr(link, "to_node", None)
        if simulation_node is None:
            continue
        if getattr(simulation_node, "bl_idname", "") == "CONTINUUM_FLOW_SIMULATION_NODE":
            return simulation_node
    return None


def _linked_domain_from_simulation(simulation_node):
    """
    Resolve the linked domain node connected to the simulation node.
    """
    socket = simulation_node.inputs.get("Domain")
    if socket is None or not socket.is_linked:
        return None

    for link in socket.links:
        domain_node = getattr(link, "from_node", None)
        if domain_node is None:
            continue
        if getattr(domain_node, "bl_idname", "") == "CONTINUUM_FLOW_DOMAIN_NODE":
            return domain_node
    return None


def _domain_dimensions(domain_node):
    return (
        float(domain_node.nx) * float(domain_node.resolution),
        float(domain_node.ny) * float(domain_node.resolution),
        float(domain_node.nz) * float(domain_node.resolution),
    )


def _domain_center(domain_node):
    width, depth, height = _domain_dimensions(domain_node)
    return (0.0, 0.0, height * 0.5)


def _domain_bounds(domain_node):
    width, depth, height = _domain_dimensions(domain_node)
    return (
        (-0.5 * width, -0.5 * depth, 0.0),
        (0.5 * width, 0.5 * depth, height),
    )


def _vector_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vector_add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _vector_scale(vec, scale):
    return (vec[0] * scale, vec[1] * scale, vec[2] * scale)


def _vector_length(vec):
    return math.sqrt((vec[0] * vec[0]) + (vec[1] * vec[1]) + (vec[2] * vec[2]))


def _vector_normalized(vec):
    length = _vector_length(vec)
    if length <= 1.0e-9:
        return None
    return (vec[0] / length, vec[1] / length, vec[2] / length)


def _vector_cross(a, b):
    return (
        (a[1] * b[2]) - (a[2] * b[1]),
        (a[2] * b[0]) - (a[0] * b[2]),
        (a[0] * b[1]) - (a[1] * b[0]),
    )


def _vector_dot(a, b):
    return (a[0] * b[0]) + (a[1] * b[1]) + (a[2] * b[2])


def _perpendicular_basis(axis):
    reference_up = (0.0, 0.0, 1.0)
    side = _vector_normalized(_vector_cross(axis, reference_up))
    if side is None:
        reference_up = (0.0, 1.0, 0.0)
        side = _vector_normalized(_vector_cross(axis, reference_up))
    if side is None:
        return None, None
    up = _vector_normalized(_vector_cross(side, axis))
    return side, up


def _arrow_segments(start, end, head_size):
    """
    Build line segments for a simple 3D arrow.
    """
    direction = _vector_normalized(_vector_sub(end, start))
    if direction is None:
        return []

    side, _up = _perpendicular_basis(direction)
    if side is None:
        return [start, end]

    head_base = _vector_add(end, _vector_scale(direction, -head_size))
    left_head = _vector_add(head_base, _vector_scale(side, head_size * 0.45))
    right_head = _vector_add(head_base, _vector_scale(side, -head_size * 0.45))

    return [
        start, end,
        left_head, end,
        right_head, end,
    ]


def _build_constant_force_segments(force_node):
    """
    Build a simple arrow in the domain center for one constant force node.
    """
    simulation_node = _linked_simulation_from_force(force_node)
    if simulation_node is None:
        return []

    domain_node = _linked_domain_from_simulation(simulation_node)
    if domain_node is None:
        return []

    force_vector = (float(force_node.fx), float(force_node.fy), float(force_node.fz))
    direction = _vector_normalized(force_vector)
    if direction is None:
        return []

    width, depth, height = _domain_dimensions(domain_node)
    min_dimension = max(min(width, depth, height), float(domain_node.resolution))
    magnitude = _vector_length(force_vector)
    arrow_length = min(min_dimension * 0.45, magnitude * min_dimension * 0.12)
    arrow_length = max(arrow_length, min_dimension * 0.05)

    center = _domain_center(domain_node)
    half_arrow = _vector_scale(direction, arrow_length * 0.5)
    start = _vector_add(center, _vector_scale(half_arrow, -1.0))
    end = _vector_add(center, half_arrow)
    head_size = min_dimension * 0.08
    return _arrow_segments(start, end, head_size)


def _draw_force_preview():
    """
    Draw the active force preview in the 3D viewport.
    """
    force_node = _resolve_force_reference(_ACTIVE_FORCE_REF)
    if not _is_valid_force_preview_node(force_node):
        disable_force_preview()
        return
    if not _view3d_overlays_visible():
        return

    force_type = getattr(force_node, "bl_idname", "")
    if force_type == "CONTINUUM_FLOW_FORCE_CONSTANT_NODE":
        segments = _build_constant_force_segments(force_node)
        if not segments:
            return
        shader = gpu.shader.from_builtin("UNIFORM_COLOR")
        gpu.state.blend_set("ALPHA")
        gpu.state.line_width_set(3.0)
        shader.bind()
        shader.uniform_float("color", (0.45, 0.65, 0.95, 1.0))
        batch_for_shader(shader, "LINES", {"pos": segments}).draw(shader)
        gpu.state.line_width_set(1.0)
        gpu.state.blend_set("NONE")
        return

    if force_type == "CONTINUUM_FLOW_FORCE_SWIRL_NODE":
        segments = _build_swirl_force_segments(force_node)
        if not segments:
            return
        shader = gpu.shader.from_builtin("UNIFORM_COLOR")
        gpu.state.blend_set("ALPHA")
        gpu.state.line_width_set(3.0)
        shader.bind()
        shader.uniform_float("color", (0.45, 0.65, 0.95, 1.0))
        batch_for_shader(shader, "LINES", {"pos": segments}).draw(shader)
        gpu.state.line_width_set(1.0)
        gpu.state.blend_set("NONE")
        return

    if force_type == "CONTINUUM_FLOW_FORCE_TURBULENCE_NODE":
        positions, colors = _build_turbulence_force_triangles(force_node)
        if not positions:
            return
        shader = gpu.shader.from_builtin("SMOOTH_COLOR")
        gpu.state.blend_set("ALPHA")
        shader.bind()
        batch_for_shader(shader, "TRIS", {"pos": positions, "color": colors}).draw(shader)
        gpu.state.blend_set("NONE")


def enable_force_preview(force_node):
    """
    Enable the preview for the given force node.
    """
    global _DRAW_HANDLER, _ACTIVE_FORCE_REF

    if not _is_valid_force_preview_node(force_node):
        disable_force_preview()
        return

    _ACTIVE_FORCE_REF = _force_node_reference(force_node)
    if _DRAW_HANDLER is None:
        _DRAW_HANDLER = bpy.types.SpaceView3D.draw_handler_add(
            _draw_force_preview, (), "WINDOW", "POST_VIEW"
        )
    _tag_view3d_redraw()


def disable_force_preview():
    """
    Disable any active force preview.
    """
    global _DRAW_HANDLER, _ACTIVE_FORCE_REF

    _ACTIVE_FORCE_REF = None
    if _DRAW_HANDLER is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_DRAW_HANDLER, "WINDOW")
        _DRAW_HANDLER = None
    _tag_view3d_redraw()


def _selected_force_nodes():
    """
    Yield currently selected Continuum Flow force nodes.
    """
    for node_tree in bpy.data.node_groups:
        if getattr(node_tree, "bl_idname", "") != "CONTINUUM_FLOW_NODE_TREE":
            continue
        for node in getattr(node_tree, "nodes", ()):
            if not getattr(node, "select", False):
                continue
            if getattr(node, "bl_idname", "") not in {
                "CONTINUUM_FLOW_FORCE_CONSTANT_NODE",
                "CONTINUUM_FLOW_FORCE_SWIRL_NODE",
                "CONTINUUM_FLOW_FORCE_TURBULENCE_NODE",
            }:
                continue
            yield node


def sync_force_preview_from_selection():
    """
    Show the first selected constant force node and hide the preview otherwise.
    """
    for force_node in _selected_force_nodes():
        enable_force_preview(force_node)
        return
    disable_force_preview()


def force_preview_timer():
    """
    Keep the force preview in sync with node selection.
    """
    sync_force_preview_from_selection()
    return 0.2

###########################
# swirl force
###########################

def _line_box_intersection(origin, direction, box_min, box_max):
    """
    Return the parameter interval where an infinite line intersects an axis-aligned box.
    """
    t_min = -float("inf")
    t_max = float("inf")

    for index in range(3):
        origin_component = origin[index]
        direction_component = direction[index]
        min_component = box_min[index]
        max_component = box_max[index]

        if abs(direction_component) <= 1.0e-9:
            if origin_component < min_component or origin_component > max_component:
                return None
            continue

        t0 = (min_component - origin_component) / direction_component
        t1 = (max_component - origin_component) / direction_component
        if t0 > t1:
            t0, t1 = t1, t0

        t_min = max(t_min, t0)
        t_max = min(t_max, t1)
        if t_min > t_max:
            return None

    return t_min, t_max


def _rim_segments(points):
    """
    Build one closed polyline from ordered rim points.
    """
    segments = []
    for index in range(len(points) - 1):
        segments.extend((points[index], points[index + 1]))
    return segments


def _build_swirl_rotation_arrow(center, axis, axis_u, radius, strength, size_scale):
    """
    Build a small arrow that indicates the swirl rotation direction.
    """
    if abs(float(strength)) <= 1.0e-9:
        return []

    sign = 1.0 if float(strength) >= 0.0 else -1.0
    radial = _vector_scale(axis_u, radius)
    arrow_center = _vector_add(center, radial)

    tangent = _vector_cross(axis, radial)
    tangent = _vector_normalized(tangent)
    if tangent is None:
        return []
    tangent = _vector_scale(tangent, sign)

    arrow_length = max(size_scale * 0.18, radius * 0.35)
    head_size = max(size_scale * 0.08, radius * 0.18)
    start = _vector_add(arrow_center, _vector_scale(tangent, -arrow_length * 0.5))
    end = _vector_add(arrow_center, _vector_scale(tangent, arrow_length * 0.5))
    return _arrow_segments(start, end, head_size)


def _sample_swirl_cylinder_rims(origin, axis, axis_u, axis_v, radius, box_min, box_max, segment_count=32):
    """
    Sample the clipped cylinder rims by intersecting offset axis lines with the domain box.
    """
    start_points = []
    end_points = []

    for index in range(segment_count + 1):
        angle = (float(index) / float(segment_count)) * (2.0 * math.pi)
        radial = _vector_add(
            _vector_scale(axis_u, math.cos(angle) * radius),
            _vector_scale(axis_v, math.sin(angle) * radius),
        )
        offset_origin = _vector_add(origin, radial)
        interval = _line_box_intersection(offset_origin, axis, box_min, box_max)
        if interval is None:
            return [], []

        t_min, t_max = interval
        start_points.append(_vector_add(offset_origin, _vector_scale(axis, t_min)))
        end_points.append(_vector_add(offset_origin, _vector_scale(axis, t_max)))

    return start_points, end_points


def _build_swirl_force_segments(force_node):
    """
    Draw the intersected cylinder outline plus one small rotation arrow.
    """
    simulation_node = _linked_simulation_from_force(force_node)
    if simulation_node is None:
        return []

    domain_node = _linked_domain_from_simulation(simulation_node)
    if domain_node is None:
        return []

    axis = _vector_normalized(tuple(float(component) for component in force_node.axis))
    if axis is None:
        return []

    radius = float(force_node.radius)
    if radius <= 0.0:
        return []

    axis_u, axis_v = _perpendicular_basis(axis)
    if axis_u is None or axis_v is None:
        return []

    origin = tuple(float(component) for component in force_node.origin)
    box_min, box_max = _domain_bounds(domain_node)
    interval = _line_box_intersection(origin, axis, box_min, box_max)
    if interval is None:
        return []

    t_min, t_max = interval
    start_center = _vector_add(origin, _vector_scale(axis, t_min))
    end_center = _vector_add(origin, _vector_scale(axis, t_max))
    start_points, end_points = _sample_swirl_cylinder_rims(
        origin,
        axis,
        axis_u,
        axis_v,
        radius,
        box_min,
        box_max,
    )
    if not start_points or not end_points:
        return []

    segments = []
    segments.extend(_rim_segments(start_points))
    segments.extend(_rim_segments(end_points))

    sample_indices = (0, 8, 16, 24)
    for sample_index in sample_indices:
        if sample_index >= len(start_points) - 1 or sample_index >= len(end_points) - 1:
            continue
        segments.extend((start_points[sample_index], end_points[sample_index]))

    width, depth, height = _domain_dimensions(domain_node)
    size_scale = max(min(width, depth, height), float(domain_node.resolution))
    mid_center = _vector_scale(_vector_add(start_center, end_center), 0.5)
    segments.extend(
        _build_swirl_rotation_arrow(
            mid_center,
            axis,
            axis_u,
            radius,
            float(force_node.strength),
            size_scale,
        )
    )
    return segments

###########################
# turbulence force
###########################

_TURBULENCE_VIEWER_CACHE = {}


def _turbulence_cache_key(force_node, domain_node, sample_count, plane):
    return (
        force_node.name,
        int(force_node.seed),
        round(float(force_node.scale), 6),
        round(float(force_node.amplitude), 6),
        round(float(force_node.frequency), 6),
        int(bpy.context.scene.frame_current),
        sample_count,
        tuple(round(x, 4) for x in plane["normal"]),
        tuple(round(x, 4) for x in _domain_center(domain_node)),
        round(float(domain_node.resolution), 6),
    )


def _current_view_direction():
    region_data = getattr(bpy.context, "region_data", None)
    if region_data is None:
        return (0.0, 0.0, -1.0)
    try:
        direction = region_data.view_rotation @ Vector((0.0, 0.0, -1.0))
        return (float(direction.x), float(direction.y), float(direction.z))
    except Exception:
        return (0.0, 0.0, -1.0)


def _solver_time_seconds():
    scene = getattr(bpy.context, "scene", None)
    if scene is None:
        return 0.0
    render = getattr(scene, "render", None)
    fps = max(1, int(getattr(render, "fps", 24)))
    return float(getattr(scene, "frame_current", 0)) / float(fps)


def _smoothstep(t):
    return t * t * (3.0 - 2.0 * t)


def _lerp(a, b, t):
    return a + t * (b - a)


def _fast_floor(x):
    i = int(x)
    if x < float(i):
        return i - 1
    return i


def _hash_noise_3d(ix, iy, iz, seed):
    n = ix * 15731 + iy * 789221 + iz * 1376312589 + seed * 1013
    n = (n << 13) ^ n
    nn = n * (n * n * 15731 + 789221) + 1376312589
    nn = nn & 0x7fffffff
    return float(nn) / 1073741824.0 - 1.0


def _value_noise_3d(x, y, z, seed):
    x0 = _fast_floor(x)
    y0 = _fast_floor(y)
    z0 = _fast_floor(z)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1
    tx = _smoothstep(x - float(x0))
    ty = _smoothstep(y - float(y0))
    tz = _smoothstep(z - float(z0))
    c000 = _hash_noise_3d(x0, y0, z0, seed)
    c100 = _hash_noise_3d(x1, y0, z0, seed)
    c010 = _hash_noise_3d(x0, y1, z0, seed)
    c110 = _hash_noise_3d(x1, y1, z0, seed)
    c001 = _hash_noise_3d(x0, y0, z1, seed)
    c101 = _hash_noise_3d(x1, y0, z1, seed)
    c011 = _hash_noise_3d(x0, y1, z1, seed)
    c111 = _hash_noise_3d(x1, y1, z1, seed)
    x00 = _lerp(c000, c100, tx)
    x10 = _lerp(c010, c110, tx)
    x01 = _lerp(c001, c101, tx)
    x11 = _lerp(c011, c111, tx)
    y0v = _lerp(x00, x10, ty)
    y1v = _lerp(x01, x11, ty)
    return _lerp(y0v, y1v, tz)


def _choose_turbulence_plane(domain_node):
    width, depth, height = _domain_dimensions(domain_node)
    center = _domain_center(domain_node)
    view_direction = _vector_normalized(_current_view_direction()) or (0.0, 0.0, -1.0)
    planes = [
        {"normal": (0.0, 0.0, 1.0), "u": (1.0, 0.0, 0.0), "v": (0.0, 1.0, 0.0), "size_u": width, "size_v": depth, "center": center},
        {"normal": (0.0, 1.0, 0.0), "u": (1.0, 0.0, 0.0), "v": (0.0, 0.0, 1.0), "size_u": width, "size_v": height, "center": center},
        {"normal": (1.0, 0.0, 0.0), "u": (0.0, 1.0, 0.0), "v": (0.0, 0.0, 1.0), "size_u": depth, "size_v": height, "center": center},
    ]
    return max(planes, key=lambda plane: abs(_vector_dot(plane["normal"], view_direction)))


def _turbulence_scalar(force_node, position, time_value):
    amplitude = float(force_node.amplitude)
    scale = float(force_node.scale)
    if abs(amplitude) <= 1.0e-9 or scale <= 1.0e-8:
        return 0.0
    inv_scale = 1.0 / scale
    frequency = float(force_node.frequency)
    seed = int(force_node.seed)
    x = position[0] * inv_scale
    y = position[1] * inv_scale
    z = position[2] * inv_scale + (time_value * frequency)
    return amplitude * _value_noise_3d(x, y, z, seed)


def _turbulence_color(value, amplitude):
    max_value = max(abs(float(amplitude)), 1.0e-6)
    factor = max(0.0, min(1.0, 0.5 + (0.5 * value / max_value)))
    gray = factor
    return (gray, gray, gray, 1.0)


def _build_turbulence_force_triangles(force_node, sample_count=64):
    simulation_node = _linked_simulation_from_force(force_node)
    if simulation_node is None:
        return [], []

    domain_node = _linked_domain_from_simulation(simulation_node)
    if domain_node is None:
        return [], []

    if float(force_node.scale) <= 1.0e-8 or abs(float(force_node.amplitude)) <= 1.0e-9:
        return [], []

    plane = _choose_turbulence_plane(domain_node)
    cache_key = _turbulence_cache_key(force_node, domain_node, sample_count, plane)

    cached = _TURBULENCE_VIEWER_CACHE.get(cache_key)
    if cached is not None:
        return cached
    time_value = _solver_time_seconds()

    size_u = max(float(plane["size_u"]), float(domain_node.resolution))
    size_v = max(float(plane["size_v"]), float(domain_node.resolution))

    half_u = _vector_scale(plane["u"], size_u * 0.5)
    half_v = _vector_scale(plane["v"], size_v * 0.5)
    origin = _vector_sub(_vector_sub(plane["center"], half_u), half_v)

    step_u = size_u / float(sample_count)
    step_v = size_v / float(sample_count)

    # Grid-Punkte einmal berechnen
    grid_positions = []
    grid_colors = []

    for iy in range(sample_count + 1):
        row_positions = []
        row_colors = []

        v_amount = float(iy) * step_v
        v_offset = _vector_scale(plane["v"], v_amount)

        for ix in range(sample_count + 1):
            u_amount = float(ix) * step_u
            u_offset = _vector_scale(plane["u"], u_amount)

            p = _vector_add(origin, _vector_add(u_offset, v_offset))
            value = _turbulence_scalar(force_node, p, time_value)
            color = _turbulence_color(value, force_node.amplitude)

            row_positions.append(p)
            row_colors.append(color)

        grid_positions.append(row_positions)
        grid_colors.append(row_colors)

    positions = []
    colors = []

    for iy in range(sample_count):
        row0_p = grid_positions[iy]
        row1_p = grid_positions[iy + 1]
        row0_c = grid_colors[iy]
        row1_c = grid_colors[iy + 1]

        for ix in range(sample_count):
            p00 = row0_p[ix]
            p10 = row0_p[ix + 1]
            p01 = row1_p[ix]
            p11 = row1_p[ix + 1]

            c00 = row0_c[ix]
            c10 = row0_c[ix + 1]
            c01 = row1_c[ix]
            c11 = row1_c[ix + 1]

            positions.extend((p00, p10, p11, p00, p11, p01))
            colors.extend((c00, c10, c11, c00, c11, c01))

    result = (positions, colors)
    _TURBULENCE_VIEWER_CACHE.clear()
    _TURBULENCE_VIEWER_CACHE[cache_key] = result
    return result