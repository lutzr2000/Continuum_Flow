import math
from . import viewer

import bpy
import gpu
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

draw_handler = None
current_drawn_force = None


#-------------- general ----------------
def force_preview_timer():
    """
    Keep the force preview in sync with node selection.
    """
    if not continuum_flow_editor_visible():
        disable_force_preview()
        return 0.5

    sync_force_preview()
    return 0.2


def continuum_flow_editor_visible():
    """
    Return whether any visible node editor is currently showing the Continuum Flow tree.
    """
    window_manager = getattr(bpy.context, "window_manager", None)
    if window_manager is None:
        return False

    for window in window_manager.windows:
        screen = getattr(window, "screen", None)
        if screen is None:
            continue
        for area in screen.areas:
            if area.type != "NODE_EDITOR":
                continue
            for space in area.spaces:
                if getattr(space, "type", "") != "NODE_EDITOR":
                    continue
                if getattr(space, "tree_type", "") == "CONTINUUM_FLOW_NODE_TREE":
                    return True
    return False


def sync_force_preview():
    """
    Show the selected force node preview
    """
    for force_node in selected_force_node():
        enable_force_preview(force_node)
        return
    disable_force_preview()


def selected_force_node():
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


def disable_force_preview():
    """
    Disable any active force preview.
    """
    global draw_handler, current_drawn_force

    current_drawn_force = None
    if draw_handler is not None:
        bpy.types.SpaceView3D.draw_handler_remove(draw_handler, "WINDOW")
        draw_handler = None
    viewer.redraw_viewport()


def enable_force_preview(force_node):
    """
    Enable the preview for the given force node.
    """
    global draw_handler, current_drawn_force

    node_tree = getattr(force_node, "id_data", None)
    if node_tree is None:
        current_drawn_force = None
        return

    current_drawn_force = {
        "node_tree_name": str(getattr(node_tree, "name", "")),
        "node_name": str(getattr(force_node, "name", "")),
    }

    if draw_handler is None:
        draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            draw_force_preview, (), "WINDOW", "POST_VIEW"
        )

    viewer.redraw_viewport()


def draw_force_preview():
    """
    Draw the active force preview in the 3D viewport.
    """
    if not current_drawn_force:
        return

    node_tree = bpy.data.node_groups.get(
        current_drawn_force.get("node_tree_name", "")
    )
    if node_tree is None:
        return

    force_node = node_tree.nodes.get(
        current_drawn_force.get("node_name", "")
    )
    if force_node is None:
        return

    if not viewer.overlay_enabeld():
        return
    
    simulation_node = get_linked_simulation_node(force_node)
    domain_node = get_linked_domain_node(simulation_node)

    force_type = getattr(force_node, "bl_idname", "")

    if force_type == "CONTINUUM_FLOW_FORCE_CONSTANT_NODE":
        segments = build_constant_force_segments(force_node,simulation_node,domain_node)
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
        segments = build_swirl_force_segments(force_node,simulation_node,domain_node)
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
        positions, colors, indices = build_turbulence_force_plane(
            force_node,
            simulation_node,
            domain_node,
        )

        if not positions:
            return

        shader = gpu.shader.from_builtin("SMOOTH_COLOR")
        gpu.state.blend_set("ALPHA")
        shader.bind()
        batch = batch_for_shader(
            shader,
            "TRIS",
            {
                "pos": positions,
                "color": colors,
            },
            indices=indices,
        )

        batch.draw(shader)
        gpu.state.blend_set("NONE")


#-------------- helper ----------------
def get_linked_simulation_node(force_node):
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


def get_linked_domain_node(simulation_node):
    """
    Resolve the linked domain node connected to the simulation node.
    """
    if simulation_node is None:
        return None

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


def domain_dimensions(domain_node):
    return (
        float(domain_node.nx) * float(domain_node.resolution),
        float(domain_node.ny) * float(domain_node.resolution),
        float(domain_node.nz) * float(domain_node.resolution),
    )


def domain_center(domain_node):
    width, depth, height = domain_dimensions(domain_node)
    return (0.0, 0.0, height * 0.5)


def domain_bounds(domain_node):
    width, depth, height = domain_dimensions(domain_node)
    return (
        (-0.5 * width, -0.5 * depth, 0.0),
        (0.5 * width, 0.5 * depth, height),
    )


def arrow_segments(start, end, head_size):
    """
    Build line segments for a simple 3D arrow.
    """
    start = Vector(start)
    end = Vector(end)

    direction = end - start

    if direction.length <= 1.0e-9:
        return []

    direction.normalize()

    side, _up = perpendicular_basis(direction)

    if side is None:
        return [start, end]

    head_base = end - direction * head_size
    left_head = head_base + side * (head_size * 0.45)
    right_head = head_base - side * (head_size * 0.45)

    return [
        start, end,
        left_head, end,
        right_head, end,
    ]


def perpendicular_basis(axis):
    axis = Vector(axis)

    reference_up = Vector((0.0, 0.0, 1.0))
    side = axis.cross(reference_up)

    if side.length <= 1.0e-9:
        reference_up = Vector((0.0, 1.0, 0.0))
        side = axis.cross(reference_up)

    if side.length <= 1.0e-9:
        return None, None

    side.normalize()

    up = side.cross(axis)

    if up.length <= 1.0e-9:
        return None, None

    up.normalize()

    return side, up


def line_box_intersection(origin, direction, box_min, box_max):
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


#-------------- constant force ----------------
def build_constant_force_segments(force_node, simulation_node, domain_node):
    """
    Build a simple arrow in the domain center for one constant force node.
    """
    if simulation_node is None:
        return []

    if domain_node is None:
        return []

    force_vector = Vector((
        float(force_node.fx),
        float(force_node.fy),
        float(force_node.fz),
    ))

    if force_vector.length <= 1.0e-9:
        return []

    direction = force_vector.normalized()

    width, depth, height = domain_dimensions(domain_node)
    min_dimension = max(
        min(width, depth, height),
        float(domain_node.resolution),
    )

    magnitude = force_vector.length
    arrow_length = min(
        min_dimension * 0.45,
        magnitude * min_dimension * 0.12,
    )
    arrow_length = max(
        arrow_length,
        min_dimension * 0.05,
    )

    center = Vector(domain_center(domain_node))
    half_arrow = direction * (arrow_length * 0.5)

    start = center - half_arrow
    end = center + half_arrow

    head_size = min_dimension * 0.08

    return arrow_segments(start, end, head_size)


#-------------- swirl force ----------------
def build_swirl_force_segments(
    force_node,
    simulation_node,
    domain_node,
    segment_count=128,
):
    """
    Draw the intersected cylinder outline plus one small rotation arrow.
    """

    if simulation_node is None:
        return []

    if domain_node is None:
        return []

    axis = Vector(force_node.axis)

    if axis.length <= 1.0e-9:
        return []

    axis.normalize()

    radius = float(force_node.radius)

    if radius <= 0.0:
        return []

    axis_u, axis_v = perpendicular_basis(axis)

    if axis_u is None or axis_v is None:
        return []

    origin = Vector(force_node.origin)

    box_min, box_max = domain_bounds(domain_node)

    start_points, end_points = sample_swirl_cylinder_rims(
        origin,
        axis,
        axis_u,
        axis_v,
        radius,
        box_min,
        box_max,
        segment_count=segment_count,
    )

    valid_start_points = [
        point
        for point in start_points
        if point is not None
    ]

    valid_end_points = [
        point
        for point in end_points
        if point is not None
    ]

    # cylinder does not intersect domain
    if not valid_start_points or not valid_end_points:
        return []

    segments = []

    segments.extend(
        rim_segments(start_points)
    )

    segments.extend(
        rim_segments(end_points)
    )

    sample_indices = (
        0,
        segment_count // 4,
        segment_count // 2,
        (segment_count * 3) // 4,
    )

    for sample_index in sample_indices:
        start = start_points[sample_index]
        end = end_points[sample_index]

        if start is None or end is None:
            continue

        segments.extend((
            start,
            end,
        ))


    width, depth, height = domain_dimensions(
        domain_node
    )

    size_scale = max(
        min(width, depth, height),
        float(domain_node.resolution),
    )

    start_center = sum(
        valid_start_points,
        Vector((0.0, 0.0, 0.0)),
    ) / len(valid_start_points)

    end_center = sum(
        valid_end_points,
        Vector((0.0, 0.0, 0.0)),
    ) / len(valid_end_points)

    mid_center = (
        start_center + end_center
    ) * 0.5

    # Rotationspfeil
    segments.extend(
        build_swirl_rotation_arrow(
            mid_center,
            axis,
            axis_u,
            radius,
            float(force_node.strength),
            size_scale,
        )
    )

    return segments


def build_swirl_rotation_arrow(
    center,
    axis,
    axis_u,
    radius,
    strength,
    size_scale,
):
    """
    Build a small arrow that indicates the swirl rotation direction.
    """
    if abs(float(strength)) <= 1.0e-9:
        return []

    sign = 1.0 if float(strength) >= 0.0 else -1.0

    center = Vector(center)
    axis = Vector(axis)
    axis_u = Vector(axis_u)

    radial = axis_u * radius
    arrow_center = center + radial

    tangent = axis.cross(radial)

    if tangent.length <= 1.0e-9:
        return []

    tangent.normalize()
    tangent *= sign

    arrow_length = max(
        size_scale * 0.18,
        radius * 0.35,
    )

    head_size = max(
        size_scale * 0.08,
        radius * 0.18,
    )

    start = arrow_center - tangent * (arrow_length * 0.5)
    end = arrow_center + tangent * (arrow_length * 0.5)

    return arrow_segments(
        start,
        end,
        head_size,
    )


def rim_segments(points):
    segments = []

    for index in range(len(points) - 1):
        p0 = points[index]
        p1 = points[index + 1]

        if p0 is None or p1 is None:
            continue

        segments.extend((
            p0,
            p1,
        ))

    return segments


def sample_swirl_cylinder_rims(
    origin,
    axis,
    axis_u,
    axis_v,
    radius,
    box_min,
    box_max,
    segment_count=128,
):
    origin = Vector(origin)
    axis = Vector(axis)
    axis_u = Vector(axis_u)
    axis_v = Vector(axis_v)

    start_points = []
    end_points = []

    for index in range(segment_count + 1):
        angle = (
            float(index)
            / float(segment_count)
        ) * (2.0 * math.pi)

        radial = (
            axis_u * (math.cos(angle) * radius)
            + axis_v * (math.sin(angle) * radius)
        )

        offset_origin = origin + radial

        interval = line_box_intersection(
            offset_origin,
            axis,
            box_min,
            box_max,
        )

        if interval is None:
            start_points.append(None)
            end_points.append(None)
            continue

        t_min, t_max = interval

        start_points.append(
            offset_origin + axis * t_min
        )

        end_points.append(
            offset_origin + axis * t_max
        )

    return start_points, end_points


#-------------- turbulence force ----------------
def build_turbulence_force_plane(
    force_node,
    simulation_node,
    domain_node,
    sample_count=128,
):
    if simulation_node is None or domain_node is None:
        return [], [], []

    amplitude = float(force_node.amplitude)
    scale = float(force_node.scale)

    if scale <= 1.0e-8 or abs(amplitude) <= 1.0e-9:
        return [], [], []

    plane = choose_turbulence_plane(domain_node)

    scene = getattr(bpy.context, "scene", None)

    if scene is None:
        time_value = 0.0
    else:
        render = getattr(scene, "render", None)
        fps = max(1, int(getattr(render, "fps", 24)))

        time_value = (
            float(getattr(scene, "frame_current", 0))
            / float(fps)
        )

    resolution = float(domain_node.resolution)

    size_u = max(
        float(plane["size_u"]),
        resolution,
    )

    size_v = max(
        float(plane["size_v"]),
        resolution,
    )

    step_u = size_u / sample_count
    step_v = size_v / sample_count

    u_vec = plane["u"]
    v_vec = plane["v"]

    half_u = u_vec * (size_u * 0.5)
    half_v = v_vec * (size_v * 0.5)

    origin = plane["center"] - half_u - half_v

    ox, oy, oz = origin
    ux, uy, uz = u_vec
    vx, vy, vz = v_vec

    positions = []
    colors = []
    indices = []

    row_size = sample_count + 1

    amplitude = float(force_node.amplitude)
    scale = float(force_node.scale)
    frequency = float(force_node.frequency)
    seed = int(force_node.seed)
    hash_cache = {}

    inv_scale = 1.0 / scale
    time_offset = time_value * frequency

    max_value = max(abs(amplitude), 1.0e-6)
    color_scale = 0.5 / max_value

    amplitude_factor = max(
        0.0,
        min(1.0, amplitude / 4.0),
    )

    u_values = [
        ix * step_u
        for ix in range(row_size)
    ]
    
    noise_x_values = [
        u * inv_scale
        for u in u_values
    ]

    for iy in range(row_size):
        v_amount = iy * step_v

        base_x = ox + vx * v_amount
        base_y = oy + vy * v_amount
        base_z = oz + vz * v_amount

        noise_y = v_amount * inv_scale + time_offset

        for ix in range(row_size):
            u_amount = u_values[ix]
            noise_x = noise_x_values[ix]

            positions.append((
                base_x + ux * u_amount,
                base_y + uy * u_amount,
                base_z + uz * u_amount,
            ))

            value = amplitude * value_noise_2d(
                noise_x,
                noise_y,
                seed,
                hash_cache,
            )

            factor = 0.5 + value * color_scale

            if factor < 0.0:
                factor = 0.0
            elif factor > 1.0:
                factor = 1.0

            factor *= amplitude_factor

            colors.append((
                factor,
                factor,
                factor,
                1.0,
            ))

    for iy in range(sample_count):
        row0 = iy * row_size
        row1 = row0 + row_size

        for ix in range(sample_count):
            i00 = row0 + ix
            i10 = i00 + 1
            i01 = row1 + ix
            i11 = i01 + 1

            indices.append((
                i00,
                i10,
                i11,
            ))

            indices.append((
                i00,
                i11,
                i01,
            ))

    return positions, colors, indices


def choose_turbulence_plane(domain_node):
    width, depth, height = domain_dimensions(domain_node)
    center = Vector(domain_center(domain_node))

    region_data = getattr(bpy.context, "region_data", None)

    if region_data is None:
        view_direction = Vector((0.0, 0.0, -1.0))
    else:
        try:
            view_direction = region_data.view_rotation @ Vector((0.0, 0.0, -1.0))
        except Exception:
            view_direction = Vector((0.0, 0.0, -1.0))

    if view_direction.length <= 1.0e-9:
        view_direction = Vector((0.0, 0.0, -1.0))
    else:
        view_direction.normalize()

    planes = [
        {
            "normal": Vector((0.0, 0.0, 1.0)),
            "u": Vector((1.0, 0.0, 0.0)),
            "v": Vector((0.0, 1.0, 0.0)),
            "size_u": width,
            "size_v": depth,
            "center": center,
        },
        {
            "normal": Vector((0.0, 1.0, 0.0)),
            "u": Vector((1.0, 0.0, 0.0)),
            "v": Vector((0.0, 0.0, 1.0)),
            "size_u": width,
            "size_v": height,
            "center": center,
        },
        {
            "normal": Vector((1.0, 0.0, 0.0)),
            "u": Vector((0.0, 1.0, 0.0)),
            "v": Vector((0.0, 0.0, 1.0)),
            "size_u": depth,
            "size_v": height,
            "center": center,
        },
    ]

    return max(
        planes,
        key=lambda plane: abs(plane["normal"].dot(view_direction)),
    )


def turbulence_cache_key(force_node, domain_node, sample_count, plane):
    return (
        force_node.name,
        int(force_node.seed),
        round(float(force_node.scale), 6),
        round(float(force_node.amplitude), 6),
        round(float(force_node.frequency), 6),
        int(bpy.context.scene.frame_current),
        sample_count,
        tuple(round(x, 4) for x in plane["normal"]),
        tuple(round(x, 4) for x in domain_center(domain_node)),
        round(float(domain_node.resolution), 6),
    )


def turbulence_color(value, amplitude):
    max_value = max(abs(float(amplitude)), 1.0e-6)
    factor = max(0.0, min(1.0, 0.5 + (0.5 * value / max_value)))
    gray = factor
    return (gray, gray, gray, 1.0)


#-------------- turbulence ----------------
def turbulence_value(force_node, u, v, time_value):
    amplitude = float(force_node.amplitude)
    scale = float(force_node.scale)

    if abs(amplitude) <= 1.0e-9 or scale <= 1.0e-8:
        return 0.0

    inv_scale = 1.0 / scale
    frequency = float(force_node.frequency)
    seed = int(force_node.seed)

    x = u * inv_scale
    y = v * inv_scale + time_value * frequency

    return amplitude * value_noise_2d(x, y, seed)


def value_noise_2d(x, y, seed, hash_cache):
    x0 = math.floor(x)
    y0 = math.floor(y)

    fx = x - x0
    fy = y - y0

    tx = fx * fx * (3.0 - 2.0 * fx)
    ty = fy * fy * (3.0 - 2.0 * fy)

    x1 = x0 + 1
    y1 = y0 + 1

    key = (x0, y0)
    c00 = hash_cache.get(key)
    if c00 is None:
        c00 = hash_noise_2d(x0, y0, seed)
        hash_cache[key] = c00

    key = (x1, y0)
    c10 = hash_cache.get(key)
    if c10 is None:
        c10 = hash_noise_2d(x1, y0, seed)
        hash_cache[key] = c10

    key = (x0, y1)
    c01 = hash_cache.get(key)
    if c01 is None:
        c01 = hash_noise_2d(x0, y1, seed)
        hash_cache[key] = c01

    key = (x1, y1)
    c11 = hash_cache.get(key)
    if c11 is None:
        c11 = hash_noise_2d(x1, y1, seed)
        hash_cache[key] = c11

    a = c00 + tx * (c10 - c00)
    b = c01 + tx * (c11 - c01)

    return a + ty * (b - a)


def hash_noise_2d(ix, iy, seed):
    n = ix * 15731 + iy * 789221 + seed * 1013
    n = (n << 13) ^ n

    nn = n * (n * n * 15731 + 789221) + 1376312589
    nn &= 0x7FFFFFFF

    return float(nn) / 1073741824.0 - 1.0



