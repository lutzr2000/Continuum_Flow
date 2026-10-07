import math
from pathlib import Path
from . import viewer
from . import reference_frame

import bpy
from .domain_grid import dimensions
import gpu
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

draw_handler = None
current_drawn_force = None
turbulence_preview_shader = None
SHADER_DIRECTORY = Path(__file__).resolve().parent / "shaders"
TURBULENCE_VERTEX_SHADER_PATH = SHADER_DIRECTORY / "turbulence_preview.vert"
TURBULENCE_FRAGMENT_SHADER_PATH = SHADER_DIRECTORY / "turbulence_preview.frag"


# -------------- general ----------------
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

    node_tree = bpy.data.node_groups.get(current_drawn_force.get("node_tree_name", ""))
    if node_tree is None:
        return

    force_node = node_tree.nodes.get(current_drawn_force.get("node_name", ""))
    if force_node is None:
        return

    if not viewer.overlay_enabeld():
        return

    simulation_node = get_linked_simulation_node(force_node)
    domain_node = get_linked_domain_node(simulation_node)

    force_type = getattr(force_node, "bl_idname", "")

    if force_type == "CONTINUUM_FLOW_FORCE_CONSTANT_NODE":
        segments = build_constant_force_segments(
            force_node, simulation_node, domain_node
        )
        if not segments:
            return
        segments = reference_frame.transform_positions(segments, simulation_node)

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
        segments = build_swirl_force_segments(force_node, simulation_node, domain_node)
        if not segments:
            return
        segments = reference_frame.transform_positions(segments, simulation_node)

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
        draw_turbulence_force_preview(force_node, simulation_node, domain_node)
        return


# -------------- helper ----------------
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
        if (
            getattr(simulation_node, "bl_idname", "")
            == "CONTINUUM_FLOW_SIMULATION_NODE"
        ):
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
    return dimensions(domain_node)


def domain_center(domain_node):
    width, depth, height = domain_dimensions(domain_node)
    return (0.0, 0.0, height * 0.5)


def domain_bounds(domain_node):
    width, depth, height = domain_dimensions(domain_node)
    return (
        (-0.5 * width, -0.5 * depth, 0.0),
        (0.5 * width, 0.5 * depth, height),
    )


def force_property_value(force_node, property_name):
    """Read the value currently displayed in the node UI."""
    from ..UI import animation_proxy

    return animation_proxy.scene_proxy_value(
        getattr(bpy.context, "scene", None),
        force_node,
        property_name,
    )


def ensure_turbulence_preview_shader():
    global turbulence_preview_shader

    if turbulence_preview_shader is not None:
        return turbulence_preview_shader

    interface = gpu.types.GPUStageInterfaceInfo(
        "continuum_flow_turbulence_preview_interface"
    )
    interface.smooth("VEC3", "noise_position")

    info = gpu.types.GPUShaderCreateInfo()
    info.push_constant("MAT4", "view_projection_matrix")
    info.push_constant("FLOAT", "scale")
    info.push_constant("FLOAT", "animation_factor")
    info.push_constant("INT", "seed")
    info.vertex_in(0, "VEC3", "position")
    info.vertex_in(1, "VEC3", "coordinate")
    info.vertex_out(interface)
    info.fragment_out(0, "VEC4", "frag_color")
    info.vertex_source(TURBULENCE_VERTEX_SHADER_PATH.read_text(encoding="utf-8"))
    info.fragment_source(TURBULENCE_FRAGMENT_SHADER_PATH.read_text(encoding="utf-8"))
    turbulence_preview_shader = gpu.shader.create_from_info(info)

    return turbulence_preview_shader


def turbulence_plane(simulation_node, domain_node):
    if simulation_node is None or domain_node is None:
        return []

    region_data = getattr(bpy.context, "region_data", None)
    if region_data is None:
        return []

    view_direction = region_data.view_rotation @ Vector((0.0, 0.0, -1.0))
    model_matrix = reference_frame.display_matrix(simulation_node)
    local_view_direction = model_matrix.inverted_safe().to_3x3() @ view_direction

    if local_view_direction.length <= 1.0e-9:
        return []

    axis = max(range(3), key=lambda index: abs(local_view_direction[index]))
    box_min, box_max = domain_bounds(domain_node)
    x0, y0, z0 = box_min
    x1, y1, z1 = box_max
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    cz = (z0 + z1) * 0.5

    if axis == 0:
        return [
            Vector((cx, y0, z0)),
            Vector((cx, y1, z0)),
            Vector((cx, y1, z1)),
            Vector((cx, y0, z1)),
        ]

    if axis == 1:
        return [
            Vector((x0, cy, z0)),
            Vector((x0, cy, z1)),
            Vector((x1, cy, z1)),
            Vector((x1, cy, z0)),
        ]

    return [
        Vector((x0, y0, cz)),
        Vector((x1, y0, cz)),
        Vector((x1, y1, cz)),
        Vector((x0, y1, cz)),
    ]


def draw_turbulence_force_preview(force_node, simulation_node, domain_node):
    local_positions = turbulence_plane(simulation_node, domain_node)
    if len(local_positions) < 3:
        return

    region_data = getattr(bpy.context, "region_data", None)
    if region_data is None:
        return

    model_matrix = reference_frame.display_matrix(simulation_node)
    world_positions = [tuple(model_matrix @ position) for position in local_positions]
    noise_coordinates = [tuple(position) for position in local_positions]
    indices = [(0, index, index + 1) for index in range(1, len(local_positions) - 1)]

    shader = ensure_turbulence_preview_shader()
    batch = batch_for_shader(
        shader,
        "TRIS",
        {
            "position": world_positions,
            "coordinate": noise_coordinates,
        },
        indices=indices,
    )
    frequency = max(abs(float(force_node.frequency)), 1.0e-6)
    scene = getattr(bpy.context, "scene", None)
    render = getattr(scene, "render", None)
    scene_fps = float(getattr(render, "fps", 24.0))
    scene_fps_base = max(float(getattr(render, "fps_base", 1.0)), 1.0e-9)
    fps = max(scene_fps / scene_fps_base, 1.0)
    start_frame = float(getattr(simulation_node, "start_frame", 1.0))
    current_frame = float(getattr(scene, "frame_current", start_frame))
    time_value = (current_frame - start_frame) / fps
    animation_factor = math.sin(time_value * frequency)

    gpu.state.blend_set("NONE")
    gpu.state.depth_test_set("LESS_EQUAL")
    shader.bind()
    shader.uniform_float("view_projection_matrix", region_data.perspective_matrix)
    shader.uniform_float("scale", float(force_node.scale))
    shader.uniform_float("animation_factor", animation_factor)
    shader.uniform_int("seed", int(force_node.seed))
    batch.draw(shader)
    gpu.state.depth_test_set("NONE")


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
        start,
        end,
        left_head,
        end,
        right_head,
        end,
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


# -------------- constant force ----------------
def build_constant_force_segments(force_node, simulation_node, domain_node):
    """
    Build a simple arrow in the domain center for one constant force node.
    """
    if simulation_node is None:
        return []

    if domain_node is None:
        return []

    force_vector = Vector(
        (
            float(force_property_value(force_node, "fx")),
            float(force_property_value(force_node, "fy")),
            float(force_property_value(force_node, "fz")),
        )
    )

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


# -------------- swirl force ----------------
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

    axis = Vector(force_property_value(force_node, "axis"))

    if axis.length <= 1.0e-9:
        return []

    axis.normalize()

    radius = float(force_property_value(force_node, "radius"))

    if radius <= 0.0:
        return []

    axis_u, axis_v = perpendicular_basis(axis)

    if axis_u is None or axis_v is None:
        return []

    origin = Vector(force_property_value(force_node, "origin"))

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

    valid_start_points = [point for point in start_points if point is not None]

    valid_end_points = [point for point in end_points if point is not None]

    # cylinder does not intersect domain
    if not valid_start_points or not valid_end_points:
        return []

    segments = []

    segments.extend(rim_segments(start_points))

    segments.extend(rim_segments(end_points))

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

        segments.extend(
            (
                start,
                end,
            )
        )

    width, depth, height = domain_dimensions(domain_node)

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

    mid_center = (start_center + end_center) * 0.5

    # Rotationspfeil
    segments.extend(
        build_swirl_rotation_arrow(
            mid_center,
            axis,
            axis_u,
            radius,
            float(force_property_value(force_node, "strength")),
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

        segments.extend(
            (
                p0,
                p1,
            )
        )

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
        angle = (float(index) / float(segment_count)) * (2.0 * math.pi)

        radial = axis_u * (math.cos(angle) * radius) + axis_v * (
            math.sin(angle) * radius
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

        start_points.append(offset_origin + axis * t_min)

        end_points.append(offset_origin + axis * t_max)

    return start_points, end_points
