"""Sparse GPU raymarch preview backed by an OpenCL snapshot."""

from pathlib import Path

import bpy
import gpu
from gpu_extras.batch import batch_for_shader

from Solver.Kernel import preview
from . import reference_frame

SHADER_DIRECTORY = Path(__file__).resolve().parent / "shaders"
VERTEX_SHADER_PATH = SHADER_DIRECTORY / "volume_preview.vert"
FRAGMENT_SHADER_PATH = SHADER_DIRECTORY / "volume_preview.frag"

smoke_density = 10.0
smoke_color = (0.32, 0.34, 0.38)
flame_density = 5.0
flame_color = (1.0, 0.12, 0.01)
draw_handler = None
shader = None
batch = None
tile_lookup_texture = None
field_atlas_texture = None
uniform_buffer = None
grid_shape = None
resolution = None
bounds_min = None
bounds_max = None
displayed_metadata = None
simulation_reference = None


def configure(new_grid_shape, new_resolution, new_simulation_reference=None):
    """Set the simulation grid geometry used by the preview."""
    global grid_shape, resolution, bounds_min, bounds_max, batch, simulation_reference

    grid_shape = tuple(new_grid_shape)
    resolution = new_resolution
    nx, ny, nz = grid_shape

    bounds_min = (-0.5 * nx * resolution, -0.5 * ny * resolution, 0.0)
    bounds_max = (
        bounds_min[0] + nx * resolution,
        bounds_min[1] + ny * resolution,
        nz * resolution,
    )

    batch = None
    simulation_reference = new_simulation_reference
    load_viewer_settings()


def get_solver_data(snapshot):
    """Upload one complete solver snapshot to Blender GPU textures."""
    global tile_lookup_texture, field_atlas_texture, displayed_metadata, draw_handler

    if (
        grid_shape != tuple(snapshot["grid_shape"])
        or resolution != snapshot["resolution"]
        or simulation_reference != snapshot["simulation_reference"]
    ):
        configure(
            snapshot["grid_shape"],
            snapshot["resolution"],
            snapshot["simulation_reference"],
        )

    lookup = snapshot["tile_lookup"]
    fields = snapshot["fields"]

    lookup_buffer = gpu.types.Buffer("FLOAT", lookup.size, lookup.ravel())
    fields_buffer = gpu.types.Buffer("FLOAT", fields.size, fields)
    tile_lookup_texture = gpu.types.GPUTexture(
        snapshot["tile_shape"], format="R32F", data=lookup_buffer
    )
    atlas_size = tuple(
        count * snapshot["tile_size"] for count in snapshot["atlas_tile_shape"]
    )
    field_atlas_texture = gpu.types.GPUTexture(
        atlas_size, format="RG32F", data=fields_buffer
    )
    field_atlas_texture.filter_mode(False)
    displayed_metadata = snapshot
    active_shader = ensure_shader()
    ensure_batch(active_shader)

    if draw_handler is None:
        draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            draw_volume, (), "WINDOW", "POST_VIEW"
        )
    redraw_viewports()


def update_live_preview():
    """Read complete snapshots without participating in solver transfer management."""
    snapshot = preview.take_snapshot()
    if snapshot is not None:
        get_solver_data(snapshot)
    elif not preview.is_enabled() and draw_handler is not None:
        clear_live_preview()

    return 0.01


def ensure_update_live_preview():
    """Ensure that the Blender-side snapshot consumer is running."""
    if not bpy.app.timers.is_registered(update_live_preview):
        bpy.app.timers.register(update_live_preview, first_interval=0.1)


def clear_live_preview():
    """Stop drawing while safely retiring any acquired OpenCL snapshot."""
    global draw_handler, tile_lookup_texture, field_atlas_texture
    global displayed_metadata, grid_shape, resolution, bounds_min, bounds_max
    global simulation_reference, batch

    if draw_handler is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(draw_handler, "WINDOW")
        except (ReferenceError, RuntimeError):
            pass
        draw_handler = None

    tile_lookup_texture = field_atlas_texture = displayed_metadata = None
    grid_shape = resolution = bounds_min = bounds_max = None
    simulation_reference = batch = None
    redraw_viewports()


def ensure_shader():
    global shader

    if shader is not None:
        return shader

    interface = gpu.types.GPUStageInterfaceInfo("continuum_flow_volume_interface")
    interface.smooth("VEC3", "world_position")
    info = gpu.types.GPUShaderCreateInfo()
    info.push_constant("MAT4", "view_projection_matrix")
    info.push_constant("MAT4", "model_matrix")
    info.typedef_source(
        """struct VolumeParameters { vec4 camera_position_and_step_size; vec4 bounds_min_and_smoke_density; vec4 bounds_max_and_flame_density; vec4 smoke_color; vec4 flame_color; vec4 tile_shape_and_size; vec4 atlas_tile_shape; };"""
    )
    info.uniform_buf(0, "VolumeParameters", "volume_parameters")
    info.sampler(0, "FLOAT_3D", "tile_lookup_texture")
    info.sampler(1, "FLOAT_3D", "field_atlas_texture")
    info.vertex_in(0, "VEC3", "position")
    info.vertex_out(interface)
    info.fragment_out(0, "VEC4", "frag_color")
    info.vertex_source(VERTEX_SHADER_PATH.read_text(encoding="utf-8"))
    info.fragment_source(FRAGMENT_SHADER_PATH.read_text(encoding="utf-8"))
    shader = gpu.shader.create_from_info(info)

    return shader


def ensure_batch(active_shader):
    global batch

    if batch is not None:
        return batch

    x0, y0, z0 = bounds_min
    x1, y1, z1 = bounds_max
    vertices = (
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    )
    indices = (
        (0, 2, 1),
        (0, 3, 2),
        (4, 5, 6),
        (4, 6, 7),
        (0, 1, 5),
        (0, 5, 4),
        (1, 2, 6),
        (1, 6, 5),
        (2, 3, 7),
        (2, 7, 6),
        (3, 0, 4),
        (3, 4, 7),
    )
    batch = batch_for_shader(
        active_shader, "TRIS", {"position": vertices}, indices=indices
    )
    return batch


def draw_volume():
    if tile_lookup_texture is None or field_atlas_texture is None:
        return

    region_data = bpy.context.region_data
    active_shader = ensure_shader()
    active_batch = ensure_batch(active_shader)
    model_matrix = reference_frame.display_matrix(resolve_simulation_node())
    camera_position = (
        model_matrix.inverted_safe() @ region_data.view_matrix.inverted().translation
    )
    diagonal = (
        sum(
            (maximum - minimum) ** 2 for minimum, maximum in zip(bounds_min, bounds_max)
        )
        ** 0.5
    )
    step_size = max(resolution * 0.75, diagonal / 256.0)
    camera_inside = all(
        minimum <= coordinate <= maximum
        for coordinate, minimum, maximum in zip(camera_position, bounds_min, bounds_max)
    )
    update_uniform_buffer(camera_position, step_size)
    gpu.state.blend_set("ALPHA_PREMULT")
    gpu.state.depth_test_set("LESS_EQUAL")
    gpu.state.depth_mask_set(False)
    gpu.state.face_culling_set("FRONT" if camera_inside else "BACK")
    try:
        active_shader.bind()
        active_shader.uniform_float(
            "view_projection_matrix", region_data.perspective_matrix
        )
        active_shader.uniform_float("model_matrix", model_matrix)
        active_shader.uniform_block("volume_parameters", uniform_buffer)
        active_shader.uniform_sampler("tile_lookup_texture", tile_lookup_texture)
        active_shader.uniform_sampler("field_atlas_texture", field_atlas_texture)
        active_batch.draw(active_shader)
    finally:
        gpu.state.face_culling_set("NONE")
        gpu.state.depth_mask_set(True)
        gpu.state.depth_test_set("NONE")
        gpu.state.blend_set("NONE")


def resolve_simulation_node():
    if not simulation_reference:
        return None

    node_tree = bpy.data.node_groups.get(simulation_reference[0])
    return None if node_tree is None else node_tree.nodes.get(simulation_reference[1])


def load_viewer_settings():
    """Load display-only settings from the connected viewer node."""
    global smoke_density, smoke_color, flame_density, flame_color

    simulation_node = resolve_simulation_node()
    result_socket = (
        None if simulation_node is None else simulation_node.outputs.get("Result")
    )
    if result_socket is None:
        return

    viewer_node = next(
        (
            link.to_node
            for link in result_socket.links
            if getattr(link.to_node, "bl_idname", "") == "CONTINUUM_FLOW_VIEWER_NODE"
            and bool(getattr(link.to_node, "live_preview", False))
        ),
        None,
    )
    if viewer_node is None:
        return

    smoke_density = float(viewer_node.preview_smoke_density)
    smoke_color = tuple(viewer_node.preview_smoke_color)
    flame_density = float(viewer_node.preview_flame_density)
    flame_color = tuple(viewer_node.preview_flame_color)


def update_uniform_buffer(camera_position, step_size):
    global uniform_buffer
    values = (
        camera_position.x,
        camera_position.y,
        camera_position.z,
        step_size,
        bounds_min[0],
        bounds_min[1],
        bounds_min[2],
        smoke_density,
        bounds_max[0],
        bounds_max[1],
        bounds_max[2],
        flame_density,
        smoke_color[0],
        smoke_color[1],
        smoke_color[2],
        0.0,
        flame_color[0],
        flame_color[1],
        flame_color[2],
        0.0,
        *displayed_metadata["tile_shape"],
        displayed_metadata["tile_size"],
        *displayed_metadata["atlas_tile_shape"],
        0.0,
    )
    buffer = gpu.types.Buffer("FLOAT", len(values), values)
    if uniform_buffer is None:
        uniform_buffer = gpu.types.GPUUniformBuf(buffer)
    else:
        uniform_buffer.update(buffer)


def redraw_viewports():
    window_manager = getattr(bpy.context, "window_manager", None)
    if window_manager is None:
        return
    for window in window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()
