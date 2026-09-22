"""GPU raymarch preview for the newest shared-memory smoke/flame frame."""

from pathlib import Path
from multiprocessing import shared_memory
import threading

import time
import bpy
import gpu
import numba
import numpy as np
from gpu_extras.batch import batch_for_shader
from . import reference_frame

# -------------- shaders ----------------
SHADER_DIRECTORY = Path(__file__).resolve().parent / "shaders"
VERTEX_SHADER_PATH = SHADER_DIRECTORY / "volume_preview.vert"
FRAGMENT_SHADER_PATH = SHADER_DIRECTORY / "volume_preview.frag"

smoke_density = 10.0
smoke_color = (0.32, 0.34, 0.38)
flame_density = 5.0
flame_color = (1.0, 0.12, 0.01)

# -------------- vars ----------------
dense_fields = None
previous_active_tiles = None
draw_handler = None
shader = None
batch = None
texture = None
uniform_buffer = None
grid_shape = None
resolution = None
bounds_min = None
bounds_max = None
capture_enabled = False
latest_frame = -1
pending_frame = None
preview_frame_busy = False
pending_lock = threading.Lock()
simulation_reference = None


# -------------- methods ----------------
def get_shared_frame(payload):
    """Copy the newest sparse fields before the writer reuses their shared memory."""
    global latest_frame, pending_frame, preview_frame_busy

    if not capture_enabled or grid_shape is None:
        return

    frame_index = int(Path(payload["output_path"]).stem.removeprefix("frame_"))

    with pending_lock:
        if frame_index <= latest_frame:
            return

    preview_grids = [
        grid for grid in payload["grids"] if grid["name"] in {"density", "flame"}
    ]
    if not preview_grids:
        return

    with pending_lock:
        if frame_index <= latest_frame or preview_frame_busy:
            return
        preview_frame_busy = True

    try:
        active_info = payload["active_tiles"]
        active_tiles = copy_shared_array(
            active_info["shm_name"],
            tuple(active_info["shape"]),
            np.int32,
            int(active_info["count"]),
        )

        pools = {}
        tile_size = int(preview_grids[0]["tile_size"])

        for grid in preview_grids:
            grid_name = grid["name"]
            field_info = next(iter(grid["fields"].values()))

            pools[grid_name] = copy_shared_array(
                field_info["shm_name"],
                tuple(field_info["shape"]),
                np.float32,
                int(grid["used_tile_count"]),
            )

        with pending_lock:
            if capture_enabled and frame_index > latest_frame:
                latest_frame = frame_index
                pending_frame = (frame_index, active_tiles, pools, tile_size)
            else:
                preview_frame_busy = False
    except Exception:
        with pending_lock:
            preview_frame_busy = False
        raise


def copy_shared_array(shm_name, shape, dtype, leading_count):
    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        source = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        return source[:leading_count].copy()
    finally:
        shm.close()


def configure(new_grid_shape, new_resolution, simulation_node=None):
    """Set the simulation grid geometry used by the preview."""
    global grid_shape, resolution, bounds_min, bounds_max, batch
    global latest_frame, pending_frame, preview_frame_busy
    global simulation_reference

    grid_shape = new_grid_shape
    resolution = new_resolution
    nx, ny, nz = grid_shape
    bounds_min = (-0.5 * nx * resolution, -0.5 * ny * resolution, 0.0)
    bounds_max = (
        bounds_min[0] + nx * resolution,
        bounds_min[1] + ny * resolution,
        nz * resolution,
    )
    batch = None
    node_tree = getattr(simulation_node, "id_data", None)
    simulation_reference = (
        str(getattr(node_tree, "name", "")),
        str(getattr(simulation_node, "name", "")),
    )
    with pending_lock:
        latest_frame = -1
        pending_frame = None
        preview_frame_busy = False


def set_enabled(enabled):
    """Enable or disable capture of incoming shared-memory frames."""
    global capture_enabled
    if enabled == capture_enabled:
        return
    capture_enabled = enabled
    if not enabled:
        clear_live_preview()


def set_preview_settings(
    new_smoke_density,
    new_smoke_color,
    new_flame_density,
    new_flame_color,
):
    global smoke_density, smoke_color, flame_density, flame_color
    smoke_density = new_smoke_density
    smoke_color = tuple(new_smoke_color)
    flame_density = new_flame_density
    flame_color = tuple(new_flame_color)


def upload_pending_frame():
    """Assemble and upload the latest captured frame on Blender's main thread."""
    global draw_handler, pending_frame, preview_frame_busy, texture

    if not capture_enabled:
        return

    with pending_lock:
        next_frame = pending_frame
        pending_frame = None

    if next_frame is None:
        return

    frame_index, active_tiles, pools, tile_size = next_frame

    try:
        fields = build_dense_texture(active_tiles, pools, tile_size)

        buffer = gpu.types.Buffer(
            "FLOAT",
            fields.size,
            fields.ravel(),
        )

        next_texture = gpu.types.GPUTexture(
            grid_shape,
            format="RG16F",
            data=buffer,
        )
        next_texture.filter_mode(True)

        texture = next_texture

        active_shader = ensure_shader()
        ensure_batch(active_shader)

        if draw_handler is None:
            draw_handler = bpy.types.SpaceView3D.draw_handler_add(
                draw_volume,
                (),
                "WINDOW",
                "POST_VIEW",
            )

        redraw_viewports()
    finally:
        with pending_lock:
            preview_frame_busy = False


def clear_live_preview():
    """Stop drawing and release the temporary GPU texture."""
    global draw_handler, texture, pending_frame, preview_frame_busy

    if draw_handler is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(draw_handler, "WINDOW")
        except (ReferenceError, RuntimeError):
            pass
        draw_handler = None

    texture = None
    with pending_lock:
        pending_frame = None
        preview_frame_busy = False
    redraw_viewports()


def build_dense_texture(active_tiles, pools, tile_size):
    global dense_fields, previous_active_tiles

    nx, ny, nz = grid_shape

    if dense_fields is None or dense_fields.shape != (nz, ny, nx, 2):
        dense_fields = np.zeros(
            (nz, ny, nx, 2),
            dtype=np.float32,
        )
        previous_active_tiles = None

    smoke_pool = pools.get("density")
    flame_pool = pools.get("flame")

    if smoke_pool is None:
        smoke_pool = np.empty(
            (0, tile_size, tile_size, tile_size),
            dtype=np.float32,
        )

    if flame_pool is None:
        flame_pool = np.empty(
            (0, tile_size, tile_size, tile_size),
            dtype=np.float32,
        )

    if previous_active_tiles is not None:
        clear_sparse_fields(
            dense_fields,
            previous_active_tiles,
            tile_size,
        )

    scatter_sparse_fields(
        dense_fields,
        active_tiles,
        smoke_pool,
        flame_pool,
        tile_size,
    )

    previous_active_tiles = active_tiles.copy()

    return dense_fields


@numba.njit(cache=True, nogil=True)
def clear_sparse_fields(
    fields,
    active_tiles,
    tile_size,
):
    for tile_number in range(active_tiles.shape[0]):
        start_x = active_tiles[tile_number, 1]
        start_y = active_tiles[tile_number, 2]
        start_z = active_tiles[tile_number, 3]

        for local_z in range(tile_size):
            z = start_z + local_z

            for local_y in range(tile_size):
                y = start_y + local_y

                for local_x in range(tile_size):
                    x = start_x + local_x

                    fields[z, y, x, 0] = 0.0
                    fields[z, y, x, 1] = 0.0


@numba.njit(cache=True, nogil=True)
def scatter_sparse_fields(
    fields,
    active_tiles,
    smoke_pool,
    flame_pool,
    tile_size,
):
    has_smoke = smoke_pool.shape[0] > 0
    has_flame = flame_pool.shape[0] > 0

    for tile_number in range(active_tiles.shape[0]):
        pool_index = active_tiles[tile_number, 0]

        start_x = active_tiles[tile_number, 1]
        start_y = active_tiles[tile_number, 2]
        start_z = active_tiles[tile_number, 3]

        for local_z in range(tile_size):
            z = start_z + local_z

            for local_y in range(tile_size):
                y = start_y + local_y

                for local_x in range(tile_size):
                    x = start_x + local_x

                    if has_smoke:
                        fields[z, y, x, 0] = smoke_pool[
                            pool_index,
                            local_x,
                            local_y,
                            local_z,
                        ]

                    if has_flame:
                        fields[z, y, x, 1] = flame_pool[
                            pool_index,
                            local_x,
                            local_y,
                            local_z,
                        ]


def ensure_shader():
    global shader

    if shader is not None:
        return shader

    vertex_source = VERTEX_SHADER_PATH.read_text(encoding="utf-8")
    fragment_source = FRAGMENT_SHADER_PATH.read_text(encoding="utf-8")

    interface = gpu.types.GPUStageInterfaceInfo("continuum_flow_volume_interface")
    interface.smooth("VEC3", "world_position")

    shader_info = gpu.types.GPUShaderCreateInfo()

    shader_info.push_constant("MAT4", "view_projection_matrix")
    shader_info.push_constant("MAT4", "model_matrix")
    shader_info.typedef_source(
        """
        struct VolumeParameters {
            vec4 camera_position_and_step_size;
            vec4 bounds_min_and_smoke_density;
            vec4 bounds_max_and_flame_density;
            vec4 smoke_color;
            vec4 flame_color;
        };
        """
    )
    shader_info.uniform_buf(0, "VolumeParameters", "volume_parameters")

    shader_info.sampler(0, "FLOAT_3D", "volume_texture")

    shader_info.vertex_in(0, "VEC3", "position")
    shader_info.vertex_out(interface)
    shader_info.fragment_out(0, "VEC4", "frag_color")

    shader_info.vertex_source(vertex_source)
    shader_info.fragment_source(fragment_source)

    shader = gpu.shader.create_from_info(shader_info)
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
    if texture is None:
        return

    region_data = bpy.context.region_data

    active_shader = ensure_shader()
    active_batch = ensure_batch(active_shader)
    simulation_node = resolve_simulation_node()
    model_matrix = reference_frame.display_matrix(simulation_node)
    camera_position = (
        model_matrix.inverted_safe() @ region_data.view_matrix.inverted().translation
    )
    diagonal = (
        sum(
            (maximum - minimum) ** 2 for minimum, maximum in zip(bounds_min, bounds_max)
        )
        ** 0.5
    )
    step_size = max(resolution * 0.75, diagonal / 512.0)
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
        active_shader.uniform_sampler("volume_texture", texture)
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
    if node_tree is None:
        return None
    return node_tree.nodes.get(simulation_reference[1])


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
        screen = window.screen
        for area in screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()
