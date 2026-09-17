"""GPU raymarch preview for the newest shared-memory smoke/flame frame."""

from pathlib import Path
from multiprocessing import shared_memory
import threading

import bpy
import gpu
import numba
import numpy as np
from gpu_extras.batch import batch_for_shader


SHADER_DIRECTORY = Path(__file__).resolve().parent / "shaders"
VERTEX_SHADER_PATH = SHADER_DIRECTORY / "volume_preview.vert"
FRAGMENT_SHADER_PATH = SHADER_DIRECTORY / "volume_preview.frag"


_draw_handler = None
_shader = None
_batch = None
_texture = None
_grid_shape = None
_resolution = None
_bounds_min = None
_bounds_max = None
_capture_enabled = False
_latest_captured_index = -1
_pending_frame = None
_pending_lock = threading.Lock()


def configure(grid_shape, resolution):
    """Set the simulation grid geometry used by the preview."""
    global _grid_shape, _resolution, _bounds_min, _bounds_max, _batch
    global _latest_captured_index, _pending_frame

    _grid_shape = tuple(int(value) for value in grid_shape)
    _resolution = float(resolution)
    nx, ny, nz = _grid_shape
    _bounds_min = (-0.5 * nx * _resolution, -0.5 * ny * _resolution, 0.0)
    _bounds_max = (
        _bounds_min[0] + nx * _resolution,
        _bounds_min[1] + ny * _resolution,
        nz * _resolution,
    )
    _batch = None
    with _pending_lock:
        _latest_captured_index = -1
        _pending_frame = None


def set_enabled(enabled):
    """Enable or disable capture of incoming shared-memory frames."""
    global _capture_enabled
    enabled = bool(enabled)
    if enabled == _capture_enabled:
        return
    _capture_enabled = enabled
    if not enabled:
        clear_live_preview()


def capture_shared_frame(payload):
    """Copy the newest sparse fields before the writer reuses their shared memory."""
    global _latest_captured_index, _pending_frame
    if not _capture_enabled or _grid_shape is None:
        return

    frame_index = _frame_index_from_payload(payload)
    with _pending_lock:
        if frame_index <= _latest_captured_index:
            return

    active_info = payload["active_tiles"]
    active_tiles = _copy_shared_array(
        active_info["shm_name"],
        tuple(active_info["shape"]),
        np.int32,
        int(active_info["count"]),
    )

    pools = {}
    tile_size = 4
    for grid in payload.get("grids", ()):
        grid_name = grid.get("name")
        if grid_name not in {"density", "smoke", "flame"}:
            continue
        field_info = next(iter((grid.get("fields") or {}).values()), None)
        if field_info is None:
            continue
        tile_size = int(grid.get("tile_size", tile_size))
        pools[grid_name] = _copy_shared_array(
            field_info["shm_name"],
            tuple(field_info["shape"]),
            np.float32,
            int(grid.get("used_tile_count", 0)),
        )

    with _pending_lock:
        if _capture_enabled and frame_index > _latest_captured_index:
            _latest_captured_index = frame_index
            _pending_frame = (active_tiles, pools, tile_size)


def upload_pending_frame():
    """Assemble and upload the latest captured frame on Blender's main thread."""
    global _draw_handler, _pending_frame, _texture
    if not _capture_enabled:
        return

    with _pending_lock:
        pending_frame = _pending_frame
        _pending_frame = None
    if pending_frame is None:
        return

    fields = _build_dense_texture(*pending_frame)
    buffer = gpu.types.Buffer("FLOAT", fields.size, fields.ravel())
    texture = gpu.types.GPUTexture(_grid_shape, format="RG16F", data=buffer)
    texture.filter_mode(True)

    shader = _ensure_shader()
    _ensure_batch(shader)
    _texture = texture
    if _draw_handler is None:
        _draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            _draw_volume, (), "WINDOW", "POST_VIEW"
        )
    _redraw_viewports()


def clear_live_preview():
    """Stop drawing and release the temporary GPU texture."""
    global _draw_handler, _texture, _pending_frame

    if _draw_handler is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_draw_handler, "WINDOW")
        except (ReferenceError, RuntimeError):
            pass
        _draw_handler = None

    _texture = None
    with _pending_lock:
        _pending_frame = None
    _redraw_viewports()


def _frame_index_from_payload(payload):
    try:
        return int(Path(payload.get("output_path", "")).stem.removeprefix("frame_"))
    except ValueError:
        return -1


def _copy_shared_array(shm_name, shape, dtype, leading_count):
    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        source = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        return source[:leading_count].copy()
    finally:
        shm.close()


def _build_dense_texture(active_tiles, pools, tile_size):
    nx, ny, nz = _grid_shape
    fields = np.zeros((nz, ny, nx, 2), dtype=np.float32)
    empty_pool = np.empty((0, tile_size, tile_size, tile_size), dtype=np.float32)
    smoke_pool = pools.get("density", pools.get("smoke", empty_pool))
    flame_pool = pools.get("flame", empty_pool)
    _scatter_sparse_fields(fields, active_tiles, smoke_pool, flame_pool, tile_size)
    return fields


@numba.njit(cache=True, nogil=True)
def _scatter_sparse_fields(fields, active_tiles, smoke_pool, flame_pool, tile_size):
    has_smoke = smoke_pool.shape[0] > 0
    has_flame = flame_pool.shape[0] > 0
    nz, ny, nx, _channels = fields.shape

    for tile_number in range(active_tiles.shape[0]):
        pool_index = int(active_tiles[tile_number, 0])
        start_x = int(active_tiles[tile_number, 1])
        start_y = int(active_tiles[tile_number, 2])
        start_z = int(active_tiles[tile_number, 3])
        if pool_index < 0:
            continue
        for local_x in range(tile_size):
            x = start_x + local_x
            if x >= nx:
                continue
            for local_y in range(tile_size):
                y = start_y + local_y
                if y >= ny:
                    continue
                for local_z in range(tile_size):
                    z = start_z + local_z
                    if z >= nz:
                        continue
                    if has_smoke and pool_index < smoke_pool.shape[0]:
                        fields[z, y, x, 0] = smoke_pool[
                            pool_index, local_x, local_y, local_z
                        ]
                    if has_flame and pool_index < flame_pool.shape[0]:
                        fields[z, y, x, 1] = flame_pool[
                            pool_index, local_x, local_y, local_z
                        ]


def _ensure_shader():
    global _shader
    if _shader is not None:
        return _shader

    interface = gpu.types.GPUStageInterfaceInfo("continuum_flow_volume_interface")
    interface.smooth("VEC3", "world_position")
    shader_info = gpu.types.GPUShaderCreateInfo()
    shader_info.push_constant("MAT4", "view_projection_matrix")
    shader_info.push_constant("VEC3", "camera_position")
    shader_info.push_constant("VEC3", "bounds_min")
    shader_info.push_constant("VEC3", "bounds_max")
    shader_info.push_constant("FLOAT", "step_size")
    shader_info.push_constant("FLOAT", "density_scale")
    shader_info.push_constant("FLOAT", "flame_scale")
    shader_info.sampler(0, "FLOAT_3D", "volume_texture")
    shader_info.vertex_in(0, "VEC3", "position")
    shader_info.vertex_out(interface)
    shader_info.fragment_out(0, "VEC4", "frag_color")
    shader_info.vertex_source(_read_shader_source(VERTEX_SHADER_PATH))
    shader_info.fragment_source(_read_shader_source(FRAGMENT_SHADER_PATH))
    _shader = gpu.shader.create_from_info(shader_info)
    return _shader


def _read_shader_source(shader_path):
    try:
        return shader_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Could not read volume shader: {shader_path}") from exc


def _ensure_batch(shader):
    global _batch
    if _batch is not None:
        return _batch

    x0, y0, z0 = _bounds_min
    x1, y1, z1 = _bounds_max
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
    _batch = batch_for_shader(shader, "TRIS", {"position": vertices}, indices=indices)
    return _batch


def _draw_volume():
    if _texture is None or _bounds_min is None:
        return

    region_data = getattr(bpy.context, "region_data", None)
    if region_data is None:
        return

    shader = _ensure_shader()
    batch = _ensure_batch(shader)
    camera_position = region_data.view_matrix.inverted().translation
    diagonal = (
        sum(
            (maximum - minimum) ** 2
            for minimum, maximum in zip(_bounds_min, _bounds_max)
        )
        ** 0.5
    )
    step_size = max(_resolution * 0.75, diagonal / 512.0, 1.0e-5)
    camera_inside = all(
        minimum <= coordinate <= maximum
        for coordinate, minimum, maximum in zip(
            camera_position, _bounds_min, _bounds_max
        )
    )

    gpu.state.blend_set("ALPHA_PREMULT")
    gpu.state.depth_test_set("LESS_EQUAL")
    gpu.state.depth_mask_set(False)
    gpu.state.face_culling_set("FRONT" if camera_inside else "BACK")
    try:
        shader.bind()
        shader.uniform_float("view_projection_matrix", region_data.perspective_matrix)
        shader.uniform_float("camera_position", camera_position)
        shader.uniform_float("bounds_min", _bounds_min)
        shader.uniform_float("bounds_max", _bounds_max)
        shader.uniform_float("step_size", step_size)
        shader.uniform_float("density_scale", 2.5)
        shader.uniform_float("flame_scale", 5.0)
        shader.uniform_sampler("volume_texture", _texture)
        batch.draw(shader)
    finally:
        gpu.state.face_culling_set("NONE")
        gpu.state.depth_mask_set(True)
        gpu.state.depth_test_set("NONE")
        gpu.state.blend_set("NONE")


def _redraw_viewports():
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
