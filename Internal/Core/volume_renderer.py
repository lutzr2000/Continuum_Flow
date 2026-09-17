"""GPU raymarch preview for the newest smoke/flame VDB frame."""

from pathlib import Path

import bpy
import gpu
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
_loaded_filepath = None


def configure(grid_shape, resolution):
    """Set the simulation grid geometry used by the preview."""
    global _grid_shape, _resolution, _bounds_min, _bounds_max, _batch

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


def show_live_preview(filepath):
    """Upload smoke and flame from one VDB and display them in the viewport."""
    global _draw_handler, _loaded_filepath, _texture

    filepath = Path(filepath).resolve()
    normalized_filepath = str(filepath)
    if not filepath.is_file() or normalized_filepath == _loaded_filepath:
        return
    if _grid_shape is None:
        raise RuntimeError("Volume preview was not configured for a simulation grid.")

    fields = _read_preview_fields(filepath)
    buffer = gpu.types.Buffer("FLOAT", fields.size, fields.ravel())
    texture = gpu.types.GPUTexture(_grid_shape, format="RG16F", data=buffer)
    texture.filter_mode(True)

    shader = _ensure_shader()
    _ensure_batch(shader)
    _texture = texture
    _loaded_filepath = normalized_filepath
    if _draw_handler is None:
        _draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            _draw_volume, (), "WINDOW", "POST_VIEW"
        )
    _redraw_viewports()


def clear_live_preview():
    """Stop drawing and release the temporary GPU texture."""
    global _draw_handler, _texture, _loaded_filepath

    if _draw_handler is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_draw_handler, "WINDOW")
        except (ReferenceError, RuntimeError):
            pass
        _draw_handler = None

    _texture = None
    _loaded_filepath = None
    _redraw_viewports()


def _read_preview_fields(filepath):
    import openvdb

    nx, ny, nz = _grid_shape
    smoke = np.zeros((nx, ny, nz), dtype=np.float32)
    flame = np.zeros((nx, ny, nz), dtype=np.float32)
    grid_names = {grid.name for grid in openvdb.readAllGridMetadata(str(filepath))}

    smoke_grid_name = next(
        (name for name in ("density", "smoke") if name in grid_names),
        None,
    )
    if smoke_grid_name is not None:
        openvdb.read(str(filepath), gridname=smoke_grid_name).copyToArray(smoke)
    if "flame" in grid_names:
        openvdb.read(str(filepath), gridname="flame").copyToArray(flame)

    # GPU textures store X as their fastest-changing spatial coordinate.
    fields = np.empty((nz, ny, nx, 2), dtype=np.float32)
    fields[:, :, :, 0] = smoke.transpose(2, 1, 0)
    fields[:, :, :, 1] = flame.transpose(2, 1, 0)
    return fields


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
