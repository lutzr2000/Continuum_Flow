import numpy as np
import trimesh
from numba import njit, prange


def combine_exported_obstacles(obstacle_entries):
    """
    Merge exported obstacle nodes into one kernel obstacle configuration.
    """
    mesh_objects = []
    for obstacle_entry in obstacle_entries:
        if obstacle_entry.get("shape") != "mesh":
            continue
        mesh_cfg = obstacle_entry.get("mesh", {})
        mesh_objects.extend(mesh_cfg.get("objects", ()))

    if mesh_objects:
        return {
            "shape": "mesh",
            "solid": True,
            "mesh": {
                "objects": mesh_objects,
            },
        }

    return {
        "shape": "empty",
        "solid": False,
        "mesh": {
            "objects": [],
        },
    }


def build_obstacle_data(domain_cfg, obstacle_entries):
    """
    Build the voxel obstacle mask from exported obstacle nodes.

    Dynamic obstacle runtime setup and host-side obstacle sampling are
    handled directly by this shared module.
    """
    obstacle_cfg = combine_exported_obstacles(obstacle_entries)
    nx = int(domain_cfg["grid"]["nx"])
    ny = int(domain_cfg["grid"]["ny"])
    nz = int(domain_cfg["grid"]["nz"])
    delta = float(domain_cfg["resolution"])
    origin_x = -0.5 * nx * delta
    origin_y = -0.5 * ny * delta
    origin_z = 0.0

    zero_velocity_x = np.zeros((nx, ny, nz), dtype=np.float32)
    zero_velocity_y = np.zeros((nx, ny, nz), dtype=np.float32)
    zero_velocity_z = np.zeros((nx, ny, nz), dtype=np.float32)

    if obstacle_cfg["shape"] == "mesh":
        mesh_cfg = obstacle_cfg.get("mesh", {})
        mesh_objects = mesh_cfg.get(
            "objects", mesh_cfg if isinstance(mesh_cfg, list) else []
        )
        obstacle_runtime = build_dynamic_runtime(
            nx,
            ny,
            nz,
            delta,
            mesh_objects,
            origin_x=origin_x,
            origin_y=origin_y,
            origin_z=origin_z,
        )
        obstacle_mask, obstacle_velocity_x, obstacle_velocity_y, obstacle_velocity_z = (
            update_dynamic_obstacle_data(
                obstacle_runtime,
                0.0,
            )
        )
        return {
            "config": obstacle_cfg,
            "mask": obstacle_mask,
            "velocity_x": obstacle_velocity_x,
            "velocity_y": obstacle_velocity_y,
            "velocity_z": obstacle_velocity_z,
            "runtime": obstacle_runtime,
            "is_animated": bool(obstacle_runtime.get("is_animated", False)),
        }

    if obstacle_cfg["shape"] == "empty":
        return {
            "config": obstacle_cfg,
            "mask": np.zeros((nx, ny, nz), dtype=np.bool_),
            "velocity_x": zero_velocity_x,
            "velocity_y": zero_velocity_y,
            "velocity_z": zero_velocity_z,
            "runtime": None,
            "is_animated": False,
        }

    raise ValueError(f"Unsupported obstacle shape '{obstacle_cfg['shape']}'")


def update_obstacle_mask(
    obstacle_data, time_value, obstacle_updater=None
):
    """
    Update the combined obstacle mask and obstacle wall velocities in-place.

    The optional obstacle_updater hook exists so CPU/GPU boundary modules can
    pass their backend-specific update function while sharing the same runtime
    bookkeeping code.
    """
    runtime = obstacle_data.get("runtime")
    if runtime is None:
        return obstacle_data["mask"]

    if obstacle_updater is None:
        obstacle_updater = update_dynamic_obstacle_data

    updated_mask, updated_velocity_x, updated_velocity_y, updated_velocity_z = (
        obstacle_updater(
            runtime,
            time_value,
            out_mask=obstacle_data["mask"],
            out_velocity_x=obstacle_data["velocity_x"],
            out_velocity_y=obstacle_data["velocity_y"],
            out_velocity_z=obstacle_data["velocity_z"],
        )
    )
    obstacle_data["mask"] = updated_mask
    obstacle_data["velocity_x"] = updated_velocity_x
    obstacle_data["velocity_y"] = updated_velocity_y
    obstacle_data["velocity_z"] = updated_velocity_z
    return updated_mask


# -----------------------------------------------------------------------------
# Small numba kernels
# -----------------------------------------------------------------------------


@njit(cache=True, parallel=True)
def _sample_mask_backwards(
    out, base, delta, ox, oy, oz, ix0, ix1, iy0, iy1, iz0, iz1, box, inv
):
    """
    Back-sample a local reference mask into the world grid.
    """
    base_ox, base_oy, base_oz = box
    bn_x, bn_y, bn_z = base.shape
    sx, sy, sz = ix1 - ix0 + 1, iy1 - iy0 + 1, iz1 - iz0 + 1

    for n in prange(sx * sy * sz):
        i = ix0 + n // (sy * sz)
        r = n % (sy * sz)
        j = iy0 + r // sz
        k = iz0 + r % sz

        x = np.float32(ox + i * delta)
        y = np.float32(oy + j * delta)
        z = np.float32(oz + k * delta)

        bx = inv[0, 0] * x + inv[0, 1] * y + inv[0, 2] * z + inv[0, 3]
        by = inv[1, 0] * x + inv[1, 1] * y + inv[1, 2] * z + inv[1, 3]
        bz = inv[2, 0] * x + inv[2, 1] * y + inv[2, 2] * z + inv[2, 3]

        bi = int(np.floor((bx - base_ox) / delta + 0.5))
        bj = int(np.floor((by - base_oy) / delta + 0.5))
        bk = int(np.floor((bz - base_oz) / delta + 0.5))

        if 0 <= bi < bn_x and 0 <= bj < bn_y and 0 <= bk < bn_z and base[bi, bj, bk]:
            out[i, j, k] = True


@njit(cache=True, parallel=True)
def _sample_obstacle_data_backwards(
    out_mask,
    out_vx,
    out_vy,
    out_vz,
    base,
    delta,
    ox,
    oy,
    oz,
    ix0,
    ix1,
    iy0,
    iy1,
    iz0,
    iz1,
    box,
    inv,
    rate,
):
    """
    Back-sample one moving obstacle mask and its wall velocity into the world grid.
    """
    base_ox, base_oy, base_oz = box
    bn_x, bn_y, bn_z = base.shape
    sx, sy, sz = ix1 - ix0 + 1, iy1 - iy0 + 1, iz1 - iz0 + 1

    for n in prange(sx * sy * sz):
        i = ix0 + n // (sy * sz)
        r = n % (sy * sz)
        j = iy0 + r // sz
        k = iz0 + r % sz

        x = np.float32(ox + i * delta)
        y = np.float32(oy + j * delta)
        z = np.float32(oz + k * delta)

        bx = inv[0, 0] * x + inv[0, 1] * y + inv[0, 2] * z + inv[0, 3]
        by = inv[1, 0] * x + inv[1, 1] * y + inv[1, 2] * z + inv[1, 3]
        bz = inv[2, 0] * x + inv[2, 1] * y + inv[2, 2] * z + inv[2, 3]

        bi = int(np.floor((bx - base_ox) / delta + 0.5))
        bj = int(np.floor((by - base_oy) / delta + 0.5))
        bk = int(np.floor((bz - base_oz) / delta + 0.5))

        if 0 <= bi < bn_x and 0 <= bj < bn_y and 0 <= bk < bn_z and base[bi, bj, bk]:
            out_mask[i, j, k] = True
            out_vx[i, j, k] = (
                rate[0, 0] * bx + rate[0, 1] * by + rate[0, 2] * bz + rate[0, 3]
            )
            out_vy[i, j, k] = (
                rate[1, 0] * bx + rate[1, 1] * by + rate[1, 2] * bz + rate[1, 3]
            )
            out_vz[i, j, k] = (
                rate[2, 0] * bx + rate[2, 1] * by + rate[2, 2] * bz + rate[2, 3]
            )


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------


def _as_f32(a):
    """
    Return a contiguous float32 view/copy of the given array-like input.
    """
    return np.ascontiguousarray(a, dtype=np.float32)


def _bounds_center_extent(bounds_min, bounds_max):
    """
    Convert min/max bounds into center/extent form for cheaper transforms.
    """
    bounds_min = np.asarray(bounds_min, dtype=np.float32)
    bounds_max = np.asarray(bounds_max, dtype=np.float32)
    center = (bounds_min + bounds_max) * np.float32(0.5)
    extent = (bounds_max - bounds_min) * np.float32(0.5)
    return _as_f32(center), _as_f32(extent)


def _bounds_to_indices(
    bounds_min, bounds_max, delta, origin=(0.0, 0.0, 0.0), shape=None
):
    """
    Convert world-space bounds into inclusive voxel index bounds, optionally clipped.
    """
    origin = np.asarray(origin, dtype=np.float32)
    lo = np.floor((bounds_min - origin) / delta).astype(np.int32)
    hi = np.ceil((bounds_max - origin) / delta).astype(np.int32)

    if shape is not None:
        hi_limit = np.asarray(shape, dtype=np.int32) - 1
        lo = np.maximum(lo, 0)
        hi = np.minimum(hi, hi_limit)

    return int(lo[0]), int(hi[0]), int(lo[1]), int(hi[1]), int(lo[2]), int(hi[2])


def _load_mesh_triangles(mesh_object):
    """
    Load one mesh object's triangle payload into contiguous float32 shape (n, 3, 3).
    """
    if mesh_object.get("triangles_file"):
        triangles = np.load(mesh_object["triangles_file"], allow_pickle=False)
    else:
        triangles = mesh_object.get("triangles", ())

    triangles = np.asarray(triangles, dtype=np.float32)
    if triangles.size == 0:
        return np.empty((0, 3, 3), dtype=np.float32)

    shape = mesh_object.get("triangles_shape") or (-1, 3, 3)
    return _as_f32(triangles.reshape(tuple(map(int, shape))))


def _voxelize_triangles(triangles, delta):
    """
    Voxelize one triangle array with trimesh and return dense local mask data.
    """
    if triangles.size == 0:
        return None

    vertices = _as_f32(triangles.reshape(-1, 3))
    faces = np.arange(vertices.shape[0], dtype=np.int64).reshape((-1, 3))
    triangle_mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False,
        validate=False,
    )
    voxel_grid = triangle_mesh.voxelized(pitch=float(delta)).fill()
    mask = np.ascontiguousarray(np.asarray(voxel_grid.matrix, dtype=np.bool_))
    if mask.size == 0 or not np.any(mask):
        return None

    origin = _as_f32(np.asarray(voxel_grid.translation, dtype=np.float32))
    bounds_max = _as_f32(
        origin
        + np.float32(delta)
        * (np.asarray(mask.shape, dtype=np.float32) - np.float32(1.0))
    )
    return {
        "mask": mask,
        "origin": origin,
        "bounds_min": origin,
        "bounds_max": bounds_max,
    }


def mesh(nx, ny, nz, delta, mesh_objects, origin_x=0.0, origin_y=0.0, origin_z=0.0):
    """
    Create one combined boolean mask for static Blender-exported triangle meshes.
    """
    out = np.zeros((nx, ny, nz), dtype=np.bool_)
    if not mesh_objects:
        return out

    origin = np.asarray((origin_x, origin_y, origin_z), dtype=np.float32)
    delta = np.float32(delta)
    identity_inv = np.eye(4, dtype=np.float32)

    for obj in mesh_objects:
        voxels = _voxelize_triangles(_load_mesh_triangles(obj), delta)
        if voxels is None:
            continue

        index_bounds = _bounds_to_indices(
            voxels["bounds_min"],
            voxels["bounds_max"],
            delta,
            origin,
            shape=out.shape,
        )
        ix0, ix1, iy0, iy1, iz0, iz1 = index_bounds
        if ix0 > ix1 or iy0 > iy1 or iz0 > iz1:
            continue

        _sample_mask_backwards(
            out,
            voxels["mask"],
            delta,
            origin[0],
            origin[1],
            origin[2],
            ix0,
            ix1,
            iy0,
            iy1,
            iz0,
            iz1,
            voxels["origin"],
            identity_inv,
        )

    return out

# -----------------------------------------------------------------------------
# Dynamic runtime
# -----------------------------------------------------------------------------


def _matrix_series(animation):
    """
    Normalize exported transform samples into a runtime-friendly series.
    """
    if not animation:
        return {
            "times": np.zeros(1, dtype=np.float32),
            "matrices_world": np.eye(4, dtype=np.float32)[None],
            "cursor": 0,
        }

    times = np.asarray(animation.get("times", (0.0,)), dtype=np.float32)
    matrices = np.asarray(
        animation.get("matrices_world", (np.eye(4, dtype=np.float32),)),
        dtype=np.float32,
    ).reshape((-1, 4, 4))
    n = min(max(times.size, 1), matrices.shape[0])
    return {
        "times": _as_f32(times[:n] if times.size else np.zeros(1, dtype=np.float32)),
        "matrices_world": _as_f32(
            matrices[:n] if matrices.size else np.eye(4, dtype=np.float32)[None]
        ),
        "cursor": 0,
    }


def _transform_series_is_animated(series):
    """
    Return whether a transform series can change over time.
    """
    times = series["times"]
    matrices = series["matrices_world"]
    if int(times.size) <= 1 or matrices.shape[0] <= 1:
        return False

    first = np.asarray(matrices[0], dtype=np.float32)
    for idx in range(1, matrices.shape[0]):
        if not np.allclose(matrices[idx], first, rtol=1.0e-5, atol=1.0e-6):
            return True
    return False


def _matrix_and_rate_at(series, t):
    """
    Return the interpolated world matrix and its piecewise-linear time derivative.
    """
    times = series["times"]
    matrices = series["matrices_world"]
    if times.size <= 1:
        return (
            np.asarray(matrices[0], dtype=np.float32),
            np.zeros((4, 4), dtype=np.float32),
        )

    if t < float(times[0]):
        return (
            np.asarray(matrices[0], dtype=np.float32),
            np.zeros((4, 4), dtype=np.float32),
        )

    cursor = min(int(series.get("cursor", 0)), int(times.size - 2))
    while cursor < times.size - 2 and t >= float(times[cursor + 1]):
        cursor += 1
    series["cursor"] = cursor

    if cursor >= times.size - 2 and t > float(times[-1]):
        return (
            np.asarray(matrices[-1], dtype=np.float32),
            np.zeros((4, 4), dtype=np.float32),
        )

    t0, t1 = float(times[cursor]), float(times[cursor + 1])
    if t1 <= t0:
        return (
            np.asarray(matrices[cursor], dtype=np.float32),
            np.zeros((4, 4), dtype=np.float32),
        )

    a = np.float32((t - t0) / (t1 - t0))
    matrix0 = np.asarray(matrices[cursor], dtype=np.float32)
    matrix1 = np.asarray(matrices[cursor + 1], dtype=np.float32)
    matrix = _as_f32(matrix0 * (1.0 - a) + matrix1 * a)
    rate = _as_f32((matrix1 - matrix0) / np.float32(t1 - t0))
    return matrix, rate


def _invert_affine_matrix(matrix):
    """
    Invert a 4x4 affine transform using only the 3x3 linear part and translation.
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    linear = matrix[:3, :3]
    translation = matrix[:3, 3]
    inv_linear = _as_f32(np.linalg.inv(linear))
    inv_translation = _as_f32(-(inv_linear @ translation))

    inv = np.eye(4, dtype=np.float32)
    inv[:3, :3] = inv_linear
    inv[:3, 3] = inv_translation
    return inv


def _transform_bounds(bounds_center, bounds_extent, matrix):
    """
    Transform an AABB using center/extent form to avoid rebuilding its eight corners.
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    linear = matrix[:3, :3]
    translation = matrix[:3, 3]
    center_world = _as_f32(linear @ bounds_center + translation)
    extent_world = _as_f32(np.abs(linear) @ bounds_extent)
    return center_world - extent_world, center_world + extent_world


def _resolve_dynamic_object_state(obj, time_value, delta, origin, shape):
    """
    Cache the expensive per-frame transform, inverse and index bounds for one object.
    """
    state = obj.get("dynamic_state")
    if state is not None and state.get("time_value") == float(time_value):
        return state

    matrix, matrix_rate = _matrix_and_rate_at(obj["transform_series"], time_value)
    bounds_min, bounds_max = _transform_bounds(
        obj["local_bounds_center"],
        obj["local_bounds_extent"],
        matrix,
    )
    index_bounds = _bounds_to_indices(
        bounds_min, bounds_max, delta, origin, shape=shape
    )
    active = not (
        index_bounds[0] > index_bounds[1]
        or index_bounds[2] > index_bounds[3]
        or index_bounds[4] > index_bounds[5]
    )

    state = {
        "time_value": float(time_value),
        "matrix": matrix,
        "matrix_rate": matrix_rate,
        "bounds_min": bounds_min,
        "bounds_max": bounds_max,
        "index_bounds": index_bounds,
        "active": bool(active),
    }
    if active:
        state["inv"] = _invert_affine_matrix(matrix)

    obj["dynamic_state"] = state
    return state


def _region_shape(index_bounds):
    """
    Return the voxel dimensions covered by one inclusive index-bounds tuple.
    """
    ix0, ix1, iy0, iy1, iz0, iz1 = index_bounds
    return (
        int(ix1 - ix0 + 1),
        int(iy1 - iy0 + 1),
        int(iz1 - iz0 + 1),
    )


def _merge_index_bounds(a, b):
    """
    Merge two inclusive index-bounds tuples into one covering both regions.
    """
    if a is None:
        return b
    if b is None:
        return a
    return (
        min(int(a[0]), int(b[0])),
        max(int(a[1]), int(b[1])),
        min(int(a[2]), int(b[2])),
        max(int(a[3]), int(b[3])),
        min(int(a[4]), int(b[4])),
        max(int(a[5]), int(b[5])),
    )


def build_dynamic_runtime(
    nx, ny, nz, delta, mesh_objects, origin_x=0.0, origin_y=0.0, origin_z=0.0
):
    """
    Precompute local masks and animation samples for dynamic obstacle/source masks.
    """
    objects = []
    has_animation = False
    for obj in mesh_objects:
        triangles = _load_mesh_triangles(obj)
        voxels = _voxelize_triangles(triangles, delta)
        if voxels is None or not np.any(voxels["mask"]):
            continue
        bounds_center, bounds_extent = _bounds_center_extent(
            voxels["bounds_min"], voxels["bounds_max"]
        )
        transform_series = _matrix_series(obj.get("transform_animation", {}))
        has_animation = has_animation or _transform_series_is_animated(transform_series)

        objects.append(
            {
                "object_name": obj.get("object_name"),
                "local_mask": voxels["mask"],
                "local_origin": voxels["origin"],
                "local_bounds_min": voxels["bounds_min"],
                "local_bounds_max": voxels["bounds_max"],
                "local_bounds_center": bounds_center,
                "local_bounds_extent": bounds_extent,
                "transform_series": transform_series,
                "dynamic_state": None,
            }
        )

    return {
        "objects": objects,
        "shape": (int(nx), int(ny), int(nz)),
        "delta": np.float32(delta),
        "origin": np.asarray((origin_x, origin_y, origin_z), dtype=np.float32),
        "is_animated": bool(has_animation),
    }


def update_dynamic_mask(runtime_data, time_value, out_mask=None):
    """
    Update one combined world-space mask by back-sampling all runtime objects.
    """
    shape = runtime_data["shape"]
    out = np.zeros(shape, dtype=np.bool_) if out_mask is None else out_mask
    out.fill(False)

    delta = np.float32(runtime_data["delta"])
    origin = np.asarray(runtime_data["origin"], dtype=np.float32)

    for obj in runtime_data["objects"]:
        state = _resolve_dynamic_object_state(obj, time_value, delta, origin, shape)
        if not state["active"]:
            continue
        ix0, ix1, iy0, iy1, iz0, iz1 = state["index_bounds"]

        _sample_mask_backwards(
            out,
            obj["local_mask"],
            delta,
            origin[0],
            origin[1],
            origin[2],
            ix0,
            ix1,
            iy0,
            iy1,
            iz0,
            iz1,
            obj["local_origin"],
            state["inv"],
        )

    return out


def update_dynamic_obstacle_data(
    runtime_data,
    time_value,
    out_mask=None,
    out_velocity_x=None,
    out_velocity_y=None,
    out_velocity_z=None,
):
    """
    Update one moving obstacle mask and its wall velocity fields on the host.
    """
    shape = runtime_data["shape"]
    out_mask = np.zeros(shape, dtype=np.bool_) if out_mask is None else out_mask
    out_velocity_x = (
        np.zeros(shape, dtype=np.float32) if out_velocity_x is None else out_velocity_x
    )
    out_velocity_y = (
        np.zeros(shape, dtype=np.float32) if out_velocity_y is None else out_velocity_y
    )
    out_velocity_z = (
        np.zeros(shape, dtype=np.float32) if out_velocity_z is None else out_velocity_z
    )

    out_mask.fill(False)
    out_velocity_x.fill(0.0)
    out_velocity_y.fill(0.0)
    out_velocity_z.fill(0.0)

    delta = np.float32(runtime_data["delta"])
    origin = np.asarray(runtime_data["origin"], dtype=np.float32)

    for obj in runtime_data["objects"]:
        state = _resolve_dynamic_object_state(obj, time_value, delta, origin, shape)
        if not state["active"]:
            continue
        ix0, ix1, iy0, iy1, iz0, iz1 = state["index_bounds"]

        _sample_obstacle_data_backwards(
            out_mask,
            out_velocity_x,
            out_velocity_y,
            out_velocity_z,
            obj["local_mask"],
            delta,
            origin[0],
            origin[1],
            origin[2],
            ix0,
            ix1,
            iy0,
            iy1,
            iz0,
            iz1,
            obj["local_origin"],
            state["inv"],
            state["matrix_rate"],
        )

    return out_mask, out_velocity_x, out_velocity_y, out_velocity_z
