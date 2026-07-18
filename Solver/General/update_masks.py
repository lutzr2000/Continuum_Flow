import numpy as np

def _as_f32(a):
    """Return a contiguous float32 view/copy of the given array-like input."""
    return np.ascontiguousarray(a, dtype=np.float32)


def _bounds_to_indices(
    bounds_min, bounds_max, delta, origin=(0.0, 0.0, 0.0), shape=None
):
    """Convert world-space bounds into inclusive voxel index bounds."""
    origin = np.asarray(origin, dtype=np.float32)
    lo = np.floor((bounds_min - origin) / delta).astype(np.int32)
    hi = np.ceil((bounds_max - origin) / delta).astype(np.int32)

    hi_limit = np.asarray(shape, dtype=np.int32) - 1
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, hi_limit)

    return int(lo[0]), int(hi[0]), int(lo[1]), int(hi[1]), int(lo[2]), int(hi[2])


def _transform_bounds(bounds_center, bounds_extent, matrix):
    """Transform one local AABB into world space using center/extent form."""
    matrix = np.asarray(matrix, dtype=np.float32)
    linear = matrix[:3, :3]
    translation = matrix[:3, 3]
    center_world = _as_f32(linear @ bounds_center + translation)
    extent_world = _as_f32(np.abs(linear) @ bounds_extent)
    return center_world - extent_world, center_world + extent_world


def _bounds_center_extent(bounds_min, bounds_max):
    """Convert min/max bounds into center/extent form for cheap affine transforms."""
    bounds_min = np.asarray(bounds_min, dtype=np.float32)
    bounds_max = np.asarray(bounds_max, dtype=np.float32)
    center = (bounds_min + bounds_max) * np.float32(0.5)
    extent = (bounds_max - bounds_min) * np.float32(0.5)
    return _as_f32(center), _as_f32(extent)


def _initial_world_matrix(mesh_object):
    """Return the first exported world transform for one mesh object."""
    animation = mesh_object.get("transform_animation") or {}
    matrices = np.asarray(
        animation.get("matrices_world", (np.eye(4, dtype=np.float32),)),
        dtype=np.float32,
    ).reshape((-1, 4, 4))
    if matrices.size == 0:
        return np.eye(4, dtype=np.float32)
    return _as_f32(matrices[0])


def _invert_affine_matrix(matrix):
    """Invert one affine 4x4 matrix using its linear part and translation."""
    matrix = np.asarray(matrix, dtype=np.float32)
    linear = matrix[:3, :3]
    translation = matrix[:3, 3]
    inv_linear = _as_f32(np.linalg.inv(linear))
    inv_translation = _as_f32(-(inv_linear @ translation))

    inv = np.eye(4, dtype=np.float32)
    inv[:3, :3] = inv_linear
    inv[:3, 3] = inv_translation
    return inv


def _animation_times(mesh_object):
    animation = mesh_object.get("transform_animation") or {}
    if "times" in animation:
        return np.asarray(animation.get("times") or (0.0,), dtype=np.float32)

    timeline = mesh_object.get("animation_timeline") or {}
    return np.asarray(timeline.get("times") or (0.0,), dtype=np.float32)


def _world_matrix_at_time(mesh_object, time_value):
    animation = mesh_object.get("transform_animation") or {}
    times = _animation_times(mesh_object)
    matrices = np.asarray(
        animation.get("matrices_world", (np.eye(4, dtype=np.float32),)),
        dtype=np.float32,
    ).reshape((-1, 4, 4))

    if matrices.size == 0:
        return np.eye(4, dtype=np.float32)
    if times.size <= 1 or matrices.shape[0] <= 1:
        return _as_f32(matrices[0])
    if time_value <= float(times[0]):
        return _as_f32(matrices[0])
    if time_value >= float(times[-1]):
        return _as_f32(matrices[min(len(matrices) - 1, len(times) - 1)])

    last_segment = min(len(times), len(matrices)) - 1
    for idx in range(last_segment):
        t0 = float(times[idx])
        t1 = float(times[idx + 1])
        if time_value <= t1:
            if t1 <= t0:
                return _as_f32(matrices[idx])
            alpha = np.float32((time_value - t0) / (t1 - t0))
            return _as_f32(matrices[idx] * (1.0 - alpha) + matrices[idx + 1] * alpha)

    return _as_f32(matrices[last_segment])


def _world_matrix_rate_at_time(mesh_object, time_value):
    animation = mesh_object.get("transform_animation") or {}
    times = _animation_times(mesh_object)
    matrices = np.asarray(
        animation.get("matrices_world", (np.eye(4, dtype=np.float32),)),
        dtype=np.float32,
    ).reshape((-1, 4, 4))

    if times.size <= 1 or matrices.shape[0] <= 1:
        return np.zeros((4, 4), dtype=np.float32)

    last_segment = min(len(times), len(matrices)) - 1
    if time_value <= float(times[0]):
        idx = 0
    elif time_value >= float(times[last_segment]):
        idx = max(0, last_segment - 1)
    else:
        idx = 0
        for candidate in range(last_segment):
            if time_value <= float(times[candidate + 1]):
                idx = candidate
                break

    t0 = float(times[idx])
    t1 = float(times[idx + 1])
    if t1 <= t0:
        return np.zeros((4, 4), dtype=np.float32)

    return _as_f32((matrices[idx + 1] - matrices[idx]) / np.float32(t1 - t0))