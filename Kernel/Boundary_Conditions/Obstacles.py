import numpy as np
from numba import njit, prange


@njit(cache=True)
def _sort_first_n(values, count):
    """In-place insertion sort for the populated prefix of a float array."""
    for i in range(1, count):
        key = values[i]
        j = i - 1
        while j >= 0 and values[j] > key:
            values[j + 1] = values[j]
            j -= 1
        values[j + 1] = key


@njit(cache=True)
def _yz_projection_barycentric(y, z, triangle, eps):
    """Evaluate barycentric coordinates of a (y, z) sample in the triangle's YZ projection."""
    y0 = triangle[0, 1]
    z0 = triangle[0, 2]
    y1 = triangle[1, 1]
    z1 = triangle[1, 2]
    y2 = triangle[2, 1]
    z2 = triangle[2, 2]

    denominator = (z1 - z2) * (y0 - y2) + (y2 - y1) * (z0 - z2)
    if abs(denominator) <= eps:
        return False, 0.0, 0.0, 0.0

    w0 = ((z1 - z2) * (y - y2) + (y2 - y1) * (z - z2)) / denominator
    w1 = ((z2 - z0) * (y - y2) + (y0 - y2) * (z - z2)) / denominator
    w2 = 1.0 - w0 - w1

    if w0 < -eps or w1 < -eps or w2 < -eps:
        return False, 0.0, 0.0, 0.0
    return True, w0, w1, w2


@njit(cache=True)
def _build_scanline_counts(local_iy_min, local_iy_max, local_iz_min, local_iz_max, z_span, line_counts):
    """Count how many candidate triangles touch each local YZ scanline."""
    triangle_count = local_iy_min.shape[0]
    for triangle_index in range(triangle_count):
        iy_start = local_iy_min[triangle_index]
        iy_end = local_iy_max[triangle_index]
        iz_start = local_iz_min[triangle_index]
        iz_end = local_iz_max[triangle_index]

        for local_iy in range(iy_start, iy_end + 1):
            line_base = local_iy * z_span
            for local_iz in range(iz_start, iz_end + 1):
                line_counts[line_base + local_iz] += 1


@njit(cache=True)
def _write_scanline_candidates(local_iy_min, local_iy_max, local_iz_min, local_iz_max, z_span, write_positions, candidate_triangle_indices):
    """Write candidate triangle indices into the flattened scanline lookup."""
    triangle_count = local_iy_min.shape[0]
    for triangle_index in range(triangle_count):
        iy_start = local_iy_min[triangle_index]
        iy_end = local_iy_max[triangle_index]
        iz_start = local_iz_min[triangle_index]
        iz_end = local_iz_max[triangle_index]

        for local_iy in range(iy_start, iy_end + 1):
            line_base = local_iy * z_span
            for local_iz in range(iz_start, iz_end + 1):
                line_index = line_base + local_iz
                write_index = write_positions[line_index]
                candidate_triangle_indices[write_index] = triangle_index
                write_positions[line_index] = write_index + 1


def _build_scanline_candidates(triangles, delta, origin_y, origin_z, iy_min, iy_max, iz_min, iz_max, eps):
    """Build one compact candidate-triangle list for every YZ scanline."""
    triangle_count = int(triangles.shape[0])
    line_count = int((iy_max - iy_min + 1) * (iz_max - iz_min + 1))
    if triangle_count == 0 or line_count <= 0:
        return np.zeros(max(line_count, 0) + 1, dtype=np.int32), np.empty(0, dtype=np.int32), np.empty((0, 3, 3), dtype=np.float32)

    triangle_y = triangles[:, :, 1]
    triangle_z = triangles[:, :, 2]
    denominator = (
        (triangle_z[:, 1] - triangle_z[:, 2]) * (triangle_y[:, 0] - triangle_y[:, 2]) +
        (triangle_y[:, 2] - triangle_y[:, 1]) * (triangle_z[:, 0] - triangle_z[:, 2])
    )
    valid_mask = np.abs(denominator) > eps
    if not np.any(valid_mask):
        return np.zeros(line_count + 1, dtype=np.int32), np.empty(0, dtype=np.int32), np.empty((0, 3, 3), dtype=np.float32)

    valid_triangles = np.ascontiguousarray(triangles[valid_mask], dtype=np.float32)
    y_min_world = triangle_y[valid_mask].min(axis=1)
    y_max_world = triangle_y[valid_mask].max(axis=1)
    z_min_world = triangle_z[valid_mask].min(axis=1)
    z_max_world = triangle_z[valid_mask].max(axis=1)

    local_iy_min = np.floor((y_min_world - origin_y - eps) / delta).astype(np.int32) - iy_min
    local_iy_max = np.ceil((y_max_world - origin_y + eps) / delta).astype(np.int32) - iy_min
    local_iz_min = np.floor((z_min_world - origin_z - eps) / delta).astype(np.int32) - iz_min
    local_iz_max = np.ceil((z_max_world - origin_z + eps) / delta).astype(np.int32) - iz_min

    local_y_span = iy_max - iy_min
    local_z_span = iz_max - iz_min
    np.clip(local_iy_min, 0, local_y_span, out=local_iy_min)
    np.clip(local_iy_max, 0, local_y_span, out=local_iy_max)
    np.clip(local_iz_min, 0, local_z_span, out=local_iz_min)
    np.clip(local_iz_max, 0, local_z_span, out=local_iz_max)

    z_span = local_z_span + 1
    line_counts = np.zeros(line_count, dtype=np.int32)
    _build_scanline_counts(local_iy_min, local_iy_max, local_iz_min, local_iz_max, z_span, line_counts)

    line_offsets = np.empty(line_count + 1, dtype=np.int32)
    line_offsets[0] = 0
    np.cumsum(line_counts, dtype=np.int32, out=line_offsets[1:])

    candidate_triangle_indices = np.empty(int(line_offsets[-1]), dtype=np.int32)
    write_positions = line_offsets[:-1].copy()
    _write_scanline_candidates(
        local_iy_min,
        local_iy_max,
        local_iz_min,
        local_iz_max,
        z_span,
        write_positions,
        candidate_triangle_indices,
    )

    return line_offsets, candidate_triangle_indices, valid_triangles


@njit(cache=True, parallel=True)
def _fill_mesh_mask(mask, triangles, delta, origin_x, origin_y, origin_z,
                    ix_min, ix_max, iy_min, iy_max, iz_min, iz_max,
                    line_offsets, candidate_triangle_indices):
    """
    Fill a closed triangle mesh into a voxel mask using YZ scanlines.

    For each (y, z) line we compute x intersections only against triangles whose
    projected YZ bounds touch that scanline.
    """
    eps = np.float32(delta * 1.0e-5 + 1.0e-7)
    z_span = iz_max - iz_min + 1
    line_count = (iy_max - iy_min + 1) * z_span

    for line_index in prange(line_count):
        j = iy_min + line_index // z_span
        k = iz_min + line_index % z_span
        y = np.float32(origin_y + j * delta)
        z = np.float32(origin_z + k * delta)

        candidate_start = line_offsets[line_index]
        candidate_end = line_offsets[line_index + 1]
        candidate_count = candidate_end - candidate_start
        if candidate_count < 2:
            continue

        intersections = np.empty(candidate_count, dtype=np.float32)
        hit_count = 0

        for candidate_index in range(candidate_start, candidate_end):
            triangle_index = candidate_triangle_indices[candidate_index]
            triangle = triangles[triangle_index]
            is_inside, w0, w1, w2 = _yz_projection_barycentric(y, z, triangle, eps)
            if not is_inside:
                continue

            x = (
                w0 * triangle[0, 0] +
                w1 * triangle[1, 0] +
                w2 * triangle[2, 0]
            )
            intersections[hit_count] = np.float32(x)
            hit_count += 1

        if hit_count < 2:
            continue

        _sort_first_n(intersections, hit_count)

        unique_count = 0
        for hit_index in range(hit_count):
            x = intersections[hit_index]
            if unique_count == 0 or abs(x - intersections[unique_count - 1]) > eps:
                intersections[unique_count] = x
                unique_count += 1

        if unique_count < 2:
            continue

        paired_count = unique_count - (unique_count % 2)
        for pair_index in range(0, paired_count, 2):
            x_start = intersections[pair_index]
            x_end = intersections[pair_index + 1]
            if x_end < x_start:
                x_start, x_end = x_end, x_start

            x_start_index = max(ix_min, int(np.ceil((x_start - origin_x - eps) / delta)))
            x_end_index = min(ix_max, int(np.floor((x_end - origin_x + eps) / delta)))
            if x_end_index < x_start_index:
                continue

            for i in range(x_start_index, x_end_index + 1):
                mask[i, j, k] = True


def mesh(nx, ny, nz, delta, mesh_objects, origin_x=0.0, origin_y=0.0, origin_z=0.0):
    """
    Create a boolean mask for one or more Blender-exported triangle meshes.

    Args:
        nx, ny, nz (int): grid resolution
        delta (float): grid spacing
        mesh_objects (list[dict]): exported geometry payload from UI/Geometry_Export.py
        origin_x, origin_y, origin_z (float): world-space domain origin for index (0, 0, 0)
    Returns:
        numpy.ndarray: combined boolean obstacle mask
    """
    mask = np.zeros((nx, ny, nz), dtype=np.bool_)
    if not mesh_objects:
        return mask

    origin_x = np.float32(origin_x)
    origin_y = np.float32(origin_y)
    origin_z = np.float32(origin_z)
    delta = np.float32(delta)

    for mesh_object in mesh_objects:
        triangle_file = mesh_object.get("triangles_file")
        if triangle_file:
            triangles = np.load(triangle_file, allow_pickle=False)
        else:
            triangles = np.asarray(mesh_object.get("triangles", ()), dtype=np.float32)

        if triangles.size == 0:
            continue

        triangle_shape = mesh_object.get("triangles_shape", ())
        if triangle_shape:
            triangles = np.asarray(triangles, dtype=np.float32).reshape(tuple(int(axis_length) for axis_length in triangle_shape))
        else:
            triangles = np.asarray(triangles, dtype=np.float32).reshape((-1, 3, 3))
        bounds = mesh_object.get("bounds", {})
        if bounds:
            bounds_min = np.asarray(bounds.get("min", (0.0, 0.0, 0.0)), dtype=np.float32)
            bounds_max = np.asarray(bounds.get("max", (0.0, 0.0, 0.0)), dtype=np.float32)
        else:
            flat_vertices = triangles.reshape(-1, 3)
            bounds_min = flat_vertices.min(axis=0)
            bounds_max = flat_vertices.max(axis=0)

        ix_min = max(0, int(np.floor((bounds_min[0] - origin_x) / delta)))
        iy_min = max(0, int(np.floor((bounds_min[1] - origin_y) / delta)))
        iz_min = max(0, int(np.floor((bounds_min[2] - origin_z) / delta)))
        ix_max = min(nx - 1, int(np.ceil((bounds_max[0] - origin_x) / delta)))
        iy_max = min(ny - 1, int(np.ceil((bounds_max[1] - origin_y) / delta)))
        iz_max = min(nz - 1, int(np.ceil((bounds_max[2] - origin_z) / delta)))

        if ix_min > ix_max or iy_min > iy_max or iz_min > iz_max:
            continue

        eps = np.float32(delta * 1.0e-5 + 1.0e-7)
        line_offsets, candidate_triangle_indices, triangles = _build_scanline_candidates(
            triangles,
            delta,
            origin_y,
            origin_z,
            iy_min,
            iy_max,
            iz_min,
            iz_max,
            eps,
        )
        if candidate_triangle_indices.size == 0:
            continue

        _fill_mesh_mask(
            mask, triangles, delta, origin_x, origin_y, origin_z,
            ix_min, ix_max, iy_min, iy_max, iz_min, iz_max,
            line_offsets, candidate_triangle_indices,
        )

    return mask


def _load_mesh_triangles(mesh_object):
    """Load one mesh object's triangle payload into a contiguous float32 array."""
    triangle_file = mesh_object.get("triangles_file")
    if triangle_file:
        triangles = np.load(triangle_file, allow_pickle=False)
    else:
        triangles = np.asarray(mesh_object.get("triangles", ()), dtype=np.float32)

    if triangles.size == 0:
        return np.empty((0, 3, 3), dtype=np.float32)

    triangle_shape = mesh_object.get("triangles_shape", ())
    if triangle_shape:
        return np.ascontiguousarray(
            np.asarray(triangles, dtype=np.float32).reshape(tuple(int(axis_length) for axis_length in triangle_shape)),
            dtype=np.float32,
        )
    return np.ascontiguousarray(np.asarray(triangles, dtype=np.float32).reshape((-1, 3, 3)), dtype=np.float32)


def _matrix_animation_to_runtime(transform_animation):
    """Convert exported matrix samples into compact runtime arrays."""
    if not transform_animation:
        return {
            "times": np.zeros(1, dtype=np.float32),
            "matrices_world": np.eye(4, dtype=np.float32).reshape(1, 4, 4),
            "cursor": 0,
        }

    times = np.asarray(transform_animation.get("times", (0.0,)), dtype=np.float32)
    matrices = np.asarray(transform_animation.get("matrices_world", (np.eye(4, dtype=np.float32),)), dtype=np.float32)
    if times.size == 0:
        times = np.zeros(1, dtype=np.float32)
    if matrices.size == 0:
        matrices = np.eye(4, dtype=np.float32).reshape(1, 4, 4)
    matrices = matrices.reshape((-1, 4, 4))

    sample_count = min(times.shape[0], matrices.shape[0])
    if sample_count <= 0:
        sample_count = 1
        times = np.zeros(1, dtype=np.float32)
        matrices = np.eye(4, dtype=np.float32).reshape(1, 4, 4)
    else:
        times = times[:sample_count]
        matrices = matrices[:sample_count]

    return {
        "times": np.ascontiguousarray(times, dtype=np.float32),
        "matrices_world": np.ascontiguousarray(matrices, dtype=np.float32),
        "cursor": 0,
    }


def _interpolate_matrix_series(series, time_value):
    """Linearly interpolate one sampled world-matrix time series."""
    times = series["times"]
    matrices = series["matrices_world"]
    if times.size == 0 or matrices.shape[0] == 0:
        return np.eye(4, dtype=np.float32)
    if times.size == 1 or time_value <= float(times[0]):
        return np.asarray(matrices[0], dtype=np.float32)

    cursor = int(series.get("cursor", 0))
    last_segment = int(times.size - 2)
    if cursor > last_segment:
        cursor = last_segment

    while cursor < last_segment and time_value >= float(times[cursor + 1]):
        cursor += 1
    series["cursor"] = cursor

    if cursor >= last_segment and time_value >= float(times[-1]):
        return np.asarray(matrices[-1], dtype=np.float32)

    t0 = float(times[cursor])
    t1 = float(times[cursor + 1])
    if t1 <= t0:
        return np.asarray(matrices[cursor], dtype=np.float32)

    alpha = np.float32((float(time_value) - t0) / (t1 - t0))
    return np.ascontiguousarray(
        matrices[cursor] * (np.float32(1.0) - alpha) + matrices[cursor + 1] * alpha,
        dtype=np.float32,
    )


def _apply_transform_to_triangles(triangles, matrix_world):
    """Apply one 4x4 transform to a triangle array."""
    if triangles.size == 0:
        return np.empty((0, 3, 3), dtype=np.float32)

    flat_vertices = triangles.reshape(-1, 3)
    homogeneous_vertices = np.ones((flat_vertices.shape[0], 4), dtype=np.float32)
    homogeneous_vertices[:, :3] = flat_vertices
    transformed_vertices = homogeneous_vertices @ np.asarray(matrix_world, dtype=np.float32).T
    return np.ascontiguousarray(transformed_vertices[:, :3].reshape(triangles.shape), dtype=np.float32)


def _voxelize_triangles_to_local_mask(triangles, delta):
    """Voxelize one triangle mesh into a cropped mask in the mesh's own coordinate space."""
    if triangles.size == 0:
        return None

    flat_vertices = triangles.reshape(-1, 3)
    bounds_min = flat_vertices.min(axis=0).astype(np.float32)
    bounds_max = flat_vertices.max(axis=0).astype(np.float32)

    ix_min = int(np.floor(bounds_min[0] / delta))
    iy_min = int(np.floor(bounds_min[1] / delta))
    iz_min = int(np.floor(bounds_min[2] / delta))
    ix_max = int(np.ceil(bounds_max[0] / delta))
    iy_max = int(np.ceil(bounds_max[1] / delta))
    iz_max = int(np.ceil(bounds_max[2] / delta))

    if ix_min > ix_max or iy_min > iy_max or iz_min > iz_max:
        return None

    local_mask = np.zeros((ix_max - ix_min + 1, iy_max - iy_min + 1, iz_max - iz_min + 1), dtype=np.bool_)
    eps = np.float32(delta * 1.0e-5 + 1.0e-7)
    line_offsets, candidate_triangle_indices, valid_triangles = _build_scanline_candidates(
        triangles,
        np.float32(delta),
        np.float32(0.0),
        np.float32(0.0),
        iy_min,
        iy_max,
        iz_min,
        iz_max,
        eps,
    )
    if candidate_triangle_indices.size == 0:
        return None

    local_origin_x = np.float32(ix_min * delta)
    local_origin_y = np.float32(iy_min * delta)
    local_origin_z = np.float32(iz_min * delta)
    _fill_mesh_mask(
        local_mask,
        valid_triangles,
        np.float32(delta),
        local_origin_x,
        local_origin_y,
        local_origin_z,
        0,
        local_mask.shape[0] - 1,
        0,
        local_mask.shape[1] - 1,
        0,
        local_mask.shape[2] - 1,
        line_offsets,
        candidate_triangle_indices,
    )

    return {
        "mask": np.ascontiguousarray(local_mask),
        "index_min": np.asarray((ix_min, iy_min, iz_min), dtype=np.int32),
        "index_max": np.asarray((ix_max, iy_max, iz_max), dtype=np.int32),
        "local_origin": np.asarray((local_origin_x, local_origin_y, local_origin_z), dtype=np.float32),
        "local_bounds_min": np.asarray(
            (local_origin_x, local_origin_y, local_origin_z),
            dtype=np.float32,
        ),
        "local_bounds_max": np.asarray(
            (
                ix_max * delta,
                iy_max * delta,
                iz_max * delta,
            ),
            dtype=np.float32,
        ),
    }


def _transform_aabb(bounds_min, bounds_max, matrix_world):
    """Transform an axis-aligned box and return its transformed axis-aligned bounds."""
    corners = np.asarray(
        [
            [bounds_min[0], bounds_min[1], bounds_min[2], 1.0],
            [bounds_min[0], bounds_min[1], bounds_max[2], 1.0],
            [bounds_min[0], bounds_max[1], bounds_min[2], 1.0],
            [bounds_min[0], bounds_max[1], bounds_max[2], 1.0],
            [bounds_max[0], bounds_min[1], bounds_min[2], 1.0],
            [bounds_max[0], bounds_min[1], bounds_max[2], 1.0],
            [bounds_max[0], bounds_max[1], bounds_min[2], 1.0],
            [bounds_max[0], bounds_max[1], bounds_max[2], 1.0],
        ],
        dtype=np.float32,
    )
    transformed = corners @ np.asarray(matrix_world, dtype=np.float32).T
    transformed_xyz = transformed[:, :3]
    return transformed_xyz.min(axis=0).astype(np.float32), transformed_xyz.max(axis=0).astype(np.float32)


@njit(cache=True, parallel=True)
def _sample_object_mask_backwards(
    output_mask,
    base_mask,
    delta,
    domain_origin_x,
    domain_origin_y,
    domain_origin_z,
    target_ix_min,
    target_ix_max,
    target_iy_min,
    target_iy_max,
    target_iz_min,
    target_iz_max,
    base_origin_x,
    base_origin_y,
    base_origin_z,
    current_to_base_matrix,
):
    """Sample one cropped base mask into the new grid via inverse world transform."""
    span_x = target_ix_max - target_ix_min + 1
    span_y = target_iy_max - target_iy_min + 1
    span_z = target_iz_max - target_iz_min + 1
    sample_count = span_x * span_y * span_z
    if sample_count <= 0:
        return

    base_nx, base_ny, base_nz = base_mask.shape
    for linear_index in prange(sample_count):
        i = target_ix_min + linear_index // (span_y * span_z)
        yz_index = linear_index % (span_y * span_z)
        j = target_iy_min + yz_index // span_z
        k = target_iz_min + yz_index % span_z

        x_world = np.float32(domain_origin_x + i * delta)
        y_world = np.float32(domain_origin_y + j * delta)
        z_world = np.float32(domain_origin_z + k * delta)

        x_base = (
            current_to_base_matrix[0, 0] * x_world +
            current_to_base_matrix[0, 1] * y_world +
            current_to_base_matrix[0, 2] * z_world +
            current_to_base_matrix[0, 3]
        )
        y_base = (
            current_to_base_matrix[1, 0] * x_world +
            current_to_base_matrix[1, 1] * y_world +
            current_to_base_matrix[1, 2] * z_world +
            current_to_base_matrix[1, 3]
        )
        z_base = (
            current_to_base_matrix[2, 0] * x_world +
            current_to_base_matrix[2, 1] * y_world +
            current_to_base_matrix[2, 2] * z_world +
            current_to_base_matrix[2, 3]
        )

        base_i = int(np.floor((x_base - base_origin_x) / delta + 0.5))
        base_j = int(np.floor((y_base - base_origin_y) / delta + 0.5))
        base_k = int(np.floor((z_base - base_origin_z) / delta + 0.5))

        if (
            0 <= base_i < base_nx and
            0 <= base_j < base_ny and
            0 <= base_k < base_nz and
            base_mask[base_i, base_j, base_k]
        ):
            output_mask[i, j, k] = True


def build_dynamic_runtime(nx, ny, nz, delta, mesh_objects, origin_x=0.0, origin_y=0.0, origin_z=0.0):
    """Build one reusable runtime payload for dynamic obstacle/source masks."""
    runtime_objects = []
    for mesh_object in mesh_objects:
        triangles_local = _load_mesh_triangles(mesh_object)
        if triangles_local.size == 0:
            continue

        transform_series = _matrix_animation_to_runtime(mesh_object.get("transform_animation", {}))
        voxelized = _voxelize_triangles_to_local_mask(
            triangles_local,
            delta,
        )
        if voxelized is None or not np.any(voxelized["mask"]):
            continue

        runtime_objects.append(
            {
                "object_name": mesh_object.get("object_name"),
                "local_mask": voxelized["mask"],
                "local_index_min": voxelized["index_min"],
                "local_index_max": voxelized["index_max"],
                "local_origin": voxelized["local_origin"],
                "local_bounds_min": voxelized["local_bounds_min"],
                "local_bounds_max": voxelized["local_bounds_max"],
                "transform_series": transform_series,
            }
        )

    return {
        "objects": runtime_objects,
        "shape": (int(nx), int(ny), int(nz)),
        "delta": np.float32(delta),
        "origin": np.asarray((origin_x, origin_y, origin_z), dtype=np.float32),
    }


def update_dynamic_mask(runtime_data, time_value, out_mask=None):
    """Update one combined mask by back-sampling all dynamic runtime objects."""
    nx, ny, nz = runtime_data["shape"]
    if out_mask is None:
        out_mask = np.zeros((nx, ny, nz), dtype=np.bool_)
    else:
        out_mask.fill(False)

    if not runtime_data["objects"]:
        return out_mask

    delta = np.float32(runtime_data["delta"])
    origin_x, origin_y, origin_z = runtime_data["origin"]

    for runtime_object in runtime_data["objects"]:
        current_matrix_world = _interpolate_matrix_series(runtime_object["transform_series"], time_value)
        current_to_local = np.ascontiguousarray(np.linalg.inv(current_matrix_world), dtype=np.float32)

        current_bounds_min, current_bounds_max = _transform_aabb(
            runtime_object["local_bounds_min"],
            runtime_object["local_bounds_max"],
            current_matrix_world,
        )

        target_ix_min = max(0, int(np.floor((current_bounds_min[0] - origin_x) / delta)))
        target_iy_min = max(0, int(np.floor((current_bounds_min[1] - origin_y) / delta)))
        target_iz_min = max(0, int(np.floor((current_bounds_min[2] - origin_z) / delta)))
        target_ix_max = min(nx - 1, int(np.ceil((current_bounds_max[0] - origin_x) / delta)))
        target_iy_max = min(ny - 1, int(np.ceil((current_bounds_max[1] - origin_y) / delta)))
        target_iz_max = min(nz - 1, int(np.ceil((current_bounds_max[2] - origin_z) / delta)))

        if target_ix_min > target_ix_max or target_iy_min > target_iy_max or target_iz_min > target_iz_max:
            continue

        _sample_object_mask_backwards(
            out_mask,
            runtime_object["local_mask"],
            delta,
            np.float32(origin_x),
            np.float32(origin_y),
            np.float32(origin_z),
            target_ix_min,
            target_ix_max,
            target_iy_min,
            target_iy_max,
            target_iz_min,
            target_iz_max,
            runtime_object["local_origin"][0],
            runtime_object["local_origin"][1],
            runtime_object["local_origin"][2],
            current_to_local,
        )

    return out_mask
