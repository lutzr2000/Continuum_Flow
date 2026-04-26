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


@njit(cache=True, parallel=True)
def _fill_mesh_mask(mask, triangles, delta, origin_x, origin_y, origin_z,
                    ix_min, ix_max, iy_min, iy_max, iz_min, iz_max):
    """
    Fill a closed triangle mesh into a voxel mask using YZ scanlines.

    For each (y, z) line we compute all x intersections with the triangle soup,
    sort them, deduplicate shared-edge hits, and fill between intersection pairs.
    """
    eps = np.float32(delta * 1.0e-5 + 1.0e-7)
    z_span = iz_max - iz_min + 1
    line_count = (iy_max - iy_min + 1) * z_span
    triangle_count = triangles.shape[0]

    for line_index in prange(line_count):
        j = iy_min + line_index // z_span
        k = iz_min + line_index % z_span
        y = np.float32(origin_y + j * delta)
        z = np.float32(origin_z + k * delta)

        intersections = np.empty(triangle_count, dtype=np.float32)
        hit_count = 0

        for triangle_index in range(triangle_count):
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

        _fill_mesh_mask(
            mask, triangles, delta, origin_x, origin_y, origin_z,
            ix_min, ix_max, iy_min, iy_max, iz_min, iz_max
        )

    return mask
