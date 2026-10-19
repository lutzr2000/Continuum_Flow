from typing import Any
from pathlib import Path

import numpy as np
import pyopencl as cl

from . import helper


def triangulate_faces(points, face_counts, face_indices, hole_indices=()):
    """Triangulate USD polygon faces once, using the first point sample."""
    triangles = []
    offset = 0
    holes = set(int(index) for index in hole_indices)

    for face_index, count in enumerate(face_counts):
        count = int(count)
        face = [int(index) for index in face_indices[offset : offset + count]]
        offset += count

        if face_index in holes or count < 3:
            continue
        if count == 3:
            triangles.append(face)
            continue

        triangles.extend(triangulate_polygon(points, face))

    if offset != len(face_indices):
        raise ValueError("Invalid USD mesh topology: face counts do not match indices")

    return np.ascontiguousarray(triangles, dtype=np.int32).reshape(-1, 3)


def triangulate_polygon(points, face):
    """Ear-clip one planar polygon, including concave Blender/USD faces."""
    polygon = np.asarray(points[face], dtype=np.float64)

    # Newell's method gives a stable face normal without assuming a triangle.
    following = np.roll(polygon, -1, axis=0)
    normal = np.array(
        [
            np.sum(
                (polygon[:, 1] - following[:, 1]) * (polygon[:, 2] + following[:, 2])
            ),
            np.sum(
                (polygon[:, 2] - following[:, 2]) * (polygon[:, 0] + following[:, 0])
            ),
            np.sum(
                (polygon[:, 0] - following[:, 0]) * (polygon[:, 1] + following[:, 1])
            ),
        ]
    )
    drop_axis = int(np.argmax(np.abs(normal)))
    projected = np.delete(polygon, drop_axis, axis=1)
    signed_area = np.sum(
        projected[:, 0] * np.roll(projected[:, 1], -1)
        - np.roll(projected[:, 0], -1) * projected[:, 1]
    )
    orientation = 1.0 if signed_area >= 0.0 else -1.0
    scale = max(float(np.ptp(projected, axis=0).max()), 1.0)
    epsilon = np.finfo(np.float64).eps * scale * scale * 32.0

    remaining = list(range(len(face)))
    result = []
    while len(remaining) > 3:
        ear_found = False
        for position, current in enumerate(remaining):
            previous = remaining[position - 1]
            following_index = remaining[(position + 1) % len(remaining)]
            a, b, c = projected[[previous, current, following_index]]
            cross = _cross_2d(b - a, c - b)
            if orientation * cross <= epsilon:
                continue

            others = [
                index
                for index in remaining
                if index not in (previous, current, following_index)
            ]
            if any(
                _point_in_triangle(projected[index], a, b, c, orientation, epsilon)
                for index in others
            ):
                continue

            result.append([face[previous], face[current], face[following_index]])
            del remaining[position]
            ear_found = True
            break

        if not ear_found:
            raise ValueError(
                "Cannot triangulate a degenerate or self-intersecting USD face"
            )

    result.append([face[index] for index in remaining])
    return result


def _point_in_triangle(point, a, b, c, orientation, epsilon):
    edges = ((a, b), (b, c), (c, a))
    return all(
        orientation * _cross_2d(end - start, point - start) >= -epsilon
        for start, end in edges
    )


def _cross_2d(a, b):
    return a[0] * b[1] - a[1] * b[0]


def load_usdc(path: Path) -> dict[str, Any]:
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(path))

    mesh = next(
        (UsdGeom.Mesh(prim) for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)),
        None,
    )

    points_attr = mesh.GetPointsAttr()
    sample_codes = points_attr.GetTimeSamples()

    points = (
        [np.asarray(points_attr.Get(code), dtype=np.float32) for code in sample_codes]
        if sample_codes
        else [np.asarray(points_attr.Get(), dtype=np.float32)]
    )

    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
    face_counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int32)
    hole_indices = mesh.GetHoleIndicesAttr().Get() or ()
    face_triangles = triangulate_faces(points[0], face_counts, indices, hole_indices)

    times_per_second = float(stage.GetTimeCodesPerSecond()) or 24.0
    start_code = float(stage.GetStartTimeCode())

    times = np.asarray(
        (
            [(float(code) - start_code) / times_per_second for code in sample_codes]
            if sample_codes
            else [0.0]
        ),
        dtype=np.float64,
    )

    return {
        "times": times,
        "points": points,
        "triangles": face_triangles,
    }


def voxelise_all_meshes(
    context: cl.Context,
    queue: cl.CommandQueue,
    surface_kernel: cl.Kernel,
    delta: float,
    geometry_inputs: list[dict[str, Any]] | None,
    bake_path: str,
) -> list[dict[str, Any]]:
    """Build base-mask entries only, preserving the mesh object metadata."""

    base_masks = []

    for mesh_object in geometry_inputs or ():
        if not mesh_object:
            continue

        path = Path(bake_path) / mesh_object.get("mesh_file")
        mesh = load_usdc(path)
        triangles = np.ascontiguousarray(
            mesh["points"][0][mesh["triangles"]], dtype=np.float32
        )

        if mesh_object.get("deforming", False):
            base_masks.append({"mesh_object": mesh_object, "mesh": mesh})
            continue

        voxels = voxelise_static(
            context,
            queue,
            surface_kernel,
            triangles,
            delta,
        )

        if voxels is not None:
            base_masks.append(
                {
                    "mesh_object": mesh_object,
                    "voxels": voxels,
                }
            )

    return base_masks


def interpolated_triangles(entry: dict[str, Any], t: float) -> np.ndarray:
    mesh = entry["mesh"]
    times = mesh["times"]
    points = mesh["points"]
    if len(points) == 1 or t <= times[0]:
        vertices = points[0]
    elif t >= times[-1]:
        vertices = points[-1]
    else:
        upper = int(np.searchsorted(times, t, side="right"))
        lower = upper - 1
        factor = np.float32((t - times[lower]) / (times[upper] - times[lower]))
        vertices = points[lower] + factor * (points[upper] - points[lower])
    return np.ascontiguousarray(vertices[mesh["triangles"]], dtype=np.float32)


def world_triangles(entry: dict[str, Any], t: float):
    triangles = interpolated_triangles(entry, t)
    matrix, rate = helper.get_matrix_data(
        entry["matrix_times"], entry["matrix_matrices"], entry["matrix_rates"], t
    )
    flat = triangles.reshape(-1, 3)
    world = flat @ np.asarray(matrix[:3, :3], dtype=np.float32).T + matrix[:3, 3]
    return np.ascontiguousarray(world.reshape(-1, 3, 3), dtype=np.float32), matrix, rate


def voxelise_deforming(
    context,
    queue,
    kernel,
    triangles,
    target_mask,
    index_tile_map,
    tile_shape,
    delta,
    origin,
    velocity_fields=None,
    velocity_coefficients=None,
) -> None:
    if triangles.size == 0:
        return
    triangles_gpu = helper.to_device(context, triangles)
    if velocity_fields is None:
        velocity_fields = (target_mask, target_mask, target_mask)
        velocity_coefficients = np.zeros((3, 4), dtype=np.float32)
        write_velocity = 0
    else:
        write_velocity = 1
    coefficients = np.asarray(velocity_coefficients, dtype=np.float32).reshape(12)
    kernel(
        queue,
        (int(triangles.shape[0]),),
        None,
        triangles_gpu,
        target_mask,
        velocity_fields[0],
        velocity_fields[1],
        velocity_fields[2],
        index_tile_map,
        np.int32(triangles.shape[0]),
        np.int32(tile_shape[0]),
        np.int32(tile_shape[1]),
        np.int32(tile_shape[2]),
        np.float32(delta),
        np.float32(origin[0]),
        np.float32(origin[1]),
        np.float32(origin[2]),
        np.int32(write_velocity),
        *[np.float32(value) for value in coefficients],
    )


def voxelise_static(
    context: cl.Context,
    queue: cl.CommandQueue,
    surface_kernel: cl.Kernel,
    triangles: np.ndarray,
    delta: float,
) -> dict[str, Any] | None:
    """
    Rasterize triangle surfaces into a padded object-local Boolean voxel mask.
    """
    if triangles.size == 0:
        return None

    triangles = np.asarray(triangles, dtype=np.float32)
    vertices = triangles.reshape(-1, 3)

    bounds_min = vertices.min(axis=0).astype(np.float32)
    bounds_max = vertices.max(axis=0).astype(np.float32)

    lo = np.floor(bounds_min / delta).astype(np.int32) - 1
    hi = np.ceil(bounds_max / delta).astype(np.int32) + 1

    shape = tuple((hi - lo + 1).tolist())

    origin = np.asarray(
        lo * delta,
        dtype=np.float32,
    )

    triangles_gpu = helper.to_device(context, triangles)

    voxel_count = int(np.prod(shape))
    mask_word_count = (voxel_count + 31) // 32

    mask = helper.device_array(context, (mask_word_count,), dtype=np.uint32)
    cl.enqueue_fill_buffer(queue, mask, np.uint32(0), 0, mask.size)

    surface_kernel(
        queue,
        (int(triangles.shape[0]),),
        None,
        triangles_gpu,
        mask,
        np.int32(triangles.shape[0]),
        np.int32(shape[0]),
        np.int32(shape[1]),
        np.int32(shape[2]),
        np.float32(delta),
        np.float32(origin[0]),
        np.float32(origin[1]),
        np.float32(origin[2]),
    )

    return {
        "mask": mask,
        "shape": shape,
        "origin": origin,
        "bounds_min": origin,
        "bounds_max": np.asarray(
            hi * delta,
            dtype=np.float32,
        ),
    }
