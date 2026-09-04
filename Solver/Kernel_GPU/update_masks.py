from pathlib import Path

import numpy as np
from Solver.Kernel_GPU import kernel_config
from numba import cuda
from pxr import Usd, UsdGeom

TILE_SIZE = kernel_config.TILE_SIZE

# ------------geometry management-------------------
def update_masks(
    t,
    simulation,
    obstacle_mask,
    source_masks,
    tile_map,
    delta,
    origin_x,
    origin_y,
    origin_z,
):
    origin = (origin_x, origin_y, origin_z)
    settings = simulation["settings"]
    timeline = simulation["animation_timeline"]
    output_path = Path(simulation["outputs"][0]["output_path"])
    start_frame = settings["start_frame"]
    end_frame = settings["end_frame"]

    frame_position = start_frame + t * timeline["fps"]
    frame_before = min(max(start_frame, int(frame_position)), end_frame - 1)
    frame_after = min(frame_before + 1, end_frame - 1)
    blend = min(1.0, max(0.0, frame_position - frame_before))

    obstacle_file = next(
        (
            output_path / geometry["mesh_file"]
            for obstacle in simulation.get("obstacles", [])
            for geometry in obstacle.get("geometry_inputs", [])
            if geometry.get("mesh_file")
        ),
        None,
    )

    if obstacle_file:
        voxelize_usd_meshes(
            load_usd_file(obstacle_file, frame_before),
            load_usd_file(obstacle_file, frame_after),
            blend,
            obstacle_mask,
            tile_map,
            origin,
            delta,
        )
    else:
        clear_mask[obstacle_mask.shape[0], (TILE_SIZE, TILE_SIZE, TILE_SIZE)](
            obstacle_mask, obstacle_mask.shape[0]
        )

    for source_index, source in enumerate(simulation.get("sources", [])):
        source_file = next(
            (
                output_path / geometry["mesh_file"]
                for geometry in source.get("geometry_inputs", [])
                if geometry.get("mesh_file")
            ),
            None,
        )
        if source_file:
            voxelize_usd_meshes(
                load_usd_file(source_file, frame_before),
                load_usd_file(source_file, frame_after),
                blend,
                source_masks[source_index],
                tile_map,
                origin,
                delta,
            )
        else:
            clear_mask[source_masks.shape[1], (TILE_SIZE, TILE_SIZE, TILE_SIZE)](
                source_masks[source_index], source_masks.shape[1]
            )


@cuda.jit(cache=True)
def clear_mask(mask_pool, active_tile_count):
    tile_index = cuda.blockIdx.x
    if tile_index < active_tile_count:
        mask_pool[
            tile_index,
            cuda.threadIdx.x,
            cuda.threadIdx.y,
            cuda.threadIdx.z,
        ] = False


def load_usd_file(filepath, frame):
    stage = Usd.Stage.Open(str(filepath))
    meshes = []

    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue

        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(frame), dtype=np.float32)
        transform = np.asarray(
            UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(frame),
            dtype=np.float32,
        )

        points = np.ascontiguousarray(
            points @ transform[:3, :3] + transform[3, :3],
            dtype=np.float32,
        )

        face_counts = np.asarray(
            mesh.GetFaceVertexCountsAttr().Get(frame), dtype=np.int32
        )
        face_indices = np.asarray(
            mesh.GetFaceVertexIndicesAttr().Get(frame), dtype=np.int32
        )

        meshes.append(
            {
                "name": str(prim.GetPath()),
                "points": points,
                "triangles": triangulate(face_counts, face_indices),
            }
        )

    return meshes


def triangulate(face_counts, face_indices):
    triangles = []
    offset = 0
    for count in face_counts:
        for corner in range(1, count - 1):
            triangles.append(
                (
                    face_indices[offset],
                    face_indices[offset + corner],
                    face_indices[offset + corner + 1],
                )
            )
        offset += count
    return np.ascontiguousarray(triangles, dtype=np.int32)


# ------------voxelisation-------------------
def voxelize_usd_meshes(
    meshes_before, meshes_after, blend, mask_pool, tile_map, origin, delta
):
    meshes_after_by_name = {mesh["name"]: mesh for mesh in meshes_after}
    clear_mask[mask_pool.shape[0], (TILE_SIZE, TILE_SIZE, TILE_SIZE)](
        mask_pool, mask_pool.shape[0]
    )

    for mesh_before in meshes_before:
        mesh_after = meshes_after_by_name.get(mesh_before["name"])
        if mesh_after is None:
            continue
        if mesh_before["points"].shape != mesh_after["points"].shape:
            raise ValueError(
                f"Animated mesh '{mesh_before['name']}' changes its vertex count."
            )
        if not len(mesh_before["triangles"]):
            continue

        mesh_tile_indices = device_tile_indices(
            mesh_before,
            mesh_after,
            tuple(size * TILE_SIZE for size in tile_map.shape),
            origin,
            delta,
        )
        if mesh_tile_indices is None:
            continue

        points_before = cuda.to_device(mesh_before["points"])
        points_after = cuda.to_device(mesh_after["points"])
        triangles = cuda.to_device(mesh_before["triangles"])
        voxelize_tiles[
            mesh_tile_indices.shape[0], (TILE_SIZE, TILE_SIZE, TILE_SIZE)
        ](
            mask_pool,
            tile_map,
            mesh_tile_indices,
            points_before,
            points_after,
            triangles,
            blend,
            origin[0],
            origin[1],
            origin[2],
            delta,
        )


@cuda.jit(device=True, inline=True)
def ray_intersection(
    x, y, z, points_before, points_after, triangles, triangle_index, blend
):
    ia = triangles[triangle_index, 0]
    ib = triangles[triangle_index, 1]
    ic = triangles[triangle_index, 2]

    ax = points_before[ia, 0] + blend * (points_after[ia, 0] - points_before[ia, 0])
    ay = points_before[ia, 1] + blend * (points_after[ia, 1] - points_before[ia, 1])
    az = points_before[ia, 2] + blend * (points_after[ia, 2] - points_before[ia, 2])
    bx = points_before[ib, 0] + blend * (points_after[ib, 0] - points_before[ib, 0])
    by = points_before[ib, 1] + blend * (points_after[ib, 1] - points_before[ib, 1])
    bz = points_before[ib, 2] + blend * (points_after[ib, 2] - points_before[ib, 2])
    cx = points_before[ic, 0] + blend * (points_after[ic, 0] - points_before[ic, 0])
    cy = points_before[ic, 1] + blend * (points_after[ic, 1] - points_before[ic, 1])
    cz = points_before[ic, 2] + blend * (points_after[ic, 2] - points_before[ic, 2])

    e1x = bx - ax
    e1y = by - ay
    e1z = bz - az
    e2x = cx - ax
    e2y = cy - ay
    e2z = cz - az
    determinant = e1z * e2y - e1y * e2z
    if -1e-7 < determinant < 1e-7:
        return False

    inv_determinant = 1.0 / determinant
    sx = x - ax
    sy = y - ay
    sz = z - az
    u = inv_determinant * (sz * e2y - sy * e2z)
    if u < 0.0 or u > 1.0:
        return False

    qx = sy * e1z - sz * e1y
    qy = sz * e1x - sx * e1z
    qz = sx * e1y - sy * e1x
    v = inv_determinant * qx
    if v < 0.0 or u + v > 1.0:
        return False

    distance = inv_determinant * (e2x * qx + e2y * qy + e2z * qz)
    return distance > 1e-7


@cuda.jit(cache=True)
def voxelize_tiles(
    mask_pool,
    mask_tile_map,
    tile_indices,
    points_before,
    points_after,
    triangles,
    blend,
    origin_x,
    origin_y,
    origin_z,
    delta,
):
    tile_index = cuda.blockIdx.x
    i = tile_indices[tile_index, 0] * TILE_SIZE + cuda.threadIdx.x
    j = tile_indices[tile_index, 1] * TILE_SIZE + cuda.threadIdx.y
    k = tile_indices[tile_index, 2] * TILE_SIZE + cuda.threadIdx.z
    x = origin_x + (i + 0.5) * delta
    y = origin_y + (j + 0.5) * delta
    z = origin_z + (k + 0.5) * delta
    intersections = 0
    for triangle_index in range(triangles.shape[0]):
        if ray_intersection(
            x,
            y,
            z,
            points_before,
            points_after,
            triangles,
            triangle_index,
            blend,
        ):
            intersections += 1

    if intersections % 2:
        pool_index = mask_tile_map[
            tile_indices[tile_index, 0],
            tile_indices[tile_index, 1],
            tile_indices[tile_index, 2],
        ]
        if pool_index >= 0:
            mask_pool[
                pool_index,
                cuda.threadIdx.x,
                cuda.threadIdx.y,
                cuda.threadIdx.z,
            ] = True


def mesh_tile_indices(mesh_before, mesh_after, mask_shape, origin, delta):
    points_min = np.minimum(
        mesh_before["points"].min(axis=0), mesh_after["points"].min(axis=0)
    )
    points_max = np.maximum(
        mesh_before["points"].max(axis=0), mesh_after["points"].max(axis=0)
    )
    cell_min = np.floor((points_min - origin) / delta).astype(np.int32)
    cell_max = np.floor((points_max - origin) / delta).astype(np.int32)
    cell_min = np.maximum(cell_min, 0)
    cell_max = np.minimum(cell_max, np.asarray(mask_shape, dtype=np.int32) - 1)
    if np.any(cell_min > cell_max):
        return np.empty((0, 3), dtype=np.int32)

    tile_min = cell_min // TILE_SIZE
    tile_max = cell_max // TILE_SIZE
    tile_indices = np.empty((np.prod(tile_max - tile_min + 1), 3), dtype=np.int32)
    index = 0
    for tile_i in range(tile_min[0], tile_max[0] + 1):
        for tile_j in range(tile_min[1], tile_max[1] + 1):
            for tile_k in range(tile_min[2], tile_max[2] + 1):
                tile_indices[index] = (tile_i, tile_j, tile_k)
                index += 1
    return tile_indices


def device_tile_indices(mesh_before, mesh_after, mask_shape, origin, delta):
    cache_key = (id(mesh_after), tuple(mask_shape), tuple(origin), delta)
    tile_cache = mesh_before.setdefault("device_tile_indices", {})
    if cache_key not in tile_cache:
        tile_indices = mesh_tile_indices(
            mesh_before, mesh_after, mask_shape, origin, delta
        )
        if not len(tile_indices):
            tile_cache[cache_key] = None
        else:
            tile_cache[cache_key] = cuda.to_device(tile_indices)
    return tile_cache[cache_key]
