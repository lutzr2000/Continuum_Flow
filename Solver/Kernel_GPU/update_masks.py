import math

import numpy as np
from numba import cuda

import Solver.Kernel_GPU.kernel_config as kernel_config


def update_source_tile_mask(
    source_tile_mask,
    source_base_masks,
    t,
    delta,
    origin,
):
    source_tile_mask.copy_to_device(np.zeros(source_tile_mask.shape, dtype=np.bool_))

    tile_size = kernel_config.TILE_SIZE

    for base_masks in source_base_masks:
        for entry in base_masks:
            mesh_object = entry["mesh_object"]
            voxels = entry["voxels"]

            matrix, _ = get_matrix_data(
                mesh_object,
                t,
            )

            bounds_min = np.asarray(voxels["bounds_min"])
            bounds_max = np.asarray(voxels["bounds_max"])

            center = (bounds_min + bounds_max) * 0.5
            extent = (bounds_max - bounds_min) * 0.5

            linear = matrix[:3, :3]
            translation = matrix[:3, 3]

            world_center = linear @ center + translation
            world_extent = np.abs(linear) @ extent

            world_min = world_center - world_extent
            world_max = world_center + world_extent

            cell_min = np.floor((world_min - origin) / delta)

            cell_max = np.ceil((world_max - origin) / delta)

            tile_min = np.floor_divide(
                cell_min,
                tile_size,
            )

            tile_max = np.floor_divide(
                cell_max,
                tile_size,
            )

            tile_min = np.maximum(tile_min, 0)
            tile_max = np.minimum(
                tile_max,
                np.asarray(source_tile_mask.shape) - 1,
            )

            mark_source_tiles[
                (
                    int(tile_max[0] - tile_min[0] + 1),
                    int(tile_max[1] - tile_min[1] + 1),
                    int(tile_max[2] - tile_min[2] + 1),
                ),
                1,
            ](
                source_tile_mask,
                int(tile_min[0]),
                int(tile_min[1]),
                int(tile_min[2]),
            )


@cuda.jit(cache=True)
def mark_source_tiles(
    source_tile_mask,
    offset_i,
    offset_j,
    offset_k,
):
    i, j, k = cuda.grid(3)

    i += offset_i
    j += offset_j
    k += offset_k

    if (
        i < source_tile_mask.shape[0]
        and j < source_tile_mask.shape[1]
        and k < source_tile_mask.shape[2]
    ):
        source_tile_mask[i, j, k] = True


def get_matrix_data(mesh_object, time_value):
    animation = mesh_object.get("transform_animation") or {}

    if "times" in animation:
        times = np.asarray(animation.get("times") or (0.0,))
    else:
        timeline = mesh_object.get("animation_timeline") or {}
        times = np.asarray(timeline.get("times") or (0.0,))

    matrices = np.asarray(animation.get("matrices_world", (np.eye(4),))).reshape(
        -1, 4, 4
    )

    if len(matrices) == 0:
        return np.eye(4), np.zeros((4, 4))

    if len(times) <= 1 or len(matrices) <= 1:
        return matrices[0], np.zeros((4, 4))

    last = min(len(times), len(matrices)) - 1

    if time_value <= times[0]:
        idx = 0
        alpha = 0.0

    elif time_value >= times[last]:
        idx = last - 1
        alpha = 1.0

    else:
        idx = np.searchsorted(times[: last + 1], time_value) - 1
        idx = max(0, idx)

        t0 = times[idx]
        t1 = times[idx + 1]

        if t1 <= t0:
            return matrices[idx], np.zeros((4, 4))

        alpha = (time_value - t0) / (t1 - t0)

    dt = times[idx + 1] - times[idx]

    if dt <= 0:
        return matrices[idx], np.zeros((4, 4))

    matrix = matrices[idx] * (1.0 - alpha) + matrices[idx + 1] * alpha
    rate = (matrices[idx + 1] - matrices[idx]) / dt

    return matrix, rate


def update_source_masks(
    source_masks,
    source_base_masks,
    t,
    delta,
    origin_x,
    origin_y,
    origin_z,
    tile_map,
):
    for source_mask, base_masks in zip(source_masks, source_base_masks):
        source_mask[:] = False

        for entry in base_masks:
            mesh_object = entry["mesh_object"]
            voxels = entry["voxels"]

            matrix, _ = get_matrix_data(mesh_object, t)
            inv = np.linalg.inv(matrix).astype(np.float32)

            local_mask = cuda.to_device(
                np.ascontiguousarray(voxels["mask"], dtype=np.bool_)
            )

            threads = kernel_config.THREADS_PER_BLOCK_3D
            blocks = kernel_config.volume_blocks_per_grid(
                tile_map.shape,
                threads,
            )

            update_source_masks_gpu[blocks, threads](
                source_mask,
                tile_map,
                local_mask,
                np.asarray(voxels["origin"], dtype=np.float32),
                inv,
                np.float32(delta),
                np.float32(origin_x),
                np.float32(origin_y),
                np.float32(origin_z),
            )


def update_obstacle_mask(
    obstacle_mask,
    obstacle_base_masks,
    t,
    delta,
    origin_x,
    origin_y,
    origin_z,
    tile_map,
    velocity_x,
    velocity_y,
    velocity_z,
):
    obstacle_mask[:] = False

    for entry in obstacle_base_masks:
        mesh_object = entry["mesh_object"]
        voxels = entry["voxels"]

        matrix, rate = get_matrix_data(mesh_object, t)
        inv = np.linalg.inv(matrix).astype(np.float32)

        local_mask = cuda.to_device(
            np.ascontiguousarray(voxels["mask"], dtype=np.bool_)
        )

        threads = kernel_config.THREADS_PER_BLOCK_3D
        blocks = kernel_config.volume_blocks_per_grid(
            tile_map.shape,
            threads,
        )

        update_obstacle_mask_gpu[
            blocks,
            threads,
        ](
            obstacle_mask,
            velocity_x,
            velocity_y,
            velocity_z,
            tile_map,
            local_mask,
            np.asarray(voxels["origin"], dtype=np.float32),
            inv,
            np.asarray(rate, dtype=np.float32),
            np.float32(delta),
            np.float32(origin_x),
            np.float32(origin_y),
            np.float32(origin_z),
        )


@cuda.jit(cache=True)
def update_source_masks_gpu(
    mask,
    tile_map,
    local_mask,
    local_origin,
    inv,
    delta,
    ox,
    oy,
    oz,
):
    ti, tj, tk = cuda.grid(3)

    if ti >= tile_map.shape[0] or tj >= tile_map.shape[1] or tk >= tile_map.shape[2]:
        return

    tile = tile_map[ti, tj, tk]

    if tile < 0:
        return

    tile_size = kernel_config.TILE_SIZE

    for i in range(tile_size):
        for j in range(tile_size):
            for k in range(tile_size):
                gi = ti * tile_size + i
                gj = tj * tile_size + j
                gk = tk * tile_size + k

                x = ox + gi * delta
                y = oy + gj * delta
                z = oz + gk * delta

                bx = inv[0, 0] * x + inv[0, 1] * y + inv[0, 2] * z + inv[0, 3]
                by = inv[1, 0] * x + inv[1, 1] * y + inv[1, 2] * z + inv[1, 3]
                bz = inv[2, 0] * x + inv[2, 1] * y + inv[2, 2] * z + inv[2, 3]

                bi = int(math.floor((bx - local_origin[0]) / delta + 0.5))
                bj = int(math.floor((by - local_origin[1]) / delta + 0.5))
                bk = int(math.floor((bz - local_origin[2]) / delta + 0.5))

                if (
                    0 <= bi < local_mask.shape[0]
                    and 0 <= bj < local_mask.shape[1]
                    and 0 <= bk < local_mask.shape[2]
                    and local_mask[bi, bj, bk]
                ):
                    mask[tile, i, j, k] = True


@cuda.jit(cache=True)
def update_obstacle_mask_gpu(
    mask,
    velocity_x,
    velocity_y,
    velocity_z,
    tile_map,
    local_mask,
    local_origin,
    inv,
    rate,
    delta,
    ox,
    oy,
    oz,
):
    ti, tj, tk = cuda.grid(3)

    if ti >= tile_map.shape[0] or tj >= tile_map.shape[1] or tk >= tile_map.shape[2]:
        return

    tile = tile_map[ti, tj, tk]

    if tile < 0:
        return

    tile_size = kernel_config.TILE_SIZE

    for i in range(tile_size):
        for j in range(tile_size):
            for k in range(tile_size):
                gi = ti * tile_size + i
                gj = tj * tile_size + j
                gk = tk * tile_size + k

                x = ox + gi * delta
                y = oy + gj * delta
                z = oz + gk * delta

                bx = inv[0, 0] * x + inv[0, 1] * y + inv[0, 2] * z + inv[0, 3]
                by = inv[1, 0] * x + inv[1, 1] * y + inv[1, 2] * z + inv[1, 3]
                bz = inv[2, 0] * x + inv[2, 1] * y + inv[2, 2] * z + inv[2, 3]

                bi = int(math.floor((bx - local_origin[0]) / delta + 0.5))
                bj = int(math.floor((by - local_origin[1]) / delta + 0.5))
                bk = int(math.floor((bz - local_origin[2]) / delta + 0.5))

                if (
                    0 <= bi < local_mask.shape[0]
                    and 0 <= bj < local_mask.shape[1]
                    and 0 <= bk < local_mask.shape[2]
                    and local_mask[bi, bj, bk]
                ):
                    mask[tile, i, j, k] = True

                    velocity_x[tile, i, j, k] = (
                        rate[0, 0] * bx + rate[0, 1] * by + rate[0, 2] * bz + rate[0, 3]
                    )

                    velocity_y[tile, i, j, k] = (
                        rate[1, 0] * bx + rate[1, 1] * by + rate[1, 2] * bz + rate[1, 3]
                    )

                    velocity_z[tile, i, j, k] = (
                        rate[2, 0] * bx + rate[2, 1] * by + rate[2, 2] * bz + rate[2, 3]
                    )
