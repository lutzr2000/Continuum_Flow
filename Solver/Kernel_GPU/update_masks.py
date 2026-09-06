import math

import numpy as np
from numba import cuda

import Solver.Kernel_GPU.kernel_config as kernel_config

IDENTITY_4 = np.eye(4)
ZERO_4 = np.zeros((4, 4))


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
                entry["matrix_times"],
                entry["matrix_matrices"],
                entry["matrix_rates"],
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


def prepare_matrix_data(mesh_object):
    animation = mesh_object.get("transform_animation") or {}

    times = animation.get("times")

    if times is None:
        timeline = mesh_object.get("animation_timeline") or {}
        times = timeline.get("times")

    if times is None:
        times = np.array([0.0], dtype=np.float64)
    else:
        times = np.asarray(times)

    matrices = animation.get("matrices_world")

    if matrices is None:
        matrices = IDENTITY_4[None, ...]
    else:
        matrices = np.asarray(matrices).reshape(-1, 4, 4)

    n = min(times.shape[0], matrices.shape[0])

    times = times[:n]
    matrices = matrices[:n]

    if n > 1:
        dt = np.diff(times)
        delta = np.diff(matrices, axis=0)

        rates = np.zeros_like(delta)

        valid = dt > 0
        rates[valid] = delta[valid] / dt[valid, None, None]

    else:
        rates = None

    return times, matrices, rates


def get_matrix_data(times, matrices, rates, time_value):
    n = matrices.shape[0]

    if n == 0:
        return IDENTITY_4, ZERO_4

    if n == 1:
        return matrices[0], ZERO_4

    if time_value <= times[0]:
        idx = 0
        alpha = 0.0

    elif time_value >= times[-1]:
        idx = n - 2
        alpha = 1.0

    else:
        idx = (
            np.searchsorted(
                times,
                time_value,
                side="right",
            )
            - 1
        )

        dt = times[idx + 1] - times[idx]

        if dt <= 0:
            return matrices[idx], ZERO_4

        alpha = (time_value - times[idx]) / dt

    rate = rates[idx]

    matrix = matrices[idx] + rate * ((times[idx + 1] - times[idx]) * alpha)

    return matrix, rate


def get_tile_bounds(
    voxels,
    matrix,
    delta,
    origin,
    tile_grid_shape,
):
    tile_size = kernel_config.TILE_SIZE

    bounds_min = np.asarray(
        voxels["bounds_min"],
        dtype=np.float32,
    )

    bounds_max = np.asarray(
        voxels["bounds_max"],
        dtype=np.float32,
    )

    center = (bounds_min + bounds_max) * 0.5
    extent = (bounds_max - bounds_min) * 0.5

    linear = matrix[:3, :3]
    translation = matrix[:3, 3]

    world_center = linear @ center + translation
    world_extent = np.abs(linear) @ extent

    world_min = world_center - world_extent
    world_max = world_center + world_extent

    cell_min = np.floor((world_min - origin) / delta).astype(np.int32)

    cell_max = np.ceil((world_max - origin) / delta).astype(np.int32)

    tile_min = np.floor_divide(
        cell_min,
        tile_size,
    )

    tile_max = np.floor_divide(
        cell_max,
        tile_size,
    )

    tile_min = np.maximum(
        tile_min,
        0,
    )

    tile_max = np.minimum(
        tile_max,
        np.asarray(
            tile_grid_shape,
            dtype=np.int32,
        )
        - 1,
    )

    return tile_min, tile_max


def prepare_cell_transform(
    inv,
    delta,
    origin_x,
    origin_y,
    origin_z,
    local_origin,
):
    inv_delta = np.float32(1.0 / delta)

    a00 = np.float32(inv[0, 0])
    a01 = np.float32(inv[0, 1])
    a02 = np.float32(inv[0, 2])

    a10 = np.float32(inv[1, 0])
    a11 = np.float32(inv[1, 1])
    a12 = np.float32(inv[1, 2])

    a20 = np.float32(inv[2, 0])
    a21 = np.float32(inv[2, 1])
    a22 = np.float32(inv[2, 2])

    c0 = np.float32(
        (
            inv[0, 0] * origin_x
            + inv[0, 1] * origin_y
            + inv[0, 2] * origin_z
            + inv[0, 3]
            - local_origin[0]
        )
        * inv_delta
    )

    c1 = np.float32(
        (
            inv[1, 0] * origin_x
            + inv[1, 1] * origin_y
            + inv[1, 2] * origin_z
            + inv[1, 3]
            - local_origin[1]
        )
        * inv_delta
    )

    c2 = np.float32(
        (
            inv[2, 0] * origin_x
            + inv[2, 1] * origin_y
            + inv[2, 2] * origin_z
            + inv[2, 3]
            - local_origin[2]
        )
        * inv_delta
    )

    return (
        c0,
        c1,
        c2,
        a00,
        a01,
        a02,
        a10,
        a11,
        a12,
        a20,
        a21,
        a22,
    )


def update_source_masks(
    source_masks,
    source_base_masks,
    animated_sources,
    initial_update,
    t,
    delta,
    origin_x,
    origin_y,
    origin_z,
    tile_map,
):
    for source_idx, (source_mask, base_masks) in enumerate(
        zip(
            source_masks,
            source_base_masks,
        )
    ):
        if not initial_update and not animated_sources[source_idx]:
            continue

        source_mask[:] = False

        for entry in base_masks:
            voxels = entry["voxels"]

            matrix, _ = get_matrix_data(
                entry["matrix_times"],
                entry["matrix_matrices"],
                entry["matrix_rates"],
                t,
            )

            inv = np.linalg.inv(matrix).astype(np.float32)

            local_mask = voxels["mask"]

            local_origin = np.asarray(
                voxels["origin"],
                dtype=np.float32,
            )

            (
                c0,
                c1,
                c2,
                a00,
                a01,
                a02,
                a10,
                a11,
                a12,
                a20,
                a21,
                a22,
            ) = prepare_cell_transform(
                inv,
                delta,
                origin_x,
                origin_y,
                origin_z,
                local_origin,
            )

            origin = np.asarray(
                (
                    origin_x,
                    origin_y,
                    origin_z,
                ),
                dtype=np.float32,
            )

            tile_min, tile_max = get_tile_bounds(
                voxels,
                matrix,
                delta,
                origin,
                tile_map.shape,
            )

            if (
                tile_min[0] > tile_max[0]
                or tile_min[1] > tile_max[1]
                or tile_min[2] > tile_max[2]
            ):
                continue

            threads = (
                kernel_config.TILE_SIZE,
                kernel_config.TILE_SIZE,
                kernel_config.TILE_SIZE,
            )

            blocks = (
                int(tile_max[0] - tile_min[0] + 1),
                int(tile_max[1] - tile_min[1] + 1),
                int(tile_max[2] - tile_min[2] + 1),
            )

            update_source_masks_gpu[
                blocks,
                threads,
            ](
                source_mask,
                tile_map,
                local_mask,
                c0,
                c1,
                c2,
                a00,
                a01,
                a02,
                a10,
                a11,
                a12,
                a20,
                a21,
                a22,
                int(tile_min[0]),
                int(tile_min[1]),
                int(tile_min[2]),
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
        voxels = entry["voxels"]

        matrix, rate = get_matrix_data(
            entry["matrix_times"],
            entry["matrix_matrices"],
            entry["matrix_rates"],
            t,
        )

        inv = np.linalg.inv(matrix).astype(np.float32)

        rate = np.asarray(
            rate,
            dtype=np.float32,
        )

        local_mask = voxels["mask"]

        local_origin = np.asarray(
            voxels["origin"],
            dtype=np.float32,
        )

        (
            c0,
            c1,
            c2,
            a00,
            a01,
            a02,
            a10,
            a11,
            a12,
            a20,
            a21,
            a22,
        ) = prepare_cell_transform(
            inv,
            delta,
            origin_x,
            origin_y,
            origin_z,
            local_origin,
        )

        origin = np.asarray(
            (
                origin_x,
                origin_y,
                origin_z,
            ),
            dtype=np.float32,
        )

        tile_min, tile_max = get_tile_bounds(
            voxels,
            matrix,
            delta,
            origin,
            tile_map.shape,
        )

        rate = np.asarray(rate, dtype=np.float32)

        velocity_transform = rate @ inv

        d = np.float32(delta)
        ox = np.float32(origin_x)
        oy = np.float32(origin_y)
        oz = np.float32(origin_z)

        # vx = vx_i * gi + vx_j * gj + vx_k * gk + vx_c
        vx_i = np.float32(velocity_transform[0, 0] * d)
        vx_j = np.float32(velocity_transform[0, 1] * d)
        vx_k = np.float32(velocity_transform[0, 2] * d)

        vx_c = np.float32(
            velocity_transform[0, 0] * ox
            + velocity_transform[0, 1] * oy
            + velocity_transform[0, 2] * oz
            + velocity_transform[0, 3]
        )

        # vy
        vy_i = np.float32(velocity_transform[1, 0] * d)
        vy_j = np.float32(velocity_transform[1, 1] * d)
        vy_k = np.float32(velocity_transform[1, 2] * d)

        vy_c = np.float32(
            velocity_transform[1, 0] * ox
            + velocity_transform[1, 1] * oy
            + velocity_transform[1, 2] * oz
            + velocity_transform[1, 3]
        )

        # vz
        vz_i = np.float32(velocity_transform[2, 0] * d)
        vz_j = np.float32(velocity_transform[2, 1] * d)
        vz_k = np.float32(velocity_transform[2, 2] * d)

        vz_c = np.float32(
            velocity_transform[2, 0] * ox
            + velocity_transform[2, 1] * oy
            + velocity_transform[2, 2] * oz
            + velocity_transform[2, 3]
        )

        if (
            tile_min[0] > tile_max[0]
            or tile_min[1] > tile_max[1]
            or tile_min[2] > tile_max[2]
        ):
            continue

        threads = (
            kernel_config.TILE_SIZE,
            kernel_config.TILE_SIZE,
            kernel_config.TILE_SIZE,
        )

        blocks = (
            int(tile_max[0] - tile_min[0] + 1),
            int(tile_max[1] - tile_min[1] + 1),
            int(tile_max[2] - tile_min[2] + 1),
        )

        update_obstacle_mask_gpu[blocks, threads](
            obstacle_mask,
            velocity_x,
            velocity_y,
            velocity_z,
            tile_map,
            local_mask,
            c0,
            c1,
            c2,
            a00,
            a01,
            a02,
            a10,
            a11,
            a12,
            a20,
            a21,
            a22,
            vx_i,
            vx_j,
            vx_k,
            vx_c,
            vy_i,
            vy_j,
            vy_k,
            vy_c,
            vz_i,
            vz_j,
            vz_k,
            vz_c,
            int(tile_min[0]),
            int(tile_min[1]),
            int(tile_min[2]),
        )


@cuda.jit(cache=True)
def update_source_masks_gpu(
    mask,
    tile_map,
    local_mask,
    c0,
    c1,
    c2,
    a00,
    a01,
    a02,
    a10,
    a11,
    a12,
    a20,
    a21,
    a22,
    offset_i,
    offset_j,
    offset_k,
):
    ti = cuda.blockIdx.x + offset_i
    tj = cuda.blockIdx.y + offset_j
    tk = cuda.blockIdx.z + offset_k

    # x dimension of CUDA warp -> contiguous k dimension
    k = cuda.threadIdx.x
    j = cuda.threadIdx.y
    i = cuda.threadIdx.z

    if ti >= tile_map.shape[0] or tj >= tile_map.shape[1] or tk >= tile_map.shape[2]:
        return

    tile = tile_map[
        ti,
        tj,
        tk,
    ]

    if tile < 0:
        return

    tile_size = kernel_config.TILE_SIZE

    if i >= tile_size or j >= tile_size or k >= tile_size:
        return

    gi = ti * tile_size + i
    gj = tj * tile_size + j
    gk = tk * tile_size + k

    fi = a00 * gi + a01 * gj + a02 * gk + c0
    fj = a10 * gi + a11 * gj + a12 * gk + c1
    fk = a20 * gi + a21 * gj + a22 * gk + c2

    bi = int(math.floor(fi + 0.5))
    bj = int(math.floor(fj + 0.5))
    bk = int(math.floor(fk + 0.5))

    if (
        0 <= bi < local_mask.shape[0]
        and 0 <= bj < local_mask.shape[1]
        and 0 <= bk < local_mask.shape[2]
        and local_mask[
            bi,
            bj,
            bk,
        ]
    ):
        mask[
            tile,
            i,
            j,
            k,
        ] = True


@cuda.jit(cache=True)
def update_obstacle_mask_gpu(
    mask,
    velocity_x,
    velocity_y,
    velocity_z,
    tile_map,
    local_mask,
    c0,
    c1,
    c2,
    a00,
    a01,
    a02,
    a10,
    a11,
    a12,
    a20,
    a21,
    a22,
    vx_i,
    vx_j,
    vx_k,
    vx_c,
    vy_i,
    vy_j,
    vy_k,
    vy_c,
    vz_i,
    vz_j,
    vz_k,
    vz_c,
    offset_i,
    offset_j,
    offset_k,
):
    ti = cuda.blockIdx.x + offset_i
    tj = cuda.blockIdx.y + offset_j
    tk = cuda.blockIdx.z + offset_k

    k = cuda.threadIdx.x
    j = cuda.threadIdx.y
    i = cuda.threadIdx.z

    if ti >= tile_map.shape[0] or tj >= tile_map.shape[1] or tk >= tile_map.shape[2]:
        return

    tile = tile_map[ti, tj, tk]

    if tile < 0:
        return

    tile_size = kernel_config.TILE_SIZE

    if i >= tile_size or j >= tile_size or k >= tile_size:
        return

    gi = ti * tile_size + i
    gj = tj * tile_size + j
    gk = tk * tile_size + k

    fi = a00 * gi + a01 * gj + a02 * gk + c0
    fj = a10 * gi + a11 * gj + a12 * gk + c1
    fk = a20 * gi + a21 * gj + a22 * gk + c2

    bi = int(math.floor(fi + 0.5))
    bj = int(math.floor(fj + 0.5))
    bk = int(math.floor(fk + 0.5))

    if (
        bi < 0
        or bi >= local_mask.shape[0]
        or bj < 0
        or bj >= local_mask.shape[1]
        or bk < 0
        or bk >= local_mask.shape[2]
    ):
        return

    if not local_mask[bi, bj, bk]:
        return

    mask[tile, i, j, k] = True

    velocity_x[tile, i, j, k] = vx_i * gi + vx_j * gj + vx_k * gk + vx_c
    velocity_y[tile, i, j, k] = vy_i * gi + vy_j * gj + vy_k * gk + vy_c
    velocity_z[tile, i, j, k] = vz_i * gi + vz_j * gj + vz_k * gk + vz_c
