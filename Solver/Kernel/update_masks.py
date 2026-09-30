from typing import Any
import numpy as np
import pyopencl as cl

import Solver.Kernel.kernel_config as kernel_config
import Solver.Kernel.helper as helper


def update_source_tile_mask(
    queue: cl.CommandQueue,
    mark_source_tiles_kernel: cl.Kernel,
    source_tile_mask: Any,
    source_base_masks: Any,
    tile_shape: tuple[int, int, int],
    t: float,
    delta: float,
    origin: tuple[int, int, int],
) -> None:
    """
    Rebuild the coarse tile-activity mask from all transformed source bounds.

    Each source mesh's animated world-space bounds are converted to clamped
    grid-tile bounds, then the enclosed tile range is marked on the GPU.
    """
    cl.enqueue_fill_buffer(
        queue,
        source_tile_mask,
        np.uint8(0),
        0,
        source_tile_mask.size,
    )

    tile_size = kernel_config.TILE_SIZE

    for base_masks in source_base_masks:
        for entry in base_masks:
            voxels = entry["voxels"]

            matrix, _ = helper.get_matrix_data(
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
                np.asarray(tile_shape) - 1,
            )

            if np.any(tile_min > tile_max):
                continue

            mark_source_tiles_kernel(
                queue,
                (
                    int(tile_max[0] - tile_min[0] + 1),
                    int(tile_max[1] - tile_min[1] + 1),
                    int(tile_max[2] - tile_min[2] + 1),
                ),
                None,
                source_tile_mask,
                np.int32(tile_shape[0]),
                np.int32(tile_shape[1]),
                np.int32(tile_shape[2]),
                np.int32(tile_min[0]),
                np.int32(tile_min[1]),
                np.int32(tile_min[2]),
            )


def update_source_masks(
    queue: cl.CommandQueue,
    update_masks_kernels: dict[str, cl.Kernel],
    source_masks: Any,
    source_base_masks: Any,
    animated_sources: Any,
    initial_update: Any,
    t: float,
    delta: float,
    origin_x: float,
    origin_y: float,
    origin_z: float,
    tile_map: Any,
    tile_shape: tuple[int, int, int],
) -> None:
    """
    Voxelize transformed source meshes into their sparse GPU masks.

    Static sources may be skipped after the initial update. Animated transforms
    are interpolated, inverted, reduced to affected tile bounds, and passed to
    the GPU mask kernel for union into each source field.
    """
    for source_idx, (source_mask, base_masks) in enumerate(
        zip(
            source_masks,
            source_base_masks,
        )
    ):
        if not initial_update and not animated_sources[source_idx]:
            continue

        cl.enqueue_fill_buffer(
            queue,
            source_mask,
            np.bool_(False),
            0,
            source_mask.size,
        )

        for entry in base_masks:
            voxels = entry["voxels"]

            matrix, _ = helper.get_matrix_data(
                entry["matrix_times"],
                entry["matrix_matrices"],
                entry["matrix_rates"],
                t,
            )

            inv = np.linalg.inv(matrix).astype(np.float32)

            local_mask = voxels["mask"]
            local_mask_shape = voxels["shape"]

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
                tile_shape,
            )

            if (
                tile_min[0] > tile_max[0]
                or tile_min[1] > tile_max[1]
                or tile_min[2] > tile_max[2]
            ):
                continue

            local_work_size = (
                kernel_config.TILE_SIZE,
                kernel_config.TILE_SIZE,
                kernel_config.TILE_SIZE,
            )

            group_count = (
                int(tile_max[0] - tile_min[0] + 1),
                int(tile_max[1] - tile_min[1] + 1),
                int(tile_max[2] - tile_min[2] + 1),
            )

            global_work_size = (
                group_count[0] * kernel_config.TILE_SIZE,
                group_count[1] * kernel_config.TILE_SIZE,
                group_count[2] * kernel_config.TILE_SIZE,
            )

            update_masks_kernels["update_source_masks"](
                queue,
                global_work_size,
                local_work_size,
                source_mask,
                tile_map,
                local_mask,
                np.float32(c0),
                np.float32(c1),
                np.float32(c2),
                np.float32(a00),
                np.float32(a01),
                np.float32(a02),
                np.float32(a10),
                np.float32(a11),
                np.float32(a12),
                np.float32(a20),
                np.float32(a21),
                np.float32(a22),
                np.int32(tile_min[0]),
                np.int32(tile_min[1]),
                np.int32(tile_min[2]),
                np.int32(tile_shape[0]),
                np.int32(tile_shape[1]),
                np.int32(tile_shape[2]),
                np.int32(local_mask_shape[0]),
                np.int32(local_mask_shape[1]),
                np.int32(local_mask_shape[2]),
            )


def get_tile_bounds(
    voxels: Any,
    matrix: Any,
    delta: float,
    origin: tuple[int, int, int],
    tile_grid_shape: Any,
) -> Any:
    """
    Transform local voxel bounds and convert them into clamped tile bounds.

    Absolute linear-transform coefficients provide a conservative world-space
    axis-aligned extent, which is mapped through cell coordinates to the sparse
    tile grid.
    """
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
    inv: Any,
    delta: float,
    origin_x: float,
    origin_y: float,
    origin_z: float,
    local_origin: Any,
) -> Any:
    """
    Pack an inverse world transform for repeated grid-cell evaluation.

    The coefficients map world-grid indices directly into the voxel mask's
    local cell coordinates, avoiding matrix operations inside CUDA kernels.
    """
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


def update_source_velocity(
    queue: cl.CommandQueue,
    update_masks_kernels: dict[str, cl.Kernel],
    base_masks: Any,
    t: float,
    delta: float,
    origin_x: float,
    origin_y: float,
    origin_z: float,
    tile_map: Any,
    tile_shape: tuple[int, int, int],
    velocity_x: Any,
    velocity_y: Any,
    velocity_z: Any,
    local_velocity_x: float,
    local_velocity_y: float,
    local_velocity_z: float,
) -> None:
    """Write one source's per-mesh local velocity into shared scratch fields."""
    local_velocity = np.asarray(
        (
            local_velocity_x,
            local_velocity_y,
            local_velocity_z,
        ),
        dtype=np.float32,
    )

    for entry in base_masks:
        voxels = entry["voxels"]

        matrix, _ = helper.get_matrix_data(
            entry["matrix_times"],
            entry["matrix_matrices"],
            entry["matrix_rates"],
            t,
        )

        inv = np.linalg.inv(matrix).astype(np.float32)

        linear = np.asarray(
            matrix[:3, :3],
            dtype=np.float32,
        )

        axis_lengths = np.linalg.norm(
            linear,
            axis=0,
        )

        rotation = linear.copy()

        for axis in range(3):
            if axis_lengths[axis] > 1.0e-8:
                rotation[:, axis] /= axis_lengths[axis]

        world_velocity = rotation @ local_velocity

        transform = prepare_cell_transform(
            inv,
            delta,
            origin_x,
            origin_y,
            origin_z,
            np.asarray(
                voxels["origin"],
                dtype=np.float32,
            ),
        )

        tile_min, tile_max = get_tile_bounds(
            voxels,
            matrix,
            delta,
            np.asarray(
                (
                    origin_x,
                    origin_y,
                    origin_z,
                ),
                dtype=np.float32,
            ),
            tile_shape,
        )

        if np.any(tile_min > tile_max):
            continue

        local_mask_shape = voxels["shape"]

        local_work_size = (
            kernel_config.TILE_SIZE,
            kernel_config.TILE_SIZE,
            kernel_config.TILE_SIZE,
        )

        group_count = tuple(int(tile_max[i] - tile_min[i] + 1) for i in range(3))

        global_work_size = (
            group_count[0] * local_work_size[0],
            group_count[1] * local_work_size[1],
            group_count[2] * local_work_size[2],
        )

        update_masks_kernels["update_source_velocity"](
            queue,
            global_work_size,
            local_work_size,
            velocity_x,
            velocity_y,
            velocity_z,
            tile_map,
            voxels["mask"],
            *[np.float32(value) for value in transform],
            np.float32(world_velocity[0]),
            np.float32(world_velocity[1]),
            np.float32(world_velocity[2]),
            np.int32(tile_min[0]),
            np.int32(tile_min[1]),
            np.int32(tile_min[2]),
            np.int32(tile_shape[0]),
            np.int32(tile_shape[1]),
            np.int32(tile_shape[2]),
            np.int32(local_mask_shape[0]),
            np.int32(local_mask_shape[1]),
            np.int32(local_mask_shape[2]),
        )


def update_obstacle_mask(
    queue: cl.CommandQueue,
    obstacle_kernels: dict[str, cl.Kernel],
    obstacle_mask: Any,
    obstacle_base_masks: Any,
    t: float,
    delta: float,
    origin_x: float,
    origin_y: float,
    origin_z: float,
    tile_map: Any,
    tile_shape: tuple[int, int, int],
    velocity_x: Any,
    velocity_y: Any,
    velocity_z: Any,
) -> None:
    """
    Rebuild the sparse obstacle mask from all transformed obstacle meshes.

    The mask is cleared before each update, then every mesh is evaluated over
    conservative tile bounds and unioned into the device field.
    """
    helper.fill_device(
        queue,
        obstacle_mask,
        0,
        dtype=np.bool_,
    )

    for entry in obstacle_base_masks:
        voxels = entry["voxels"]

        matrix, rate = helper.get_matrix_data(
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
            tile_shape,
        )

        velocity_transform = rate @ inv

        d = np.float32(delta)
        ox = np.float32(origin_x)
        oy = np.float32(origin_y)
        oz = np.float32(origin_z)

        vx_i = np.float32(velocity_transform[0, 0] * d)
        vx_j = np.float32(velocity_transform[0, 1] * d)
        vx_k = np.float32(velocity_transform[0, 2] * d)

        vx_c = np.float32(
            velocity_transform[0, 0] * ox
            + velocity_transform[0, 1] * oy
            + velocity_transform[0, 2] * oz
            + velocity_transform[0, 3]
        )

        vy_i = np.float32(velocity_transform[1, 0] * d)
        vy_j = np.float32(velocity_transform[1, 1] * d)
        vy_k = np.float32(velocity_transform[1, 2] * d)

        vy_c = np.float32(
            velocity_transform[1, 0] * ox
            + velocity_transform[1, 1] * oy
            + velocity_transform[1, 2] * oz
            + velocity_transform[1, 3]
        )

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

        local_mask_shape = voxels["shape"]

        local_work_size = (
            kernel_config.TILE_SIZE,
            kernel_config.TILE_SIZE,
            kernel_config.TILE_SIZE,
        )

        group_count = (
            int(tile_max[0] - tile_min[0] + 1),
            int(tile_max[1] - tile_min[1] + 1),
            int(tile_max[2] - tile_min[2] + 1),
        )

        global_work_size = (
            group_count[0] * local_work_size[0],
            group_count[1] * local_work_size[1],
            group_count[2] * local_work_size[2],
        )

        obstacle_kernels["update_obstacle_mask"](
            queue,
            global_work_size,
            local_work_size,
            obstacle_mask,
            velocity_x,
            velocity_y,
            velocity_z,
            tile_map,
            local_mask,
            np.float32(c0),
            np.float32(c1),
            np.float32(c2),
            np.float32(a00),
            np.float32(a01),
            np.float32(a02),
            np.float32(a10),
            np.float32(a11),
            np.float32(a12),
            np.float32(a20),
            np.float32(a21),
            np.float32(a22),
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
            np.int32(tile_min[0]),
            np.int32(tile_min[1]),
            np.int32(tile_min[2]),
            np.int32(tile_shape[0]),
            np.int32(tile_shape[1]),
            np.int32(tile_shape[2]),
            np.int32(local_mask_shape[0]),
            np.int32(local_mask_shape[1]),
            np.int32(local_mask_shape[2]),
        )
