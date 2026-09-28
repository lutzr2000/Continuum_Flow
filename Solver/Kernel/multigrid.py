from typing import Any

import numpy as np
import pyopencl as cl

import Solver.Kernel.kernel_config as kernel_config
import Solver.Kernel.helper as helper

FIELD_DTYPE = kernel_config.FIELD_DTYPE


def create_multigrid_levels(
    context: cl.Context,
    shape: tuple[int, int, int],
    delta: float,
    level_0_pool_capacity: int,
    min_size: int = 8,
) -> Any:
    """
    Allocate the sparse coarse levels used below the sparse simulation grid.

    Every level halves each dimension with upward rounding and doubles the
    physical cell spacing. Pressure, right-hand-side, tile maps, active-tile
    lists are allocated until any dimension would fall below ``min_size``.
    """
    p_levels = []
    b_levels = []
    delta_levels = []
    tile_maps = []
    active_tiles = []
    active_tile_counts = []
    level_shapes = []

    nx = (shape[0] + 1) // 2
    ny = (shape[1] + 1) // 2
    nz = (shape[2] + 1) // 2

    level = 1
    parent_pool_capacity = level_0_pool_capacity

    while nx >= min_size and ny >= min_size and nz >= min_size:
        level_shape = (nx, ny, nz)

        tile_shape = (
            (nx + kernel_config.TILE_SIZE - 1) // kernel_config.TILE_SIZE,
            (ny + kernel_config.TILE_SIZE - 1) // kernel_config.TILE_SIZE,
            (nz + kernel_config.TILE_SIZE - 1) // kernel_config.TILE_SIZE,
        )

        total_tile_count = tile_shape[0] * tile_shape[1] * tile_shape[2]

        pool_capacity = min(
            parent_pool_capacity,
            total_tile_count,
        )

        pool_shape = (
            pool_capacity,
            kernel_config.TILE_SIZE,
            kernel_config.TILE_SIZE,
            kernel_config.TILE_SIZE,
        )

        p_levels.append(
            helper.device_array(
                context,
                pool_shape,
                dtype=FIELD_DTYPE,
            )
        )

        b_levels.append(
            helper.device_array(
                context,
                pool_shape,
                dtype=FIELD_DTYPE,
            )
        )

        tile_maps.append(
            helper.to_device(
                context,
                np.full(
                    tile_shape,
                    -1,
                    dtype=np.int32,
                ),
            )
        )

        active_tiles.append(
            helper.device_array(
                context,
                (pool_capacity, 3),
                dtype=np.int32,
            )
        )

        active_tile_counts.append(
            helper.to_device(context, np.zeros(1, dtype=np.int32))
        )

        level_shapes.append(level_shape)
        delta_levels.append(delta * (2**level))

        parent_pool_capacity = pool_capacity

        nx = (nx + 1) // 2
        ny = (ny + 1) // 2
        nz = (nz + 1) // 2
        level += 1

    return (
        p_levels,
        b_levels,
        delta_levels,
        tile_maps,
        active_tiles,
        active_tile_counts,
        level_shapes,
    )
