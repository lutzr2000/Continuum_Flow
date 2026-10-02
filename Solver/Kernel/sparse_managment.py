import pyopencl as cl
import numpy as np

from typing import Any

import Solver.Kernel.kernel_config as kernel_config
import Solver.Kernel.helper as helper

FIELD_DTYPE = kernel_config.FIELD_DTYPE


def reset_pools(queue: cl.CommandQueue, dst_pools: Any, fill_pool: Any) -> None:
    """
    Reset scratch pools from a shared fill pool.
    """
    for dst_pool in dst_pools:
        cl.enqueue_copy(
            queue,
            dst_pool,
            fill_pool,
        )


def reset_reused_pool_slots(
    queue: cl.CommandQueue,
    sparse_managment_kernels: dict[str, cl.Kernel],
    pool_specs: Any,
    reused_slot_list: Any,
    reused_slot_count: int,
) -> None:
    """
    Restore field-specific defaults in every sparse slot reused this step.
    """
    if reused_slot_count <= 0:
        return

    cells_per_tile = (
        kernel_config.TILE_SIZE * kernel_config.TILE_SIZE * kernel_config.TILE_SIZE
    )

    total_cell_count = int(reused_slot_count) * cells_per_tile

    for pool_tile_buffer, fill_value, dtype in pool_specs:
        dtype = np.dtype(dtype)

        if dtype == np.dtype(np.bool_) or dtype == np.dtype(np.uint8):
            kernel = sparse_managment_kernels["fill_sparse_tile_slots_uchar"]
            kernel_fill_value = np.uint8(fill_value)
        elif dtype == np.dtype(FIELD_DTYPE):
            kernel = sparse_managment_kernels["fill_sparse_tile_slots"]
            kernel_fill_value = FIELD_DTYPE(fill_value)
        else:
            raise TypeError(f"Unsupported sparse pool dtype: {dtype}")

        kernel(
            queue,
            (total_cell_count,),
            None,
            pool_tile_buffer,
            reused_slot_list,
            np.int32(reused_slot_count),
            kernel_fill_value,
        )


def required_pool_capacity(
    current_capacity_tiles: Any,
    required_capacity_tiles: Any,
    tile_growth_size: Any,
) -> Any:
    """
    Compute a geometrically grown capacity that covers the required slot count.

    Growth proceeds in fixed increments and never exceeds the total number of
    logical tiles.
    """
    required_capacity_tiles = int(required_capacity_tiles)
    current_capacity_tiles = int(current_capacity_tiles)
    tile_growth_size = max(int(tile_growth_size), 1)

    if required_capacity_tiles <= current_capacity_tiles:
        return current_capacity_tiles

    new_capacity_tiles = max(current_capacity_tiles, tile_growth_size)
    while required_capacity_tiles > new_capacity_tiles:
        new_capacity_tiles += tile_growth_size

    return new_capacity_tiles


def ensure_pool_capacities(
    context: cl.Context,
    queue: cl.CommandQueue,
    pool_specs: Any,
    current_capacity_tiles: int,
    target_capacity_tiles: int,
) -> Any:
    """
    Grow undersized GPU field pools while preserving all allocated tile data.

    Pools already large enough are returned unchanged; replacement buffers are
    initialized and populated through device-to-device copies.
    """
    if target_capacity_tiles == current_capacity_tiles:
        return [pool for pool, _fill_value, _dtype in pool_specs]

    resized_pools = []

    cells_per_tile = (
        kernel_config.TILE_SIZE * kernel_config.TILE_SIZE * kernel_config.TILE_SIZE
    )

    for pool_tile_buffer, fill_value, dtype in pool_specs:
        itemsize = np.dtype(dtype).itemsize

        new_pool_tile_buffer = helper.device_array(
            context,
            (
                target_capacity_tiles,
                kernel_config.TILE_SIZE,
                kernel_config.TILE_SIZE,
                kernel_config.TILE_SIZE,
            ),
            dtype=dtype,
        )

        if current_capacity_tiles > 0:
            copy_size = current_capacity_tiles * cells_per_tile * itemsize

            cl.enqueue_copy(
                queue,
                new_pool_tile_buffer,
                pool_tile_buffer,
                byte_count=copy_size,
            )

        if target_capacity_tiles > current_capacity_tiles:
            fill_offset = current_capacity_tiles * cells_per_tile * itemsize

            fill_size = (
                (target_capacity_tiles - current_capacity_tiles)
                * cells_per_tile
                * itemsize
            )

            cl.enqueue_fill_buffer(
                queue,
                new_pool_tile_buffer,
                np.asarray(fill_value, dtype=dtype),
                fill_offset,
                fill_size,
            )

        resized_pools.append(new_pool_tile_buffer)

    return resized_pools


def copy_pools(
    queue: cl.CommandQueue,
    dst_src_pairs: Any,
    active_tile_count: int,
) -> None:
    """
    Copy the allocated portion of several sparse GPU pools.
    """
    if active_tile_count <= 0:
        return

    cells_per_tile = kernel_config.TILE_SIZE**3
    copy_size = active_tile_count * cells_per_tile * np.dtype(FIELD_DTYPE).itemsize

    for dst_pool, src_pool in dst_src_pairs:
        cl.enqueue_copy(
            queue,
            dst_pool,
            src_pool,
            byte_count=copy_size,
        )
