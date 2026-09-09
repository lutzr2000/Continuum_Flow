from typing import Any

import numpy as np
from numba import njit, prange

import Solver.Kernel_CPU.kernel_config as kernel_config

tile_size = kernel_config.TILE_SIZE


@njit(device=False, inline="always", cache=True)
def tile_to_index(
    tile_flat: int,
    local_i: int,
    local_j: int,
    local_k: int,
    tiles_x: int,
    tiles_y: int,
    tiles_z: int,
) -> Any:
    """
    Convert flattened CPU tile coordinates and tile-local coordinates into
    tile-local and global cell indices.
    """
    tiles_per_yz = tiles_y * tiles_z

    tile_i = tile_flat // tiles_per_yz
    remainder = tile_flat % tiles_per_yz

    tile_j = remainder // tiles_z
    tile_k = remainder % tiles_z

    i = tile_i * kernel_config.TILE_SIZE + local_i
    j = tile_j * kernel_config.TILE_SIZE + local_j
    k = tile_k * kernel_config.TILE_SIZE + local_k

    return (
        tile_i,
        tile_j,
        tile_k,
        local_i,
        local_j,
        local_k,
        i,
        j,
        k,
    )


@njit(inline="always", cache=True)
def get_pool_value(
    field: Any, tile_map: Any, i: int, j: int, k: int, default_value: float
) -> Any:
    """
    Read a sparse field value or return the background value for an inactive tile.
    """
    tile_i = i // tile_size
    tile_j = j // tile_size
    tile_k = k // tile_size

    tile_index = tile_map[tile_i, tile_j, tile_k]

    if tile_index == -1:
        return default_value

    local_i = i - tile_i * tile_size
    local_j = j - tile_j * tile_size
    local_k = k - tile_k * tile_size

    return field[tile_index, local_i, local_j, local_k]


@njit(cache=True, parallel=True)
def build_activity_mask(
    smoke: Any,
    fuel: Any,
    flame: Any,
    tile_map: Any,
    source_tile_mask: Any,
    base_tile_map: Any,
    threshold: float,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    """
    Mark tiles active when they contain source, smoke, fuel, or flame data.

    The base activity map is rebuilt in parallel. Source tiles are activated
    unconditionally; allocated tiles remain active only when at least one cell
    exceeds the configured scalar threshold.
    """
    tiles_x, tiles_y, tiles_z = base_tile_map.shape
    total_tiles = tiles_x * tiles_y * tiles_z

    for tile_flat in prange(total_tiles):
        tile_i = tile_flat // (tiles_y * tiles_z)
        remainder = tile_flat % (tiles_y * tiles_z)
        tile_j = remainder // tiles_z
        tile_k = remainder % tiles_z

        base_tile_map[tile_i, tile_j, tile_k] = -1

        # Sources activate a tile regardless of existing sparse allocation.
        if source_tile_mask[tile_i, tile_j, tile_k]:
            base_tile_map[tile_i, tile_j, tile_k] = 1
            continue

        tile_index = tile_map[tile_i, tile_j, tile_k]

        if tile_index == -1:
            continue

        cell_i_start = tile_i * tile_size
        cell_j_start = tile_j * tile_size
        cell_k_start = tile_k * tile_size

        active = False

        for local_i in range(tile_size):
            i = cell_i_start + local_i

            if i >= nx:
                break

            for local_j in range(tile_size):
                j = cell_j_start + local_j

                if j >= ny:
                    break

                for local_k in range(tile_size):
                    k = cell_k_start + local_k

                    if k >= nz:
                        break

                    if (
                        smoke[tile_index, local_i, local_j, local_k] >= threshold
                        or fuel[tile_index, local_i, local_j, local_k] >= threshold
                        or flame[tile_index, local_i, local_j, local_k] >= threshold
                    ):
                        active = True
                        break

                if active:
                    break

            if active:
                break

        if active:
            base_tile_map[tile_i, tile_j, tile_k] = 1


@njit(cache=True)
def compact_active_tile_map(
    current_tile_map: Any,
    previous_tile_map: Any,
    compacted_tile_map: Any,
    previous_index_lookup: Any,
    next_tile_index_counter: Any,
) -> None:
    """
    Assign contiguous pool indices to active tiles and record their old slots.
    """
    tiles_x, tiles_y, tiles_z = current_tile_map.shape

    compacted_index = 0

    for tile_i in range(tiles_x):
        for tile_j in range(tiles_y):
            for tile_k in range(tiles_z):
                current_index = current_tile_map[tile_i, tile_j, tile_k]

                if current_index == -1:
                    compacted_tile_map[tile_i, tile_j, tile_k] = -1
                    continue

                compacted_tile_map[tile_i, tile_j, tile_k] = compacted_index

                previous_index_lookup[compacted_index] = previous_tile_map[
                    tile_i,
                    tile_j,
                    tile_k,
                ]

                compacted_index += 1

    next_tile_index_counter[0] = compacted_index


@njit(cache=True, parallel=True)
def remap_sparse_pool(
    old_pool: Any,
    new_pool: Any,
    previous_index_lookup: Any,
    active_tile_count: int,
) -> None:
    """
    Copy active tile data from old sparse slots into their compacted slots.

    Newly activated tiles without a previous slot are initialized to zero.
    """
    cells_per_tile = tile_size * tile_size * tile_size
    total_cell_count = active_tile_count * cells_per_tile

    for flat_index in prange(total_cell_count):
        compacted_index = flat_index // cells_per_tile
        local_flat_index = flat_index % cells_per_tile

        local_i = local_flat_index // (tile_size * tile_size)
        remainder = local_flat_index % (tile_size * tile_size)
        local_j = remainder // tile_size
        local_k = remainder % tile_size

        previous_index = previous_index_lookup[compacted_index]

        if previous_index == -1:
            new_pool[compacted_index, local_i, local_j, local_k] = 0.0
            continue

        new_pool[compacted_index, local_i, local_j, local_k] = old_pool[
            previous_index,
            local_i,
            local_j,
            local_k,
        ]


@njit(cache=True, parallel=True)
def fill_sparse_tile_buffer_range(
    pool: Any,
    start_tile: int,
    fill_value: float,
) -> None:
    """
    Fill every cell from ``start_tile`` through the end of a sparse pool.
    """
    cells_per_tile = tile_size * tile_size * tile_size
    tile_count = pool.shape[0] - start_tile
    total_cell_count = tile_count * cells_per_tile

    for flat_index in prange(total_cell_count):
        tile_offset = flat_index // cells_per_tile
        local_flat_index = flat_index % cells_per_tile
        tile_index = start_tile + tile_offset

        local_i = local_flat_index // (tile_size * tile_size)
        remainder = local_flat_index % (tile_size * tile_size)
        local_j = remainder // tile_size
        local_k = remainder % tile_size

        pool[tile_index, local_i, local_j, local_k] = fill_value


@njit(inline="always", cache=True)
def tile_is_active_in_margin(
    base_tile_map: Any,
    tile_i: Any,
    tile_j: Any,
    tile_k: Any,
    margin: int,
) -> Any:
    """
    Check whether an active base tile lies within a bounded neighborhood.
    """
    tiles_x, tiles_y, tiles_z = base_tile_map.shape

    for di in range(-margin, margin + 1):
        ni = tile_i + di

        if ni < 0 or ni >= tiles_x:
            continue

        for dj in range(-margin, margin + 1):
            nj = tile_j + dj

            if nj < 0 or nj >= tiles_y:
                continue

            for dk in range(-margin, margin + 1):
                nk = tile_k + dk

                if nk < 0 or nk >= tiles_z:
                    continue

                if base_tile_map[ni, nj, nk] != -1:
                    return True

    return False


@njit(cache=True, parallel=True)
def copy_sparse_tile_buffer_range(
    src_pool: Any,
    dst_pool: Any,
    tile_count: int,
) -> None:
    """
    Copy all cells belonging to the first ``tile_count`` sparse slots.
    """
    cells_per_tile = tile_size * tile_size * tile_size
    total_cell_count = tile_count * cells_per_tile

    for flat_index in prange(total_cell_count):
        tile_index = flat_index // cells_per_tile
        local_flat_index = flat_index % cells_per_tile

        local_i = local_flat_index // (tile_size * tile_size)
        remainder = local_flat_index % (tile_size * tile_size)
        local_j = remainder // tile_size
        local_k = remainder % tile_size

        dst_pool[tile_index, local_i, local_j, local_k] = src_pool[
            tile_index,
            local_i,
            local_j,
            local_k,
        ]


@njit(cache=True)
def release_inactive_tile_slots(
    base_tile_map: Any,
    tile_map: Any,
    margin: int,
    free_slot_stack: Any,
    free_slot_count: Any,
) -> None:
    """
    Remove inactive tiles from the map and push their slots onto the free stack.

    Activity is dilated by ``margin`` so tiles near active content remain
    allocated and can support neighboring stencil and advection samples.
    """
    tiles_x, tiles_y, tiles_z = tile_map.shape

    stack_index = int(free_slot_count[0])

    for tile_i in range(tiles_x):
        for tile_j in range(tiles_y):
            for tile_k in range(tiles_z):
                if tile_map[tile_i, tile_j, tile_k] == -1:
                    continue

                if tile_is_active_in_margin(
                    base_tile_map,
                    tile_i,
                    tile_j,
                    tile_k,
                    margin,
                ):
                    continue

                released_slot = tile_map[tile_i, tile_j, tile_k]

                tile_map[tile_i, tile_j, tile_k] = -1

                free_slot_stack[stack_index] = released_slot
                stack_index += 1

    free_slot_count[0] = stack_index


@njit(cache=True)
def activate_tiles_with_reuse(
    base_tile_map: Any,
    tile_map: Any,
    margin: int,
    free_slot_stack: Any,
    free_slot_count: Any,
    reused_slot_stack: Any,
    reused_slot_count: Any,
    next_tile_index_counter: Any,
    active_tile_counter: Any,
) -> None:
    """
    Allocate required tiles from freed slots before extending the sparse pool.

    Reused indices are recorded separately so all associated field buffers can
    be reset before simulation kernels access the newly activated tiles.
    """
    tiles_x, tiles_y, tiles_z = tile_map.shape

    free_count = int(free_slot_count[0])
    reused_count = int(reused_slot_count[0])
    next_tile_index = int(next_tile_index_counter[0])
    active_count = 0

    for tile_i in range(tiles_x):
        for tile_j in range(tiles_y):
            for tile_k in range(tiles_z):
                if not tile_is_active_in_margin(
                    base_tile_map,
                    tile_i,
                    tile_j,
                    tile_k,
                    margin,
                ):
                    continue

                active_count += 1

                if tile_map[tile_i, tile_j, tile_k] != -1:
                    continue

                if free_count > 0:
                    free_count -= 1

                    slot_index = free_slot_stack[free_count]

                    tile_map[tile_i, tile_j, tile_k] = slot_index

                    reused_slot_stack[reused_count] = slot_index
                    reused_count += 1

                    continue

                tile_map[tile_i, tile_j, tile_k] = next_tile_index
                next_tile_index += 1

    free_slot_count[0] = free_count
    reused_slot_count[0] = reused_count
    next_tile_index_counter[0] = next_tile_index
    active_tile_counter[0] = active_count


@njit(cache=True, parallel=True)
def fill_sparse_tile_slots(
    pool: Any,
    slot_indices: Any,
    slot_count: int,
    fill_value: float,
) -> None:
    """
    Initialize the listed sparse pool slots with one scalar value.
    """
    cells_per_tile = tile_size * tile_size * tile_size
    total_cell_count = slot_count * cells_per_tile

    for flat_index in prange(total_cell_count):
        slot_offset = flat_index // cells_per_tile
        local_flat_index = flat_index % cells_per_tile
        tile_index = slot_indices[slot_offset]

        local_i = local_flat_index // (tile_size * tile_size)
        remainder = local_flat_index % (tile_size * tile_size)
        local_j = remainder // tile_size
        local_k = remainder % tile_size

        pool[tile_index, local_i, local_j, local_k] = fill_value


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
    pool_specs: Any,
    current_capacity_tiles: Any,
    target_capacity_tiles: Any,
) -> Any:
    """
    Grow undersized CPU field pools while preserving all allocated tile data.

    Pools already large enough are returned unchanged; replacement buffers are
    initialized and populated with NumPy arrays.
    """
    if target_capacity_tiles == current_capacity_tiles:
        return [pool for pool, _fill_value in pool_specs]

    resized_pools = []

    for pool_tile_buffer, fill_value in pool_specs:
        new_pool_tile_buffer = np.full(
            (
                target_capacity_tiles,
                tile_size,
                tile_size,
                tile_size,
            ),
            fill_value,
            dtype=pool_tile_buffer.dtype,
        )

        if current_capacity_tiles > 0:
            new_pool_tile_buffer[:current_capacity_tiles] = pool_tile_buffer[
                :current_capacity_tiles
            ]

        resized_pools.append(new_pool_tile_buffer)

    return resized_pools


def reset_reused_pool_slots(
    pool_specs: Any,
    reused_slot_stack: Any,
    reused_slot_count: int,
) -> None:
    """
    Restore field-specific defaults in every sparse slot reused this step.
    """
    if reused_slot_count <= 0:
        return

    for pool_tile_buffer, fill_value in pool_specs:
        fill_sparse_tile_slots(
            pool_tile_buffer,
            reused_slot_stack,
            reused_slot_count,
            fill_value,
        )


def copy_pools(dst_src_pairs: Any, active_tile_count: int) -> None:
    """
    Copy the allocated portion of several sparse CPU pools in one operation.
    """
    if active_tile_count <= 0:
        return

    for dst_pool, src_pool in dst_src_pairs:
        dst_pool[:active_tile_count] = src_pool[:active_tile_count]


def reset_pools(
    dst_pools: Any,
    fill_pool: Any,
    active_tile_count: int,
) -> None:
    """
    Reset scratch pools from a shared fill pool over the allocated tile range.
    """
    if active_tile_count <= 0:
        return

    for dst_pool in dst_pools:
        dst_pool[:active_tile_count] = fill_pool[:active_tile_count]
