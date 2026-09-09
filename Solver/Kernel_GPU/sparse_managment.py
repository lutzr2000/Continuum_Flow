from numba import cuda
from typing import Any
import Solver.Kernel_GPU.kernel_config as kernel_config

tile_size = kernel_config.TILE_SIZE


@cuda.jit(device=True, inline=True, cache=True)
def tile_to_index() -> Any:
    """
    Convert CUDA block and thread coordinates into tile-local and global cells.
    """
    tile_i = cuda.blockIdx.x
    tile_j = cuda.blockIdx.y
    tile_k = cuda.blockIdx.z

    local_i = cuda.threadIdx.x
    local_j = cuda.threadIdx.y
    local_k = cuda.threadIdx.z

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


@cuda.jit(device=True, inline=True, cache=True)
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


@cuda.jit(cache=True)
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
    tile_i, tile_j, tile_k = cuda.grid(3)

    if (
        tile_i >= base_tile_map.shape[0]
        or tile_j >= base_tile_map.shape[1]
        or tile_k >= base_tile_map.shape[2]
    ):
        return

    base_tile_map[tile_i, tile_j, tile_k] = -1

    # Sources activate a tile regardless of existing sparse allocation.
    if source_tile_mask[tile_i, tile_j, tile_k]:
        base_tile_map[tile_i, tile_j, tile_k] = 1
        return

    tile_index = tile_map[tile_i, tile_j, tile_k]

    if tile_index == -1:
        return

    cell_i_start = tile_i * tile_size
    cell_j_start = tile_j * tile_size
    cell_k_start = tile_k * tile_size

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
                    base_tile_map[tile_i, tile_j, tile_k] = 1
                    return


@cuda.jit(cache=True)
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
    tile_i, tile_j, tile_k = cuda.grid(3)
    tiles_x, tiles_y, tiles_z = current_tile_map.shape

    if tile_i >= tiles_x or tile_j >= tiles_y or tile_k >= tiles_z:
        return

    current_index = current_tile_map[tile_i, tile_j, tile_k]
    if current_index == -1:
        compacted_tile_map[tile_i, tile_j, tile_k] = -1
        return

    compacted_index = cuda.atomic.add(next_tile_index_counter, 0, 1)
    compacted_tile_map[tile_i, tile_j, tile_k] = compacted_index
    previous_index_lookup[compacted_index] = previous_tile_map[tile_i, tile_j, tile_k]


@cuda.jit(cache=True)
def remap_sparse_pool(
    old_pool: Any, new_pool: Any, previous_index_lookup: Any, active_tile_count: int
) -> None:
    """
    Copy active tile data from old sparse slots into their compacted slots.

    Newly activated tiles without a previous slot are initialized to zero.
    """
    flat_index = cuda.grid(1)
    cells_per_tile = tile_size * tile_size * tile_size
    total_cell_count = active_tile_count * cells_per_tile

    if flat_index >= total_cell_count:
        return

    compacted_index = flat_index // cells_per_tile
    local_flat_index = flat_index % cells_per_tile

    local_i = local_flat_index // (tile_size * tile_size)
    remainder = local_flat_index % (tile_size * tile_size)
    local_j = remainder // tile_size
    local_k = remainder % tile_size

    previous_index = previous_index_lookup[compacted_index]
    if previous_index == -1:
        new_pool[compacted_index, local_i, local_j, local_k] = 0.0
        return

    new_pool[compacted_index, local_i, local_j, local_k] = old_pool[
        previous_index,
        local_i,
        local_j,
        local_k,
    ]


@cuda.jit(cache=True)
def fill_sparse_tile_buffer_range(
    pool: Any, start_tile: int, fill_value: float
) -> None:
    """
    Fill every cell from ``start_tile`` through the end of a sparse pool.
    """
    flat_index = cuda.grid(1)
    cells_per_tile = tile_size * tile_size * tile_size
    tile_count = pool.shape[0] - start_tile
    total_cell_count = tile_count * cells_per_tile

    if flat_index >= total_cell_count:
        return

    tile_offset = flat_index // cells_per_tile
    local_flat_index = flat_index % cells_per_tile
    tile_index = start_tile + tile_offset

    local_i = local_flat_index // (tile_size * tile_size)
    remainder = local_flat_index % (tile_size * tile_size)
    local_j = remainder // tile_size
    local_k = remainder % tile_size

    pool[tile_index, local_i, local_j, local_k] = fill_value


@cuda.jit(device=True, inline=True, cache=True)
def tile_is_active_in_margin(
    base_tile_map: Any, tile_i: Any, tile_j: Any, tile_k: Any, margin: int
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


@cuda.jit(cache=True)
def copy_sparse_tile_buffer_range(
    src_pool: Any, dst_pool: Any, tile_count: int
) -> None:
    """
    Copy all cells belonging to the first ``tile_count`` sparse slots.
    """
    flat_index = cuda.grid(1)
    cells_per_tile = tile_size * tile_size * tile_size
    total_cell_count = tile_count * cells_per_tile

    if flat_index >= total_cell_count:
        return

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


@cuda.jit(cache=True)
def release_inactive_tile_slots(
    base_tile_map: Any,
    tile_map: Any,
    margin: int,
    free_slot_stack: Any,
    free_slot_count: int,
) -> None:
    """
    Remove inactive tiles from the map and push their slots onto the free stack.

    Activity is dilated by ``margin`` so tiles near active content remain
    allocated and can support neighboring stencil and advection samples.
    """
    tile_i, tile_j, tile_k = cuda.grid(3)
    tiles_x, tiles_y, tiles_z = tile_map.shape

    if tile_i >= tiles_x or tile_j >= tiles_y or tile_k >= tiles_z:
        return

    if tile_map[tile_i, tile_j, tile_k] == -1:
        return

    if tile_is_active_in_margin(base_tile_map, tile_i, tile_j, tile_k, margin):
        return

    released_slot = tile_map[tile_i, tile_j, tile_k]
    tile_map[tile_i, tile_j, tile_k] = -1
    stack_index = cuda.atomic.add(free_slot_count, 0, 1)
    free_slot_stack[stack_index] = released_slot


@cuda.jit(cache=True)
def activate_tiles_with_reuse(
    base_tile_map: Any,
    tile_map: Any,
    margin: int,
    free_slot_stack: Any,
    free_slot_count: int,
    reused_slot_stack: Any,
    reused_slot_count: int,
    next_tile_index_counter: Any,
    active_tile_counter: Any,
) -> None:
    """
    Allocate required tiles from freed slots before extending the sparse pool.

    Reused indices are recorded separately so all associated field buffers can
    be reset before simulation kernels access the newly activated tiles.
    """
    tile_i, tile_j, tile_k = cuda.grid(3)
    tiles_x, tiles_y, tiles_z = tile_map.shape

    if tile_i >= tiles_x or tile_j >= tiles_y or tile_k >= tiles_z:
        return

    if not tile_is_active_in_margin(base_tile_map, tile_i, tile_j, tile_k, margin):
        return

    cuda.atomic.add(active_tile_counter, 0, 1)

    if tile_map[tile_i, tile_j, tile_k] != -1:
        return

    previous_free_count = cuda.atomic.add(free_slot_count, 0, -1)
    if previous_free_count > 0:
        slot_index = free_slot_stack[previous_free_count - 1]
        tile_map[tile_i, tile_j, tile_k] = slot_index
        reused_index = cuda.atomic.add(reused_slot_count, 0, 1)
        reused_slot_stack[reused_index] = slot_index
        return

    cuda.atomic.add(free_slot_count, 0, 1)
    tile_map[tile_i, tile_j, tile_k] = cuda.atomic.add(next_tile_index_counter, 0, 1)


@cuda.jit(cache=True)
def fill_sparse_tile_slots(
    pool: Any, slot_indices: Any, slot_count: int, fill_value: float
) -> None:
    """
    Initialize the listed sparse pool slots with one scalar value.
    """
    flat_index = cuda.grid(1)
    cells_per_tile = tile_size * tile_size * tile_size
    total_cell_count = slot_count * cells_per_tile

    if flat_index >= total_cell_count:
        return

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
    Grow undersized GPU field pools while preserving all allocated tile data.

    Pools already large enough are returned unchanged; replacement buffers are
    initialized and populated through device-to-device copies.
    """
    if target_capacity_tiles == current_capacity_tiles:
        return [pool for pool, _fill_value in pool_specs]

    resized_pools = []
    cells_per_tile = tile_size * tile_size * tile_size
    threads_per_block = 256

    for pool_tile_buffer, fill_value in pool_specs:
        new_pool_tile_buffer = cuda.device_array(
            (
                target_capacity_tiles,
                tile_size,
                tile_size,
                tile_size,
            ),
            dtype=pool_tile_buffer.dtype,
        )

        if current_capacity_tiles > 0:
            copy_cell_count = current_capacity_tiles * cells_per_tile
            copy_blocks = (copy_cell_count + threads_per_block - 1) // threads_per_block
            copy_sparse_tile_buffer_range[copy_blocks, threads_per_block](
                pool_tile_buffer,
                new_pool_tile_buffer,
                current_capacity_tiles,
            )

        if target_capacity_tiles > current_capacity_tiles:
            fill_cell_count = (
                target_capacity_tiles - current_capacity_tiles
            ) * cells_per_tile
            fill_blocks = (fill_cell_count + threads_per_block - 1) // threads_per_block
            fill_sparse_tile_buffer_range[fill_blocks, threads_per_block](
                new_pool_tile_buffer,
                current_capacity_tiles,
                fill_value,
            )

        resized_pools.append(new_pool_tile_buffer)

    return resized_pools


def reset_reused_pool_slots(
    pool_specs: Any, reused_slot_stack: Any, reused_slot_count: int
) -> None:
    """
    Restore field-specific defaults in every sparse slot reused this step.
    """
    if reused_slot_count <= 0:
        return

    cells_per_tile = tile_size * tile_size * tile_size
    threads_per_block = 256
    total_cell_count = int(reused_slot_count) * cells_per_tile
    blocks = (total_cell_count + threads_per_block - 1) // threads_per_block

    for pool_tile_buffer, fill_value in pool_specs:
        fill_sparse_tile_slots[blocks, threads_per_block](
            pool_tile_buffer,
            reused_slot_stack,
            reused_slot_count,
            fill_value,
        )


def copy_pools(dst_src_pairs: Any, active_tile_count: int) -> None:
    """
    Copy the allocated portion of several sparse GPU pools in one operation.
    """
    if active_tile_count <= 0:
        return

    for dst_pool, src_pool in dst_src_pairs:
        dst_pool[:active_tile_count].copy_to_device(src_pool[:active_tile_count])


def reset_pools(dst_pools: Any, fill_pool: Any, active_tile_count: int) -> None:
    """
    Reset scratch pools from a shared fill pool over the allocated tile range.
    """
    if active_tile_count <= 0:
        return

    for dst_pool in dst_pools:
        dst_pool[:active_tile_count].copy_to_device(fill_pool[:active_tile_count])
