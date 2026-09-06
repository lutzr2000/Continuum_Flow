from numba import cuda
from typing import Any
import Solver.Kernel_GPU.kernel_config as kernel_config

tile_size = kernel_config.TILE_SIZE

@cuda.jit(device=True, inline=True, cache=True)
def tile_to_index():
    """
    Map tiles to cell indices.
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
def get_pool_value(field, tile_map, i, j, k, default_value):
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
    smoke,
    fuel,
    flame,
    tile_map,
    source_tile_mask,
    base_tile_map,
    threshold,
    nx,
    ny,
    nz,
):
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
    current_tile_map,
    previous_tile_map,
    compacted_tile_map,
    previous_index_lookup,
    next_tile_index_counter,
):
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
def remap_sparse_pool(old_pool, new_pool, previous_index_lookup, active_tile_count):
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
def fill_sparse_tile_buffer_range(pool, start_tile, fill_value):
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
def tile_is_active_in_margin(base_tile_map, tile_i, tile_j, tile_k, margin):
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
def copy_sparse_tile_buffer_range(src_pool, dst_pool, tile_count):
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
    base_tile_map,
    tile_map,
    margin,
    free_slot_stack,
    free_slot_count,
):
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
    base_tile_map,
    tile_map,
    margin,
    free_slot_stack,
    free_slot_count,
    reused_slot_stack,
    reused_slot_count,
    next_tile_index_counter,
    active_tile_counter,
):
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
def fill_sparse_tile_slots(pool, slot_indices, slot_count, fill_value):
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
    current_capacity_tiles,
    required_capacity_tiles,
    tile_growth_size,
):
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
    pool_specs,
    current_capacity_tiles,
    target_capacity_tiles,
):
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


def reset_reused_pool_slots(pool_specs, reused_slot_stack, reused_slot_count):
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


def copy_pools(dst_src_pairs, active_tile_count):
    if active_tile_count <= 0:
        return

    for dst_pool, src_pool in dst_src_pairs:
        dst_pool[:active_tile_count].copy_to_device(src_pool[:active_tile_count])


def reset_pools(dst_pools, fill_pool, active_tile_count):
    if active_tile_count <= 0:
        return

    for dst_pool in dst_pools:
        dst_pool[:active_tile_count].copy_to_device(fill_pool[:active_tile_count])
