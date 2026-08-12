from numba import cuda
import numpy as np
import Solver.Kernel_GPU_sparse.kernel_config as kernel_config


@cuda.jit(device=True, inline=True, cache=True)
def tile_to_index(field_shape):
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

    nx, ny, nz = field_shape

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
        nx,
        ny,
        nz,
    )


@cuda.jit(device=True, inline=True, cache=True)
def _sample_sparse_cell(field, tile_map, i, j, k, default_value):
    tile_size = kernel_config.TILE_SIZE

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
    temperature,
    smoke,
    fuel,
    flame,
    tile_map,
    source_mask,
    base_tile_map,
    threshold,
    ref_temp,
):
    tile_i, tile_j, tile_k = cuda.grid(3)
    tiles_x, tiles_y, tiles_z = base_tile_map.shape

    if tile_i >= tiles_x or tile_j >= tiles_y or tile_k >= tiles_z:
        return

    base_tile_map[tile_i, tile_j, tile_k] = -1

    tile_size = kernel_config.TILE_SIZE
    cell_i_start = tile_i * tile_size
    cell_j_start = tile_j * tile_size
    cell_k_start = tile_k * tile_size

    nx, ny, nz = source_mask.shape

    tile_index = tile_map[tile_i, tile_j, tile_k]

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

                if source_mask[i, j, k]:
                    base_tile_map[tile_i, tile_j, tile_k] = 1
                    return

                if tile_index == -1:
                    continue

                if (
                    abs(temperature[tile_index, local_i, local_j, local_k] - ref_temp) >= threshold
                    or smoke[tile_index, local_i, local_j, local_k] >= threshold
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
    tile_size = kernel_config.TILE_SIZE
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
def dilate_tile_map_persistent(
    base_tile_map,
    tile_map,
    margin,
    next_tile_index_counter,
    active_tile_counter,
):
    tile_i, tile_j, tile_k = cuda.grid(3)
    tiles_x, tiles_y, tiles_z = tile_map.shape

    if tile_i >= tiles_x or tile_j >= tiles_y or tile_k >= tiles_z:
        return

    is_active = False

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
                    is_active = True
                    break
            if is_active:
                break
        if is_active:
            break

    if not is_active:
        tile_map[tile_i, tile_j, tile_k] = -1
        return

    cuda.atomic.add(active_tile_counter, 0, 1)

    if tile_map[tile_i, tile_j, tile_k] != -1:
        return

    tile_map[tile_i, tile_j, tile_k] = cuda.atomic.add(next_tile_index_counter, 0, 1)


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


def ensure_pool_capacity(
    pool_tile_buffer,
    current_capacity_tiles,
    target_capacity_tiles,
    fill_value,
):
    if target_capacity_tiles == current_capacity_tiles:
        return pool_tile_buffer

    new_pool_tile_buffer = cuda.to_device(
        np.full(
            (
                target_capacity_tiles,
                kernel_config.TILE_SIZE,
                kernel_config.TILE_SIZE,
                kernel_config.TILE_SIZE,
            ),
            fill_value,
            dtype=kernel_config.GPU_FIELD_DTYPE,
        )
    )

    if current_capacity_tiles > 0:
        new_pool_tile_buffer[:current_capacity_tiles].copy_to_device(
            pool_tile_buffer[:current_capacity_tiles]
        )

    return new_pool_tile_buffer


def copy_pool(dst_pool, src_pool, active_tile_count):
    if active_tile_count <= 0:
        return

    dst_pool[:active_tile_count].copy_to_device(src_pool[:active_tile_count])


def reset_pool(dst_pool, fill_pool, active_tile_count):
    if active_tile_count <= 0:
        return

    dst_pool[:active_tile_count].copy_to_device(fill_pool[:active_tile_count])
