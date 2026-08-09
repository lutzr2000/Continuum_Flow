from numba import cuda

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


@cuda.jit(cache=True)
def build_base_tile_map(smoke, fuel, flame, base_tile_map, threshold):
    tile_i, tile_j, tile_k = cuda.grid(3)
    tiles_x, tiles_y, tiles_z = base_tile_map.shape

    if tile_i >= tiles_x or tile_j >= tiles_y or tile_k >= tiles_z:
        return

    nx, ny, nz = smoke.shape
    tile_size = kernel_config.TILE_SIZE

    cell_i_start = tile_i * tile_size
    cell_j_start = tile_j * tile_size
    cell_k_start = tile_k * tile_size

    base_tile_map[tile_i, tile_j, tile_k] = -1

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
                    smoke[i, j, k] >= threshold
                    or fuel[i, j, k] >= threshold
                    or flame[i, j, k] >= threshold
                ):
                    base_tile_map[tile_i, tile_j, tile_k] = 1
                    return


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


def ensure_pool_capacity(
    pool_tile_buffer,
    current_capacity_tiles,
    required_capacity_tiles,
    tile_growth_size,
):
    """
    Grow a pool tile buffer in fixed-size chunkgs until it can hold all requiered tiles.
    Copies the original data into the new pool.
    """
    required_capacity_tiles = int(required_capacity_tiles)
    current_capacity_tiles = int(current_capacity_tiles)
    tile_growth_size = max(int(tile_growth_size), 1)

    if required_capacity_tiles <= current_capacity_tiles:
        return pool_tile_buffer, current_capacity_tiles

    new_capacity_tiles = max(current_capacity_tiles, tile_growth_size)
    while required_capacity_tiles > new_capacity_tiles:
        new_capacity_tiles += tile_growth_size

    new_pool_tile_buffer = cuda.device_array(
        (
            new_capacity_tiles,
            kernel_config.TILE_SIZE,
            kernel_config.TILE_SIZE,
            kernel_config.TILE_SIZE,
        ),
        dtype=kernel_config.GPU_FIELD_DTYPE,
    )

    # Copy existing tile data into the newly allocated larger pool.
    if current_capacity_tiles > 0:
        new_pool_tile_buffer[:current_capacity_tiles].copy_to_device(
            pool_tile_buffer[:current_capacity_tiles]
        )

    pool_tile_buffer = new_pool_tile_buffer

    return pool_tile_buffer, new_capacity_tiles

