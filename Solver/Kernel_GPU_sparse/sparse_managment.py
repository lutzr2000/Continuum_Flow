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
    return tile_i, tile_j, tile_k, i, j, k, nx, ny, nz


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
def dilate_tile_map(base_tile_map, tile_map, margin):
    tile_i, tile_j, tile_k = cuda.grid(3)
    tiles_x, tiles_y, tiles_z = tile_map.shape

    if tile_i >= tiles_x or tile_j >= tiles_y or tile_k >= tiles_z:
        return

    tile_map[tile_i, tile_j, tile_k] = -1

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
                    tile_map[tile_i, tile_j, tile_k] = 1
                    return