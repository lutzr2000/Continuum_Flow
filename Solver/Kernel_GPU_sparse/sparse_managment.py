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


def update_tile_map(smoke,fuel,flame,tile_map):
    print("nothing implemented yet")
