import numpy as np
from numba import cuda

from Solver.Kernel_GPU_sparse.kernel_config import (
    REDUCTION_THREADS_PER_BLOCK,
    reduction_blocks_per_grid,
    TILE_SIZE,
)

@cuda.jit
def velocity_maxima_timestep(u, v, w, tile_map, maxima, total_tile_count):
    s_u = cuda.shared.array(REDUCTION_THREADS_PER_BLOCK, dtype=np.float32)
    s_v = cuda.shared.array(REDUCTION_THREADS_PER_BLOCK, dtype=np.float32)
    s_w = cuda.shared.array(REDUCTION_THREADS_PER_BLOCK, dtype=np.float32)

    tid = cuda.threadIdx.x
    stride = cuda.blockDim.x * cuda.gridDim.x
    idx = cuda.grid(1)

    tile_size = TILE_SIZE
    cells_per_tile = tile_size * tile_size * tile_size

    tiles_x, tiles_y, tiles_z = tile_map.shape
    tiles_per_yz = tiles_y * tiles_z

    max_u = np.float32(0.0)
    max_v = np.float32(0.0)
    max_w = np.float32(0.0)

    while idx < total_tile_count:
        tile_i = idx // tiles_per_yz
        remainder = idx % tiles_per_yz
        tile_j = remainder // tiles_z
        tile_k = remainder % tiles_z

        tile_index = tile_map[tile_i, tile_j, tile_k]
        if tile_index != -1:
            for local_flat in range(cells_per_tile):
                local_i = local_flat // (tile_size * tile_size)
                remainder2 = local_flat % (tile_size * tile_size)
                local_j = remainder2 // tile_size
                local_k = remainder2 % tile_size

                val_u = abs(u[tile_index, local_i, local_j, local_k])
                val_v = abs(v[tile_index, local_i, local_j, local_k])
                val_w = abs(w[tile_index, local_i, local_j, local_k])

                if val_u > max_u:
                    max_u = val_u
                if val_v > max_v:
                    max_v = val_v
                if val_w > max_w:
                    max_w = val_w

        idx += stride

    s_u[tid] = max_u
    s_v[tid] = max_v
    s_w[tid] = max_w
    cuda.syncthreads()

    offset = cuda.blockDim.x // 2
    while offset > 0:
        if tid < offset:
            if s_u[tid + offset] > s_u[tid]:
                s_u[tid] = s_u[tid + offset]
            if s_v[tid + offset] > s_v[tid]:
                s_v[tid] = s_v[tid + offset]
            if s_w[tid + offset] > s_w[tid]:
                s_w[tid] = s_w[tid + offset]
        cuda.syncthreads()
        offset //= 2

    if tid == 0:
        cuda.atomic.max(maxima, 0, s_u[0])
        cuda.atomic.max(maxima, 1, s_v[0])
        cuda.atomic.max(maxima, 2, s_w[0])


def compute_new_timestep_gpu(
    u, v, w, tile_map, active_tile_count, maxima, delta, cfl_max, max_dt=None
):
    eps = 1e-12

    if active_tile_count <= 0:
        return float(max_dt)

    total_tile_count = tile_map.size
    blockspergrid = reduction_blocks_per_grid(total_tile_count)

    velocity_maxima_timestep[blockspergrid, REDUCTION_THREADS_PER_BLOCK](
        u, v, w, tile_map, maxima, total_tile_count
    )

    abs_u_max, abs_v_max, abs_w_max = maxima.copy_to_host()

    cfl_delta = cfl_max * delta
    dt_conv = min(
        cfl_delta / max(float(abs_u_max), eps),
        cfl_delta / max(float(abs_v_max), eps),
        cfl_delta / max(float(abs_w_max), eps),
    )

    return min(dt_conv, float(max_dt))