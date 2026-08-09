import math
import Solver.Kernel_GPU_sparse.sparse_managment as sparse_managment
from numba import cuda
from numba import njit, prange

@cuda.jit(cache=True)
def obstacle_bc_kernel(
    u,
    v,
    w,
    smoke,
    fuel,
    flame,
    mask,
    obstacle_velocity_x,
    obstacle_velocity_y,
    obstacle_velocity_z,
    tile_map,
):
    """
    Apply obstacle boundary conditions.
    """
    (
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
    ) = sparse_managment.tile_to_index(mask.shape)

    if i >= nx or j >= ny or k >= nz:
        return

    if not mask[i, j, k]:
        return

    u[i, j, k] = obstacle_velocity_x[i, j, k]
    v[i, j, k] = obstacle_velocity_y[i, j, k]
    w[i, j, k] = obstacle_velocity_z[i, j, k]

    smoke[i, j, k] = 0.0
    fuel[i, j, k] = 0.0

    tile_index = tile_map[tile_i, tile_j, tile_k]
    if tile_index == -1:
        return

    flame[tile_index, local_i, local_j, local_k] = 0.0