import Solver.Kernel_GPU.sparse_managment as sparse_managment
from numba import cuda


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
    use_obstacle_velocity,
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
    ) = sparse_managment.tile_to_index()

    if not mask[i, j, k]:
        return

    tile_index = tile_map[tile_i, tile_j, tile_k]
    if tile_index == -1:
        return

    if use_obstacle_velocity:
        u[tile_index, local_i, local_j, local_k] = obstacle_velocity_x[
            tile_index, local_i, local_j, local_k
        ]
        v[tile_index, local_i, local_j, local_k] = obstacle_velocity_y[
            tile_index, local_i, local_j, local_k
        ]
        w[tile_index, local_i, local_j, local_k] = obstacle_velocity_z[
            tile_index, local_i, local_j, local_k
        ]
    else:
        u[tile_index, local_i, local_j, local_k] = 0.0
        v[tile_index, local_i, local_j, local_k] = 0.0
        w[tile_index, local_i, local_j, local_k] = 0.0

    smoke[tile_index, local_i, local_j, local_k] = 0.0
    fuel[tile_index, local_i, local_j, local_k] = 0.0
    flame[tile_index, local_i, local_j, local_k] = 0.0
