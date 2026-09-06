import Solver.Kernel_GPU.sparse_managment as sparse_managment
from numba import cuda


@cuda.jit(cache=True)
def obstacle_bc(
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

    tile_index = tile_map[tile_i, tile_j, tile_k]

    if tile_index == -1:
        return

    if not mask[tile_index, local_i, local_j, local_k]:
        return

    u[tile_index, local_i, local_j, local_k] = obstacle_velocity_x[
        tile_index, local_i, local_j, local_k
    ]

    v[tile_index, local_i, local_j, local_k] = obstacle_velocity_y[
        tile_index, local_i, local_j, local_k
    ]

    w[tile_index, local_i, local_j, local_k] = obstacle_velocity_z[
        tile_index, local_i, local_j, local_k
    ]

    smoke[tile_index, local_i, local_j, local_k] = 0.0
    fuel[tile_index, local_i, local_j, local_k] = 0.0
    flame[tile_index, local_i, local_j, local_k] = 0.0
