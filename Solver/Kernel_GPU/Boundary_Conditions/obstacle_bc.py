from typing import Any

import Solver.Kernel_GPU.sparse_managment as sparse_managment
from numba import cuda


@cuda.jit(cache=True)
def obstacle_bc(
    u: Any,
    v: Any,
    w: Any,
    smoke: Any,
    fuel: Any,
    flame: Any,
    mask: Any,
    obstacle_velocity_x: Any,
    obstacle_velocity_y: Any,
    obstacle_velocity_z: Any,
    tile_map: Any,
) -> None:
    """
    Obstacle bc.
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
