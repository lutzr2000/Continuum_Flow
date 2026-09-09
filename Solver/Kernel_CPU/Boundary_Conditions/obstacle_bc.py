from typing import Any

from numba import njit, prange

import Solver.Kernel_CPU.kernel_config as kernel_config
import Solver.Kernel_CPU.sparse_managment as sparse_managment


@njit(cache=True, parallel=True)
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
    Impose obstacle velocity and clear combustible scalars inside solid cells.

    Active cells covered by the obstacle mask receive the voxelized obstacle
    velocity, producing the moving-wall condition. Smoke, fuel, and flame are
    reset so scalar material cannot remain inside solid geometry.
    """

    total_tiles = tile_map.shape[0] * tile_map.shape[1] * tile_map.shape[2]

    for tile_flat in prange(total_tiles):
        for local_i in range(kernel_config.TILE_SIZE):
            for local_j in range(kernel_config.TILE_SIZE):
                for local_k in range(kernel_config.TILE_SIZE):
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
                    ) = sparse_managment.tile_to_index(
                        tile_flat,
                        local_i,
                        local_j,
                        local_k,
                        tile_map.shape[0],
                        tile_map.shape[1],
                        tile_map.shape[2],
                    )

                    tile_index = tile_map[tile_i, tile_j, tile_k]

                    if tile_index == -1:
                        continue

                    if not mask[tile_index, local_i, local_j, local_k]:
                        continue

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
