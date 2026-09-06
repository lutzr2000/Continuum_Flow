from numba import cuda

import Solver.Kernel_GPU.sparse_managment as sparse_managment


@cuda.jit(cache=True)
def source_bc(
    u,
    v,
    w,
    T,
    smoke,
    fuel,
    tile_map,
    source_mask,
    temperature_value,
    smoke_value,
    fuel_value,
    velocity_x_value,
    velocity_y_value,
    velocity_z_value,
    dt,
):
    """
    Add source velocity/temperature and inject smoke/fuel rates on the GPU.
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

    if not source_mask[tile_index, local_i, local_j, local_k]:
        return

    u[tile_index, local_i, local_j, local_k] += velocity_x_value
    v[tile_index, local_i, local_j, local_k] += velocity_y_value
    w[tile_index, local_i, local_j, local_k] += velocity_z_value

    T[tile_index, local_i, local_j, local_k] = max(
        temperature_value,
        0.0,
    )

    smoke[tile_index, local_i, local_j, local_k] = min(
        max(
            smoke[tile_index, local_i, local_j, local_k] + smoke_value * dt,
            0.0,
        ),
        100.0,
    )

    fuel[tile_index, local_i, local_j, local_k] = min(
        max(
            fuel[tile_index, local_i, local_j, local_k] + fuel_value * dt,
            0.0,
        ),
        100.0,
    )
