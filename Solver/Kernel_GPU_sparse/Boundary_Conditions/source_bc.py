from numba import cuda

import Solver.Kernel_GPU_sparse.sparse_managment as sparse_managment


@cuda.jit(cache=True)
def source_bc_kernel(
    u,
    v,
    w,
    T,
    smoke,
    fuel,
    tile_map,
    source_mask,
    source_noise,
    temperature_value,
    smoke_value,
    fuel_value,
    velocity_x_value,
    velocity_y_value,
    velocity_z_value,
    noise_amplitude,
    dt,
):
    """
    Apply source velocity/temperature and inject smoke/fuel rates on the GPU.
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

    if not source_mask[i, j, k]:
        return

    tile_index = tile_map[tile_i, tile_j, tile_k]
    if tile_index == -1:
        return

    scalar_multiplier = min(
        max(1.0 + noise_amplitude * source_noise[i, j, k], 0.0),
        2.0,
    )

    if velocity_x_value != 0:
        u[tile_index, local_i, local_j, local_k] = velocity_x_value
    if velocity_y_value != 0:
        v[tile_index, local_i, local_j, local_k] = velocity_y_value
    if velocity_z_value != 0:
        w[tile_index, local_i, local_j, local_k] = velocity_z_value

    temperature = temperature_value * scalar_multiplier
    if temperature < 0.0:
        temperature = 0.0
    T[tile_index, local_i, local_j, local_k] = temperature

    smoke[tile_index, local_i, local_j, local_k] = min(
        smoke[tile_index, local_i, local_j, local_k] + smoke_value * dt * scalar_multiplier,
        100.0,
    )
    fuel[tile_index, local_i, local_j, local_k] = min(
        fuel[tile_index, local_i, local_j, local_k] + fuel_value * dt * scalar_multiplier,
        100.0,
    )
