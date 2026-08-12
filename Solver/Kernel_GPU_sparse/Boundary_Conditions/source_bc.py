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
        nx,
        ny,
        nz,
    ) = sparse_managment.tile_to_index(source_mask.shape)

    if i >= nx or j >= ny or k >= nz:
        return

    if i == 0 or i == nx - 1 or j == 0 or j == ny - 1 or k == 0 or k == nz - 1:
        return

    if not source_mask[i, j, k]:
        return

    tile_index = tile_map[tile_i, tile_j, tile_k]
    if tile_index == -1:
        return

    scalar_multiplier = 1.0 + noise_amplitude * source_noise[i, j, k]
    if scalar_multiplier < 0.0:
        scalar_multiplier = 0.0
    elif scalar_multiplier > 2.0:
        scalar_multiplier = 2.0

    if velocity_x_value != 0:
        u[i, j, k] = velocity_x_value
    if velocity_y_value != 0:
        v[i, j, k] = velocity_y_value
    if velocity_z_value != 0:
        w[i, j, k] = velocity_z_value

    temperature = temperature_value * scalar_multiplier
    if temperature < 0.0:
        temperature = 0.0
    T[tile_index, local_i, local_j, local_k] = temperature

    if smoke_value != 0:
        smoke_updated = (
            smoke[tile_index, local_i, local_j, local_k]
            + dt * 10.0 * smoke_value * scalar_multiplier
        )
        if smoke_updated < 0.0:
            smoke_updated = 0.0
        elif smoke_updated > 100.0:
            smoke_updated = 100.0
        smoke[tile_index, local_i, local_j, local_k] = smoke_updated

    if fuel_value != 0:
        fuel_updated = (
            fuel[tile_index, local_i, local_j, local_k]
            + dt * 10.0 * fuel_value * scalar_multiplier
        )
        if fuel_updated < 0.0:
            fuel_updated = 0.0
        elif fuel_updated > 100.0:
            fuel_updated = 100.0
        fuel[tile_index, local_i, local_j, local_k] = fuel_updated