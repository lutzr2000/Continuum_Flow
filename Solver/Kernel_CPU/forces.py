from numba import njit, prange

import Solver.Kernel_CPU.kernel_config as kernel_config


@njit(cache=True, parallel=True, fastmath=True)
def update_force_fields(
    Fx_base,
    Fy_base,
    Fz_base,
    turbulence_Fx,
    turbulence_Fy,
    turbulence_Fz,
    turbulence_signed_amplitudes,
    turbulence_count,
    animated_force_x,
    animated_force_y,
    animated_force_z,
    Fx,
    Fy,
    Fz,
    active_tile_mask,
):
    """
    Update body-force fields from base, animated and turbulence contributions.
    """
    nx, ny, nz = Fx.shape
    tile_size = kernel_config.ACTIVE_TILE_SIZE
    for i in prange(nx):
        tile_i = i // tile_size
        for j in range(ny):
            tile_j = j // tile_size
            for k in range(nz):
                tile_k = k // tile_size
                if not active_tile_mask[tile_i, tile_j, tile_k]:
                    continue
                fx = Fx_base[i, j, k] + animated_force_x
                fy = Fy_base[i, j, k] + animated_force_y
                fz = Fz_base[i, j, k] + animated_force_z

                for turbulence_index in range(turbulence_count):
                    amplitude = turbulence_signed_amplitudes[turbulence_index]
                    fx += amplitude * turbulence_Fx[turbulence_index, i, j, k]
                    fy += amplitude * turbulence_Fy[turbulence_index, i, j, k]
                    fz += amplitude * turbulence_Fz[turbulence_index, i, j, k]

                Fx[i, j, k] = fx
                Fy[i, j, k] = fy
                Fz[i, j, k] = fz


@njit(cache=True, parallel=True, fastmath=True)
def buoyancy_approximation(T, Fz, buoyancy_factor, t_reference, active_tile_mask):
    """
    Accumulate Boussinesq buoyancy into the z-force field on the CPU.
    """
    nx, ny, nz = T.shape
    g = 9.81
    tile_size = kernel_config.ACTIVE_TILE_SIZE
    for i in prange(1, nx - 1):
        tile_i = i // tile_size
        for j in range(1, ny - 1):
            tile_j = j // tile_size
            for k in range(1, nz - 1):
                tile_k = k // tile_size
                if not active_tile_mask[tile_i, tile_j, tile_k]:
                    continue
                Fz[i, j, k] += g * buoyancy_factor * (T[i, j, k] - t_reference)
