from numba import njit, prange


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
):
    """
    Update body-force fields from base, animated and turbulence contributions.
    """
    nx, ny, nz = Fx.shape
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
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
def buoyancy_approximation(T, Fz, buoyancy_factor, t_reference):
    """
    Accumulate Boussinesq buoyancy into the z-force field on the CPU.
    """
    nx, ny, nz = T.shape
    g = 9.81
    for i in prange(1, nx - 1):
        for j in range(1, ny - 1):
            for k in range(1, nz - 1):
                Fz[i, j, k] += g * buoyancy_factor * (T[i, j, k] - t_reference)
