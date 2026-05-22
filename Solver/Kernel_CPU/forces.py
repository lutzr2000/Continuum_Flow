from numba import njit, prange


@njit(cache=True, parallel=True, fastmath=True)
def update_force_fields(
    Fx_base,
    Fy_base,
    Fz_base,
    turbulence_Fx_a,
    turbulence_Fy_a,
    turbulence_Fz_a,
    turbulence_Fx_b,
    turbulence_Fy_b,
    turbulence_Fz_b,
    turbulence_amplitudes,
    turbulence_mix_factors,
    turbulence_count,
    animated_force_x,
    animated_force_y,
    animated_force_z,
    Fx,
    Fy,
    Fz,
):
    """Update body-force fields from base, animated and turbulence contributions."""
    nx, ny, nz = Fx.shape
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                fx = Fx_base[i, j, k] + animated_force_x
                fy = Fy_base[i, j, k] + animated_force_y
                fz = Fz_base[i, j, k] + animated_force_z

                for turbulence_index in range(turbulence_count):
                    amplitude = turbulence_amplitudes[turbulence_index]
                    mix_factor = turbulence_mix_factors[turbulence_index]
                    inverse_mix_factor = 1.0 - mix_factor
                    fx += (
                        amplitude * (
                            mix_factor * turbulence_Fx_a[turbulence_index, i, j, k] +
                            inverse_mix_factor * turbulence_Fx_b[turbulence_index, i, j, k]
                        )
                    )
                    fy += (
                        amplitude * (
                            mix_factor * turbulence_Fy_a[turbulence_index, i, j, k] +
                            inverse_mix_factor * turbulence_Fy_b[turbulence_index, i, j, k]
                        )
                    )
                    fz += (
                        amplitude * (
                            mix_factor * turbulence_Fz_a[turbulence_index, i, j, k] +
                            inverse_mix_factor * turbulence_Fz_b[turbulence_index, i, j, k]
                        )
                    )

                Fx[i, j, k] = fx
                Fy[i, j, k] = fy
                Fz[i, j, k] = fz


@njit(cache=True, parallel=True, fastmath=True)
def buoyancy_approximation(T, Fz, buoyancy_factor, t_reference):
    """Accumulate Boussinesq buoyancy into the z-force field on the CPU."""
    nx, ny, nz = T.shape
    g = 9.81
    for i in prange(1, nx - 1):
        for j in range(1, ny - 1):
            for k in range(1, nz - 1):
                Fz[i, j, k] += g * buoyancy_factor * (T[i, j, k] - t_reference)
