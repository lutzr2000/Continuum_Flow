from numba import cuda


@cuda.jit(cache=True)
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
    """
    Update body-force fields from base fields and precomputed turbulence fields.

    The expensive smooth turbulence fields are precomputed on the host. Per
    timestep this kernel only blends the two stored fields with one
    host-computed mix factor per turbulence node.
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = Fx.shape

    if i >= nx or j >= ny or k >= nz:
        return

    fx = Fx_base[i, j, k] + animated_force_x
    fy = Fy_base[i, j, k] + animated_force_y
    fz = Fz_base[i, j, k] + animated_force_z

    for turbulence_index in range(turbulence_count):
        amplitude = turbulence_amplitudes[turbulence_index]
        mix_factor = turbulence_mix_factors[turbulence_index]
        inverse_mix_factor = 1.0 - mix_factor
        fx += amplitude * (
            mix_factor * turbulence_Fx_a[turbulence_index, i, j, k]
            + inverse_mix_factor * turbulence_Fx_b[turbulence_index, i, j, k]
        )
        fy += amplitude * (
            mix_factor * turbulence_Fy_a[turbulence_index, i, j, k]
            + inverse_mix_factor * turbulence_Fy_b[turbulence_index, i, j, k]
        )
        fz += amplitude * (
            mix_factor * turbulence_Fz_a[turbulence_index, i, j, k]
            + inverse_mix_factor * turbulence_Fz_b[turbulence_index, i, j, k]
        )

    Fx[i, j, k] = fx
    Fy[i, j, k] = fy
    Fz[i, j, k] = fz


@cuda.jit(cache=True)
def buoyancy_approximation(T, Fz, buoyancy_factor, t_reference):
    """
    computes the buoyancy force in z-direction with the Boussinesq approximation on the GPU.

    Each thread updates one interior cell of the z-direction body-force field.
    The force is derived from the local temperature difference to the reference
    temperature and scaled by gravity and the configured buoyancy factor.

    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = T.shape

    if i < 1 or j < 1 or k < 1 or i >= nx - 1 or j >= ny - 1 or k >= nz - 1:
        return

    g = 9.81
    Fz[i, j, k] += g * buoyancy_factor * (T[i, j, k] - t_reference)
