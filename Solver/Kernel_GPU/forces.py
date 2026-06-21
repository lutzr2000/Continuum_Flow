from numba import cuda

from Solver.Kernel_GPU.scalar_update import _active_tile_cell_indices


@cuda.jit(cache=True)
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
    Update body-force fields from base fields and precomputed turbulence fields.

    The expensive smooth turbulence fields are precomputed on the host. Per
    timestep this kernel only applies one host-computed signed amplitude per
    turbulence node.
    """
    tile_i, tile_j, tile_k, i, j, k, nx, ny, nz = _active_tile_cell_indices(Fx.shape)

    if (
        tile_i >= active_tile_mask.shape[0]
        or tile_j >= active_tile_mask.shape[1]
        or tile_k >= active_tile_mask.shape[2]
    ):
        return
    if active_tile_mask[tile_i, tile_j, tile_k] == 0:
        return
    if i >= nx or j >= ny or k >= nz:
        return

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


@cuda.jit(device=True, inline=True, cache=True)
def buoyancy_approximation(
    T,
    i,
    j,
    k,
    buoyancy_factor,
    t_reference,
):
    """
    computes the buoyancy force in z-direction with the Boussinesq approximation on the GPU.

    Each thread updates one interior cell of the z-direction body-force field.
    The force is derived from the local temperature difference to the reference
    temperature and scaled by gravity and the configured buoyancy factor.

    """
    g = 9.81
    return g * buoyancy_factor * (T[i, j, k] - t_reference)
