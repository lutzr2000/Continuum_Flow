from numba import cuda

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
