from numba import cuda

THREADS_PER_BLOCK_3D = (8, 8, 8)


@cuda.jit
def obstacle_boundary_conditions_velocity(u, v, w, mask):
    """
    applies no-slip velocity conditions inside a 3D obstacle mask on the GPU.

    Each thread checks one obstacle cell and sets all three velocity components
    to zero when the mask marks that cell as solid.

    Args:
        u (device array): x-velocity field
        v (device array): y-velocity field
        w (device array): z-velocity field
        mask (device array): boolean obstacle mask
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = mask.shape

    if i >= nx or j >= ny or k >= nz:
        return

    if mask[i, j, k]:
        u[i, j, k] = 0.0
        v[i, j, k] = 0.0
        w[i, j, k] = 0.0


@cuda.jit
def obstacle_boundary_conditions_pressure(p, mask):
    """
    applies zero pressure inside a 3D obstacle mask on the GPU.

    Each thread checks one obstacle cell and sets the pressure to zero when the
    mask marks that cell as part of the obstacle region.

    Args:
        p (device array): pressure field
        mask (device array): boolean obstacle mask
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = mask.shape

    if i >= nx or j >= ny or k >= nz:
        return

    if mask[i, j, k]:
        p[i, j, k] = 0.0


@cuda.jit
def obstacle_boundary_conditions_scalars(smoke, fuel, flame, mask):
    """
    applies zero scalar conditions inside a 3D obstacle mask on the GPU.

    Each thread checks one obstacle cell and clears smoke, fuel and flame when
    the mask marks that cell as part of the obstacle region.

    Args:
        smoke (device array): smoke field
        fuel (device array): fuel field
        flame (device array): flame indicator field
        mask (device array): boolean obstacle mask
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = mask.shape

    if i >= nx or j >= ny or k >= nz:
        return

    if mask[i, j, k]:
        smoke[i, j, k] = 0.0
        fuel[i, j, k] = 0.0
        flame[i, j, k] = 0.0


def obstacle_velocity_bc(u, v, w, mask, threadsperblock=None):
    """
    launches the obstacle no-slip velocity kernel on the GPU.

    Args:
        u (device array): x-velocity field
        v (device array): y-velocity field
        w (device array): z-velocity field
        mask (device array): boolean obstacle mask
        threadsperblock (tuple[int, int, int], optional): CUDA block shape for volume kernels
    Returns:
        tuple: updated velocity fields
    """
    if threadsperblock is None:
        threadsperblock = THREADS_PER_BLOCK_3D
    blockspergrid = (
        (mask.shape[0] + threadsperblock[0] - 1) // threadsperblock[0],
        (mask.shape[1] + threadsperblock[1] - 1) // threadsperblock[1],
        (mask.shape[2] + threadsperblock[2] - 1) // threadsperblock[2],
    )
    obstacle_boundary_conditions_velocity[blockspergrid, threadsperblock](u, v, w, mask)
    return u, v, w


def obstacle_pressure_bc(p, mask, threadsperblock=None):
    """
    launches the obstacle pressure boundary kernel on the GPU.

    Args:
        p (device array): pressure field
        mask (device array): boolean obstacle mask
        threadsperblock (tuple[int, int, int], optional): CUDA block shape for volume kernels
    Returns:
        device array: the updated pressure field
    """
    if threadsperblock is None:
        threadsperblock = THREADS_PER_BLOCK_3D
    blockspergrid = (
        (mask.shape[0] + threadsperblock[0] - 1) // threadsperblock[0],
        (mask.shape[1] + threadsperblock[1] - 1) // threadsperblock[1],
        (mask.shape[2] + threadsperblock[2] - 1) // threadsperblock[2],
    )
    obstacle_boundary_conditions_pressure[blockspergrid, threadsperblock](p, mask)
    return p


def obstacle_scalar_bc(smoke, fuel, flame, mask, threadsperblock=None):
    """
    launches the obstacle scalar-clearing kernel on the GPU.

    Args:
        smoke (device array): smoke field
        fuel (device array): fuel field
        flame (device array): flame indicator field
        mask (device array): boolean obstacle mask
        threadsperblock (tuple[int, int, int], optional): CUDA block shape for volume kernels
    Returns:
        tuple: updated scalar fields
    """
    if threadsperblock is None:
        threadsperblock = THREADS_PER_BLOCK_3D
    blockspergrid = (
        (mask.shape[0] + threadsperblock[0] - 1) // threadsperblock[0],
        (mask.shape[1] + threadsperblock[1] - 1) // threadsperblock[1],
        (mask.shape[2] + threadsperblock[2] - 1) // threadsperblock[2],
    )
    obstacle_boundary_conditions_scalars[blockspergrid, threadsperblock](smoke, fuel, flame, mask)
    return smoke, fuel, flame


def apply_all_obstacle_BCs(u, v, w, p, T, smoke, fuel, flame, obstacle_mask, obstacle_solid):
    """
    applies all obstacle boundary conditions to the GPU field state.

    Velocity components are forced to zero inside solid obstacles and pressure
    is set to zero there as well. Smoke, fuel and flame are always cleared
    inside obstacle cells so obstacle regions stay empty.

    Args:
        u (device array): x-velocity field
        v (device array): y-velocity field
        w (device array): z-velocity field
        p (device array): pressure field
        T (device array): temperature field
        smoke (device array): smoke field
        fuel (device array): fuel field
        flame (device array): flame indicator field
        obstacle_mask (device array): boolean obstacle mask
        obstacle_solid (bool): whether obstacle velocity conditions should be applied
    Returns:
        tuple: updated velocity, pressure and scalar fields
    """
    smoke, fuel, flame = obstacle_scalar_bc(smoke, fuel, flame, obstacle_mask)

    if obstacle_solid:
        u, v, w = obstacle_velocity_bc(u, v, w, obstacle_mask)
        p = obstacle_pressure_bc(p, obstacle_mask)

    return u, v, w, p, T, smoke, fuel, flame
