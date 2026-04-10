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
def obstacle_boundary_conditions_scalar(phi, mask, value):
    """
    applies a fixed scalar value inside a 3D obstacle mask on the GPU.

    Each thread checks one obstacle cell and writes the prescribed scalar value
    when the cell belongs to the obstacle region.

    Args:
        phi (device array): scalar field that will be clamped inside the obstacle
        mask (device array): boolean obstacle mask
        value (float): scalar value prescribed inside the obstacle
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = mask.shape

    if i >= nx or j >= ny or k >= nz:
        return

    if mask[i, j, k]:
        phi[i, j, k] = value


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


def obstacle_scalar_bc(phi, mask, value, threadsperblock=None):
    """
    launches the obstacle scalar boundary kernel on the GPU.

    Args:
        phi (device array): scalar field that will be clamped inside the obstacle
        mask (device array): boolean obstacle mask
        value (float): scalar value prescribed inside the obstacle
        threadsperblock (tuple[int, int, int], optional): CUDA block shape for volume kernels
    Returns:
        device array: the updated scalar field
    """
    if threadsperblock is None:
        threadsperblock = THREADS_PER_BLOCK_3D
    blockspergrid = (
        (mask.shape[0] + threadsperblock[0] - 1) // threadsperblock[0],
        (mask.shape[1] + threadsperblock[1] - 1) // threadsperblock[1],
        (mask.shape[2] + threadsperblock[2] - 1) // threadsperblock[2],
    )
    obstacle_boundary_conditions_scalar[blockspergrid, threadsperblock](phi, mask, value)
    return phi


def apply_all_obstacle_BCs(u, v, w, p, T, smoke, fuel, obstacle_mask, obstacle_solid,
                           obstacle_initial_temperature, obstacle_initial_smoke,
                           obstacle_initial_fuel):
    """
    applies all obstacle boundary conditions to the GPU field state.

    Velocity components are forced to zero inside solid obstacles, pressure is
    set to zero there as well and scalar fields are clamped to their configured
    obstacle values.

    Args:
        u (device array): x-velocity field
        v (device array): y-velocity field
        w (device array): z-velocity field
        p (device array): pressure field
        T (device array): temperature field
        smoke (device array): smoke field
        fuel (device array): fuel field
        obstacle_mask (device array): boolean obstacle mask
        obstacle_solid (bool): whether obstacle velocity conditions should be applied
        obstacle_initial_temperature (float): temperature prescribed inside the obstacle
        obstacle_initial_smoke (float): smoke value prescribed inside the obstacle
        obstacle_initial_fuel (float): fuel value prescribed inside the obstacle
    Returns:
        tuple: updated velocity, pressure and scalar fields
    """
    if obstacle_solid:
        u, v, w = obstacle_velocity_bc(u, v, w, obstacle_mask)
        p = obstacle_pressure_bc(p, obstacle_mask)

    T = obstacle_scalar_bc(T, obstacle_mask, obstacle_initial_temperature)
    smoke = obstacle_scalar_bc(smoke, obstacle_mask, obstacle_initial_smoke)
    fuel = obstacle_scalar_bc(fuel, obstacle_mask, obstacle_initial_fuel)

    return u, v, w, p, T, smoke, fuel
