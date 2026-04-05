from numba import njit, prange


@njit(parallel=False, cache=True)
def obstacle_boundary_conditions_velocity(u, v, w, mask):
    """
    applies no-slip velocity conditions inside a 3D obstacle mask.

    Args:
        u (3d-array): x-velocity field
        v (3d-array): y-velocity field
        w (3d-array): z-velocity field
        mask (3d-array): boolean obstacle mask
    Returns:
        u (3d-array): updated x-velocity field
        v (3d-array): updated y-velocity field
        w (3d-array): updated z-velocity field
    """
    nx, ny, nz = mask.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if mask[i, j, k]:
                    u[i, j, k] = 0.0
                    v[i, j, k] = 0.0
                    w[i, j, k] = 0.0
    return u, v, w


@njit(parallel=False, cache=True)
def obstacle_boundary_conditions_pressure(p, mask):
    """
    applies zero pressure inside a 3D obstacle mask.

    Args:
        p (3d-array): pressure field
        mask (3d-array): boolean obstacle mask
    Returns:
        p (3d-array): updated pressure field
    """
    for i in prange(p.shape[0]):
        for j in range(p.shape[1]):
            for k in range(p.shape[2]):
                if mask[i, j, k]:
                    p[i, j, k] = 0.0

    return p


@njit(parallel=False, cache=True)
def obstacle_boundary_conditions_scalar(phi, mask, value):
    """
    applies a fixed scalar value inside a 3D obstacle mask.

    Args:
        phi (3d-array): scalar field
        mask (3d-array): boolean obstacle mask
        value (float): scalar value inside the obstacle
    Returns:
        phi (3d-array): updated scalar field
    """
    for i in range(phi.shape[0]):
        for j in range(phi.shape[1]):
            for k in range(phi.shape[2]):
                if mask[i, j, k]:
                    phi[i, j, k] = value

    return phi
