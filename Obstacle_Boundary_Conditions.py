from numba import njit, prange

@njit
def obstacle_boundary_conditions_velocity(u,v,mask):
    """
    Applies no slip conditions to a given mask

    Args:
        u (2d-array): u-velocity field
        v (2d-array): v-velocity field
        mask (2d-array): mask for obstacle

    Returns:
        u (2d-array): u-velocity field
        v (2d-array): v-velocity field
    """
    nx, ny = mask.shape
    for i in range(nx):
        for j in range(ny):
            if mask[i, j]:
                u[i, j] = 0.0
                v[i, j] = 0.0
    return u, v

@njit
def obstacle_boundary_conditions_pressure(p,mask):
    """
    Applies zero pressure to a given mask

    Args:
        p (2d-array): pressure field
        mask (2d-array): mask for obstacle

    Returns:
        p (2d-array): pressure field
    """
    for i in prange(p.shape[0]):
        for j in range(p.shape[1]):
            if mask[i, j]:
                p[i, j] = 0.0

    return p

@njit
def obstacle_boundary_conditions_scalar(phi,mask,value):
    """
    Applies a value to the scalar field phi at an obstacle.

    Args:
        phi (2d-array): scalar field
        mask (2d-array): mask for obstacle
        value (2d_array): obstacle scalar value

    Returns:
        phi (2d-array): scalar field
    """
    for i in range(phi.shape[0]):
        for j in range(phi.shape[1]):
            if mask[i, j]:
                phi[i, j] = value

    return phi