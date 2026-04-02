from numba import njit, prange


@njit
def obstacle_boundary_conditions_velocity(u, v, w, mask):
    """
    Applies no-slip conditions inside a 3D obstacle mask.
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


@njit
def obstacle_boundary_conditions_pressure(p, mask):
    """
    Applies zero pressure inside a 3D obstacle mask.
    """
    for i in prange(p.shape[0]):
        for j in range(p.shape[1]):
            for k in range(p.shape[2]):
                if mask[i, j, k]:
                    p[i, j, k] = 0.0

    return p


@njit
def obstacle_boundary_conditions_scalar(phi, mask, value):
    """
    Applies a fixed scalar value inside a 3D obstacle mask.
    """
    for i in range(phi.shape[0]):
        for j in range(phi.shape[1]):
            for k in range(phi.shape[2]):
                if mask[i, j, k]:
                    phi[i, j, k] = value

    return phi
