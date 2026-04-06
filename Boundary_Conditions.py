from numba import njit, prange


@njit(parallel=False, cache=True)
def neumann_boundary_condition(field, side):
    """
    applies a zero-gradient boundary condition to one side of a 3D field.

    Args:
        field (3d-array): field to update
        side (str): boundary side identifier
    Returns:
        field (3d-array): field with applied boundary condition
    """
    nx, ny, nz = field.shape

    if side == "x_low":
        for j in prange(ny):
            for k in range(nz):
                field[0, j, k] = field[1, j, k]
    elif side == "x_high":
        for j in prange(ny):
            for k in range(nz):
                field[nx - 1, j, k] = field[nx - 2, j, k]
    elif side == "y_low":
        for i in prange(nx):
            for k in range(nz):
                field[i, 0, k] = field[i, 1, k]
    elif side == "y_high":
        for i in prange(nx):
            for k in range(nz):
                field[i, ny - 1, k] = field[i, ny - 2, k]
    elif side == "z_low":
        for i in prange(nx):
            for j in range(ny):
                field[i, j, 0] = field[i, j, 1]
    elif side == "z_high":
        for i in prange(nx):
            for j in range(ny):
                field[i, j, nz - 1] = field[i, j, nz - 2]

    return field


@njit(parallel=False, cache=True)
def dirichlet_boundary_condition(field, side, value):
    """
    applies a fixed-value boundary condition to one side of a 3D field.

    Args:
        field (3d-array): field to update
        side (str): boundary side identifier
        value (float): prescribed boundary value
    Returns:
        field (3d-array): field with applied boundary condition
    """
    nx, ny, nz = field.shape

    if side == "x_low":
        for j in prange(ny):
            for k in range(nz):
                field[0, j, k] = value
    elif side == "x_high":
        for j in prange(ny):
            for k in range(nz):
                field[nx - 1, j, k] = value
    elif side == "y_low":
        for i in prange(nx):
            for k in range(nz):
                field[i, 0, k] = value
    elif side == "y_high":
        for i in prange(nx):
            for k in range(nz):
                field[i, ny - 1, k] = value
    elif side == "z_low":
        for i in prange(nx):
            for j in range(ny):
                field[i, j, 0] = value
    elif side == "z_high":
        for i in prange(nx):
            for j in range(ny):
                field[i, j, nz - 1] = value

    return field


@njit(parallel=False, cache=True)
def inflow_BC(u, v, w, p, T, side, u_inflow, v_inflow, w_inflow, T_inflow=None):
    """
    applies inflow boundary conditions to a given side of the domain.

    Args:
        u (3d-array): x-velocity field
        v (3d-array): y-velocity field
        w (3d-array): z-velocity field
        p (3d-array): pressure field
        T (3d-array): temperature field
        side (str): boundary side identifier
        u_inflow (float): inflow velocity in x-direction
        v_inflow (float): inflow velocity in y-direction
        w_inflow (float): inflow velocity in z-direction
        T_inflow (float, optional): prescribed inflow temperature
    Returns:
        u (3d-array): updated x-velocity field
        v (3d-array): updated y-velocity field
        w (3d-array): updated z-velocity field
        p (3d-array): updated pressure field
        T (3d-array): updated temperature field
    """
    u = dirichlet_boundary_condition(u, side, u_inflow)
    v = dirichlet_boundary_condition(v, side, v_inflow)
    w = dirichlet_boundary_condition(w, side, w_inflow)
    p = neumann_boundary_condition(p, side)

    if T_inflow is None:
        T = neumann_boundary_condition(T, side)
    else:
        T = dirichlet_boundary_condition(T, side, T_inflow)

    return u, v, w, p, T


@njit(parallel=False, cache=True)
def outflow_BC(u, v, w, p, T, side):
    """
    applies outflow boundary conditions to a given side of the domain.

    Args:
        u (3d-array): x-velocity field
        v (3d-array): y-velocity field
        w (3d-array): z-velocity field
        p (3d-array): pressure field
        T (3d-array): temperature field
        side (str): boundary side identifier
    Returns:
        u (3d-array): updated x-velocity field
        v (3d-array): updated y-velocity field
        w (3d-array): updated z-velocity field
        p (3d-array): updated pressure field
        T (3d-array): updated temperature field
    """
    u = neumann_boundary_condition(u, side)
    v = neumann_boundary_condition(v, side)
    w = neumann_boundary_condition(w, side)
    p = neumann_boundary_condition(p, side)
    T = neumann_boundary_condition(T, side)

    return u, v, w, p, T


@njit(parallel=False, cache=True)
def slip_wall_BC(u, v, w, p, T, side, T_Wall=None):
    """
    applies slip wall boundary conditions to a given side of the 3D domain.

    Args:
        u (3d-array): x-velocity field
        v (3d-array): y-velocity field
        w (3d-array): z-velocity field
        p (3d-array): pressure field
        T (3d-array): temperature field
        side (str): boundary side identifier
        T_Wall (float, optional): prescribed wall temperature
    Returns:
        u (3d-array): updated x-velocity field
        v (3d-array): updated y-velocity field
        w (3d-array): updated z-velocity field
        p (3d-array): updated pressure field
        T (3d-array): updated temperature field
    """
    if side == "x_low" or side == "x_high":
        u = dirichlet_boundary_condition(u, side, 0.0)
        v = neumann_boundary_condition(v, side)
        w = neumann_boundary_condition(w, side)
    elif side == "y_low" or side == "y_high":
        u = neumann_boundary_condition(u, side)
        v = dirichlet_boundary_condition(v, side, 0.0)
        w = neumann_boundary_condition(w, side)
    else:
        u = neumann_boundary_condition(u, side)
        v = neumann_boundary_condition(v, side)
        w = dirichlet_boundary_condition(w, side, 0.0)

    p = neumann_boundary_condition(p, side)
    if T_Wall is None:
        T = neumann_boundary_condition(T, side)
    else:
        T = dirichlet_boundary_condition(T, side, T_Wall)

    return u, v, w, p, T


@njit(parallel=False, cache=True)
def no_slip_wall_BC(u, v, w, p, T, side, T_Wall=None):
    """
    applies no-slip wall boundary conditions to a given side of the 3D domain.

    Args:
        u (3d-array): x-velocity field
        v (3d-array): y-velocity field
        w (3d-array): z-velocity field
        p (3d-array): pressure field
        T (3d-array): temperature field
        side (str): boundary side identifier
        T_Wall (float, optional): prescribed wall temperature
    Returns:
        u (3d-array): updated x-velocity field
        v (3d-array): updated y-velocity field
        w (3d-array): updated z-velocity field
        p (3d-array): updated pressure field
        T (3d-array): updated temperature field
    """
    u = dirichlet_boundary_condition(u, side, 0.0)
    v = dirichlet_boundary_condition(v, side, 0.0)
    w = dirichlet_boundary_condition(w, side, 0.0)
    p = neumann_boundary_condition(p, side)

    if T_Wall is None:
        T = neumann_boundary_condition(T, side)
    else:
        T = dirichlet_boundary_condition(T, side, T_Wall)

    return u, v, w, p, T
