from numba import njit


@njit
def neumann_boundary_condition(field, side):
    """
    Applies a zero-gradient boundary condition to one side of a 3D field.
    """
    nx, ny, nz = field.shape

    if side == "x_low":
        for j in range(ny):
            for k in range(nz):
                field[0, j, k] = field[1, j, k]
    elif side == "x_high":
        for j in range(ny):
            for k in range(nz):
                field[nx - 1, j, k] = field[nx - 2, j, k]
    elif side == "y_low":
        for i in range(nx):
            for k in range(nz):
                field[i, 0, k] = field[i, 1, k]
    elif side == "y_high":
        for i in range(nx):
            for k in range(nz):
                field[i, ny - 1, k] = field[i, ny - 2, k]
    elif side == "z_low":
        for i in range(nx):
            for j in range(ny):
                field[i, j, 0] = field[i, j, 1]
    elif side == "z_high":
        for i in range(nx):
            for j in range(ny):
                field[i, j, nz - 1] = field[i, j, nz - 2]

    return field


@njit
def dirichlet_boundary_condition(field, side, value):
    """
    Applies a fixed-value boundary condition to one side of a 3D field.
    """
    nx, ny, nz = field.shape

    if side == "x_low":
        for j in range(ny):
            for k in range(nz):
                field[0, j, k] = value
    elif side == "x_high":
        for j in range(ny):
            for k in range(nz):
                field[nx - 1, j, k] = value
    elif side == "y_low":
        for i in range(nx):
            for k in range(nz):
                field[i, 0, k] = value
    elif side == "y_high":
        for i in range(nx):
            for k in range(nz):
                field[i, ny - 1, k] = value
    elif side == "z_low":
        for i in range(nx):
            for j in range(ny):
                field[i, j, 0] = value
    elif side == "z_high":
        for i in range(nx):
            for j in range(ny):
                field[i, j, nz - 1] = value

    return field


@njit
def inflow_BC(u, v, w, p, T, side, u_inflow, v_inflow, w_inflow, T_inflow=None):
    """
    Applies inflow boundary conditions to a given side.
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


@njit
def outflow_BC(u, v, w, p, T, side):
    """
    Applies outflow boundary conditions to a given side.
    """
    u = neumann_boundary_condition(u, side)
    v = neumann_boundary_condition(v, side)
    w = neumann_boundary_condition(w, side)
    p = neumann_boundary_condition(p, side)
    T = neumann_boundary_condition(T, side)

    return u, v, w, p, T


@njit
def slip_wall_BC(u, v, w, p, T, side, T_Wall=None):
    """
    Applies slip wall boundary conditions to a given side of a 3D domain.
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


@njit
def no_slip_wall_BC(u, v, w, p, T, side, T_Wall=None):
    """
    Applies no-slip wall boundary conditions to a given side of a 3D domain.
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
