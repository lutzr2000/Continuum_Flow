from numba import njit, prange

@njit
def neumann_boundary_condition(field, side):
    """
    Applies a neumann boundary condition to a given field and side

    Args:
        field (2d-array): field to apply BC for
        side (string): keyword for side: "y_low", "y_high", "x_low", "x_high" 

    Returns:
        field (2d-array): field with BC applied
    """
    ny, nx = field.shape

    if side == "y_low":
        for j in range(nx):
            field[0, j] = field[1, j]
    elif side == "y_high":
        for j in range(nx):
            field[-1, j] = field[-2, j]
    elif side == "x_low":
        for i in range(ny):
            field[i, 0] = field[i, 1]
    elif side == "x_high":
        for i in range(ny):
            field[i, -1] = field[i, -2]

    return field

@njit
def dirichlet_boundary_condition(field, side, value):
    """
    Applies a diriclet boundary condition to a given field and side

    Args:
        field (2d-array): field to apply BC for
        side (string): keyword for side: "y_low", "y_high", "x_low", "x_high" 
        value (float): value of the BC

    Returns:
        field (2d-array): field with BC applied
    """
    nx, ny = field.shape
    if side == "y_low":
        for j in range(ny):
            field[0, j] = value
    elif side == "y_high":
        for j in range(ny):
            field[nx-1, j] = value
    elif side == "x_low":
        for i in range(nx):
            field[i, 0] = value
    elif side == "x_high":
        for i in range(nx):
            field[i, ny-1] = value
    return field

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

@njit(parallel=True)
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
def pressure_poisson_boundary_conditions(p, obstacle_i, obstacle_j):
    """
    Applies the pressure BCs used inside the Poisson solve.

    Obstacle cells are passed as index arrays so sparse obstacles do not force
    a full-domain mask scan on every Jacobi iteration.
    """
    nx, ny = p.shape

    for j in range(ny):
        p[0, j] = p[1, j]
        p[nx - 1, j] = p[nx - 2, j]

    for i in range(nx):
        p[i, 0] = p[i, 1]
        p[i, ny - 1] = p[i, ny - 2]

    for k in range(obstacle_i.shape[0]):
        p[obstacle_i[k], obstacle_j[k]] = 0.0

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

@njit
def inflow_BC(u,v,p,T,side,u_inflow,v_inflow,T_inflow=None):
    """
    Applies inflow boundary conditions to a given side.

    Args:
        u (2d-array): u-velocity field
        v (2d-array): v-velocity field
        p (2d-array): pressure field
        T (2d-array): temperature field
        side (string): which side to aplly to: "y_low", "y_high", "x_low", "x_high" 
        u_inflow (float): inflow u-velocity
        v_inflow (float): inflow v-velocity
        T_inflow (float,optional): inflow temperature

    Returns:
        u (2d-array): u-velocity field
        v (2d-array): v-velocity field
        p (2d-array): pressure field
        T (2d-array): temperature field
    """

    u = dirichlet_boundary_condition(u, side, u_inflow)
    v = dirichlet_boundary_condition(v, side, v_inflow) 
    p = neumann_boundary_condition(p, side) 

    if T_inflow is None:
        T = neumann_boundary_condition(T, side) 
    else:
        T = dirichlet_boundary_condition(T, side, T_inflow) 

    return u,v,p,T

@njit
def outflow_BC(u,v,p,T,side):
    """
    Applies outflow boundary conditions to a given side.

    Args:
        u (2d-array): u-velocity field
        v (2d-array): v-velocity field
        p (2d-array): pressure field
        T (2d-array): temperature field
        side (string): which side to aplly to: "y_low", "y_high", "x_low", "x_high" 

    Returns:
        u (2d-array): u-velocity field
        v (2d-array): v-velocity field
        p (2d-array): pressure field
        T (2d-array): temperature field
    """
    u = neumann_boundary_condition(u, side)
    v = neumann_boundary_condition(v, side)
    p = neumann_boundary_condition(p, side)
    T = neumann_boundary_condition(T, side)   

    return u,v,p,T

@njit
def slip_wall_BC(u,v,p,T,side,T_Wall=None):
    """
    Applies slip wall boundary conditions to a given side.

    Args:
        u (2d-array): u-velocity field
        v (2d-array): v-velocity field
        p (2d-array): pressure field
        T (2d-array): temperature field
        side (string): which side to aplly to: "y_low", "y_high", "x_low", "x_high" 
        T_Wall (float,optinal): wall temperatue, if none => neumann BC

    Returns:
        u (2d-array): u-velocity field
        v (2d-array): v-velocity field
        p (2d-array): pressure field
        T (2d-array): temperature field
    """

    u = neumann_boundary_condition(u, side)
    v = dirichlet_boundary_condition(v, side, 0) 
    p = neumann_boundary_condition(p, side)
    if T_Wall is None:
        T = neumann_boundary_condition(T, side) 
    else:
        T = dirichlet_boundary_condition(T, side, T_Wall) 

    return u,v,p,T

@njit
def no_slip_wall_BC(u,v,p,T,side,T_Wall=None):
    """
    Applies no slip wall boundary conditions to a given side.

    Args:
        u (2d-array): u-velocity field
        v (2d-array): v-velocity field
        p (2d-array): pressure field
        T (2d-array): temperature field
        side (string): which side to aplly to: "y_low", "y_high", "x_low", "x_high" 
        T_Wall (float,optinal): wall temperatue, if none => neumann BC

    Returns:
        u (2d-array): u-velocity field
        v (2d-array): v-velocity field
        p (2d-array): pressure field
        T (2d-array): temperature field
    """

    u = dirichlet_boundary_condition(u, side, 0)
    v = dirichlet_boundary_condition(v, side, 0) 
    p = neumann_boundary_condition(p, side)
    if T_Wall is None:
        T = neumann_boundary_condition(T, side) 
    else:
        T = dirichlet_boundary_condition(T, side, T_Wall) 

    return u,v,p,T
