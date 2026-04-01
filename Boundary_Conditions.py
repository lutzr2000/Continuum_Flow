from numba import njit

@njit
def neumann_boundary_condition(field, side):
    """
    Applies a neumann boundary condition to a given field and side

    Args:
        field (2d-array): field to apply BC for
        side (string): keyword for side: "bottom", "top", "left", "right" 

    Returns:
        field (2d-array): field with BC applied
    """
    ny, nx = field.shape

    if side == "bottom":
        for j in range(nx):
            field[0, j] = field[1, j]
    elif side == "top":
        for j in range(nx):
            field[-1, j] = field[-2, j]
    elif side == "left":
        for i in range(ny):
            field[i, 0] = field[i, 1]
    elif side == "right":
        for i in range(ny):
            field[i, -1] = field[i, -2]

    return field

@njit
def dirichlet_boundary_condition(field, side, value):
    """
    Applies a diriclet boundary condition to a given field and side

    Args:
        field (2d-array): field to apply BC for
        side (string): keyword for side: "bottom", "top", "left", "right" 
        value (float): value of the BC

    Returns:
        field (2d-array): field with BC applied
    """
    nx, ny = field.shape
    if side == "bottom":
        for j in range(ny):
            field[0, j] = value
    elif side == "top":
        for j in range(ny):
            field[nx-1, j] = value
    elif side == "left":
        for i in range(nx):
            field[i, 0] = value
    elif side == "right":
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
    for i in range(p.shape[0]):
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

@njit
def apply_velocity_BC(u,v):
    """
    Applies a set of velocity boundary conditions to all sides

    Args:
        u (2d-array): u-velocity field
        v (2d-array): v-velocity field

    Returns:
        u (2d-array): u-velocity field
        v (2d-array): v-velocity field
    """
    v = dirichlet_boundary_condition(v, "bottom", 0.0) 
    u = dirichlet_boundary_condition(u, "bottom", 0.0) 

    v = dirichlet_boundary_condition(v, "top", 0.0)  
    u = dirichlet_boundary_condition(u, "top", 0.0)  

    u = dirichlet_boundary_condition(u, "left", 5.0)
    v = dirichlet_boundary_condition(v, "left", 0.0)

    u = neumann_boundary_condition(u, "right")
    v = neumann_boundary_condition(v, "right")

    return u,v

@njit
def apply_pressure_BC(p):
    """
    Applies a set of pressure boundary conditions to all sides

    Args:
        p (2d-array): pressure field

    Returns:
        p (2d-array): pressure field
    """
    p = neumann_boundary_condition(p, "bottom") 
    p = neumann_boundary_condition(p, "top") 
    p = neumann_boundary_condition(p, "left")  
    p = neumann_boundary_condition(p, "right") 
    return p

@njit
def apply_temperature_BC(T):
    """
    Applies a set of temperature boundary conditions to all sides

    Args:
        T (2d-array): temperature field

    Returns:
        T (2d-array): temperature field
    """
    T = neumann_boundary_condition(T, "bottom") 
    T = neumann_boundary_condition(T, "top") 
    T = dirichlet_boundary_condition(T, "left", 300)  
    T = neumann_boundary_condition(T, "right") 
    return T