from numba import njit, prange

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
    if side == "bottom":  
        field[0, :] = field[1, :]
    elif side == "top":  
        field[-1, :] = field[-2, :]
    elif side == "left":  
        field[:, 0] = field[:, 1]
    elif side == "right":  
        field[:, -1] = field[:, -2]
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
    if side == "bottom": 
        field[0, :] = value
    elif side == "top":  
        field[-1, :] = value
    elif side == "left":  
        field[:, 0] = value
    elif side == "right":  
        field[:, -1] = value
    return field

def obstacle_boundary_conditions_velocity(u,v,mask):
    """
    Applies no slip conditions on a given mask

    Args:
        u (2d-array): u-velocity field
        v (2d-array): v-velocity field
        mask (2d-array): mask for obstacle

    Returns:
        u (2d-array): u-velocity field
        v (2d-array): v-velocity field
    """
    u[mask] = 0.0
    v[mask] = 0.0

    return u,v

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
    # the loops are necessary because of numba, maybe remove later
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            if mask[i, j]:
                p[i, j] = 0.0

    return p