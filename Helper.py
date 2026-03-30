import numpy as np
from numba import njit

@njit
def compute_CFL(u,v,dt,delta):
    """
    computes the CFL condition

    Args:
        u (2d-array): u-velocity field
        v (2d-array): v-velocity field
        dt (float): time step size
        delta (float): space resolution

    Returns:
        b (2d-array): right hand side of pressure poisson euaqtion
    """
    max_u = 0.0
    max_v = 0.0
    nx, ny = u.shape

    for i in range(nx):
        for j in range(ny):
            if abs(u[i,j]) > max_u:
                max_u = abs(u[i,j])
            if abs(v[i,j]) > max_v:
                max_v = abs(v[i,j])

    CFL_x = max_u * dt / delta
    CFL_y = max_v * dt / delta
    return max(CFL_x, CFL_y)

@njit
def compute_new_timestep(u, v, delta, CFL_max):
    """
    Computes a stable time step based on maximum CFL condition.
    
    Args:
        u (2d-array): u-velocity field
        v (2d-array): v-velocity field
        delta (float): space resolution
        CFL_max (float): desired CFL number

    Returns:
        dt (float): stable time step
    """
    eps = 1e-12
    nx, ny = u.shape
    
    abs_u_max = 0.0
    abs_v_max = 0.0
    
    for i in range(nx):
        for j in range(ny):
            if abs(u[i,j]) > abs_u_max:
                abs_u_max = abs(u[i,j])
            if abs(v[i,j]) > abs_v_max:
                abs_v_max = abs(v[i,j])
    
    dt_x = CFL_max * delta / max(abs_u_max, eps)
    dt_y = CFL_max * delta / max(abs_v_max, eps)
    
    dt = min(dt_x, dt_y)
    return dt

@njit
def compute_divergence(u, v, delta):
    """
    computes divergence of a 2-d velocity field without the edges

    Args:
        u (2d-array): u-velocity field
        v (2d-array): v-velocity field
        delta (float): space resolution

    Returns:
        div (2d-array): divergence
        div_l1 (2d-array): L1 norm of divergence
    """
    nx, ny = u.shape
    div = np.zeros((nx, ny))
    sum_div = 0.0

    for i in range(1, nx-1):
        for j in range(1, ny-1):
            dudx = (u[i, j+1] - u[i, j-1]) / (2*delta)
            dvdy = (v[i+1, j] - v[i-1, j]) / (2*delta)
            div[i, j] = dudx + dvdy
            sum_div += abs(div[i,j])

    div_l1 = sum_div / ((nx-2)*(ny-2))
    return div, div_l1