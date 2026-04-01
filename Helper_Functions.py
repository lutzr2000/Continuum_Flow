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
        CFL (flaot): CFL number
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
def compute_new_timestep(u, v, F, RHO, delta, nu, CFL_max):
    """
    Computes a stable time step based on maximum CFL condition for convection and diffusion.
    
    Args:
        u (2d-array): u-velocity field
        v (2d-array): v-velocity field
        delta (float): space resolution
        nu (float): viscosity
        CFL_max (float): maximum CFL number

    Returns:
        dt (float): stable time step
    """
    EPS = 1e-12
    nx, ny = u.shape
    
    abs_u_max = 0.0
    abs_v_max = 0.0
    abs_F_max = 0.0
    
    for i in range(nx):
        for j in range(ny):
            if abs(u[i,j]) > abs_u_max:
                abs_u_max = abs(u[i,j])
            if abs(v[i,j]) > abs_v_max:
                abs_v_max = abs(v[i,j])
            if abs(F[i,j]) > abs_F_max:
                abs_F_max = abs(F[i,j])
    
    dt_x = CFL_max * delta / max(abs_u_max, EPS)
    dt_y = CFL_max * delta / max(abs_v_max, EPS)
    
    dt_conv = min(dt_x, dt_y)
    dt_diff = 0.25 * delta**2 / nu
    dt_forcing = CFL_max * delta / max(abs_F_max/RHO, EPS)
    dt = min(dt_conv, dt_diff, dt_forcing)
    return dt

@njit
def compute_divergence(u, v, delta):
    """
    computes divergence of a 2-d velocity field without the boundary

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