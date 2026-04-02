import numpy as np
from numba import njit, prange

@njit(parallel=True)
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

@njit(parallel=True)
def compute_new_timestep(u, v, F, RHO, delta, nu, CFL_max, row_max_u, row_max_v, row_max_F):
    """
    Computes a stable time step based on maximum CFL condition for convection and diffusion.
    
    Args:
        u (2d-array): u-velocity field
        v (2d-array): v-velocity field
        F (2d-array): force field used in the timestep limiter
        delta (float): space resolution
        nu (float): viscosity
        CFL_max (float): maximum CFL number
        row_max_u (1d-array): preallocated row maxima workspace for u
        row_max_v (1d-array): preallocated row maxima workspace for v
        row_max_F (1d-array): preallocated row maxima workspace for F

    Returns:
        dt (float): stable time step
    """
    EPS = 1e-12
    nx, ny = u.shape

    for i in prange(nx):
        max_u_row = 0.0
        max_v_row = 0.0
        max_F_row = 0.0

        for j in range(ny):
            uij = u[i, j]
            vij = v[i, j]
            Fij = F[i, j]

            abs_u = uij if uij >= 0.0 else -uij
            abs_v = vij if vij >= 0.0 else -vij
            abs_F = Fij if Fij >= 0.0 else -Fij

            if abs_u > max_u_row:
                max_u_row = abs_u
            if abs_v > max_v_row:
                max_v_row = abs_v
            if abs_F > max_F_row:
                max_F_row = abs_F

        row_max_u[i] = max_u_row
        row_max_v[i] = max_v_row
        row_max_F[i] = max_F_row

    abs_u_max = 0.0
    abs_v_max = 0.0
    abs_F_max = 0.0

    for i in range(nx):
        if row_max_u[i] > abs_u_max:
            abs_u_max = row_max_u[i]
        if row_max_v[i] > abs_v_max:
            abs_v_max = row_max_v[i]
        if row_max_F[i] > abs_F_max:
            abs_F_max = row_max_F[i]

    cfl_delta = CFL_max * delta
    dt_x = CFL_max * delta / max(abs_u_max, EPS)
    dt_y = CFL_max * delta / max(abs_v_max, EPS)
    
    dt_conv = min(dt_x, dt_y)
    dt_diff = 0.25 * delta * delta / nu
    dt_forcing = cfl_delta * RHO / max(abs_F_max, EPS)
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
