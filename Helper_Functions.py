import numpy as np
from numba import njit, prange

@njit(parallel=False, cache=True)
def compute_CFL(u, v, w, dt, delta):
    """
    computes the CFL number of the 3D velocity field.

    Args:
        u (3d-array): x-velocity field
        v (3d-array): y-velocity field
        w (3d-array): z-velocity field
        dt (float): time step size
        delta (float): space resolution

    Returns:
        CFL (float): CFL number
    """
    max_u = 0.0
    max_v = 0.0
    max_w = 0.0
    nx, ny, nz = u.shape

    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                if abs(u[i, j, k]) > max_u:
                    max_u = abs(u[i, j, k])
                if abs(v[i, j, k]) > max_v:
                    max_v = abs(v[i, j, k])
                if abs(w[i, j, k]) > max_w:
                    max_w = abs(w[i, j, k])

    CFL_x = max_u * dt / delta
    CFL_y = max_v * dt / delta
    CFL_z = max_w * dt / delta
    return max(CFL_x, CFL_y, CFL_z)

@njit(parallel=True, cache=True)
def compute_new_timestep(u, v, w, Fx, Fy, Fz, RHO, delta, nu, CFL_max,
                         plane_max_u, plane_max_v, plane_max_w,
                         plane_max_Fx, plane_max_Fy, plane_max_Fz):
    """
    computes a stable timestep based on convection, diffusion and forcing limits.

    Args:
        u (3d-array): x-velocity field
        v (3d-array): y-velocity field
        w (3d-array): z-velocity field
        Fx (3d-array): x-direction force field used in the timestep limiter
        Fy (3d-array): y-direction force field used in the timestep limiter
        Fz (3d-array): z-direction force field used in the timestep limiter
        RHO (float): fluid density
        delta (float): space resolution
        nu (float): viscosity
        CFL_max (float): maximum CFL number
        plane_max_u (1d-array): preallocated plane maxima workspace for u
        plane_max_v (1d-array): preallocated plane maxima workspace for v
        plane_max_w (1d-array): preallocated plane maxima workspace for w
        plane_max_Fx (1d-array): preallocated plane maxima workspace for Fx
        plane_max_Fy (1d-array): preallocated plane maxima workspace for Fy
        plane_max_Fz (1d-array): preallocated plane maxima workspace for Fz
    Returns:
        dt (float): stable time step
    """
    EPS = 1e-12
    nx, ny, nz = u.shape

    for i in prange(nx):
        max_u_plane = 0.0
        max_v_plane = 0.0
        max_w_plane = 0.0
        max_Fx_plane = 0.0
        max_Fy_plane = 0.0
        max_Fz_plane = 0.0

        for j in range(ny):
            for k in range(nz):
                uij = u[i, j, k]
                vij = v[i, j, k]
                wij = w[i, j, k]
                Fxij = Fx[i, j, k]
                Fyij = Fy[i, j, k]
                Fzij = Fz[i, j, k]

                abs_u = uij if uij >= 0.0 else -uij
                abs_v = vij if vij >= 0.0 else -vij
                abs_w = wij if wij >= 0.0 else -wij
                abs_Fx = Fxij if Fxij >= 0.0 else -Fxij
                abs_Fy = Fyij if Fyij >= 0.0 else -Fyij
                abs_Fz = Fzij if Fzij >= 0.0 else -Fzij

                if abs_u > max_u_plane:
                    max_u_plane = abs_u
                if abs_v > max_v_plane:
                    max_v_plane = abs_v
                if abs_w > max_w_plane:
                    max_w_plane = abs_w
                if abs_Fx > max_Fx_plane:
                    max_Fx_plane = abs_Fx
                if abs_Fy > max_Fy_plane:
                    max_Fy_plane = abs_Fy
                if abs_Fz > max_Fz_plane:
                    max_Fz_plane = abs_Fz

        plane_max_u[i] = max_u_plane
        plane_max_v[i] = max_v_plane
        plane_max_w[i] = max_w_plane
        plane_max_Fx[i] = max_Fx_plane
        plane_max_Fy[i] = max_Fy_plane
        plane_max_Fz[i] = max_Fz_plane

    abs_u_max = 0.0
    abs_v_max = 0.0
    abs_w_max = 0.0
    abs_Fx_max = 0.0
    abs_Fy_max = 0.0
    abs_Fz_max = 0.0

    for i in range(nx):
        if plane_max_u[i] > abs_u_max:
            abs_u_max = plane_max_u[i]
        if plane_max_v[i] > abs_v_max:
            abs_v_max = plane_max_v[i]
        if plane_max_w[i] > abs_w_max:
            abs_w_max = plane_max_w[i]
        if plane_max_Fx[i] > abs_Fx_max:
            abs_Fx_max = plane_max_Fx[i]
        if plane_max_Fy[i] > abs_Fy_max:
            abs_Fy_max = plane_max_Fy[i]
        if plane_max_Fz[i] > abs_Fz_max:
            abs_Fz_max = plane_max_Fz[i]

    cfl_delta = CFL_max * delta
    dt_x = cfl_delta / max(abs_u_max, EPS)
    dt_y = cfl_delta / max(abs_v_max, EPS)
    dt_z = cfl_delta / max(abs_w_max, EPS)
    
    dt_conv = min(dt_x, dt_y, dt_z)
    dt_diff = delta * delta / (6.0 * nu)
    dt_force_x = cfl_delta * RHO / max(abs_Fx_max, EPS)
    dt_force_y = cfl_delta * RHO / max(abs_Fy_max, EPS)
    dt_force_z = cfl_delta * RHO / max(abs_Fz_max, EPS)
    dt_forcing = min(dt_force_x, dt_force_y, dt_force_z)
    return min(dt_conv, dt_diff, dt_forcing)

@njit(parallel=False, cache=True)
def compute_divergence(u, v, w, delta):
    """
    computes the divergence of a 3D velocity field excluding the boundary cells.

    Args:
        u (3d-array): x-velocity field
        v (3d-array): y-velocity field
        w (3d-array): z-velocity field
        delta (float): space resolution

    Returns:
        div (3d-array): divergence field
        div_l1 (float): L1 norm of the divergence field
    """
    nx, ny, nz = u.shape
    div = np.zeros((nx, ny, nz))
    sum_div = 0.0

    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            for k in range(1, nz - 1):
                dudx = (u[i + 1, j, k] - u[i - 1, j, k]) / (2 * delta)
                dvdy = (v[i, j + 1, k] - v[i, j - 1, k]) / (2 * delta)
                dwdz = (w[i, j, k + 1] - w[i, j, k - 1]) / (2 * delta)
                div[i, j, k] = dudx + dvdy + dwdz
                sum_div += abs(div[i, j, k])

    div_l1 = sum_div / ((nx - 2) * (ny - 2) * (nz - 2))
    return div, div_l1
