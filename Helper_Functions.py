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
def compute_new_timestep(u, v, w, Fx, Fy, Fz, RHO, delta, nu, CFL_max):
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
    Returns:
        dt (float): stable time step
    """
    EPS = 1e-12
    nx, ny, nz = u.shape

    max_u = np.empty(nx, dtype=u.dtype)
    max_v = np.empty(nx, dtype=u.dtype)
    max_w = np.empty(nx, dtype=u.dtype)
    max_Fx = np.empty(nx, dtype=u.dtype)
    max_Fy = np.empty(nx, dtype=u.dtype)
    max_Fz = np.empty(nx, dtype=u.dtype)

    for i in prange(nx):
        mu = 0.0
        mv = 0.0
        mw = 0.0
        mFx = 0.0
        mFy = 0.0
        mFz = 0.0

        for j in range(ny):
            for k in range(nz):
                au = abs(u[i, j, k])
                av = abs(v[i, j, k])
                aw = abs(w[i, j, k])
                aFx = abs(Fx[i, j, k])
                aFy = abs(Fy[i, j, k])
                aFz = abs(Fz[i, j, k])

                if au > mu:
                    mu = au
                if av > mv:
                    mv = av
                if aw > mw:
                    mw = aw
                if aFx > mFx:
                    mFx = aFx
                if aFy > mFy:
                    mFy = aFy
                if aFz > mFz:
                    mFz = aFz

        max_u[i] = mu
        max_v[i] = mv
        max_w[i] = mw
        max_Fx[i] = mFx
        max_Fy[i] = mFy
        max_Fz[i] = mFz

    abs_u_max = 0.0
    abs_v_max = 0.0
    abs_w_max = 0.0
    abs_Fx_max = 0.0
    abs_Fy_max = 0.0
    abs_Fz_max = 0.0

    for i in range(nx):
        if max_u[i] > abs_u_max:
            abs_u_max = max_u[i]
        if max_v[i] > abs_v_max:
            abs_v_max = max_v[i]
        if max_w[i] > abs_w_max:
            abs_w_max = max_w[i]
        if max_Fx[i] > abs_Fx_max:
            abs_Fx_max = max_Fx[i]
        if max_Fy[i] > abs_Fy_max:
            abs_Fy_max = max_Fy[i]
        if max_Fz[i] > abs_Fz_max:
            abs_Fz_max = max_Fz[i]

    cfl_delta = CFL_max * delta
    dt_conv = min(
        cfl_delta / max(abs_u_max, EPS),
        cfl_delta / max(abs_v_max, EPS),
        cfl_delta / max(abs_w_max, EPS),
    )
    dt_diff = delta * delta / (6.0 * nu)
    dt_forcing = min(
        cfl_delta * RHO / max(abs_Fx_max, EPS),
        cfl_delta * RHO / max(abs_Fy_max, EPS),
        cfl_delta * RHO / max(abs_Fz_max, EPS),
    )

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
