from numba import njit, prange

@njit
def pressure_BC(p):
    p[0, :] = p[1, :]   # dp/dy = 0
    p[-1, :] = p[-2, :] # dp/dy = 0
    p[:, 0] = p[:, 1]
    p[:, -1] = p[:, -2] # 0
    return p
@njit
def velocity_BC(u, v):
    u[0, :] = 0       # y = 0
    u[-1, :] = 0      # y = H
    v[0, :] = 0
    v[-1, :] = 0
    u[:, 0] = 2
    v[:, 0] = 0
    u[:, -1] = u[:, -2]  # ∂u/∂x = 0
    v[:, -1] = v[:, -2]  # ∂v/∂x = 0
    return u, v