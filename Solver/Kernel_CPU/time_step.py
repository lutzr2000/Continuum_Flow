import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True, fastmath=True)
def _compute_velocity_maxima_and_divergence(u, v, w):
    """
    Scan all velocity components once and return maxima plus divergence state.
    """
    nx, ny, nz = u.shape
    partial_u = np.empty(nx, dtype=u.dtype)
    partial_v = np.empty(nx, dtype=v.dtype)
    partial_w = np.empty(nx, dtype=w.dtype)
    partial_invalid = np.zeros(nx, dtype=np.uint8)

    for i in prange(nx):
        max_u = 0.0
        max_v = 0.0
        max_w = 0.0
        invalid = 0

        for j in range(ny):
            for k in range(nz):
                u_value = u[i, j, k]
                v_value = v[i, j, k]
                w_value = w[i, j, k]

                if np.isnan(u_value) or np.isinf(u_value):
                    invalid = 1
                else:
                    abs_u = abs(u_value)
                    if abs_u > max_u:
                        max_u = abs_u

                if np.isnan(v_value) or np.isinf(v_value):
                    invalid = 1
                else:
                    abs_v = abs(v_value)
                    if abs_v > max_v:
                        max_v = abs_v

                if np.isnan(w_value) or np.isinf(w_value):
                    invalid = 1
                else:
                    abs_w = abs(w_value)
                    if abs_w > max_w:
                        max_w = abs_w

        partial_u[i] = max_u
        partial_v[i] = max_v
        partial_w[i] = max_w
        partial_invalid[i] = invalid

    abs_u_max = 0.0
    abs_v_max = 0.0
    abs_w_max = 0.0
    solver_diverged = False

    for i in range(nx):
        if partial_invalid[i] != 0:
            solver_diverged = True

        if partial_u[i] > abs_u_max:
            abs_u_max = partial_u[i]
        if partial_v[i] > abs_v_max:
            abs_v_max = partial_v[i]
        if partial_w[i] > abs_w_max:
            abs_w_max = partial_w[i]

    return abs_u_max, abs_v_max, abs_w_max, solver_diverged


def compute_new_timestep(
    u, v, w, maxima, delta, cfl_max, max_dt
):
    """
    Compute a stable timestep from convection limits on the CPU.
    """
    eps = 1e-12

    abs_u_max, abs_v_max, abs_w_max, solver_diverged = (
        _compute_velocity_maxima_and_divergence(u, v, w)
    )
    maxima[0] = abs_u_max
    maxima[1] = abs_v_max
    maxima[2] = abs_w_max

    if solver_diverged:
        return 0.0, True

    cfl_delta = cfl_max * delta
    dt = min(
        cfl_delta / max(float(abs_u_max), eps),
        cfl_delta / max(float(abs_v_max), eps),
        cfl_delta / max(float(abs_w_max), eps),
    )

    dt = min(dt, float(max_dt))

    return dt