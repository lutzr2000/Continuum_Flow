import numpy as np


def reset_velocity_maxima(maxima, host_zeros):
    """
    Reset the persistent velocity-maxima reduction buffer to zero.
    """
    maxima[...] = host_zeros


def compute_new_timestep(
    u, v, w, maxima, delta, cfl_max, max_dt=None
):
    """
    Compute a stable timestep from the velocity field.
    """
    eps = 1e-12

    abs_u_max = float(np.max(np.abs(u)))
    abs_v_max = float(np.max(np.abs(v)))
    abs_w_max = float(np.max(np.abs(w)))

    if maxima.ndim == 1:
        maxima[0] = abs_u_max
        maxima[1] = abs_v_max
        maxima[2] = abs_w_max
    else:
        maxima[...] = 0.0
        if maxima.shape[0] > 0:
            maxima[0, 0] = abs_u_max
            maxima[0, 1] = abs_v_max
            maxima[0, 2] = abs_w_max

    cfl_delta = cfl_max * delta
    dt_conv = min(
        cfl_delta / max(abs_u_max, eps),
        cfl_delta / max(abs_v_max, eps),
        cfl_delta / max(abs_w_max, eps),
    )

    if max_dt is None:
        return dt_conv

    return min(dt_conv, float(max_dt))
