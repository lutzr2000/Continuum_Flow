import numpy as np



def compute_new_timestep(
    u, v, w, delta, cfl_max, max_dt=None
):
    """
    Compute a stable timestep from the velocity field.
    """
    eps = 1e-12

    abs_u_max = float(np.max(np.abs(u)))
    abs_v_max = float(np.max(np.abs(v)))
    abs_w_max = float(np.max(np.abs(w)))

    cfl_delta = cfl_max * delta
    dt_conv = min(
        cfl_delta / max(abs_u_max, eps),
        cfl_delta / max(abs_v_max, eps),
        cfl_delta / max(abs_w_max, eps),
    )

    if max_dt is None:
        return dt_conv

    return min(dt_conv, float(max_dt))
