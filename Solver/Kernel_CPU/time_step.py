import numpy as np


def reset_velocity_maxima(maxima, host_zeros):
    """Reset the persistent velocity-maxima buffer to zero."""
    maxima[...] = host_zeros


def compute_new_timestep_cpu(
    u, v, w, maxima, fx_max, fy_max, fz_max, rho, delta, nu, cfl_max, max_dt=None
):
    """
    Compute a stable timestep from convection, diffusion and forcing limits on the CPU.

    Args:
        u (ndarray): x-velocity field
        v (ndarray): y-velocity field
        w (ndarray): z-velocity field
        maxima (ndarray): persistent output array with shape (3,) for the maxima
        fx_max (float): maximum absolute x-direction force used in the timestep limiter
        fy_max (float): maximum absolute y-direction force used in the timestep limiter
        fz_max (float): maximum absolute z-direction force used in the timestep limiter
        rho (float): fluid density
        delta (float): grid spacing
        nu (float): kinematic viscosity
        cfl_max (float): maximum admissible CFL number
        max_dt (float, optional): upper bound for the returned timestep
    Returns:
        tuple[float, bool]: stable timestep and divergence flag
    """
    eps = 1e-12

    abs_u_max = float(np.max(np.abs(u)))
    abs_v_max = float(np.max(np.abs(v)))
    abs_w_max = float(np.max(np.abs(w)))
    maxima[0] = abs_u_max
    maxima[1] = abs_v_max
    maxima[2] = abs_w_max

    solver_diverged = bool(
        np.isnan(abs_u_max) or
        np.isnan(abs_v_max) or
        np.isnan(abs_w_max) or
        np.isinf(abs_u_max) or
        np.isinf(abs_v_max) or
        np.isinf(abs_w_max)
    )
    if solver_diverged:
        return 0.0, True

    cfl_delta = cfl_max * delta
    dt_conv = min(
        cfl_delta / max(float(abs_u_max), eps),
        cfl_delta / max(float(abs_v_max), eps),
        cfl_delta / max(float(abs_w_max), eps),
    )
    dt_diff = delta * delta / (6.0 * nu)
    dt_forcing = min(
        cfl_delta * rho / max(abs(float(fx_max)), eps),
        cfl_delta * rho / max(abs(float(fy_max)), eps),
        cfl_delta * rho / max(abs(float(fz_max)), eps),
    )

    dt = min(dt_conv, dt_diff, dt_forcing)
    if max_dt is not None:
        dt = min(dt, float(max_dt))

    if np.isnan(dt) or np.isinf(dt) or float(dt) < 1.0e-7:
        return 0.0, True

    return dt, False
