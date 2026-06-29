import numpy as np
from numba import njit, prange

from Solver.Kernel_CPU.kernel_config import reduction_blocks_per_grid


def reset_velocity_maxima(maxima, host_zeros):
    """
    Reset the persistent velocity-maxima reduction buffer to zero.
    """
    maxima[...] = host_zeros


@njit(cache=True, parallel=True)
def _velocity_maxima_timestep(u, v, w, maxima, total_size):
    """
    computes global velocity maxima in one CPU kernel.

    The flattened velocity fields are scanned in parallel chunks and reduced
    into one global maximum per velocity component.
    """
    partial_count = maxima.shape[0]
    u_flat = u.reshape(total_size)
    v_flat = v.reshape(total_size)
    w_flat = w.reshape(total_size)

    for part_idx in prange(partial_count):
        max_u = 0.0
        max_v = 0.0
        max_w = 0.0

        idx = part_idx
        while idx < total_size:
            val_u = abs(u_flat[idx])
            val_v = abs(v_flat[idx])
            val_w = abs(w_flat[idx])

            if val_u > max_u:
                max_u = val_u
            if val_v > max_v:
                max_v = val_v
            if val_w > max_w:
                max_w = val_w

            idx += partial_count

        maxima[part_idx, 0] = max_u
        maxima[part_idx, 1] = max_v
        maxima[part_idx, 2] = max_w


def compute_new_timestep_gpu(
    u, v, w, maxima, delta, cfl_max, max_dt=None
):
    """
    computes a stable timestep from convection, diffusion and forcing limits on the CPU.

    A CPU reduction pass determines the maximum absolute values of the three
    velocity components. The smallest of the convective restrictions is
    returned, capped by `max_dt`.
    """
    eps = 1e-12
    total_size = u.size
    reduction_blocks = reduction_blocks_per_grid(total_size)

    if maxima.ndim == 1:
        local_maxima = np.zeros((1, 3), dtype=maxima.dtype)
    else:
        local_maxima = maxima

    if local_maxima.shape[0] != reduction_blocks:
        local_maxima = np.zeros((reduction_blocks, 3), dtype=local_maxima.dtype)
    else:
        local_maxima[...] = 0.0

    _velocity_maxima_timestep(u, v, w, local_maxima, total_size)

    abs_u_max = float(np.max(local_maxima[:, 0]))
    abs_v_max = float(np.max(local_maxima[:, 1]))
    abs_w_max = float(np.max(local_maxima[:, 2]))

    if maxima.ndim == 1:
        maxima[0] = abs_u_max
        maxima[1] = abs_v_max
        maxima[2] = abs_w_max
    else:
        maxima[...] = local_maxima

    cfl_delta = cfl_max * delta
    dt_conv = min(
        cfl_delta / max(abs_u_max, eps),
        cfl_delta / max(abs_v_max, eps),
        cfl_delta / max(abs_w_max, eps),
    )

    if max_dt is None:
        return dt_conv

    return min(dt_conv, float(max_dt))
