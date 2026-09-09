from typing import Any

import numpy as np
from numba import njit, prange

from Solver.Kernel_CPU.kernel_config import (
    CPU_FIELD_DTYPE,
    TILE_SIZE,
)


@njit(cache=True, parallel=True)
def velocity_maxima_timestep(
    u: Any,
    v: Any,
    w: Any,
    tile_map: Any,
    maxima_per_tile: Any,
    total_tile_count: int,
) -> None:
    r"""
    Reduce the absolute velocity maxima of all active sparse cells.

    Each CPU worker computes component-wise maxima for one sparse tile.
    The per-tile results are written to ``maxima_per_tile``.
    """
    tile_size = TILE_SIZE
    cells_per_tile = tile_size * tile_size * tile_size

    tiles_x, tiles_y, tiles_z = tile_map.shape
    tiles_per_yz = tiles_y * tiles_z

    for idx in prange(total_tile_count):
        tile_i = idx // tiles_per_yz
        remainder = idx % tiles_per_yz
        tile_j = remainder // tiles_z
        tile_k = remainder % tiles_z

        max_u = CPU_FIELD_DTYPE(0.0)
        max_v = CPU_FIELD_DTYPE(0.0)
        max_w = CPU_FIELD_DTYPE(0.0)

        tile_index = tile_map[tile_i, tile_j, tile_k]

        if tile_index != -1:
            for local_flat in range(cells_per_tile):
                local_i = local_flat // (tile_size * tile_size)
                remainder2 = local_flat % (tile_size * tile_size)
                local_j = remainder2 // tile_size
                local_k = remainder2 % tile_size

                val_u = abs(u[tile_index, local_i, local_j, local_k])
                val_v = abs(v[tile_index, local_i, local_j, local_k])
                val_w = abs(w[tile_index, local_i, local_j, local_k])

                if val_u > max_u:
                    max_u = val_u
                if val_v > max_v:
                    max_v = val_v
                if val_w > max_w:
                    max_w = val_w

        maxima_per_tile[idx, 0] = max_u
        maxima_per_tile[idx, 1] = max_v
        maxima_per_tile[idx, 2] = max_w


def compute_new_timestep_cpu(
    u: Any,
    v: Any,
    w: Any,
    tile_map: Any,
    active_tile_count: int,
    maxima: Any,
    delta: float,
    cfl_max: float,
    max_dt: float | None = None,
) -> float:
    r"""
    Compute a stable timestep from the component-wise velocity maxima.

    The timestep satisfies the configured CFL limit independently along every
    coordinate axis:

    .. math::

        \Delta t = \min\left(
            \frac{C_{\max}\,\Delta x}{\max |u|},
            \frac{C_{\max}\,\Delta x}{\max |v|},
            \frac{C_{\max}\,\Delta x}{\max |w|},
            \Delta t_{\max}
        \right).

    A small denominator bound prevents division by zero in stationary fields.
    """
    eps = 1e-12

    if active_tile_count <= 0:
        return float(max_dt)

    total_tile_count = tile_map.size

    maxima_per_tile = np.zeros(
        (total_tile_count, 3),
        dtype=CPU_FIELD_DTYPE,
    )

    velocity_maxima_timestep(
        u,
        v,
        w,
        tile_map,
        maxima_per_tile,
        total_tile_count,
    )

    abs_u_max = np.max(maxima_per_tile[:, 0])
    abs_v_max = np.max(maxima_per_tile[:, 1])
    abs_w_max = np.max(maxima_per_tile[:, 2])

    maxima[0] = abs_u_max
    maxima[1] = abs_v_max
    maxima[2] = abs_w_max

    cfl_delta = cfl_max * delta
    dt_conv = min(
        cfl_delta / max(float(abs_u_max), eps),
        cfl_delta / max(float(abs_v_max), eps),
        cfl_delta / max(float(abs_w_max), eps),
    )

    return min(dt_conv, float(max_dt))
