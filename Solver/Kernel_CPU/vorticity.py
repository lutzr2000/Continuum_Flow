import math
from typing import Any

from numba import njit, prange

import Solver.Kernel_CPU.sparse_managment as sparse_managment
import Solver.Kernel_CPU.kernel_config as kernel_config


@njit(cache=True, parallel=True)
def compute_vorticity(
    u: Any,
    v: Any,
    w: Any,
    u_initial: float,
    v_initial: float,
    w_initial: float,
    obstacle_mask: Any,
    vorticity_magnitude: Any,
    delta: float,
    tile_map: Any,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    r"""
    Compute the vorticity magnitude of the velocity field for one grid cell.

    Vorticity is the curl of the velocity field
    :math:`\mathbf{u} = (u, v, w)`:

    .. math::

        \boldsymbol{\omega} = \nabla \times \mathbf{u}
        = \begin{pmatrix}
            \partial_y w - \partial_z v \\
            \partial_z u - \partial_x w \\
            \partial_x v - \partial_y u
          \end{pmatrix}.

    Each derivative is approximated with a second-order centered difference.

    .. math::

        \partial_y w_{i,j,k}
        \approx \frac{w_{i,j+1,k} - w_{i,j-1,k}}{2\,\Delta x}.

    The resulting magnitude is written to ``vorticity_magnitude``:

    .. math::

        \lVert\boldsymbol{\omega}\rVert
        = \sqrt{\omega_x^2 + \omega_y^2 + \omega_z^2}.

    Cells outside the active sparse domain, cells on the outer grid boundary,
    and obstacle cells are skipped or assigned zero vorticity.
    """
    total_tiles = tile_map.shape[0] * tile_map.shape[1] * tile_map.shape[2]

    for tile_flat in prange(total_tiles):
        for local_i in range(kernel_config.TILE_SIZE):
            for local_j in range(kernel_config.TILE_SIZE):
                for local_k in range(kernel_config.TILE_SIZE):
                    (
                        tile_i,
                        tile_j,
                        tile_k,
                        local_i,
                        local_j,
                        local_k,
                        i,
                        j,
                        k,
                    ) = sparse_managment.tile_to_index(
                        tile_flat,
                        local_i,
                        local_j,
                        local_k,
                        tile_map.shape[0],
                        tile_map.shape[1],
                        tile_map.shape[2],
                    )

                    tile_index = tile_map[tile_i, tile_j, tile_k]

                    if tile_index == -1:
                        continue

                    if (
                        i < 1
                        or j < 1
                        or k < 1
                        or i >= nx - 1
                        or j >= ny - 1
                        or k >= nz - 1
                    ):
                        vorticity_magnitude[tile_index, local_i, local_j, local_k] = 0.0
                        continue

                    if obstacle_mask[tile_index, local_i, local_j, local_k]:
                        vorticity_magnitude[tile_index, local_i, local_j, local_k] = 0.0
                        continue

                    half_inv_delta = 0.5 / delta

                    du_dy = (
                        sparse_managment.get_pool_value(
                            u, tile_map, i, j + 1, k, u_initial
                        )
                        - sparse_managment.get_pool_value(
                            u, tile_map, i, j - 1, k, u_initial
                        )
                    ) * half_inv_delta
                    du_dz = (
                        sparse_managment.get_pool_value(
                            u, tile_map, i, j, k + 1, u_initial
                        )
                        - sparse_managment.get_pool_value(
                            u, tile_map, i, j, k - 1, u_initial
                        )
                    ) * half_inv_delta

                    dv_dx = (
                        sparse_managment.get_pool_value(
                            v, tile_map, i + 1, j, k, v_initial
                        )
                        - sparse_managment.get_pool_value(
                            v, tile_map, i - 1, j, k, v_initial
                        )
                    ) * half_inv_delta
                    dv_dz = (
                        sparse_managment.get_pool_value(
                            v, tile_map, i, j, k + 1, v_initial
                        )
                        - sparse_managment.get_pool_value(
                            v, tile_map, i, j, k - 1, v_initial
                        )
                    ) * half_inv_delta

                    dw_dx = (
                        sparse_managment.get_pool_value(
                            w, tile_map, i + 1, j, k, w_initial
                        )
                        - sparse_managment.get_pool_value(
                            w, tile_map, i - 1, j, k, w_initial
                        )
                    ) * half_inv_delta
                    dw_dy = (
                        sparse_managment.get_pool_value(
                            w, tile_map, i, j + 1, k, w_initial
                        )
                        - sparse_managment.get_pool_value(
                            w, tile_map, i, j - 1, k, w_initial
                        )
                    ) * half_inv_delta

                    wx = dw_dy - dv_dz
                    wy = du_dz - dw_dx
                    wz = dv_dx - du_dy

                    vorticity_magnitude[tile_index, local_i, local_j, local_k] = (
                        math.sqrt(wx * wx + wy * wy + wz * wz)
                    )


@njit(inline="always", cache=True)
def apply_vorticity_confinement(
    u: Any,
    v: Any,
    w: Any,
    obstacle_mask: Any,
    omega_magnitude: Any,
    i: int,
    j: int,
    k: int,
    delta: float,
    vorticity_strength: float,
    tile_map: Any,
    u_initial: float,
    v_initial: float,
    w_initial: float,
    nx: int,
    ny: int,
    nz: int,
) -> tuple[float, float, float]:
    r"""
    Compute the vorticity-confinement force for one grid cell.

    First, the centered-difference gradient of the vorticity magnitude is
    evaluated and normalized:

    .. math::

        \mathbf{N}
        = \frac{\nabla \lVert\boldsymbol{\omega}\rVert}
               {\lVert\nabla \lVert\boldsymbol{\omega}\rVert\rVert}.

    The local vorticity vector is reconstructed from the velocity curl:

    .. math::

        \boldsymbol{\omega} = \nabla \times \mathbf{u}.

    Finally, the confinement force points perpendicular to both the magnitude
    gradient and the vorticity vector:

    .. math::

        \mathbf{f}_{\mathrm{conf}}
        = \varepsilon\,\mathbf{N} \times \boldsymbol{\omega},

    where :math:`\varepsilon` is ``vorticity_strength``. This force restores
    rotational detail that numerical dissipation would otherwise remove. A
    zero force is returned for inactive tiles, cells near the domain boundary,
    obstacle cells, and cells whose magnitude gradient cannot be normalized.
    """
    tile_size = kernel_config.TILE_SIZE

    tile_i = i // tile_size
    tile_j = j // tile_size
    tile_k = k // tile_size

    local_i = i % tile_size
    local_j = j % tile_size
    local_k = k % tile_size

    tile_index = tile_map[tile_i, tile_j, tile_k]

    if tile_index == -1:
        return 0.0, 0.0, 0.0

    if (
        i < 2
        or j < 2
        or k < 2
        or i >= nx - 2
        or j >= ny - 2
        or k >= nz - 2
        or obstacle_mask[tile_index, local_i, local_j, local_k]
    ):
        return 0.0, 0.0, 0.0

    half_inv_delta = 0.5 / delta

    grad_x = (
        sparse_managment.get_pool_value(omega_magnitude, tile_map, i + 1, j, k, 0.0)
        - sparse_managment.get_pool_value(omega_magnitude, tile_map, i - 1, j, k, 0.0)
    ) * half_inv_delta

    grad_y = (
        sparse_managment.get_pool_value(omega_magnitude, tile_map, i, j + 1, k, 0.0)
        - sparse_managment.get_pool_value(omega_magnitude, tile_map, i, j - 1, k, 0.0)
    ) * half_inv_delta

    grad_z = (
        sparse_managment.get_pool_value(omega_magnitude, tile_map, i, j, k + 1, 0.0)
        - sparse_managment.get_pool_value(omega_magnitude, tile_map, i, j, k - 1, 0.0)
    ) * half_inv_delta

    grad_length = math.sqrt(grad_x * grad_x + grad_y * grad_y + grad_z * grad_z)

    if grad_length <= 1.0e-12:
        return 0.0, 0.0, 0.0

    nx_dir = grad_x / grad_length
    ny_dir = grad_y / grad_length
    nz_dir = grad_z / grad_length

    du_dy = (
        sparse_managment.get_pool_value(u, tile_map, i, j + 1, k, u_initial)
        - sparse_managment.get_pool_value(u, tile_map, i, j - 1, k, u_initial)
    ) * half_inv_delta
    du_dz = (
        sparse_managment.get_pool_value(u, tile_map, i, j, k + 1, u_initial)
        - sparse_managment.get_pool_value(u, tile_map, i, j, k - 1, u_initial)
    ) * half_inv_delta

    dv_dx = (
        sparse_managment.get_pool_value(v, tile_map, i + 1, j, k, v_initial)
        - sparse_managment.get_pool_value(v, tile_map, i - 1, j, k, v_initial)
    ) * half_inv_delta
    dv_dz = (
        sparse_managment.get_pool_value(v, tile_map, i, j, k + 1, v_initial)
        - sparse_managment.get_pool_value(v, tile_map, i, j, k - 1, v_initial)
    ) * half_inv_delta

    dw_dx = (
        sparse_managment.get_pool_value(w, tile_map, i + 1, j, k, w_initial)
        - sparse_managment.get_pool_value(w, tile_map, i - 1, j, k, w_initial)
    ) * half_inv_delta
    dw_dy = (
        sparse_managment.get_pool_value(w, tile_map, i, j + 1, k, w_initial)
        - sparse_managment.get_pool_value(w, tile_map, i, j - 1, k, w_initial)
    ) * half_inv_delta

    wx = dw_dy - dv_dz
    wy = du_dz - dw_dx
    wz = dv_dx - du_dy

    fx = vorticity_strength * (ny_dir * wz - nz_dir * wy)
    fy = vorticity_strength * (nz_dir * wx - nx_dir * wz)
    fz = vorticity_strength * (nx_dir * wy - ny_dir * wx)

    return fx, fy, fz
