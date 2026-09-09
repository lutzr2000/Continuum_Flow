from typing import Any

from Solver.Kernel_CPU.timing import profiled_run
import numpy as np
from numba import njit, prange

import Solver.Kernel_CPU.kernel_config as kernel_config
import Solver.Kernel_CPU.sparse_managment as sparse_managment
import Solver.Kernel_CPU.multigrid as multigrid
import Solver.Kernel_CPU.noise as noise

CPU_FIELD_DTYPE = kernel_config.CPU_FIELD_DTYPE


@njit(cache=True, parallel=True)
def pressure_equation_right_side(
    u: Any,
    v: Any,
    w: Any,
    b: Any,
    dt: float,
    delta: float,
    rho: float,
    tile_map: Any,
    u_initial: float,
    v_initial: float,
    w_initial: float,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    r"""
    Assemble the pressure-Poisson right-hand side from velocity divergence.

    Centered differences approximate the divergence and form

    .. math::

        b = \frac{\rho}{\Delta t}\,\nabla\!\cdot\mathbf{u}
          = \frac{\rho}{\Delta t}
            \left(\partial_x u + \partial_y v + \partial_z w\right).

    Boundary cells are assigned zero because their pressure conditions are
    handled separately.
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
                        b[tile_index, local_i, local_j, local_k] = 0.0
                        continue

                    half_inv_delta = 0.5 / delta
                    rho_over_dt = rho / dt

                    du_dx = (
                        sparse_managment.get_pool_value(
                            u, tile_map, i + 1, j, k, u_initial
                        )
                        - sparse_managment.get_pool_value(
                            u, tile_map, i - 1, j, k, u_initial
                        )
                    ) * half_inv_delta

                    dv_dy = (
                        sparse_managment.get_pool_value(
                            v, tile_map, i, j + 1, k, v_initial
                        )
                        - sparse_managment.get_pool_value(
                            v, tile_map, i, j - 1, k, v_initial
                        )
                    ) * half_inv_delta

                    dw_dz = (
                        sparse_managment.get_pool_value(
                            w, tile_map, i, j, k + 1, w_initial
                        )
                        - sparse_managment.get_pool_value(
                            w, tile_map, i, j, k - 1, w_initial
                        )
                    ) * half_inv_delta

                    b[tile_index, local_i, local_j, local_k] = rho_over_dt * (
                        du_dx + dv_dy + dw_dz
                    )


@njit(cache=True, parallel=True)
def rhs_sum_count_partial_kernel(
    b: Any,
    tile_map: Any,
    partial_sums: Any,
    partial_counts: Any,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    """
    Reduce active interior right-hand-side values into per-tile sums and counts.
    """
    total_tiles = tile_map.shape[0] * tile_map.shape[1] * tile_map.shape[2]

    for tile_flat in prange(total_tiles):
        tiles_y = tile_map.shape[1]
        tiles_z = tile_map.shape[2]
        tiles_per_yz = tiles_y * tiles_z

        tile_i = tile_flat // tiles_per_yz
        remainder = tile_flat % tiles_per_yz
        tile_j = remainder // tiles_z
        tile_k = remainder % tiles_z

        tile_index = tile_map[tile_i, tile_j, tile_k]

        local_sum = 0.0
        local_count = 0.0

        if tile_index != -1:
            for local_i in range(kernel_config.TILE_SIZE):
                for local_j in range(kernel_config.TILE_SIZE):
                    for local_k in range(kernel_config.TILE_SIZE):
                        i = tile_i * kernel_config.TILE_SIZE + local_i
                        j = tile_j * kernel_config.TILE_SIZE + local_j
                        k = tile_k * kernel_config.TILE_SIZE + local_k

                        if (
                            i >= 1
                            and j >= 1
                            and k >= 1
                            and i < nx - 1
                            and j < ny - 1
                            and k < nz - 1
                        ):
                            local_sum += b[
                                tile_index,
                                local_i,
                                local_j,
                                local_k,
                            ]

                            local_count += 1.0

        partial_sums[tile_flat] = local_sum
        partial_counts[tile_flat] = local_count


@njit(cache=True, parallel=True)
def count_rhs_active_partial_kernel(
    tile_map: Any,
    partial_counts: Any,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    """
    Count active interior RHS cells per sparse tile.
    """
    total_tiles = tile_map.shape[0] * tile_map.shape[1] * tile_map.shape[2]

    for tile_flat in prange(total_tiles):
        tiles_y = tile_map.shape[1]
        tiles_z = tile_map.shape[2]
        tiles_per_yz = tiles_y * tiles_z

        tile_i = tile_flat // tiles_per_yz
        remainder = tile_flat % tiles_per_yz
        tile_j = remainder // tiles_z
        tile_k = remainder % tiles_z

        tile_index = tile_map[tile_i, tile_j, tile_k]

        local_count = 0.0

        if tile_index != -1:
            for local_i in range(kernel_config.TILE_SIZE):
                for local_j in range(kernel_config.TILE_SIZE):
                    for local_k in range(kernel_config.TILE_SIZE):
                        i = tile_i * kernel_config.TILE_SIZE + local_i
                        j = tile_j * kernel_config.TILE_SIZE + local_j
                        k = tile_k * kernel_config.TILE_SIZE + local_k

                        if (
                            i >= 1
                            and j >= 1
                            and k >= 1
                            and i < nx - 1
                            and j < ny - 1
                            and k < nz - 1
                        ):
                            local_count += 1.0

        partial_counts[tile_flat] = local_count


@njit(cache=True)
def sum_partial_sums_kernel(
    partial_sums: Any,
    partial_count: int,
    rhs_sum: Any,
) -> None:
    """
    Reduce the partial sums into one scalar sum.
    """
    total_sum = 0.0

    for idx in range(partial_count):
        total_sum += partial_sums[idx]

    rhs_sum[0] = total_sum


@njit(cache=True)
def rhs_mean_kernel(
    partial_sums: Any,
    partial_counts: Any,
    partial_count: int,
    rhs_mean: Any,
) -> None:
    r"""
    Divide the global right-hand-side sum by the active-cell count.

    .. math::

        \bar b = \frac{1}{N}\sum_{i=1}^{N} b_i.
    """
    total_sum = 0.0
    total_count = 0.0

    for idx in range(partial_count):
        total_sum += partial_sums[idx]
        total_count += partial_counts[idx]

    if total_count > 0.0:
        rhs_mean[0] = total_sum / total_count
    else:
        rhs_mean[0] = 0.0


@njit(cache=True, parallel=True)
def subtract_rhs_mean_kernel(
    b: Any,
    rhs_mean: Any,
    tile_map: Any,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    r"""
    Subtract the active-domain mean from every pressure right-hand-side cell.

    .. math::

        b_i \leftarrow b_i - \bar b.

    This compatibility correction makes the Neumann Poisson problem solvable
    by ensuring that the discrete right-hand side has zero mean.
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
                        continue

                    b[tile_index, local_i, local_j, local_k] -= rhs_mean[0]


@njit(cache=True, parallel=True)
def reset_inactive_pressure(
    p: Any,
    tile_map: Any,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    """
    Clear pressure values belonging to inactive or out-of-domain sparse cells.
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
                        p[tile_index, local_i, local_j, local_k] = 0.0


def remove_rhs_mean(
    b: Any,
    tile_map: Any,
    rhs_partial_sums: Any,
    rhs_partial_counts: Any,
    rhs_mean_buffer: Any,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    """
    Compute and remove the mean of the active pressure right-hand side.
    """
    total_tile_count = tile_map.size

    rhs_sum_count_partial_kernel(
        b,
        tile_map,
        rhs_partial_sums,
        rhs_partial_counts,
        nx,
        ny,
        nz,
    )

    rhs_mean_kernel(
        rhs_partial_sums,
        rhs_partial_counts,
        total_tile_count,
        rhs_mean_buffer,
    )

    subtract_rhs_mean_kernel(
        b,
        rhs_mean_buffer,
        tile_map,
        nx,
        ny,
        nz,
    )


@njit(cache=True, parallel=True)
def project_velocity_kernel(
    u: Any,
    v: Any,
    w: Any,
    p: Any,
    obstacle_mask: Any,
    dt: float,
    delta: float,
    rho: Any,
    tile_map: Any,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    """
    Apply the pressure projection `u <- u - dt/rho * grad(p)` to one interior cell.

    Obstacle cells are skipped because their wall velocities are restored by the
    obstacle boundary conditions after the projection pass.
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
                        continue

                    if obstacle_mask[tile_index, local_i, local_j, local_k]:
                        continue

                    pressure_coeff = dt / (2.0 * rho * delta)

                    px1 = sparse_managment.get_pool_value(p, tile_map, i + 1, j, k, 0.0)
                    px0 = sparse_managment.get_pool_value(p, tile_map, i - 1, j, k, 0.0)
                    py1 = sparse_managment.get_pool_value(p, tile_map, i, j + 1, k, 0.0)
                    py0 = sparse_managment.get_pool_value(p, tile_map, i, j - 1, k, 0.0)
                    pz1 = sparse_managment.get_pool_value(p, tile_map, i, j, k + 1, 0.0)
                    pz0 = sparse_managment.get_pool_value(p, tile_map, i, j, k - 1, 0.0)

                    u[tile_index, local_i, local_j, local_k] -= pressure_coeff * (
                        px1 - px0
                    )
                    v[tile_index, local_i, local_j, local_k] -= pressure_coeff * (
                        py1 - py0
                    )
                    w[tile_index, local_i, local_j, local_k] -= pressure_coeff * (
                        pz1 - pz0
                    )


@njit(cache=True, parallel=True)
def add_artifical_divergence(
    T: Any,
    source_mask: Any,
    source_extra_pressure: Any,
    noise_scale: float,
    noise_amplitude: Any,
    noise_seed: Any,
    expansion_rate: float,
    t_reference: float,
    b: Any,
    tile_map: Any,
    rho: Any,
    delta: float,
    nx: int,
    ny: int,
    nz: int,
    dt: float,
) -> None:
    r"""
    Add thermal expansion and source pressure to the Poisson right-hand side.

    The correction subtracted from ``b`` is

    .. math::

        \frac{\rho}{\Delta x}\left[
            \alpha(T-T_{\mathrm{ref}}) + p_{\mathrm{source}}
        \right],

    with optional procedural-noise modulation of the source term.
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
                        continue

                    thermal_divergence = expansion_rate * (
                        T[tile_index, local_i, local_j, local_k] - t_reference
                    )

                    extra_pressure_term = 0.0

                    if source_mask[tile_index, local_i, local_j, local_k]:
                        scalar_multiplier = 1.0

                        if noise_amplitude != 0.0:
                            scale = max(noise_scale, 1.0e-6)

                            noise_value = noise._value_noise_3d(
                                i / scale,
                                j / scale,
                                k / scale,
                                noise_seed,
                            )

                            scalar_multiplier = max(
                                1.0 + noise_value * noise_amplitude,
                                0.0,
                            )

                        extra_pressure_term = (
                            1 / dt * source_extra_pressure * scalar_multiplier
                        )

                    b[tile_index, local_i, local_j, local_k] -= (
                        rho / delta * (thermal_divergence + extra_pressure_term)
                    )


@profiled_run
def pressure_poisson_multigrid(
    u: Any,
    v: Any,
    w: Any,
    p: Any,
    T: Any,
    b: Any,
    dt: float,
    source_masks: Any,
    source_noise_scales: Any,
    source_noise_amplitudes: Any,
    source_noise_seeds: Any,
    extra_pressure: Any,
    delta: float,
    rho: Any,
    expansion_rate: float,
    t_reference: float,
    tile_map: Any,
    tile_shape: tuple[int, int, int],
    u_initial: float,
    v_initial: float,
    w_initial: float,
    p_levels: Any,
    b_levels: Any,
    delta_levels: Any,
    num_vcycles: Any,
    rhs_partial_sums: Any,
    rhs_partial_counts: Any,
    rhs_mean_buffer: Any,
    zero_levels: Any,
    nx: int,
    ny: int,
    nz: int,
    *,
    timings: Any = None,
) -> Any:
    r"""
    Solve the pressure Poisson equation with repeated multigrid V-cycles.

    The projection pressure satisfies

    .. math::

        \nabla^2 p = \frac{\rho}{\Delta t}\,\nabla\!\cdot\mathbf{u}.

    The right-hand side is assembled and made compatible with Neumann
    boundary conditions before the multigrid hierarchy reduces the residual.
    """
    with timings.section("pressure_poisson_multigrid", "pressure_equation_right_side"):
        pressure_equation_right_side(
            u,
            v,
            w,
            b,
            dt,
            delta,
            rho,
            tile_map,
            u_initial,
            v_initial,
            w_initial,
            nx,
            ny,
            nz,
        )

    with timings.section("pressure_poisson_multigrid", "reset_inactive_pressure"):
        reset_inactive_pressure(
            p,
            tile_map,
            nx,
            ny,
            nz,
        )

    for source_idx, source_mask in enumerate(source_masks):
        with timings.section("pressure_poisson_multigrid", "add_artifical_divergence"):
            add_artifical_divergence(
                T,
                source_mask,
                extra_pressure[source_idx],
                source_noise_scales[source_idx],
                source_noise_amplitudes[source_idx],
                source_noise_seeds[source_idx],
                expansion_rate,
                t_reference,
                b,
                tile_map,
                rho,
                delta,
                nx,
                ny,
                nz,
                dt,
            )

    with timings.section("pressure_poisson_multigrid", "remove_rhs_mean"):
        remove_rhs_mean(
            b,
            tile_map,
            rhs_partial_sums,
            rhs_partial_counts,
            rhs_mean_buffer,
            nx,
            ny,
            nz,
        )

    for _ in range(num_vcycles):
        with timings.section("pressure_poisson_multigrid", "multigrid.v_cycle"):
            multigrid.v_cycle(
                0,
                p_levels,
                b_levels,
                p,
                b,
                zero_levels,
                delta,
                delta_levels,
                pre_smooth=2,
                post_smooth=4,
                coarse_smooth=20,
                nx=nx,
                ny=ny,
                nz=nz,
                tile_map=tile_map,
            )

    return p
