import numpy as np
from numba import njit, prange

import Solver.Kernel_CPU.Boundary_Conditions.domain_bc as BC
import Solver.Kernel_CPU.kernel_config as kernel_config

REDUCTION_THREADS_PER_BLOCK = kernel_config.REDUCTION_THREADS_PER_BLOCK


@njit(cache=True, inline="always")
def _is_active_pressure_cell(active_tile_mask, i, j, k):
    tile_size = kernel_config.ACTIVE_TILE_SIZE
    return active_tile_mask[i // tile_size, j // tile_size, k // tile_size] != 0


@njit(cache=True, parallel=True)
def pressure_equation_right_side(
    u,
    v,
    w,
    b,
    dt,
    delta,
    rho,
    active_tile_mask,
):
    """
    CPU kernel that computes the right hand side of the pressure Poisson equation.
    Only the divergence of the velocity field is used, we neglect non linear terms.
    """
    nx, ny, nz = u.shape
    total = nx * ny * nz
    half_inv_delta = 0.5 / delta
    rho_over_dt = rho / dt

    for n in prange(total):
        i = n // (ny * nz)
        rem = n - i * ny * nz
        j = rem // nz
        k = rem - j * nz

        if (
            i < 1
            or j < 1
            or k < 1
            or i >= nx - 1
            or j >= ny - 1
            or k >= nz - 1
            or not _is_active_pressure_cell(active_tile_mask, i, j, k)
        ):
            b[i, j, k] = 0.0
            continue

        du_dx = (u[i + 1, j, k] - u[i - 1, j, k]) * half_inv_delta
        dv_dy = (v[i, j + 1, k] - v[i, j - 1, k]) * half_inv_delta
        dw_dz = (w[i, j, k + 1] - w[i, j, k - 1]) * half_inv_delta

        b[i, j, k] = rho_over_dt * (du_dx + dv_dy + dw_dz)


@njit(cache=True, parallel=True)
def _sum_rhs_partial_kernel(b, active_tile_mask, partial_sums):
    nx, ny, nz = b.shape
    interior_nx = nx - 2
    interior_ny = ny - 2
    interior_nz = nz - 2
    interior_cell_count = interior_nx * interior_ny * interior_nz
    partial_count = partial_sums.shape[0]

    if interior_cell_count <= 0:
        for idx in prange(partial_count):
            partial_sums[idx] = 0.0
        return

    plane_size = interior_ny * interior_nz

    for part_idx in prange(partial_count):
        local_sum = np.float32(0.0)
        flat_idx = part_idx
        while flat_idx < interior_cell_count:
            i = flat_idx // plane_size + 1
            remainder = flat_idx % plane_size
            j = remainder // interior_nz + 1
            k = remainder % interior_nz + 1
            if _is_active_pressure_cell(active_tile_mask, i, j, k):
                local_sum += b[i, j, k]
            flat_idx += partial_count
        partial_sums[part_idx] = local_sum


@njit(cache=True, parallel=True)
def _count_rhs_active_partial_kernel(b, active_tile_mask, partial_counts):
    nx, ny, nz = b.shape
    interior_nx = nx - 2
    interior_ny = ny - 2
    interior_nz = nz - 2
    interior_cell_count = interior_nx * interior_ny * interior_nz
    partial_count = partial_counts.shape[0]

    if interior_cell_count <= 0:
        for idx in prange(partial_count):
            partial_counts[idx] = 0.0
        return

    plane_size = interior_ny * interior_nz

    for part_idx in prange(partial_count):
        local_count = np.float32(0.0)
        flat_idx = part_idx
        while flat_idx < interior_cell_count:
            i = flat_idx // plane_size + 1
            remainder = flat_idx % plane_size
            j = remainder // interior_nz + 1
            k = remainder % interior_nz + 1
            if _is_active_pressure_cell(active_tile_mask, i, j, k):
                local_count += np.float32(1.0)
            flat_idx += partial_count
        partial_counts[part_idx] = local_count


@njit(cache=True)
def _sum_partial_sums_kernel(partial_sums, partial_count, rhs_sum):
    total_sum = np.float32(0.0)
    for idx in range(partial_count):
        total_sum += partial_sums[idx]
    rhs_sum[0] = total_sum


@njit(cache=True, parallel=True)
def _subtract_rhs_mean_kernel(b, rhs_mean, active_tile_mask):
    nx, ny, nz = b.shape
    total = nx * ny * nz

    for n in prange(total):
        i = n // (ny * nz)
        rem = n - i * ny * nz
        j = rem // nz
        k = rem - j * nz

        if (
            i < 1
            or j < 1
            or k < 1
            or i >= nx - 1
            or j >= ny - 1
            or k >= nz - 1
            or not _is_active_pressure_cell(active_tile_mask, i, j, k)
        ):
            continue

        b[i, j, k] -= rhs_mean


@njit(cache=True, parallel=True)
def _reset_inactive_pressure_kernel(p, active_tile_mask):
    nx, ny, nz = p.shape
    total = nx * ny * nz

    for n in prange(total):
        i = n // (ny * nz)
        rem = n - i * ny * nz
        j = rem // nz
        k = rem - j * nz

        if (
            i < 1
            or j < 1
            or k < 1
            or i >= nx - 1
            or j >= ny - 1
            or k >= nz - 1
            or not _is_active_pressure_cell(active_tile_mask, i, j, k)
        ):
            p[i, j, k] = 0.0


def _remove_rhs_mean(
    b,
    active_tile_mask,
    threadsperblock_3d,
    rhs_partial_sums,
    rhs_sum_buffer,
):
    """
    This is a wrapper method for enforcing the Neumann compatibility condition by removing the RHS mean.
    """
    nx, ny, nz = b.shape
    interior_cell_count = max((nx - 2) * (ny - 2) * (nz - 2), 1)
    reduction_blocks = kernel_config.reduction_blocks_per_grid(interior_cell_count)

    _sum_rhs_partial_kernel(b, active_tile_mask, rhs_partial_sums)
    _sum_partial_sums_kernel(rhs_partial_sums, reduction_blocks, rhs_sum_buffer)
    rhs_sum = float(rhs_sum_buffer[0])

    _count_rhs_active_partial_kernel(b, active_tile_mask, rhs_partial_sums)
    _sum_partial_sums_kernel(rhs_partial_sums, reduction_blocks, rhs_sum_buffer)
    active_cell_count = int(rhs_sum_buffer[0])
    if active_cell_count <= 0:
        return

    rhs_mean = rhs_sum / float(active_cell_count)
    if abs(rhs_mean) <= 1.0e-12:
        return

    _subtract_rhs_mean_kernel(b, rhs_mean, active_tile_mask)


@njit(cache=True, parallel=True)
def project_velocity_kernel(
    u, v, w, p, obstacle_mask, dt, delta, rho, active_tile_mask
):
    """
    Apply the pressure projection `u <- u - dt/rho * grad(p)` to one interior cell.

    Obstacle cells are skipped because their wall velocities are restored by the
    obstacle boundary conditions after the projection pass.
    """
    nx, ny, nz = u.shape
    total = nx * ny * nz
    pressure_coeff = dt / (2.0 * rho * delta)

    for n in prange(total):
        i = n // (ny * nz)
        rem = n - i * ny * nz
        j = rem // nz
        k = rem - j * nz

        if i < 1 or j < 1 or k < 1 or i >= nx - 1 or j >= ny - 1 or k >= nz - 1:
            continue

        if not _is_active_pressure_cell(active_tile_mask, i, j, k) or obstacle_mask[i, j, k]:
            continue

        u[i, j, k] -= pressure_coeff * (p[i + 1, j, k] - p[i - 1, j, k])
        v[i, j, k] -= pressure_coeff * (p[i, j + 1, k] - p[i, j - 1, k])
        w[i, j, k] -= pressure_coeff * (p[i, j, k + 1] - p[i, j, k - 1])


@njit(cache=True, parallel=True)
def add_artifical_divergence(
    T,
    source_masks,
    extra_pressure,
    source_noise,
    noise_amplitudes,
    expansion_rate,
    t_reference,
    b,
    active_tile_mask,
    rho,
    dt,
):
    nx, ny, nz = b.shape
    total = nx * ny * nz
    rho_over_dt = rho / dt
    source_count = source_masks.shape[0]

    for n in prange(total):
        i = n // (ny * nz)
        rem = n - i * ny * nz
        j = rem // nz
        k = rem - j * nz

        if (
            i < 1
            or j < 1
            or k < 1
            or i >= nx - 1
            or j >= ny - 1
            or k >= nz - 1
            or not _is_active_pressure_cell(active_tile_mask, i, j, k)
        ):
            continue

        thermal_divergence = expansion_rate * (T[i, j, k] - t_reference)
        extra_pressure_term = 0.0

        for source_idx in range(source_count):
            if not source_masks[source_idx, i, j, k]:
                continue
            source_extra_pressure = extra_pressure[source_idx]
            source_extra_pressure *= min(
                max(1.0 + noise_amplitudes[source_idx] * source_noise[source_idx, i, j, k], 0.0),
                2.0,
            )
            if source_extra_pressure > extra_pressure_term:
                extra_pressure_term = source_extra_pressure

        b[i, j, k] -= rho_over_dt * (thermal_divergence + extra_pressure_term)


@njit(cache=True, parallel=True)
def mg_restrict_residual_8cell(p, b, coarse_b, delta):
    cnx, cny, cnz = coarse_b.shape
    nx, ny, nz = p.shape
    total = cnx * cny * cnz
    inv_delta2 = 1.0 / (delta * delta)

    for n in prange(total):
        I = n // (cny * cnz)
        rem = n - I * cny * cnz
        J = rem // cnz
        K = rem - J * cnz

        i0 = 2 * I
        j0 = 2 * J
        k0 = 2 * K

        s = 0.0
        count = 0.0

        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    i = i0 + di
                    j = j0 + dj
                    k = k0 + dk

                    if i >= 1 and j >= 1 and k >= 1 and i < nx - 1 and j < ny - 1 and k < nz - 1:
                        lap = (
                            p[i + 1, j, k] + p[i - 1, j, k]
                            + p[i, j + 1, k] + p[i, j - 1, k]
                            + p[i, j, k + 1] + p[i, j, k - 1]
                            - 6.0 * p[i, j, k]
                        ) * inv_delta2

                        s += b[i, j, k] - lap
                        count += 1.0

        coarse_b[I, J, K] = s / count if count > 0.0 else 0.0


@njit(cache=True, parallel=True)
def mg_prolongate_add_nearest_sparse_level0(coarse_e, fine_p, active_tile_mask):
    cnx, cny, cnz = coarse_e.shape
    fnx, fny, fnz = fine_p.shape
    total = cnx * cny * cnz

    for n in prange(total):
        I = n // (cny * cnz)
        rem = n - I * cny * cnz
        J = rem // cnz
        K = rem - J * cnz

        e = 0.25 * coarse_e[I, J, K]
        i0 = 2 * I
        j0 = 2 * J
        k0 = 2 * K

        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    i = i0 + di
                    j = j0 + dj
                    k = k0 + dk

                    if i < fnx and j < fny and k < fnz and _is_active_pressure_cell(active_tile_mask, i, j, k):
                        fine_p[i, j, k] += e


@njit(cache=True, parallel=True)
def mg_restrict_8cell(fine_r, coarse_b):
    cnx, cny, cnz = coarse_b.shape
    fnx, fny, fnz = fine_r.shape
    total = cnx * cny * cnz

    for n in prange(total):
        I = n // (cny * cnz)
        rem = n - I * cny * cnz
        J = rem // cnz
        K = rem - J * cnz

        i0 = 2 * I
        j0 = 2 * J
        k0 = 2 * K

        s = 0.0
        count = 0.0

        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    i = i0 + di
                    j = j0 + dj
                    k = k0 + dk

                    if i < fnx and j < fny and k < fnz:
                        s += fine_r[i, j, k]
                        count += 1.0

        coarse_b[I, J, K] = s / count if count > 0.0 else 0.0


@njit(cache=True, parallel=True)
def mg_restrict_residual_8cell_sparse_level0(p, b, coarse_b, delta, active_tile_mask):
    cnx, cny, cnz = coarse_b.shape
    nx, ny, nz = p.shape
    total = cnx * cny * cnz
    inv_delta2 = 1.0 / (delta * delta)

    for n in prange(total):
        I = n // (cny * cnz)
        rem = n - I * cny * cnz
        J = rem // cnz
        K = rem - J * cnz

        i0 = 2 * I
        j0 = 2 * J
        k0 = 2 * K

        s = 0.0
        count = 0.0

        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    i = i0 + di
                    j = j0 + dj
                    k = k0 + dk

                    if (
                        i >= 1 and j >= 1 and k >= 1
                        and i < nx - 1 and j < ny - 1 and k < nz - 1
                        and _is_active_pressure_cell(active_tile_mask, i, j, k)
                    ):
                        lap = (
                            p[i + 1, j, k] + p[i - 1, j, k]
                            + p[i, j + 1, k] + p[i, j - 1, k]
                            + p[i, j, k + 1] + p[i, j, k - 1]
                            - 6.0 * p[i, j, k]
                        ) * inv_delta2

                        s += b[i, j, k] - lap
                        count += 1.0

        coarse_b[I, J, K] = s / count if count > 0.0 else 0.0


@njit(cache=True, parallel=True)
def mg_prolongate_add_nearest(coarse_e, fine_p):
    cnx, cny, cnz = coarse_e.shape
    fnx, fny, fnz = fine_p.shape
    total = cnx * cny * cnz

    for n in prange(total):
        I = n // (cny * cnz)
        rem = n - I * cny * cnz
        J = rem // cnz
        K = rem - J * cnz

        e = 0.25 * coarse_e[I, J, K]
        i0 = 2 * I
        j0 = 2 * J
        k0 = 2 * K

        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    i = i0 + di
                    j = j0 + dj
                    k = k0 + dk

                    if i < fnx and j < fny and k < fnz:
                        fine_p[i, j, k] += e


@njit(cache=True, parallel=True)
def mg_rbgs_step(p, b, delta, parity):
    nx, ny, nz = p.shape
    total = nx * ny * nz
    delta2 = delta * delta

    for n in prange(total):
        i = n // (ny * nz)
        rem = n - i * ny * nz
        j = rem // nz
        k = rem - j * nz

        if i < 1 or j < 1 or k < 1 or i >= nx - 1 or j >= ny - 1 or k >= nz - 1:
            continue

        if ((i + j + k) & 1) != parity:
            continue

        p[i, j, k] = (
            p[i + 1, j, k] + p[i - 1, j, k]
            + p[i, j + 1, k] + p[i, j - 1, k]
            + p[i, j, k + 1] + p[i, j, k - 1]
            - delta2 * b[i, j, k]
        ) / 6.0


@njit(cache=True, parallel=True)
def mg_rbgs_step_sparse_level0(p, b, delta, parity, active_tile_mask):
    nx, ny, nz = p.shape
    total = nx * ny * nz
    delta2 = delta * delta

    for n in prange(total):
        i = n // (ny * nz)
        rem = n - i * ny * nz
        j = rem // nz
        k = rem - j * nz

        if (
            i < 1 or j < 1 or k < 1
            or i >= nx - 1 or j >= ny - 1 or k >= nz - 1
            or ((i + j + k) & 1) != parity
            or not _is_active_pressure_cell(active_tile_mask, i, j, k)
        ):
            continue

        p[i, j, k] = (
            p[i + 1, j, k] + p[i - 1, j, k]
            + p[i, j + 1, k] + p[i, j - 1, k]
            + p[i, j, k + 1] + p[i, j, k - 1]
            - delta2 * b[i, j, k]
        ) / 6.0


def _mg_smooth(p, b, delta, iterations, level=0, active_tile_mask=None):
    use_sparse = level == 0 and active_tile_mask is not None

    for _ in range(iterations):
        if use_sparse:
            mg_rbgs_step_sparse_level0(p, b, delta, 0, active_tile_mask)
            mg_rbgs_step_sparse_level0(p, b, delta, 1, active_tile_mask)
        else:
            mg_rbgs_step(p, b, delta, 0)
            mg_rbgs_step(p, b, delta, 1)

    BC._pressure_poisson_apply_neumann_bcs(p)


def _mg_vcycle(
    level,
    p_levels,
    b_levels,
    zero_levels,
    delta_levels,
    pre_smooth,
    post_smooth,
    coarse_smooth,
    active_tile_mask=None,
):
    p = p_levels[level]
    b = b_levels[level]
    delta = delta_levels[level]

    _mg_smooth(
        p,
        b,
        delta,
        pre_smooth,
        level=level,
        active_tile_mask=active_tile_mask,
    )

    last_level = len(p_levels) - 1

    if level == last_level:
        _mg_smooth(
            p,
            b,
            delta,
            coarse_smooth,
            level=level,
            active_tile_mask=active_tile_mask,
        )
        return

    coarse_p = p_levels[level + 1]
    coarse_b = b_levels[level + 1]

    coarse_p[:] = zero_levels[level + 1]

    if level == 0 and active_tile_mask is not None:
        mg_restrict_residual_8cell_sparse_level0(
            p,
            b,
            coarse_b,
            delta,
            active_tile_mask,
        )
    else:
        mg_restrict_residual_8cell(
            p,
            b,
            coarse_b,
            delta,
        )

    _mg_vcycle(
        level + 1,
        p_levels,
        b_levels,
        zero_levels,
        delta_levels,
        pre_smooth,
        post_smooth,
        coarse_smooth,
        active_tile_mask=None,
    )

    if level == 0 and active_tile_mask is not None:
        mg_prolongate_add_nearest_sparse_level0(
            coarse_p,
            p,
            active_tile_mask,
        )
    else:
        mg_prolongate_add_nearest(
            coarse_p,
            p,
        )

    _mg_smooth(
        p,
        b,
        delta,
        post_smooth,
        level=level,
        active_tile_mask=active_tile_mask,
    )


def pressure_poisson_multigrid(
    u,
    v,
    w,
    p,
    T,
    b,
    dt,
    source_masks,
    extra_pressure,
    source_noise,
    noise_amplitudes,
    delta,
    rho,
    expansion_rate,
    t_reference,
    active_tile_mask,
    p_levels,
    b_levels,
    delta_levels,
    num_vcycles,
    rhs_partial_sums,
    rhs_sum_buffer,
    zero_levels,
):
    threadsperblock_3d = kernel_config.THREADS_PER_BLOCK_3D
    _ = kernel_config.volume_blocks_per_grid(u.shape, threadsperblock_3d)

    p_levels[0] = p
    b_levels[0] = b

    pressure_equation_right_side(
        u,
        v,
        w,
        b_levels[0],
        dt,
        delta,
        rho,
        active_tile_mask,
    )

    _remove_rhs_mean(
        b_levels[0],
        active_tile_mask,
        threadsperblock_3d,
        rhs_partial_sums,
        rhs_sum_buffer,
    )

    _reset_inactive_pressure_kernel(
        p_levels[0],
        active_tile_mask,
    )

    add_artifical_divergence(
        T,
        source_masks,
        extra_pressure,
        source_noise,
        noise_amplitudes,
        expansion_rate,
        t_reference,
        b_levels[0],
        active_tile_mask,
        rho,
        dt,
    )

    for _ in range(num_vcycles):
        _mg_vcycle(
            0,
            p_levels,
            b_levels,
            zero_levels,
            delta_levels,
            pre_smooth=2,
            post_smooth=4,
            coarse_smooth=20,
            active_tile_mask=active_tile_mask,
        )

    return p_levels[0]
