import numpy as np
from numba import cuda
from time import perf_counter

import Solver.Kernel_GPU.Boundary_Conditions.domain_bc as BC
import Solver.Kernel_GPU.kernel_config as kernel_config
import Solver.Kernel_GPU.sparse_managment as sparse_managment
import Solver.Kernel_GPU.multigrid as multigrid

REDUCTION_THREADS_PER_BLOCK = (
    kernel_config.REDUCTION_THREADS_PER_BLOCK
)  # this is needed because if this is added directly inline, cuda crashes i do not know why


def _profile_pressure_section(profile_stats, name, callback, synchronize_cuda=False):
    if profile_stats is None:
        return callback()

    if synchronize_cuda:
        cuda.synchronize()

    start_time = perf_counter()
    result = callback()

    if synchronize_cuda:
        cuda.synchronize()

    elapsed = perf_counter() - start_time
    entry = profile_stats.setdefault(name, {"total_runtime": 0.0, "call_count": 0})
    entry["total_runtime"] += elapsed
    entry["call_count"] += 1

    return result


def print_profile_summary(profile_stats):
    if not profile_stats:
        return

    total_profiled_runtime = sum(
        entry["total_runtime"] for entry in profile_stats.values()
    )
    name_width = max(len("Section"), max(len(name) for name in profile_stats))
    total_width = len("Total Runtime [s]")
    calls_width = len("N Calls")
    avg_width = len("Average Runtime [ms]")
    share_width = len("% Total Runtime")

    divider = (
        f"+-{'-' * name_width}-+-{'-' * total_width}-+-{'-' * calls_width}"
        f"-+-{'-' * avg_width}-+-{'-' * share_width}-+"
    )

    print("Pressure / Multigrid profiling summary")
    print(divider)
    print(
        f"| {'Section':<{name_width}} | {'Total Runtime [s]':>{total_width}} "
        f"| {'N Calls':>{calls_width}} | {'Average Runtime [ms]':>{avg_width}} "
        f"| {'% Total Runtime':>{share_width}} |"
    )
    print(divider)

    for name, entry in sorted(
        profile_stats.items(),
        key=lambda item: item[1]["total_runtime"],
        reverse=True,
    ):
        total_runtime = entry["total_runtime"]
        call_count = entry["call_count"]
        average_runtime_ms = (total_runtime / call_count) * 1000.0 if call_count else 0.0
        runtime_share = (
            (total_runtime / total_profiled_runtime) * 100.0
            if total_profiled_runtime > 0.0
            else 0.0
        )
        print(
            f"| {name:<{name_width}} | {total_runtime:>{total_width}.6f} "
            f"| {call_count:>{calls_width}d} | {average_runtime_ms:>{avg_width}.3f} "
            f"| {runtime_share:>{share_width}.2f} |"
        )

    print(divider)


@cuda.jit(cache=True)
def pressure_equation_right_side(
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
):
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
    ) = sparse_managment.tile_to_index()

    tile_index = tile_map[tile_i, tile_j, tile_k]

    if tile_index == -1:
        return

    if i < 1 or j < 1 or k < 1 or i >= nx - 1 or j >= ny - 1 or k >= nz - 1:
        b[tile_index, local_i, local_j, local_k] = 0.0
        return

    half_inv_delta = 0.5 / delta
    rho_over_dt = rho / dt

    du_dx = (
        sparse_managment._sample_sparse_cell(u, tile_map, i + 1, j, k, u_initial)
        - sparse_managment._sample_sparse_cell(u, tile_map, i - 1, j, k, u_initial)
    ) * half_inv_delta

    dv_dy = (
        sparse_managment._sample_sparse_cell(v, tile_map, i, j + 1, k, v_initial)
        - sparse_managment._sample_sparse_cell(v, tile_map, i, j - 1, k, v_initial)
    ) * half_inv_delta

    dw_dz = (
        sparse_managment._sample_sparse_cell(w, tile_map, i, j, k + 1, w_initial)
        - sparse_managment._sample_sparse_cell(w, tile_map, i, j, k - 1, w_initial)
    ) * half_inv_delta

    b[tile_index, local_i, local_j, local_k] = rho_over_dt * (du_dx + dv_dy + dw_dz)


@cuda.jit(cache=True)
def sum_rhs_partial_kernel(b, tile_map, partial_sums, nx, ny, nz):
    """
    Reduce the interior RHS into one partial sum per CUDA block.
    """
    interior_nx = nx - 2
    interior_ny = ny - 2
    interior_nz = nz - 2
    interior_cell_count = interior_nx * interior_ny * interior_nz

    if interior_cell_count <= 0:
        return

    tid = cuda.threadIdx.x
    block_size = cuda.blockDim.x
    global_idx = cuda.grid(1)
    stride = cuda.gridsize(1)

    shared_sums = cuda.shared.array(
        shape=REDUCTION_THREADS_PER_BLOCK,
        dtype=np.float32,
    )

    local_sum = np.float32(0.0)
    flat_idx = global_idx
    plane_size = interior_ny * interior_nz

    while flat_idx < interior_cell_count:
        i = flat_idx // plane_size + 1
        remainder = flat_idx % plane_size
        j = remainder // interior_nz + 1
        k = remainder % interior_nz + 1

        tile_i = i // kernel_config.TILE_SIZE
        tile_j = j // kernel_config.TILE_SIZE
        tile_k = k // kernel_config.TILE_SIZE
        tile_index = tile_map[tile_i, tile_j, tile_k]
        if tile_index != -1:
            local_i = i - tile_i * kernel_config.TILE_SIZE
            local_j = j - tile_j * kernel_config.TILE_SIZE
            local_k = k - tile_k * kernel_config.TILE_SIZE
            local_sum += b[tile_index, local_i, local_j, local_k]

        flat_idx += stride

    shared_sums[tid] = local_sum
    cuda.syncthreads()

    offset = block_size >> 1
    while offset > 0:
        if tid < offset:
            shared_sums[tid] += shared_sums[tid + offset]
        cuda.syncthreads()
        offset >>= 1

    if tid == 0:
        partial_sums[cuda.blockIdx.x] = shared_sums[0]


@cuda.jit(cache=True)
def count_rhs_active_partial_kernel(b, tile_map, partial_counts, nx, ny, nz):
    """
    Reduce the number of active RHS cells into one partial count per CUDA block.
    """
    interior_nx = nx - 2
    interior_ny = ny - 2
    interior_nz = nz - 2
    interior_cell_count = interior_nx * interior_ny * interior_nz

    if interior_cell_count <= 0:
        return

    tid = cuda.threadIdx.x
    block_size = cuda.blockDim.x
    global_idx = cuda.grid(1)
    stride = cuda.gridsize(1)

    shared_counts = cuda.shared.array(
        shape=REDUCTION_THREADS_PER_BLOCK,
        dtype=np.float32,
    )

    local_count = np.float32(0.0)
    flat_idx = global_idx
    plane_size = interior_ny * interior_nz

    while flat_idx < interior_cell_count:
        i = flat_idx // plane_size + 1
        remainder = flat_idx % plane_size
        j = remainder // interior_nz + 1
        k = remainder % interior_nz + 1

        tile_i = i // kernel_config.TILE_SIZE
        tile_j = j // kernel_config.TILE_SIZE
        tile_k = k // kernel_config.TILE_SIZE
        if tile_map[tile_i, tile_j, tile_k] != -1:
            local_count += np.float32(1.0)

        flat_idx += stride

    shared_counts[tid] = local_count
    cuda.syncthreads()

    offset = block_size >> 1
    while offset > 0:
        if tid < offset:
            shared_counts[tid] += shared_counts[tid + offset]
        cuda.syncthreads()
        offset >>= 1

    if tid == 0:
        partial_counts[cuda.blockIdx.x] = shared_counts[0]


@cuda.jit(cache=True)
def sum_partial_sums_kernel(partial_sums, partial_count, rhs_sum):
    """
    Reduce the block partial sums into one scalar sum on the GPU. This is needed
    for computing RHS mean.
    """
    tid = cuda.threadIdx.x
    stride = cuda.blockDim.x
    shared_sums = cuda.shared.array(
        shape=REDUCTION_THREADS_PER_BLOCK,
        dtype=np.float32,
    )

    local_sum = np.float32(0.0)
    idx = tid
    while idx < partial_count:
        local_sum += partial_sums[idx]
        idx += stride

    shared_sums[tid] = local_sum
    cuda.syncthreads()

    offset = stride >> 1
    while offset > 0:
        if tid < offset:
            shared_sums[tid] += shared_sums[tid + offset]
        cuda.syncthreads()
        offset >>= 1

    if tid == 0:
        rhs_sum[0] = shared_sums[0]


@cuda.jit(cache=True)
def subtract_rhs_mean_kernel(b, rhs_mean, tile_map, nx, ny, nz):
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
    ) = sparse_managment.tile_to_index()

    tile_index = tile_map[tile_i, tile_j, tile_k]
    if tile_index == -1:
        return

    if i < 1 or j < 1 or k < 1 or i >= nx - 1 or j >= ny - 1 or k >= nz - 1:
        return

    b[tile_index, local_i, local_j, local_k] -= rhs_mean


@cuda.jit(cache=True)
def reset_inactive_pressure(p, tile_map, nx, ny, nz):
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
    ) = sparse_managment.tile_to_index()

    tile_index = tile_map[tile_i, tile_j, tile_k]
    if tile_index == -1:
        return

    if i < 1 or j < 1 or k < 1 or i >= nx - 1 or j >= ny - 1 or k >= nz - 1:
        p[tile_index, local_i, local_j, local_k] = 0.0


def remove_rhs_mean(
    b,
    tile_map,
    rhs_partial_sums,
    rhs_sum_buffer,
    nx,
    ny,
    nz,
    profile_stats=None,
):
    """
    Enforce the Neumann compatibility condition by removing
    the mean RHS over active interior cells.
    """

    interior_cell_count = max((nx - 2) * (ny - 2) * (nz - 2), 1)

    reduction_blocks = kernel_config.reduction_blocks_per_grid(interior_cell_count)
    reduction_threads = kernel_config.REDUCTION_THREADS_PER_BLOCK

    blockspergrid_3d = kernel_config.volume_blocks_per_grid(
        (nx, ny, nz),
        kernel_config.THREADS_PER_BLOCK_3D,
    )

    _profile_pressure_section(
        profile_stats,
        "remove_rhs_mean.sum_rhs_partial_kernel",
        lambda: sum_rhs_partial_kernel[reduction_blocks, reduction_threads](
            b,
            tile_map,
            rhs_partial_sums,
            nx,
            ny,
            nz,
        ),
        synchronize_cuda=True,
    )

    _profile_pressure_section(
        profile_stats,
        "remove_rhs_mean.sum_partial_sums_rhs",
        lambda: sum_partial_sums_kernel[1, reduction_threads](
            rhs_partial_sums,
            reduction_blocks,
            rhs_sum_buffer,
        ),
        synchronize_cuda=True,
    )

    rhs_sum = float(rhs_sum_buffer.copy_to_host()[0])

    _profile_pressure_section(
        profile_stats,
        "remove_rhs_mean.count_rhs_active_partial_kernel",
        lambda: count_rhs_active_partial_kernel[reduction_blocks, reduction_threads](
            b,
            tile_map,
            rhs_partial_sums,
            nx,
            ny,
            nz,
        ),
        synchronize_cuda=True,
    )

    _profile_pressure_section(
        profile_stats,
        "remove_rhs_mean.sum_partial_sums_count",
        lambda: sum_partial_sums_kernel[1, reduction_threads](
            rhs_partial_sums,
            reduction_blocks,
            rhs_sum_buffer,
        ),
        synchronize_cuda=True,
    )

    active_cell_count = int(rhs_sum_buffer.copy_to_host()[0])

    if active_cell_count <= 0:
        return

    rhs_mean = rhs_sum / float(active_cell_count)

    if abs(rhs_mean) <= 1.0e-12:
        return

    _profile_pressure_section(
        profile_stats,
        "remove_rhs_mean.subtract_rhs_mean_kernel",
        lambda: subtract_rhs_mean_kernel[
            blockspergrid_3d,
            kernel_config.THREADS_PER_BLOCK_3D,
        ](
            b,
            rhs_mean,
            tile_map,
            nx,
            ny,
            nz,
        ),
        synchronize_cuda=True,
    )


@cuda.jit(cache=True)
def project_velocity_kernel(
    u,
    v,
    w,
    p,
    obstacle_mask,
    dt,
    delta,
    rho,
    tile_map,
    nx,
    ny,
    nz,
):
    """
    Apply the pressure projection `u <- u - dt/rho * grad(p)` to one interior cell.

    Obstacle cells are skipped because their wall velocities are restored by the
    obstacle boundary conditions after the projection pass.
    """
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
    ) = sparse_managment.tile_to_index()

    tile_index = tile_map[tile_i, tile_j, tile_k]
    if tile_index == -1:
        return

    if i < 1 or j < 1 or k < 1 or i >= nx - 1 or j >= ny - 1 or k >= nz - 1:
        return

    if obstacle_mask[i, j, k]:
        return

    pressure_coeff = dt / (2.0 * rho * delta)

    px1 = sparse_managment._sample_sparse_cell(p, tile_map, i + 1, j, k, 0.0)
    px0 = sparse_managment._sample_sparse_cell(p, tile_map, i - 1, j, k, 0.0)
    py1 = sparse_managment._sample_sparse_cell(p, tile_map, i, j + 1, k, 0.0)
    py0 = sparse_managment._sample_sparse_cell(p, tile_map, i, j - 1, k, 0.0)
    pz1 = sparse_managment._sample_sparse_cell(p, tile_map, i, j, k + 1, 0.0)
    pz0 = sparse_managment._sample_sparse_cell(p, tile_map, i, j, k - 1, 0.0)

    u[tile_index, local_i, local_j, local_k] -= pressure_coeff * (px1 - px0)
    v[tile_index, local_i, local_j, local_k] -= pressure_coeff * (py1 - py0)
    w[tile_index, local_i, local_j, local_k] -= pressure_coeff * (pz1 - pz0)


@cuda.jit(cache=True)
def add_artifical_divergence(
    T,
    source_masks,
    extra_pressure,
    source_noise,
    noise_amplitudes,
    expansion_rate,
    t_reference,
    b,
    tile_map,
    rho,
    dt,
    nx,
    ny,
    nz,
):
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
    ) = sparse_managment.tile_to_index()

    tile_index = tile_map[tile_i, tile_j, tile_k]
    if tile_index == -1:
        return

    if i < 1 or j < 1 or k < 1 or i >= nx - 1 or j >= ny - 1 or k >= nz - 1:
        return

    rho_over_dt = rho / dt

    thermal_divergence = expansion_rate * (
        T[tile_index, local_i, local_j, local_k] - t_reference
    )

    extra_pressure_term = 0.0
    source_count = source_masks.shape[0]
    for source_idx in range(source_count):
        if not source_masks[source_idx, i, j, k]:
            continue
        source_extra_pressure = extra_pressure[source_idx]
        source_extra_pressure *= min(
            max(
                1.0 + noise_amplitudes[source_idx] * source_noise[source_idx, i, j, k],
                0.0,
            ),
            2.0,
        )
        if abs(source_extra_pressure) > abs(extra_pressure_term):
            extra_pressure_term = source_extra_pressure

    b[tile_index, local_i, local_j, local_k] -= rho_over_dt * (
        thermal_divergence + extra_pressure_term
    )


def multigrid_smooth(
    p,
    b,
    delta,
    iterations,
    level=0,
    tile_map=None,
    nx=None,
    ny=None,
    nz=None,
    profile_stats=None,
    phase="smooth",
):
    use_sparse = level == 0

    if use_sparse:
        blocks = kernel_config.volume_blocks_per_grid(
            (nx, ny, nz),
            kernel_config.THREADS_PER_BLOCK_3D,
        )
    else:
        blocks = kernel_config.volume_blocks_per_grid(
            p.shape,
            kernel_config.THREADS_PER_BLOCK_3D,
        )

    def _run_smooth():
        for _ in range(iterations):
            if use_sparse:
                multigrid.mg_rbgs_step_sparse_level0[blocks, kernel_config.THREADS_PER_BLOCK_3D](
                    p, b, delta, 0, tile_map, nx, ny, nz
                )
                multigrid.mg_rbgs_step_sparse_level0[blocks, kernel_config.THREADS_PER_BLOCK_3D](
                    p, b, delta, 1, tile_map, nx, ny, nz
                )
            else:
                multigrid.mg_rbgs_step[blocks, kernel_config.THREADS_PER_BLOCK_3D](p, b, delta, 0)
                multigrid.mg_rbgs_step[blocks, kernel_config.THREADS_PER_BLOCK_3D](p, b, delta, 1)

        if use_sparse:
            BC.pressure_poisson_apply_neumann_bcs[
                blocks, kernel_config.THREADS_PER_BLOCK_3D
            ](p, tile_map, nx, ny, nz)
        else:
            BC.pressure_poisson_apply_neumann_bcs_dense[
                blocks, kernel_config.THREADS_PER_BLOCK_3D
            ](p)

    return _profile_pressure_section(
        profile_stats,
        f"multigrid_smooth.level_{level}.{phase}",
        _run_smooth,
        synchronize_cuda=True,
    )


def multigrid_vcycle(
    level,
    p_levels,
    b_levels,
    p_level0,
    b_level0,
    zero_levels,
    delta_levels,
    pre_smooth,
    post_smooth,
    coarse_smooth,
    nx,
    ny,
    nz,
    tile_map=None,
    profile_stats=None,
):
    def _run_vcycle():
        p = p_level0 if level == 0 else p_levels[level]
        b = b_level0 if level == 0 else b_levels[level]
        delta = delta_levels[level]

        multigrid_smooth(
            p,
            b,
            delta,
            pre_smooth,
            level=level,
            tile_map=tile_map,
            nx=nx,
            ny=ny,
            nz=nz,
            profile_stats=profile_stats,
            phase="pre",
        )

        last_level = len(p_levels) - 1

        if level == last_level:
            multigrid_smooth(
                p,
                b,
                delta,
                coarse_smooth,
                level=level,
                tile_map=tile_map,
                nx=nx,
                ny=ny,
                nz=nz,
                profile_stats=profile_stats,
                phase="coarse",
            )
            return

        coarse_p = p_levels[level + 1]
        coarse_b = b_levels[level + 1]

        coarse_blocks = kernel_config.volume_blocks_per_grid(
            coarse_p.shape,
            kernel_config.THREADS_PER_BLOCK_3D,
        )

        _profile_pressure_section(
            profile_stats,
            f"multigrid_vcycle.level_{level}.coarse_p_reset",
            lambda: coarse_p.copy_to_device(zero_levels[level + 1]),
            synchronize_cuda=True,
        )

        if level == 0 and tile_map is not None:
            _profile_pressure_section(
                profile_stats,
                f"multigrid_vcycle.level_{level}.restrict_sparse",
                lambda: multigrid.mg_restrict_residual_8cell_sparse_level0[
                    coarse_blocks,
                    kernel_config.THREADS_PER_BLOCK_3D,
                ](
                    p,
                    b,
                    coarse_b,
                    delta,
                    tile_map,
                    nx,
                    ny,
                    nz,
                ),
                synchronize_cuda=True,
            )
        else:
            _profile_pressure_section(
                profile_stats,
                f"multigrid_vcycle.level_{level}.restrict_dense",
                lambda: multigrid.mg_restrict_residual_8cell[
                    coarse_blocks,
                    kernel_config.THREADS_PER_BLOCK_3D,
                ](
                    p,
                    b,
                    coarse_b,
                    delta,
                    nx,
                    ny,
                    nz,
                ),
                synchronize_cuda=True,
            )

        multigrid_vcycle(
            level + 1,
            p_levels,
            b_levels,
            p_level0,
            b_level0,
            zero_levels,
            delta_levels,
            pre_smooth,
            post_smooth,
            coarse_smooth,
            nx,
            ny,
            nz,
            tile_map=None,
            profile_stats=profile_stats,
        )

        if level == 0 and tile_map is not None:
            _profile_pressure_section(
                profile_stats,
                f"multigrid_vcycle.level_{level}.prolongate_sparse",
                lambda: multigrid.mg_prolongate_add_nearest_sparse_level0[
                    coarse_blocks,
                    kernel_config.THREADS_PER_BLOCK_3D,
                ](coarse_p, p, tile_map, (nx, ny, nz)),
                synchronize_cuda=True,
            )
        else:
            _profile_pressure_section(
                profile_stats,
                f"multigrid_vcycle.level_{level}.prolongate_dense",
                lambda: multigrid.mg_prolongate_add_nearest[
                    coarse_blocks,
                    kernel_config.THREADS_PER_BLOCK_3D,
                ](
                    coarse_p,
                    p,
                ),
                synchronize_cuda=True,
            )

        multigrid_smooth(
            p,
            b,
            delta,
            post_smooth,
            level=level,
            tile_map=tile_map,
            nx=nx,
            ny=ny,
            nz=nz,
            profile_stats=profile_stats,
            phase="post",
        )

    return _profile_pressure_section(
        profile_stats,
        f"multigrid_vcycle.level_{level}.total",
        _run_vcycle,
        synchronize_cuda=False,
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
    tile_map,
    tile_shape,
    u_initial,
    v_initial,
    w_initial,
    p_levels,
    b_levels,
    delta_levels,
    num_vcycles,
    rhs_partial_sums,
    rhs_sum_buffer,
    zero_levels,
    nx,
    ny,
    nz,
    profile_stats=None,
):
    def _run_pressure_poisson():
        _profile_pressure_section(
            profile_stats,
            "pressure_poisson_multigrid.pressure_equation_right_side",
            lambda: pressure_equation_right_side[
                tile_shape, kernel_config.THREADS_PER_BLOCK_3D
            ](
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
            ),
            synchronize_cuda=True,
        )

        _profile_pressure_section(
            profile_stats,
            "pressure_poisson_multigrid.remove_rhs_mean",
            lambda: remove_rhs_mean(
                b,
                tile_map,
                rhs_partial_sums,
                rhs_sum_buffer,
                nx,
                ny,
                nz,
                profile_stats=profile_stats,
            ),
            synchronize_cuda=False,
        )

        _profile_pressure_section(
            profile_stats,
            "pressure_poisson_multigrid.reset_inactive_pressure",
            lambda: reset_inactive_pressure[
                tile_shape, kernel_config.THREADS_PER_BLOCK_3D
            ](
                p,
                tile_map,
                nx,
                ny,
                nz,
            ),
            synchronize_cuda=True,
        )

        _profile_pressure_section(
            profile_stats,
            "pressure_poisson_multigrid.add_artifical_divergence",
            lambda: add_artifical_divergence[
                tile_shape, kernel_config.THREADS_PER_BLOCK_3D
            ](
                T,
                source_masks,
                extra_pressure,
                source_noise,
                noise_amplitudes,
                expansion_rate,
                t_reference,
                b,
                tile_map,
                rho,
                dt,
                nx,
                ny,
                nz,
            ),
            synchronize_cuda=True,
        )

        for _ in range(num_vcycles):
            multigrid_vcycle(
                0,
                p_levels,
                b_levels,
                p,
                b,
                zero_levels,
                delta_levels,
                pre_smooth=2,
                post_smooth=4,
                coarse_smooth=20,
                nx=nx,
                ny=ny,
                nz=nz,
                tile_map=tile_map,
                profile_stats=profile_stats,
            )

        return p

    return _profile_pressure_section(
        profile_stats,
        "pressure_poisson_multigrid.total",
        _run_pressure_poisson,
        synchronize_cuda=False,
    )
