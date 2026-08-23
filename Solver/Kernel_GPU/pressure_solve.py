import numpy as np
from numba import cuda

import Solver.Kernel_GPU.kernel_config as kernel_config
import Solver.Kernel_GPU.sparse_managment as sparse_managment
import Solver.Kernel_GPU.multigrid as multigrid

GPU_FIELD_DTYPE = kernel_config.GPU_FIELD_DTYPE

REDUCTION_THREADS_PER_BLOCK = (
    kernel_config.REDUCTION_THREADS_PER_BLOCK
)  # this is needed because if this is added directly inline, cuda crashes i do not know why


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
        sparse_managment.get_pool_value(u, tile_map, i + 1, j, k, u_initial)
        - sparse_managment.get_pool_value(u, tile_map, i - 1, j, k, u_initial)
    ) * half_inv_delta

    dv_dy = (
        sparse_managment.get_pool_value(v, tile_map, i, j + 1, k, v_initial)
        - sparse_managment.get_pool_value(v, tile_map, i, j - 1, k, v_initial)
    ) * half_inv_delta

    dw_dz = (
        sparse_managment.get_pool_value(w, tile_map, i, j, k + 1, w_initial)
        - sparse_managment.get_pool_value(w, tile_map, i, j, k - 1, w_initial)
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
        dtype=GPU_FIELD_DTYPE,
    )

    local_sum = GPU_FIELD_DTYPE(0.0)
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
        dtype=GPU_FIELD_DTYPE,
    )

    local_count = GPU_FIELD_DTYPE(0.0)
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
            local_count += GPU_FIELD_DTYPE(1.0)

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
        dtype=GPU_FIELD_DTYPE,
    )

    local_sum = GPU_FIELD_DTYPE(0.0)
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
):
    """
    Enforce the Neumann compatibility condition by removing
    the mean RHS over active interior cells.
    """

    interior_cell_count = max((nx - 2) * (ny - 2) * (nz - 2), 1)

    reduction_blocks = kernel_config.reduction_blocks_per_grid(interior_cell_count)

    blockspergrid_3d = kernel_config.volume_blocks_per_grid(
        (nx, ny, nz),
        kernel_config.THREADS_PER_BLOCK_3D,
    )

    sum_rhs_partial_kernel[reduction_blocks, REDUCTION_THREADS_PER_BLOCK](
        b,
        tile_map,
        rhs_partial_sums,
        nx,
        ny,
        nz,
    )

    sum_partial_sums_kernel[1, REDUCTION_THREADS_PER_BLOCK](
        rhs_partial_sums,
        reduction_blocks,
        rhs_sum_buffer,
    )

    rhs_sum = float(rhs_sum_buffer.copy_to_host()[0])

    count_rhs_active_partial_kernel[reduction_blocks, REDUCTION_THREADS_PER_BLOCK](
        b,
        tile_map,
        rhs_partial_sums,
        nx,
        ny,
        nz,
    )

    sum_partial_sums_kernel[1, REDUCTION_THREADS_PER_BLOCK](
        rhs_partial_sums,
        reduction_blocks,
        rhs_sum_buffer,
    )

    active_cell_count = int(rhs_sum_buffer.copy_to_host()[0])

    if active_cell_count <= 0:
        return

    rhs_mean = rhs_sum / float(active_cell_count)

    if abs(rhs_mean) <= 1.0e-12:
        return

    subtract_rhs_mean_kernel[
        blockspergrid_3d,
        kernel_config.THREADS_PER_BLOCK_3D,
    ](
        b,
        rhs_mean,
        tile_map,
        nx,
        ny,
        nz,
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

    px1 = sparse_managment.get_pool_value(p, tile_map, i + 1, j, k, 0.0)
    px0 = sparse_managment.get_pool_value(p, tile_map, i - 1, j, k, 0.0)
    py1 = sparse_managment.get_pool_value(p, tile_map, i, j + 1, k, 0.0)
    py0 = sparse_managment.get_pool_value(p, tile_map, i, j - 1, k, 0.0)
    pz1 = sparse_managment.get_pool_value(p, tile_map, i, j, k + 1, 0.0)
    pz0 = sparse_managment.get_pool_value(p, tile_map, i, j, k - 1, 0.0)

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
    delta,
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

    b[tile_index, local_i, local_j, local_k] -= rho/delta * (
        thermal_divergence + extra_pressure_term
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
):
    pressure_equation_right_side[tile_shape, kernel_config.THREADS_PER_BLOCK_3D](
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

    reset_inactive_pressure[tile_shape, kernel_config.THREADS_PER_BLOCK_3D](
        p,
        tile_map,
        nx,
        ny,
        nz,
    )

    add_artifical_divergence[tile_shape, kernel_config.THREADS_PER_BLOCK_3D](
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
        delta,
        nx,
        ny,
        nz,
    )

    remove_rhs_mean(
        b,
        tile_map,
        rhs_partial_sums,
        rhs_sum_buffer,
        nx,
        ny,
        nz,
    )


    for _ in range(num_vcycles):
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
