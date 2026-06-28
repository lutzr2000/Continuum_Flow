import numpy as np
from numba import cuda

import Solver.Kernel_GPU.Boundary_Conditions.domain_bc as BC
import Solver.Kernel_GPU.kernel_config as kernel_config
from Solver.Kernel_GPU.scalar_update import _active_tile_cell_indices

REDUCTION_THREADS_PER_BLOCK = kernel_config.REDUCTION_THREADS_PER_BLOCK


@cuda.jit(device=True, inline=True, cache=True)
def _is_active_pressure_cell(active_tile_mask, i, j, k):
    tile_size = kernel_config.ACTIVE_TILE_SIZE
    return active_tile_mask[i // tile_size, j // tile_size, k // tile_size] != 0


@cuda.jit(cache=True)
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
    CUDA kernel that computes the right hand side of the pressure Poisson equation.
    Only the divergence of the velociy field is used, we neglect non linear terms.

    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = u.shape

    if i >= nx or j >= ny or k >= nz:
        return

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
        return

    half_inv_delta = 0.5 / delta
    rho_over_dt = rho / dt

    # ------------Derivatives on main diagonal-------------------
    du_dx = (u[i + 1, j, k] - u[i - 1, j, k]) * half_inv_delta
    dv_dy = (v[i, j + 1, k] - v[i, j - 1, k]) * half_inv_delta
    dw_dz = (w[i, j, k + 1] - w[i, j, k - 1]) * half_inv_delta

    divergence = du_dx + dv_dy + dw_dz

    # ------------Right hand side-------------------
    b[i, j, k] = rho_over_dt * divergence


@cuda.jit(cache=True)
def _sum_rhs_partial_kernel(b, active_tile_mask, partial_sums):
    """
    Reduce the interior RHS into one partial sum per CUDA block. This is needed
    for computing RHS mean.
    """
    nx, ny, nz = b.shape
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
        if _is_active_pressure_cell(active_tile_mask, i, j, k):
            local_sum += b[i, j, k]
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
def _count_rhs_active_partial_kernel(b, active_tile_mask, partial_counts):
    """
    Reduce the number of active RHS cells into one partial count per CUDA block.
    """
    nx, ny, nz = b.shape
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
        if _is_active_pressure_cell(active_tile_mask, i, j, k):
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
def _sum_partial_sums_kernel(partial_sums, partial_count, rhs_sum):
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
def _subtract_rhs_mean_kernel(b, rhs_mean, active_tile_mask):
    """
    Subtract the interior RHS mean from interior cells only.
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = b.shape

    if (
        i < 1
        or j < 1
        or k < 1
        or i >= nx - 1
        or j >= ny - 1
        or k >= nz - 1
        or not _is_active_pressure_cell(active_tile_mask, i, j, k)
    ):
        return

    b[i, j, k] -= rhs_mean


@cuda.jit(cache=True)
def _reset_inactive_pressure_kernel(p, active_tile_mask):
    i, j, k = cuda.grid(3)
    nx, ny, nz = p.shape

    if i >= nx or j >= ny or k >= nz:
        return

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
    rhs_partial_sums=None,
    rhs_sum_buffer=None,
):
    """
    This is a wrapper method for enforcing the Neumann compatibility condition by removing the RHS mean.

    """
    nx, ny, nz = b.shape
    interior_cell_count = max((nx - 2) * (ny - 2) * (nz - 2), 1)
    reduction_blocks = kernel_config.reduction_blocks_per_grid(interior_cell_count)
    reduction_threads = REDUCTION_THREADS_PER_BLOCK

    if rhs_partial_sums is None:
        rhs_partial_sums = cuda.device_array(reduction_blocks, dtype=np.float32)
    if rhs_sum_buffer is None:
        rhs_sum_buffer = cuda.device_array(1, dtype=np.float32)

    _sum_rhs_partial_kernel[reduction_blocks, reduction_threads](
        b, active_tile_mask, rhs_partial_sums
    )
    _sum_partial_sums_kernel[1, reduction_threads](
        rhs_partial_sums, reduction_blocks, rhs_sum_buffer
    )
    rhs_sum = float(rhs_sum_buffer.copy_to_host()[0])

    _count_rhs_active_partial_kernel[reduction_blocks, reduction_threads](
        b, active_tile_mask, rhs_partial_sums
    )
    _sum_partial_sums_kernel[1, reduction_threads](
        rhs_partial_sums, reduction_blocks, rhs_sum_buffer
    )
    active_cell_count = int(rhs_sum_buffer.copy_to_host()[0])
    if active_cell_count <= 0:
        return

    rhs_mean = rhs_sum / float(active_cell_count)
    if abs(rhs_mean) <= 1.0e-12:
        return

    blockspergrid_3d = kernel_config.volume_blocks_per_grid(b.shape, threadsperblock_3d)
    _subtract_rhs_mean_kernel[blockspergrid_3d, threadsperblock_3d](
        b, rhs_mean, active_tile_mask
    )


@cuda.jit(cache=True)
def _pressure_poisson_red_black_gauss_seidel_step(
    p, b, delta, parity, active_tile_mask
):
    """
    Perform one in-place red-black Gauss-Seidel color pass of the 3D pressure Poisson
    equation on the GPU.

    Only interior cells whose `(i + j + k) % 2` matches `parity` are updated in
    this launch. Each update computes the red-black Gauss-Seidel value and
    stores it in place.

    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = p.shape

    if i >= nx or j >= ny or k >= nz:
        return

    if (
        i < 1
        or j < 1
        or k < 1
        or i >= nx - 1
        or j >= ny - 1
        or k >= nz - 1
        or ((i + j + k) & 1) != parity
        or not _is_active_pressure_cell(active_tile_mask, i, j, k)
    ):
        return

    delta2 = delta * delta
    # nabla^2p = b
    p[i, j, k] = (
        p[i + 1, j, k]
        + p[i - 1, j, k]
        + p[i, j + 1, k]
        + p[i, j - 1, k]
        + p[i, j, k + 1]
        + p[i, j, k - 1]
        - delta2 * b[i, j, k]
    ) / 6.0


@cuda.jit(cache=True)
def project_velocity_kernel(
    u, v, w, p, obstacle_mask, dt, delta, rho, active_tile_mask
):
    """
    Apply the pressure projection `u <- u - dt/rho * grad(p)` to one interior cell.

    Obstacle cells are skipped because their wall velocities are restored by the
    obstacle boundary conditions after the projection pass.
    """
    tile_i, tile_j, tile_k, i, j, k, nx, ny, nz = _active_tile_cell_indices(u.shape)

    if (
        tile_i >= active_tile_mask.shape[0]
        or tile_j >= active_tile_mask.shape[1]
        or tile_k >= active_tile_mask.shape[2]
    ):
        return
    if i < 1 or j < 1 or k < 1 or i >= nx - 1 or j >= ny - 1 or k >= nz - 1:
        return

    if active_tile_mask[tile_i, tile_j, tile_k] == 0 or obstacle_mask[i, j, k]:
        return

    pressure_coeff = dt / (2.0 * rho * delta)
    # u <= u - dp/delta * dt/(2*rho)
    u[i, j, k] -= pressure_coeff * (p[i + 1, j, k] - p[i - 1, j, k])
    v[i, j, k] -= pressure_coeff * (p[i, j + 1, k] - p[i, j - 1, k])
    w[i, j, k] -= pressure_coeff * (p[i, j, k + 1] - p[i, j, k - 1])


@cuda.jit(cache=True)
def add_artifical_divergence(
    T,
    source_masks,
    extra_pressure,
    expansion_rate,
    t_reference,
    b,
    active_tile_mask,
    rho,
    dt
):
    i, j, k = cuda.grid(3)
    nx, ny, nz = b.shape

    if i >= nx or j >= ny or k >= nz:
        return

    if (
        i < 1
        or j < 1
        or k < 1
        or i >= nx - 1
        or j >= ny - 1
        or k >= nz - 1
        or not _is_active_pressure_cell(active_tile_mask, i, j, k)
    ):
        return
    
    rho_over_dt = rho / dt

    thermal_divergence = expansion_rate * (T[i, j, k] - t_reference)
    extra_pressure_term = 0.0
    source_count = source_masks.shape[0]
    for source_idx in range(source_count):
        if not source_masks[source_idx, i, j, k]:
            continue
        source_extra_pressure = extra_pressure[source_idx]
        if source_extra_pressure > extra_pressure_term:
            extra_pressure_term = source_extra_pressure


    b[i, j, k] -= rho_over_dt * (thermal_divergence + extra_pressure_term)



def pressure_poisson(
    u,
    v,
    w,
    p,
    T,
    b,
    dt,
    source_masks,
    extra_pressure,
    delta,
    rho,
    expansion_rate,
    t_reference,
    active_tile_mask,
    max_iter=10,
    threadsperblock_3d=None,
    rhs_partial_sums=None,
    rhs_sum_buffer=None,
):
    """
    Host-side pressure Poisson solve that launches CUDA kernels for the RHS,
    red-black Gauss-Seidel iterations and Neumann boundary conditions.

    """
    if threadsperblock_3d is None:
        threadsperblock_3d = kernel_config.THREADS_PER_BLOCK_3D
    blockspergrid_3d = kernel_config.volume_blocks_per_grid(u.shape, threadsperblock_3d)

    pressure_equation_right_side[blockspergrid_3d, threadsperblock_3d](
        u,
        v,
        w,
        b,
        dt,
        delta,
        rho,
        active_tile_mask,
    )
    _remove_rhs_mean(
        b,
        active_tile_mask,
        threadsperblock_3d,
        rhs_partial_sums,
        rhs_sum_buffer,
    )
    p_old = p
    _reset_inactive_pressure_kernel[blockspergrid_3d, threadsperblock_3d](
        p_old, active_tile_mask
    )

    add_artifical_divergence[blockspergrid_3d, threadsperblock_3d](
        T,
        source_masks,
        extra_pressure,
        expansion_rate,
        t_reference,
        b,
        active_tile_mask,
        rho,
        dt
    )

    for _ in range(max_iter):
        _pressure_poisson_red_black_gauss_seidel_step[
            blockspergrid_3d, threadsperblock_3d
        ](p_old, b, delta, 0, active_tile_mask)
        _pressure_poisson_red_black_gauss_seidel_step[
            blockspergrid_3d, threadsperblock_3d
        ](p_old, b, delta, 1, active_tile_mask)
        BC._pressure_poisson_apply_neumann_bcs[blockspergrid_3d, threadsperblock_3d](
            p_old
        )

    return p_old



