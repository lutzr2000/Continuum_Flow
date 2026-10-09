#include "noise.cl"

#include "helper.cl"

__kernel void project_velocity_kernel(__global float *u,
                                      __global float *v,
                                      __global float *w,
                                      __global const float *p,
                                      __global const uchar *obstacle_mask,
                                      const float dt,
                                      const float delta,
                                      const float rho,
                                      __global const int *index_tile_map,
                                      const int nx,
                                      const int ny,
                                      const int nz,
                                      const int tiles_x,
                                      const int tiles_y,
                                      const int tiles_z) {
    /*
    In this solver the Navier-Stokes equations are solved according to Chorins projection.
    After computing a pressure field we correct the intermediate velocity field by the pressure gradient.
    The pressure gradient is computed with central differences.
    */
    const SparseCell cell = get_sparse_cell(index_tile_map, tiles_x, tiles_y, tiles_z);

    if (!cell.valid)
        return;

    const int i = cell.i;
    const int j = cell.j;
    const int k = cell.k;

    if (i < 1 || j < 1 || k < 1 || i >= nx - 1 || j >= ny - 1 || k >= nz - 1)
        return;

    const int index = cell.cell_index;

    if (obstacle_mask[index])
        return;

    const float half_inv_delta = 0.5f / delta;
    const float pressure_coeff = dt / rho;
    const float3 pressure_gradient =
        central_gradient_sparse(p, index_tile_map, i, j, k, half_inv_delta, 0.0f, tiles_y, tiles_z);

    u[index] -= pressure_coeff * pressure_gradient.x;
    v[index] -= pressure_coeff * pressure_gradient.y;
    w[index] -= pressure_coeff * pressure_gradient.z;
}

__kernel void pressure_equation_right_side(__global const float *u,
                                           __global const float *v,
                                           __global const float *w,
                                           __global float *b,
                                           const float dt,
                                           const float delta,
                                           const float rho,
                                           __global const int *index_tile_map,
                                           const float u_initial,
                                           const float v_initial,
                                           const float w_initial,
                                           const int nx,
                                           const int ny,
                                           const int nz,
                                           const int tiles_x,
                                           const int tiles_y,
                                           const int tiles_z) {
    /*
    This kernel computes the right hand side (rhs or b) of the pressure poisson equation.
    In Chorins projections this is simply the divergence of the intermediate velocity field.
    */
    const SparseCell cell = get_sparse_cell(index_tile_map, tiles_x, tiles_y, tiles_z);

    if (!cell.valid)
        return;

    const int i = cell.i;
    const int j = cell.j;
    const int k = cell.k;
    const int index = cell.cell_index;

    if (i < 1 || j < 1 || k < 1 || i >= nx - 1 || j >= ny - 1 || k >= nz - 1) {
        b[index] = 0.0f;
        return;
    }

    const float half_inv_delta = 0.5f / delta;

    const float rho_over_dt = rho / dt;

    const float du_dx = central_difference_sparse(u, index_tile_map, i, j, k, (int3)(1, 0, 0), half_inv_delta,
                                                  u_initial, tiles_y, tiles_z);
    const float dv_dy = central_difference_sparse(v, index_tile_map, i, j, k, (int3)(0, 1, 0), half_inv_delta,
                                                  v_initial, tiles_y, tiles_z);
    const float dw_dz = central_difference_sparse(w, index_tile_map, i, j, k, (int3)(0, 0, 1), half_inv_delta,
                                                  w_initial, tiles_y, tiles_z);

    b[index] = rho_over_dt * (du_dx + dv_dy + dw_dz);
}

__kernel void reset_inactive_pressure(__global float *p,
                                      __global const int *index_tile_map,
                                      const int nx,
                                      const int ny,
                                      const int nz,
                                      const int tiles_x,
                                      const int tiles_y,
                                      const int tiles_z) {
    /*
    Reset inactive pressure cells to 0
    */
    const SparseCell cell = get_sparse_cell(index_tile_map, tiles_x, tiles_y, tiles_z);

    if (!cell.valid)
        return;

    const int i = cell.i;
    const int j = cell.j;
    const int k = cell.k;

    if (i < 1 || j < 1 || k < 1 || i >= nx - 1 || j >= ny - 1 || k >= nz - 1) {
        p[cell.cell_index] = 0.0f;
    }
}

__kernel void rhs_sum_count_partial_kernel(__global const float *b,
                                           __global const int *index_tile_map,
                                           __global float *partial_sums,
                                           __global float *partial_counts,
                                           const int nx,
                                           const int ny,
                                           const int nz,
                                           const int tiles_y,
                                           const int tiles_z) {
    /*
    Helper kernel for computing the mean of the right hand side (rhs or b)
    */
    const int interior_nx = nx - 2;
    const int interior_ny = ny - 2;
    const int interior_nz = nz - 2;

    const int interior_cell_count = interior_nx * interior_ny * interior_nz;

    const int tid = get_local_id(0);

    const int global_idx = get_global_id(0);

    const int stride = get_global_size(0);

    __local float shared_sums[REDUCTION_THREADS_PER_BLOCK];

    __local float shared_counts[REDUCTION_THREADS_PER_BLOCK];

    float local_sum = 0.0f;
    float local_count = 0.0f;

    const int plane_size = interior_ny * interior_nz;

    int flat_idx = global_idx;

    while (flat_idx < interior_cell_count) {
        const int i = flat_idx / plane_size + 1;
        const int remainder = flat_idx % plane_size;
        const int j = remainder / interior_nz + 1;
        const int k = remainder % interior_nz + 1;

        const SparseCell cell = get_sparse_cell_at(index_tile_map, i, j, k, tiles_y, tiles_z);

        if (cell.valid) {
            local_sum += b[cell.cell_index];
            local_count += 1.0f;
        }

        flat_idx += stride;
    }

    shared_sums[tid] = local_sum;
    shared_counts[tid] = local_count;

    barrier(CLK_LOCAL_MEM_FENCE);

    int offset = get_local_size(0) >> 1;

    while (offset > 0) {
        if (tid < offset) {
            shared_sums[tid] += shared_sums[tid + offset];

            shared_counts[tid] += shared_counts[tid + offset];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        offset >>= 1;
    }

    if (tid == 0) {
        const int group_idx = get_group_id(0);

        partial_sums[group_idx] = shared_sums[0];

        partial_counts[group_idx] = shared_counts[0];
    }
}

__kernel void rhs_mean_kernel(__global const float *partial_sums,
                              __global const float *partial_counts,
                              const int partial_count,
                              __global float *rhs_mean) {
    /*
    Compute the mean of the rhs of the pressure poisson equation
    */
    const int tid = get_local_id(0);

    const int local_size = get_local_size(0);

    __local float shared_sums[REDUCTION_THREADS_PER_BLOCK];

    __local float shared_counts[REDUCTION_THREADS_PER_BLOCK];

    float total_sum = 0.0f;
    float total_count = 0.0f;

    int idx = tid;

    while (idx < partial_count) {
        total_sum += partial_sums[idx];

        total_count += partial_counts[idx];

        idx += local_size;
    }

    shared_sums[tid] = total_sum;

    shared_counts[tid] = total_count;

    barrier(CLK_LOCAL_MEM_FENCE);

    int offset = local_size >> 1;

    while (offset > 0) {
        if (tid < offset) {
            shared_sums[tid] += shared_sums[tid + offset];

            shared_counts[tid] += shared_counts[tid + offset];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        offset >>= 1;
    }

    if (tid == 0) {
        if (shared_counts[0] > 0.0f) {
            rhs_mean[0] = shared_sums[0] / shared_counts[0];
        } else {
            rhs_mean[0] = 0.0f;
        }
    }
}

__kernel void subtract_rhs_mean_kernel(__global float *b,
                                       __global const float *rhs_mean,
                                       __global const int *index_tile_map,
                                       const int nx,
                                       const int ny,
                                       const int nz,
                                       const int tiles_x,
                                       const int tiles_y,
                                       const int tiles_z) {
    /*
    The mean of b is substracted because the pressure only has Neumann boundary conditions.
    This makes the absolute value of p undefined hence the mean is subsrtacted to avoud "drifting"
    of the pressure field and to improve solver convergence.
    */
    const SparseCell cell = get_sparse_cell(index_tile_map, tiles_x, tiles_y, tiles_z);

    if (!cell.valid)
        return;

    const int i = cell.i;
    const int j = cell.j;
    const int k = cell.k;

    if (i < 1 || j < 1 || k < 1 || i >= nx - 1 || j >= ny - 1 || k >= nz - 1)
        return;

    b[cell.cell_index] -= rhs_mean[0];
}

__kernel void add_thermal_divergence(__global const float *T,
                                     const float expansion_rate,
                                     const float t_reference,
                                     __global float *b,
                                     __global const int *index_tile_map,
                                     const float rho,
                                     const float dt,
                                     const int nx,
                                     const int ny,
                                     const int nz,
                                     const int tiles_x,
                                     const int tiles_y,
                                     const int tiles_z) {
    /*
    This kernel adds artificial divergence based on the expansion rate and temperature difference
    to the reference temperature.
    */
    const int tile_i = get_group_id(0);
    const int tile_j = get_group_id(1);
    const int tile_k = get_group_id(2);

    const int local_i = get_local_id(0);
    const int local_j = get_local_id(1);
    const int local_k = get_local_id(2);

    if (tile_i >= tiles_x || tile_j >= tiles_y || tile_k >= tiles_z)
        return;

    const int i = tile_i * TILE_SIZE + local_i;
    const int j = tile_j * TILE_SIZE + local_j;
    const int k = tile_k * TILE_SIZE + local_k;

    const SparseCell cell = get_sparse_cell_at(index_tile_map, i, j, k, tiles_y, tiles_z);

    if (!cell.valid)
        return;

    if (i < 1 || j < 1 || k < 1 || i >= nx - 1 || j >= ny - 1 || k >= nz - 1)
        return;

    const int index = cell.cell_index;

    b[index] -= (rho / dt) * expansion_rate * (T[index] - t_reference);
}

__kernel void add_source_extra_pressure(__global const uchar *source_mask,
                                        const float source_extra_pressure,
                                        const float randomness_scale,
                                        const int randomness_seed,
                                        const float pressure_randomness,
                                        __global float *b,
                                        __global const int *index_tile_map,
                                        const float rho,
                                        const int nx,
                                        const int ny,
                                        const int nz,
                                        const float dt,
                                        const float delta,
                                        const float origin_x,
                                        const float origin_y,
                                        const float origin_z,
                                        const int tiles_x,
                                        const int tiles_y,
                                        const int tiles_z) {
    /*
    This kernel adds extra divergence ("pressure" is technically not 100% correct here) to
    the flow.
    */
    const int tile_i = get_group_id(0);
    const int tile_j = get_group_id(1);
    const int tile_k = get_group_id(2);

    const int local_i = get_local_id(0);
    const int local_j = get_local_id(1);
    const int local_k = get_local_id(2);

    if (tile_i >= tiles_x || tile_j >= tiles_y || tile_k >= tiles_z)
        return;

    const int i = tile_i * TILE_SIZE + local_i;
    const int j = tile_j * TILE_SIZE + local_j;
    const int k = tile_k * TILE_SIZE + local_k;

    const SparseCell cell = get_sparse_cell_at(index_tile_map, i, j, k, tiles_y, tiles_z);

    if (!cell.valid)
        return;

    if (i < 1 || j < 1 || k < 1 || i >= nx - 1 || j >= ny - 1 || k >= nz - 1)
        return;

    const int index = cell.cell_index;

    float extra_pressure_term = 0.0f;

    if (source_mask[index]) {
        float noise = 0.0f;
        if (pressure_randomness != 0.0f) {
            noise = gradient_noise_3d(origin_x + (float)i * delta, origin_y + (float)j * delta,
                                      origin_z + (float)k * delta, randomness_seed, randomness_scale);
        }
        const float multiplier = noise_amplitude_multiplier(noise, pressure_randomness);
        extra_pressure_term = source_extra_pressure * multiplier;
    }

    b[index] -= (rho / dt) * extra_pressure_term;
}
