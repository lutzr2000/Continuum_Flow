#ifndef TILE_SIZE
#define TILE_SIZE 4
#endif

#include "helper.cl"

__kernel void build_coarse_tile_level(__global const int *fine_tile_map,
                                      __global int *coarse_tile_map,
                                      __global int *coarse_active_tiles,
                                      __global int *coarse_active_tile_count,
                                      const int coarse_active_tile_capacity,
                                      const int fine_tiles_x,
                                      const int fine_tiles_y,
                                      const int fine_tiles_z,
                                      const int coarse_tiles_x,
                                      const int coarse_tiles_y,
                                      const int coarse_tiles_z) {
    /*
    Build the sparse tile structure for coarser multigrid levels
    */
    const int coarse_i = get_global_id(0);
    const int coarse_j = get_global_id(1);
    const int coarse_k = get_global_id(2);

    if (coarse_i >= coarse_tiles_x || coarse_j >= coarse_tiles_y || coarse_k >= coarse_tiles_z)
        return;

    const int fine_i_start = coarse_i * 2;
    const int fine_j_start = coarse_j * 2;
    const int fine_k_start = coarse_k * 2;

    int is_active = 0;

    for (int offset_i = 0; offset_i < 2; ++offset_i) {
        const int fine_i = fine_i_start + offset_i;

        if (fine_i >= fine_tiles_x)
            continue;

        for (int offset_j = 0; offset_j < 2; ++offset_j) {
            const int fine_j = fine_j_start + offset_j;

            if (fine_j >= fine_tiles_y)
                continue;

            for (int offset_k = 0; offset_k < 2; ++offset_k) {
                const int fine_k = fine_k_start + offset_k;

                if (fine_k >= fine_tiles_z)
                    continue;

                const int fine_index = (fine_i * fine_tiles_y + fine_j) * fine_tiles_z + fine_k;

                if (fine_tile_map[fine_index] != -1) {
                    is_active = 1;
                    break;
                }
            }

            if (is_active)
                break;
        }

        if (is_active)
            break;
    }

    if (!is_active)
        return;

    const int pool_index = atomic_add((volatile __global int *)coarse_active_tile_count, 1);

    if (pool_index >= coarse_active_tile_capacity)
        return;

    const int coarse_index = (coarse_i * coarse_tiles_y + coarse_j) * coarse_tiles_z + coarse_k;

    coarse_tile_map[coarse_index] = pool_index;

    const int active_index = pool_index * 3;

    coarse_active_tiles[active_index + 0] = coarse_i;
    coarse_active_tiles[active_index + 1] = coarse_j;
    coarse_active_tiles[active_index + 2] = coarse_k;
}

inline float residual_sparse(__global const float *p,
                             __global const float *b,
                             const float inv_delta2,
                             __global const int *index_tile_map,
                             const int i,
                             const int j,
                             const int k,
                             const int nx,
                             const int ny,
                             const int nz,
                             const int tiles_y,
                             const int tiles_z,
                             int *valid) {
    /*
    Compute the residual of the pressure equation. In general we solve laplace(p)=b.
    In discretized form this is A*p = b with A beeing the coefficent matrix. Rearranging yields
    the residual r = b - A*p. In the code the right hand side (rhs) is b and A*p is the discrete
    laplacian with central differences.
    */
    const SparseCell cell = get_sparse_cell_at(index_tile_map, i, j, k, tiles_y, tiles_z);

    if (!cell.valid) {
        *valid = 0;
        return 0.0f;
    }

    const int index = cell.cell_index;

    const float laplace = laplacian_sparse(p, index_tile_map, i, j, k, inv_delta2, 0.0f,
                                           nx, ny, nz, tiles_y, tiles_z);

    const float rhs = b[index];

    *valid = 1;

    return rhs - laplace;
}

__kernel void restrict_residual_sparse(__global const float *fine_p,
                                       __global const float *fine_b,
                                       __global float *coarse_p,
                                       __global float *coarse_b,
                                       const float fine_delta,
                                       __global const int *fine_tile_map,
                                       __global const int *coarse_tile_map,
                                       __global const int *coarse_active_tiles,
                                       __global const int *coarse_active_tile_count,
                                       const int fine_nx,
                                       const int fine_ny,
                                       const int fine_nz,
                                       const int coarse_nx,
                                       const int coarse_ny,
                                       const int coarse_nz,
                                       const int fine_tiles_y,
                                       const int fine_tiles_z,
                                       const int coarse_tiles_y,
                                       const int coarse_tiles_z) {
    /*
    When moving between grid levels a transfer from fine to coarse is done. This is called restriciton.
    In this case a coarser cell contains 8 (2x2x2) finer cells. There residual is averaged for the residual
    on the coarser level
    */
    const int coarse_pool_index = get_group_id(0);

    if (coarse_pool_index >= coarse_active_tile_count[0])
        return;

    const int local_i = get_local_id(0);
    const int local_j = get_local_id(1);
    const int local_k = get_local_id(2);

    const int active_index = coarse_pool_index * 3;

    const int coarse_tile_i = coarse_active_tiles[active_index + 0];
    const int coarse_tile_j = coarse_active_tiles[active_index + 1];
    const int coarse_tile_k = coarse_active_tiles[active_index + 2];

    const int I = coarse_tile_i * TILE_SIZE + local_i;
    const int J = coarse_tile_j * TILE_SIZE + local_j;
    const int K = coarse_tile_k * TILE_SIZE + local_k;

    if (I >= coarse_nx || J >= coarse_ny || K >= coarse_nz)
        return;

    const SparseCell coarse_cell = get_sparse_cell_at(coarse_tile_map, I, J, K, coarse_tiles_y, coarse_tiles_z);

    if (!coarse_cell.valid)
        return;

    const float inv_delta2 = 1.0f / (fine_delta * fine_delta);

    const int fine_i_start = 2 * I;
    const int fine_j_start = 2 * J;
    const int fine_k_start = 2 * K;

    float residual_sum = 0.0f;
    float residual_count = 0.0f;

    for (int offset_i = 0; offset_i < 2; ++offset_i) {
        for (int offset_j = 0; offset_j < 2; ++offset_j) {
            for (int offset_k = 0; offset_k < 2; ++offset_k) {
                const int i = fine_i_start + offset_i;
                const int j = fine_j_start + offset_j;
                const int k = fine_k_start + offset_k;

                if (i < 1 || j < 1 || k < 1 || i >= fine_nx - 1 || j >= fine_ny - 1 || k >= fine_nz - 1)
                    continue;

                int valid;

                const float residual_value =
                    residual_sparse(fine_p, fine_b, inv_delta2, fine_tile_map, i, j, k, fine_nx, fine_ny, fine_nz,
                                    fine_tiles_y, fine_tiles_z, &valid);

                if (!valid)
                    continue;

                residual_sum += residual_value;
                residual_count += 1.0f;
            }
        }
    }

    const int index = coarse_cell.cell_index;

    coarse_p[index] = 0.0f;

    if (residual_count > 0.0f) {
        coarse_b[index] = residual_sum / residual_count;
    } else {
        coarse_b[index] = 0.0f;
    }
}

__kernel void prolongate_sparse(__global const float *coarse_e,
                                __global float *fine_p,
                                __global const int *coarse_tile_map,
                                __global const int *fine_tile_map,
                                __global const int *coarse_active_tiles,
                                __global const int *coarse_active_tile_count,
                                const int coarse_nx,
                                const int coarse_ny,
                                const int coarse_nz,
                                const int fine_nx,
                                const int fine_ny,
                                const int fine_nz,
                                const int coarse_tiles_y,
                                const int coarse_tiles_z,
                                const int fine_tiles_y,
                                const int fine_tiles_z) {
    /*
    Now we have the opposite situation from restrict_residual_sparse. We now want to move
    from coarse to fine. We computed an error on the coarser level and now want to correct
    by this error on the finer level.
    */
    const int coarse_pool_index = get_group_id(0);

    if (coarse_pool_index >= coarse_active_tile_count[0])
        return;

    const int local_i = get_local_id(0);
    const int local_j = get_local_id(1);
    const int local_k = get_local_id(2);

    const int active_index = coarse_pool_index * 3;

    const int coarse_tile_i = coarse_active_tiles[active_index + 0];
    const int coarse_tile_j = coarse_active_tiles[active_index + 1];
    const int coarse_tile_k = coarse_active_tiles[active_index + 2];

    const int I = coarse_tile_i * TILE_SIZE + local_i;
    const int J = coarse_tile_j * TILE_SIZE + local_j;
    const int K = coarse_tile_k * TILE_SIZE + local_k;

    if (I >= coarse_nx || J >= coarse_ny || K >= coarse_nz)
        return;

    const SparseCell coarse_cell = get_sparse_cell_at(coarse_tile_map, I, J, K, coarse_tiles_y, coarse_tiles_z);

    if (!coarse_cell.valid)
        return;

    const float error =
        0.25f * coarse_e[coarse_cell.cell_index]; // !!! The *0.25 is only there for stability reasons, the multigird
                                                  // diverges with higher values, mathematical default would be 1 !!!

    const int fine_i_start = 2 * I;
    const int fine_j_start = 2 * J;
    const int fine_k_start = 2 * K;

    for (int offset_i = 0; offset_i < 2; ++offset_i) {
        for (int offset_j = 0; offset_j < 2; ++offset_j) {
            for (int offset_k = 0; offset_k < 2; ++offset_k) {
                const int i = fine_i_start + offset_i;
                const int j = fine_j_start + offset_j;
                const int k = fine_k_start + offset_k;

                if (i >= fine_nx || j >= fine_ny || k >= fine_nz)
                    continue;

                const SparseCell fine_cell = get_sparse_cell_at(fine_tile_map, i, j, k, fine_tiles_y, fine_tiles_z);

                if (!fine_cell.valid)
                    continue;

                fine_p[fine_cell.cell_index] += error;
            }
        }
    }
}

__kernel void rbgs_step_sparse(__global float *p,
                               __global const float *b,
                               const float delta,
                               const int parity,
                               __global const int *index_tile_map,
                               __global const int *active_tiles,
                               __global const int *active_tile_count,
                               const int nx,
                               const int ny,
                               const int nz,
                               const int tiles_y,
                               const int tiles_z) {
    /*
    Perform a red black Gauss Seidel step. The equation is obtained by rearanging
    the discrete version of laplace(p) = b. Red and black Gauss Seidel updates
    in a checkerboard pattern. In a red run all values are independent and can be
    computed in parallel, same in the black case.
    */
    const int active_index = get_group_id(0);

    if (active_index >= active_tile_count[0])
        return;

    const int local_i = get_local_id(0);
    const int local_j = get_local_id(1);
    const int local_k = get_local_id(2);

    const int active_tile_index = active_index * 3;

    const int tile_i = active_tiles[active_tile_index + 0];
    const int tile_j = active_tiles[active_tile_index + 1];
    const int tile_k = active_tiles[active_tile_index + 2];

    const int i = tile_i * TILE_SIZE + local_i;
    const int j = tile_j * TILE_SIZE + local_j;
    const int k = tile_k * TILE_SIZE + local_k;

    const SparseCell cell = get_sparse_cell_at(index_tile_map, i, j, k, tiles_y, tiles_z);

    if (!cell.valid)
        return;

    if (i < 1 || j < 1 || k < 1 || i >= nx - 1 || j >= ny - 1 || k >= nz - 1)
        return;

    if (((i + j + k) & 1) != parity)
        return;

    const float delta2 = delta * delta;

    const int index = cell.cell_index;

    const float neighbour_sum = neighbor_sum_sparse(p, index_tile_map, i, j, k, 0.0f,
                                                     nx, ny, nz, tiles_y, tiles_z);

    p[index] = (neighbour_sum - delta2 * b[index]) / 6.0f;
}

__kernel void coarse_smooth(__global float *p,
                            __global const float *b,
                            const float delta,
                            __global const int *index_tile_map,
                            __global const int *active_tiles,
                            __global const int *active_tile_count,
                            const int active_tile_capacity,
                            const int iterations,
                            const int nx,
                            const int ny,
                            const int nz,
                            const int tiles_y,
                            const int tiles_z) {
    /*
    Perform multiple Red-Black Gauss-Seidel iterations on the coarsest
    multigrid level within a single kernel launch to reduce overhead.
    The entire coarse grid is processed by a single work-group,
    allowing barriers to synchronize all participating work-items
    between red and black updates.
    */
    const int thread_index = get_local_id(0);
    const int thread_count = get_local_size(0);
    const int tile_count = min(active_tile_count[0], active_tile_capacity);
    const int cells_per_tile = TILE_SIZE * TILE_SIZE * TILE_SIZE;
    const int cell_count = tile_count * cells_per_tile;
    const float delta2 = delta * delta;

    for (int iteration = 0; iteration < iterations; ++iteration) {
        for (int parity = 0; parity < 2; ++parity) {
            for (int cell = thread_index; cell < cell_count; cell += thread_count) {
                const int active_index = cell / cells_per_tile;
                const int local_index = cell - active_index * cells_per_tile;
                const int local_i = local_index / (TILE_SIZE * TILE_SIZE);
                const int local_j = (local_index / TILE_SIZE) % TILE_SIZE;
                const int local_k = local_index % TILE_SIZE;
                const int active_tile_index = active_index * 3;
                const int tile_i = active_tiles[active_tile_index + 0];
                const int tile_j = active_tiles[active_tile_index + 1];
                const int tile_k = active_tiles[active_tile_index + 2];
                const int i = tile_i * TILE_SIZE + local_i;
                const int j = tile_j * TILE_SIZE + local_j;
                const int k = tile_k * TILE_SIZE + local_k;

                if (i < 1 || j < 1 || k < 1 || i >= nx - 1 || j >= ny - 1 || k >= nz - 1 || ((i + j + k) & 1) != parity)
                    continue;

                const SparseCell sparse_cell = get_sparse_cell_at(index_tile_map, i, j, k, tiles_y, tiles_z);

                if (!sparse_cell.valid)
                    continue;

                const int index = sparse_cell.cell_index;

                const float neighbour_sum = neighbor_sum_sparse(p, index_tile_map, i, j, k, 0.0f,
                                                                 nx, ny, nz, tiles_y, tiles_z);

                p[index] = (neighbour_sum - delta2 * b[index]) / 6.0f;
            }

            barrier(CLK_GLOBAL_MEM_FENCE);
        }
    }
}

__kernel void rbgs_step_level_0(__global float *p,
                                __global const float *b,
                                const float delta,
                                const int parity,
                                __global const int *index_tile_map,
                                const int nx,
                                const int ny,
                                const int nz,
                                const int tiles_x,
                                const int tiles_y,
                                const int tiles_z) {
    /*
    Essentially the same as rbgs_step_sparse just for the coarsest level.
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

    if (i < 1 || j < 1 || k < 1 || i >= nx - 1 || j >= ny - 1 || k >= nz - 1 || ((i + j + k) & 1) != parity)
        return;

    const float delta2 = delta * delta;

    const int index = cell.cell_index;

    const float neighbour_sum = neighbor_sum_sparse(p, index_tile_map, i, j, k, 0.0f,
                                                     nx, ny, nz, tiles_y, tiles_z);
    const float center = (neighbour_sum -
                          delta2 * get_pool_value(b, index_tile_map, i, j, k, 0.0f, tiles_y, tiles_z)) /
                         6.0f;

    p[index] = center;
}

__kernel void pressure_poisson_neumann(__global float *p,
                                       __global const int *index_tile_map,
                                       const int nx,
                                       const int ny,
                                       const int nz,
                                       const int tiles_y,
                                       const int tiles_z) {
    /*
    Apply Neumann boundary conditions for pressure on the six domain sides.
    */
    const int boundary_index = get_global_id(0);
    const int x_face_count = 2 * ny * nz;
    const int y_face_count = 2 * max(nx - 2, 0) * nz;
    const int z_face_count = 2 * max(nx - 2, 0) * max(ny - 2, 0);
    const int boundary_cell_count = x_face_count + y_face_count + z_face_count;

    if (boundary_index >= boundary_cell_count)
        return;

    int i;
    int j;
    int k;

    /* Assign every boundary cell to exactly one work-item. Edges belong to
       the X faces and the remaining Z-face edges belong to the Y faces. */
    if (boundary_index < x_face_count) {
        const int face_area = ny * nz;
        const int face_index = boundary_index % face_area;
        i = boundary_index < face_area ? 0 : nx - 1;
        j = face_index / nz;
        k = face_index - j * nz;
    } else if (boundary_index < x_face_count + y_face_count) {
        const int local_index = boundary_index - x_face_count;
        const int face_area = (nx - 2) * nz;
        const int face_index = local_index % face_area;
        i = 1 + face_index / nz;
        j = local_index < face_area ? 0 : ny - 1;
        k = face_index - (i - 1) * nz;
    } else {
        const int local_index = boundary_index - x_face_count - y_face_count;
        const int face_area = (nx - 2) * (ny - 2);
        const int face_index = local_index % face_area;
        i = 1 + face_index / (ny - 2);
        j = 1 + face_index - (i - 1) * (ny - 2);
        k = local_index < face_area ? 0 : nz - 1;
    }

    const SparseCell cell = get_sparse_cell_at(index_tile_map, i, j, k, tiles_y, tiles_z);

    if (!cell.valid)
        return;

    const int source_i = clamp(i, 1, nx - 2);
    const int source_j = clamp(j, 1, ny - 2);
    const int source_k = clamp(k, 1, nz - 2);

    p[cell.cell_index] = get_pool_value(p, index_tile_map, source_i, source_j, source_k, 0.0f, tiles_y, tiles_z);
}
