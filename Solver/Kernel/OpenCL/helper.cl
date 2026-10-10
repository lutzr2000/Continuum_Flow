#ifndef HELPER_CL
#define HELPER_CL

#include "sparse_managment.cl"

inline float central_difference_sparse(__global const float *field,
                                       __global const int *index_tile_map,
                                       const int i,
                                       const int j,
                                       const int k,
                                       const int3 direction,
                                       const float half_inv_delta,
                                       const float default_value,
                                       const int tiles_y,
                                       const int tiles_z) {
    /*
    Compute a central difference along the supplied grid direction.
    */
    const float upper = get_pool_value(field, index_tile_map, i + direction.x, j + direction.y, k + direction.z,
                                       default_value, tiles_y, tiles_z);
    const float lower = get_pool_value(field, index_tile_map, i - direction.x, j - direction.y, k - direction.z,
                                       default_value, tiles_y, tiles_z);

    return (upper - lower) * half_inv_delta;
}

inline float neighbor_sum_sparse(__global const float *field,
                                 __global const int *index_tile_map,
                                 const int i,
                                 const int j,
                                 const int k,
                                 const float default_value,
                                 const int nx,
                                 const int ny,
                                 const int nz,
                                 const int tiles_y,
                                 const int tiles_z) {
    /*
    Sum the six axis-aligned neighbours used by the 3D seven-point stencil.
    Clamping to the closest interior cell implements the pressure solver's
    homogeneous Neumann condition at the domain boundary.
    */
    return get_pool_value(field, index_tile_map, clamp(i + 1, 1, nx - 2), j, k, default_value, tiles_y, tiles_z) +
           get_pool_value(field, index_tile_map, clamp(i - 1, 1, nx - 2), j, k, default_value, tiles_y, tiles_z) +
           get_pool_value(field, index_tile_map, i, clamp(j + 1, 1, ny - 2), k, default_value, tiles_y, tiles_z) +
           get_pool_value(field, index_tile_map, i, clamp(j - 1, 1, ny - 2), k, default_value, tiles_y, tiles_z) +
           get_pool_value(field, index_tile_map, i, j, clamp(k + 1, 1, nz - 2), default_value, tiles_y, tiles_z) +
           get_pool_value(field, index_tile_map, i, j, clamp(k - 1, 1, nz - 2), default_value, tiles_y, tiles_z);
}

inline float laplacian_sparse(__global const float *field,
                              __global const int *index_tile_map,
                              const int i,
                              const int j,
                              const int k,
                              const float inv_delta2,
                              const float default_value,
                              const int nx,
                              const int ny,
                              const int nz,
                              const int tiles_y,
                              const int tiles_z) {
    /*
    Compute the 3D seven-point discrete Laplacian.
    */
    const SparseCell cell = get_sparse_cell_at(index_tile_map, i, j, k, tiles_y, tiles_z);

    if (!cell.valid)
        return 0.0f;

    const float neighbour_sum =
        neighbor_sum_sparse(field, index_tile_map, i, j, k, default_value, nx, ny, nz, tiles_y, tiles_z);

    return (neighbour_sum - 6.0f * field[cell.cell_index]) * inv_delta2;
}

inline float3 central_gradient_sparse(__global const float *field,
                                      __global const int *index_tile_map,
                                      const int i,
                                      const int j,
                                      const int k,
                                      const float half_inv_delta,
                                      const float default_value,
                                      const int tiles_y,
                                      const int tiles_z) {
    /*
    Compute the central-difference gradient of a sparse scalar field.
    */
    const float dx = central_difference_sparse(field, index_tile_map, i, j, k, (int3)(1, 0, 0), half_inv_delta,
                                               default_value, tiles_y, tiles_z);
    const float dy = central_difference_sparse(field, index_tile_map, i, j, k, (int3)(0, 1, 0), half_inv_delta,
                                               default_value, tiles_y, tiles_z);
    const float dz = central_difference_sparse(field, index_tile_map, i, j, k, (int3)(0, 0, 1), half_inv_delta,
                                               default_value, tiles_y, tiles_z);

    return (float3)(dx, dy, dz);
}

inline float3 curl_sparse(__global const float *u,
                          __global const float *v,
                          __global const float *w,
                          __global const int *index_tile_map,
                          const int i,
                          const int j,
                          const int k,
                          const float half_inv_delta,
                          const float u_default,
                          const float v_default,
                          const float w_default,
                          const int tiles_y,
                          const int tiles_z) {
    /*
    Compute the curl of a sparse vector field with central differences.
    */
    const float du_dy = central_difference_sparse(u, index_tile_map, i, j, k, (int3)(0, 1, 0), half_inv_delta,
                                                  u_default, tiles_y, tiles_z);
    const float du_dz = central_difference_sparse(u, index_tile_map, i, j, k, (int3)(0, 0, 1), half_inv_delta,
                                                  u_default, tiles_y, tiles_z);
    const float dv_dx = central_difference_sparse(v, index_tile_map, i, j, k, (int3)(1, 0, 0), half_inv_delta,
                                                  v_default, tiles_y, tiles_z);
    const float dv_dz = central_difference_sparse(v, index_tile_map, i, j, k, (int3)(0, 0, 1), half_inv_delta,
                                                  v_default, tiles_y, tiles_z);
    const float dw_dx = central_difference_sparse(w, index_tile_map, i, j, k, (int3)(1, 0, 0), half_inv_delta,
                                                  w_default, tiles_y, tiles_z);
    const float dw_dy = central_difference_sparse(w, index_tile_map, i, j, k, (int3)(0, 1, 0), half_inv_delta,
                                                  w_default, tiles_y, tiles_z);

    return (float3)(dw_dy - dv_dz, du_dz - dw_dx, dv_dx - du_dy);
}

#endif // HELPER_CL
