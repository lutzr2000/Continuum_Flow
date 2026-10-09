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
