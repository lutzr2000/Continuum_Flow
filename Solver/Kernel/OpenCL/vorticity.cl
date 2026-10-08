#ifndef TILE_SIZE
#define TILE_SIZE 4
#endif

#include "sparse_managment.cl"

__kernel void compute_vorticity(__global const float *u,
                                __global const float *v,
                                __global const float *w,
                                const float u_initial,
                                const float v_initial,
                                const float w_initial,
                                __global const uchar *obstacle_mask,
                                __global float *vorticity_magnitude,
                                const float delta,
                                __global const int *index_tile_map,
                                const int nx,
                                const int ny,
                                const int nz,
                                const int tiles_x,
                                const int tiles_y,
                                const int tiles_z) {
    const int tile_i = get_group_id(0);
    const int tile_j = get_group_id(1);
    const int tile_k = get_group_id(2);

    const int local_k = get_local_id(0);
    const int local_j = get_local_id(1);
    const int local_i = get_local_id(2);

    if (tile_i >= tiles_x || tile_j >= tiles_y || tile_k >= tiles_z)
        return;

    const int i = tile_i * TILE_SIZE + local_i;
    const int j = tile_j * TILE_SIZE + local_j;
    const int k = tile_k * TILE_SIZE + local_k;

    const int tile_map_index = (tile_i * tiles_y + tile_j) * tiles_z + tile_k;

    const int tile_index = index_tile_map[tile_map_index];

    if (tile_index == -1)
        return;

    const int index = ((tile_index * TILE_SIZE + local_i) * TILE_SIZE + local_j) * TILE_SIZE + local_k;

    if (i < 1 || j < 1 || k < 1 || i >= nx - 1 || j >= ny - 1 || k >= nz - 1) {
        vorticity_magnitude[index] = 0.0f;
        return;
    }

    if (obstacle_mask[index]) {
        vorticity_magnitude[index] = 0.0f;
        return;
    }

    const float half_inv_delta = 0.5f / delta;

    const float du_dy = (get_pool_value(u, index_tile_map, i, j + 1, k, u_initial, tiles_y, tiles_z) -
                         get_pool_value(u, index_tile_map, i, j - 1, k, u_initial, tiles_y, tiles_z)) *
                        half_inv_delta;

    const float du_dz = (get_pool_value(u, index_tile_map, i, j, k + 1, u_initial, tiles_y, tiles_z) -
                         get_pool_value(u, index_tile_map, i, j, k - 1, u_initial, tiles_y, tiles_z)) *
                        half_inv_delta;

    const float dv_dx = (get_pool_value(v, index_tile_map, i + 1, j, k, v_initial, tiles_y, tiles_z) -
                         get_pool_value(v, index_tile_map, i - 1, j, k, v_initial, tiles_y, tiles_z)) *
                        half_inv_delta;

    const float dv_dz = (get_pool_value(v, index_tile_map, i, j, k + 1, v_initial, tiles_y, tiles_z) -
                         get_pool_value(v, index_tile_map, i, j, k - 1, v_initial, tiles_y, tiles_z)) *
                        half_inv_delta;

    const float dw_dx = (get_pool_value(w, index_tile_map, i + 1, j, k, w_initial, tiles_y, tiles_z) -
                         get_pool_value(w, index_tile_map, i - 1, j, k, w_initial, tiles_y, tiles_z)) *
                        half_inv_delta;

    const float dw_dy = (get_pool_value(w, index_tile_map, i, j + 1, k, w_initial, tiles_y, tiles_z) -
                         get_pool_value(w, index_tile_map, i, j - 1, k, w_initial, tiles_y, tiles_z)) *
                        half_inv_delta;

    const float wx = dw_dy - dv_dz;
    const float wy = du_dz - dw_dx;
    const float wz = dv_dx - du_dy;

    vorticity_magnitude[index] = sqrt(wx * wx + wy * wy + wz * wz);
}

inline void apply_vorticity_confinement(__global const float *u,
                                        __global const float *v,
                                        __global const float *w,
                                        __global const uchar *obstacle_mask,
                                        __global const float *omega_magnitude,
                                        const int i,
                                        const int j,
                                        const int k,
                                        const float delta,
                                        const float vorticity_strength,
                                        __global const int *index_tile_map,
                                        const float u_initial,
                                        const float v_initial,
                                        const float w_initial,
                                        const int nx,
                                        const int ny,
                                        const int nz,
                                        const int tiles_y,
                                        const int tiles_z,
                                        float *fx,
                                        float *fy,
                                        float *fz) {
    const int tile_i = i / TILE_SIZE;
    const int tile_j = j / TILE_SIZE;
    const int tile_k = k / TILE_SIZE;

    const int local_i = i - tile_i * TILE_SIZE;
    const int local_j = j - tile_j * TILE_SIZE;
    const int local_k = k - tile_k * TILE_SIZE;

    const int tile_map_index = (tile_i * tiles_y + tile_j) * tiles_z + tile_k;

    const int tile_index = index_tile_map[tile_map_index];

    if (tile_index == -1) {
        *fx = 0.0f;
        *fy = 0.0f;
        *fz = 0.0f;
        return;
    }

    const int index = ((tile_index * TILE_SIZE + local_i) * TILE_SIZE + local_j) * TILE_SIZE + local_k;

    if (i < 2 || j < 2 || k < 2 || i >= nx - 2 || j >= ny - 2 || k >= nz - 2 || obstacle_mask[index]) {
        *fx = 0.0f;
        *fy = 0.0f;
        *fz = 0.0f;
        return;
    }

    const float half_inv_delta = 0.5f / delta;

    const float grad_x = (get_pool_value(omega_magnitude, index_tile_map, i + 1, j, k, 0.0f, tiles_y, tiles_z) -
                          get_pool_value(omega_magnitude, index_tile_map, i - 1, j, k, 0.0f, tiles_y, tiles_z)) *
                         half_inv_delta;

    const float grad_y = (get_pool_value(omega_magnitude, index_tile_map, i, j + 1, k, 0.0f, tiles_y, tiles_z) -
                          get_pool_value(omega_magnitude, index_tile_map, i, j - 1, k, 0.0f, tiles_y, tiles_z)) *
                         half_inv_delta;

    const float grad_z = (get_pool_value(omega_magnitude, index_tile_map, i, j, k + 1, 0.0f, tiles_y, tiles_z) -
                          get_pool_value(omega_magnitude, index_tile_map, i, j, k - 1, 0.0f, tiles_y, tiles_z)) *
                         half_inv_delta;

    const float grad_length = sqrt(grad_x * grad_x + grad_y * grad_y + grad_z * grad_z);

    if (grad_length <= 1.0e-12f) {
        *fx = 0.0f;
        *fy = 0.0f;
        *fz = 0.0f;
        return;
    }

    const float nx_dir = grad_x / grad_length;
    const float ny_dir = grad_y / grad_length;
    const float nz_dir = grad_z / grad_length;

    const float du_dy = (get_pool_value(u, index_tile_map, i, j + 1, k, u_initial, tiles_y, tiles_z) -
                         get_pool_value(u, index_tile_map, i, j - 1, k, u_initial, tiles_y, tiles_z)) *
                        half_inv_delta;

    const float du_dz = (get_pool_value(u, index_tile_map, i, j, k + 1, u_initial, tiles_y, tiles_z) -
                         get_pool_value(u, index_tile_map, i, j, k - 1, u_initial, tiles_y, tiles_z)) *
                        half_inv_delta;

    const float dv_dx = (get_pool_value(v, index_tile_map, i + 1, j, k, v_initial, tiles_y, tiles_z) -
                         get_pool_value(v, index_tile_map, i - 1, j, k, v_initial, tiles_y, tiles_z)) *
                        half_inv_delta;

    const float dv_dz = (get_pool_value(v, index_tile_map, i, j, k + 1, v_initial, tiles_y, tiles_z) -
                         get_pool_value(v, index_tile_map, i, j, k - 1, v_initial, tiles_y, tiles_z)) *
                        half_inv_delta;

    const float dw_dx = (get_pool_value(w, index_tile_map, i + 1, j, k, w_initial, tiles_y, tiles_z) -
                         get_pool_value(w, index_tile_map, i - 1, j, k, w_initial, tiles_y, tiles_z)) *
                        half_inv_delta;

    const float dw_dy = (get_pool_value(w, index_tile_map, i, j + 1, k, w_initial, tiles_y, tiles_z) -
                         get_pool_value(w, index_tile_map, i, j - 1, k, w_initial, tiles_y, tiles_z)) *
                        half_inv_delta;

    const float wx = dw_dy - dv_dz;
    const float wy = du_dz - dw_dx;
    const float wz = dv_dx - du_dy;

    *fx = vorticity_strength * (ny_dir * wz - nz_dir * wy);
    *fy = vorticity_strength * (nz_dir * wx - nx_dir * wz);
    *fz = vorticity_strength * (nx_dir * wy - ny_dir * wx);
}