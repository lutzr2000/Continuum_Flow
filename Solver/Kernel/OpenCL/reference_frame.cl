#include "sparse_managment.cl"

__kernel void transfer_velocity(__global float *u,
                                __global float *v,
                                __global float *w,
                                __global const int *index_tile_map,
                                const float a00,
                                const float a01,
                                const float a02,
                                const float a03,
                                const float a10,
                                const float a11,
                                const float a12,
                                const float a13,
                                const float a20,
                                const float a21,
                                const float a22,
                                const float a23,
                                const float origin_x,
                                const float origin_y,
                                const float origin_z,
                                const float delta,
                                const int nx,
                                const int ny,
                                const int nz,
                                const int tiles_y,
                                const int tiles_z) {
    /*
    This kernel transfers the reference frame velocity to the flow
    */
    const int tile_i = get_group_id(0);
    const int tile_j = get_group_id(1);
    const int tile_k = get_group_id(2);

    const int local_i = get_local_id(0);
    const int local_j = get_local_id(1);
    const int local_k = get_local_id(2);

    const int i = tile_i * TILE_SIZE + local_i;
    const int j = tile_j * TILE_SIZE + local_j;
    const int k = tile_k * TILE_SIZE + local_k;

    if (i < 1 || j < 1 || k < 1 || i >= nx - 1 || j >= ny - 1 || k >= nz - 1)
        return;

    const int tile_map_index = (tile_i * tiles_y + tile_j) * tiles_z + tile_k;

    const int tile_index = index_tile_map[tile_map_index];

    if (tile_index == -1)
        return;

    const float x = origin_x + (float)i * delta;
    const float y = origin_y + (float)j * delta;
    const float z = origin_z + (float)k * delta;

    const int index = ((tile_index * TILE_SIZE + local_i) * TILE_SIZE + local_j) * TILE_SIZE + local_k;

    u[index] = get_pool_value(u, index_tile_map, i, j, k, 0.0f, tiles_y, tiles_z) + a00 * x + a01 * y + a02 * z + a03;
    v[index] = get_pool_value(v, index_tile_map, i, j, k, 0.0f, tiles_y, tiles_z) + a10 * x + a11 * y + a12 * z + a13;
    w[index] = get_pool_value(w, index_tile_map, i, j, k, 0.0f, tiles_y, tiles_z) + a20 * x + a21 * y + a22 * z + a23;
}
