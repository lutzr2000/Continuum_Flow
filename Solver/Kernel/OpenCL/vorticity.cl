#ifndef TILE_SIZE
#define TILE_SIZE 4
#endif

#include "sparse_managment.cl"

__kernel void compute_vorticity(
    __global const float *u,
    __global const float *v,
    __global const float *w,
    const float u_initial,
    const float v_initial,
    const float w_initial,
    __global const uchar *obstacle_mask,
    __global float *vorticity_magnitude,
    const float delta,
    __global const int *tile_map,
    const int nx,
    const int ny,
    const int nz,
    const int tiles_x,
    const int tiles_y,
    const int tiles_z
)
{
    const int tile_i = get_group_id(0);
    const int tile_j = get_group_id(1);
    const int tile_k = get_group_id(2);

    const int local_k = get_local_id(0);
    const int local_j = get_local_id(1);
    const int local_i = get_local_id(2);

    if (
        tile_i >= tiles_x ||
        tile_j >= tiles_y ||
        tile_k >= tiles_z
    )
        return;

    const int i =
        tile_i * TILE_SIZE + local_i;

    const int j =
        tile_j * TILE_SIZE + local_j;

    const int k =
        tile_k * TILE_SIZE + local_k;

    const int tile_map_index =
        (tile_i * tiles_y + tile_j)
        * tiles_z + tile_k;

    const int tile_index =
        tile_map[tile_map_index];

    if (tile_index == -1)
        return;

    const int index =
        ((tile_index * TILE_SIZE + local_i)
        * TILE_SIZE + local_j)
        * TILE_SIZE + local_k;

    if (
        i < 1 ||
        j < 1 ||
        k < 1 ||
        i >= nx - 1 ||
        j >= ny - 1 ||
        k >= nz - 1
    )
    {
        vorticity_magnitude[index] = 0.0f;
        return;
    }

    if (obstacle_mask[index])
    {
        vorticity_magnitude[index] = 0.0f;
        return;
    }

    const float half_inv_delta =
        0.5f / delta;

    const float du_dy =
        (
            get_pool_value(
                u,
                tile_map,
                i,
                j + 1,
                k,
                u_initial,
                tiles_y,
                tiles_z
            )
            -
            get_pool_value(
                u,
                tile_map,
                i,
                j - 1,
                k,
                u_initial,
                tiles_y,
                tiles_z
            )
        )
        * half_inv_delta;

    const float du_dz =
        (
            get_pool_value(
                u,
                tile_map,
                i,
                j,
                k + 1,
                u_initial,
                tiles_y,
                tiles_z
            )
            -
            get_pool_value(
                u,
                tile_map,
                i,
                j,
                k - 1,
                u_initial,
                tiles_y,
                tiles_z
            )
        )
        * half_inv_delta;

    const float dv_dx =
        (
            get_pool_value(
                v,
                tile_map,
                i + 1,
                j,
                k,
                v_initial,
                tiles_y,
                tiles_z
            )
            -
            get_pool_value(
                v,
                tile_map,
                i - 1,
                j,
                k,
                v_initial,
                tiles_y,
                tiles_z
            )
        )
        * half_inv_delta;

    const float dv_dz =
        (
            get_pool_value(
                v,
                tile_map,
                i,
                j,
                k + 1,
                v_initial,
                tiles_y,
                tiles_z
            )
            -
            get_pool_value(
                v,
                tile_map,
                i,
                j,
                k - 1,
                v_initial,
                tiles_y,
                tiles_z
            )
        )
        * half_inv_delta;

    const float dw_dx =
        (
            get_pool_value(
                w,
                tile_map,
                i + 1,
                j,
                k,
                w_initial,
                tiles_y,
                tiles_z
            )
            -
            get_pool_value(
                w,
                tile_map,
                i - 1,
                j,
                k,
                w_initial,
                tiles_y,
                tiles_z
            )
        )
        * half_inv_delta;

    const float dw_dy =
        (
            get_pool_value(
                w,
                tile_map,
                i,
                j + 1,
                k,
                w_initial,
                tiles_y,
                tiles_z
            )
            -
            get_pool_value(
                w,
                tile_map,
                i,
                j - 1,
                k,
                w_initial,
                tiles_y,
                tiles_z
            )
        )
        * half_inv_delta;

    const float wx = dw_dy - dv_dz;
    const float wy = du_dz - dw_dx;
    const float wz = dv_dx - du_dy;

    vorticity_magnitude[index] =
        sqrt(
            wx * wx +
            wy * wy +
            wz * wz
        );
}
