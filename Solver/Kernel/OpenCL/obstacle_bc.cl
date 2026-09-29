#ifndef TILE_SIZE
#define TILE_SIZE 4
#endif

__kernel void obstacle_bc(
    __global float *u,
    __global float *v,
    __global float *w,
    __global float *smoke,
    __global float *fuel,
    __global float *flame,
    __global const uchar *mask,
    __global const float *obstacle_velocity_x,
    __global const float *obstacle_velocity_y,
    __global const float *obstacle_velocity_z,
    __global const int *tile_map,
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

    if (
        local_i >= TILE_SIZE ||
        local_j >= TILE_SIZE ||
        local_k >= TILE_SIZE
    )
        return;

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

    if (!mask[index])
        return;

    u[index] =
        obstacle_velocity_x[index];

    v[index] =
        obstacle_velocity_y[index];

    w[index] =
        obstacle_velocity_z[index];

    smoke[index] = 0.0f;
    fuel[index] = 0.0f;
    flame[index] = 0.0f;
}