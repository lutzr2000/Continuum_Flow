#include "sparse_managment.cl"

__kernel void obstacle_bc(__global float *u,
                          __global float *v,
                          __global float *w,
                          __global float *smoke,
                          __global float *fuel,
                          __global float *oxygen,
                          __global float *flame,
                          __global const uchar *mask,
                          __global const float *obstacle_velocity_x,
                          __global const float *obstacle_velocity_y,
                          __global const float *obstacle_velocity_z,
                          __global const int *index_tile_map,
                          const int tiles_x,
                          const int tiles_y,
                          const int tiles_z) {
    /*
    Apply boundary conditions for obstacles. Fuel, smoke, oxygen and flame are
    zero within an obstacle. The velocity is set to the computed obstacle velocity.
    Pressure and temperature are not touched by the obstacle.
    */
    const SparseCell cell = get_sparse_cell(index_tile_map, tiles_x, tiles_y, tiles_z);

    if (!cell.valid)
        return;

    const int index = cell.cell_index;

    if (!mask[index])
        return;

    u[index] = obstacle_velocity_x[index];
    v[index] = obstacle_velocity_y[index];
    w[index] = obstacle_velocity_z[index];

    smoke[index] = 0.0f;
    fuel[index] = 0.0f;
    oxygen[index] = 0.0f;
    flame[index] = 0.0f;
}
