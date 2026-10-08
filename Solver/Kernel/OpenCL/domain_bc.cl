#ifndef TILE_SIZE
#define TILE_SIZE 4
#endif

#include "sparse_managment.cl"

inline void apply_face_state(
    __global float *u,
    __global float *v,
    __global float *w,
    __global float *p,
    __global float *T,
    __global float *smoke,
    __global float *fuel,
    __global const int *index_tile_map,
    const float ref_temp,
    const float u_initial,
    const float v_initial,
    const float w_initial,
    const int i,
    const int j,
    const int k,
    const int src_i,
    const int src_j,
    const int src_k,
    const int axis,
    const int side_index,
    const int bc_mode,
    const float u_value,
    const float v_value,
    const float w_value,
    const float temp_value,
    const int use_temp,
    const int tiles_y,
    const int tiles_z
)
{
    const int src_tile_i =
        src_i / TILE_SIZE;

    const int src_tile_j =
        src_j / TILE_SIZE;

    const int src_tile_k =
        src_k / TILE_SIZE;

    const int src_tile_map_index =
        (src_tile_i * tiles_y + src_tile_j)
        * tiles_z + src_tile_k;

    const int src_tile_index =
        index_tile_map[src_tile_map_index];

    float neighbor_u;
    float neighbor_v;
    float neighbor_w;
    float neighbor_T;
    float neighbor_smoke;
    float neighbor_fuel;

    if (src_tile_index == -1)
    {
        neighbor_u = u_initial;
        neighbor_v = v_initial;
        neighbor_w = w_initial;

        neighbor_T =
            use_temp ? temp_value : ref_temp;

        neighbor_smoke = 0.0f;
        neighbor_fuel = 0.0f;
    }
    else
    {
        const int src_local_i =
            src_i - src_tile_i * TILE_SIZE;

        const int src_local_j =
            src_j - src_tile_j * TILE_SIZE;

        const int src_local_k =
            src_k - src_tile_k * TILE_SIZE;

        const int src_index =
            ((src_tile_index * TILE_SIZE + src_local_i)
            * TILE_SIZE + src_local_j)
            * TILE_SIZE + src_local_k;

        neighbor_u = u[src_index];
        neighbor_v = v[src_index];
        neighbor_w = w[src_index];

        neighbor_T = T[src_index];
        neighbor_smoke = smoke[src_index];
        neighbor_fuel = fuel[src_index];
    }

    const int dst_tile_i =
        i / TILE_SIZE;

    const int dst_tile_j =
        j / TILE_SIZE;

    const int dst_tile_k =
        k / TILE_SIZE;

    const int dst_tile_map_index =
        (dst_tile_i * tiles_y + dst_tile_j)
        * tiles_z + dst_tile_k;

    const int dst_tile_index =
        index_tile_map[dst_tile_map_index];

    if (dst_tile_index == -1)
        return;

    const int dst_local_i =
        i - dst_tile_i * TILE_SIZE;

    const int dst_local_j =
        j - dst_tile_j * TILE_SIZE;

    const int dst_local_k =
        k - dst_tile_k * TILE_SIZE;

    const int dst_index =
        ((dst_tile_index * TILE_SIZE + dst_local_i)
        * TILE_SIZE + dst_local_j)
        * TILE_SIZE + dst_local_k;

    if (bc_mode == 0)
    {
        u[dst_index] = neighbor_u;
        v[dst_index] = neighbor_v;
        w[dst_index] = neighbor_w;

        if (axis == 0)
        {
            u[dst_index] =
                side_index == 0
                ? fmin(neighbor_u, 0.0f)
                : fmax(neighbor_u, 0.0f);
        }
        else if (axis == 1)
        {
            v[dst_index] =
                side_index == 0
                ? fmin(neighbor_v, 0.0f)
                : fmax(neighbor_v, 0.0f);
        }
        else
        {
            w[dst_index] =
                side_index == 0
                ? fmin(neighbor_w, 0.0f)
                : fmax(neighbor_w, 0.0f);
        }
    }
    else if (bc_mode == 1)
    {
        u[dst_index] = u_value;
        v[dst_index] = v_value;
        w[dst_index] = w_value;
    }
    else if (bc_mode == 2)
    {
        u[dst_index] = 0.0f;
        v[dst_index] = 0.0f;
        w[dst_index] = 0.0f;
    }
    else
    {
        u[dst_index] =
            axis == 0 ? 0.0f : neighbor_u;

        v[dst_index] =
            axis == 1 ? 0.0f : neighbor_v;

        w[dst_index] =
            axis == 2 ? 0.0f : neighbor_w;
    }

    p[dst_index] = get_pool_value(
        p,
        index_tile_map,
        src_i,
        src_j,
        src_k,
        0.0f,
        tiles_y,
        tiles_z
    );

    T[dst_index] =
        use_temp ? temp_value : neighbor_T;

    smoke[dst_index] = neighbor_smoke;
    fuel[dst_index] = neighbor_fuel;
}


__kernel void domain_bc(
    __global float *u,
    __global float *v,
    __global float *w,
    __global float *p,
    __global float *T,
    __global float *smoke,
    __global float *fuel,
    __global const int *index_tile_map,
    const float ref_temp,
    const float u_initial,
    const float v_initial,
    const float w_initial,

    const int x_low_mode,
    const float x_low_u,
    const float x_low_v,
    const float x_low_w,
    const float x_low_temp,
    const int x_low_use_temp,

    const int x_high_mode,
    const float x_high_u,
    const float x_high_v,
    const float x_high_w,
    const float x_high_temp,
    const int x_high_use_temp,

    const int y_low_mode,
    const float y_low_u,
    const float y_low_v,
    const float y_low_w,
    const float y_low_temp,
    const int y_low_use_temp,

    const int y_high_mode,
    const float y_high_u,
    const float y_high_v,
    const float y_high_w,
    const float y_high_temp,
    const int y_high_use_temp,

    const int z_low_mode,
    const float z_low_u,
    const float z_low_v,
    const float z_low_w,
    const float z_low_temp,
    const int z_low_use_temp,

    const int z_high_mode,
    const float z_high_u,
    const float z_high_v,
    const float z_high_w,
    const float z_high_temp,
    const int z_high_use_temp,

    const int nx,
    const int ny,
    const int nz,
    const int tiles_y,
    const int tiles_z
)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);
    const int k = get_global_id(2);

    if (
        i >= nx ||
        j >= ny ||
        k >= nz
    )
        return;

    if (
        i > 0 && i < nx - 1 &&
        j > 0 && j < ny - 1 &&
        k > 0 && k < nz - 1
    )
        return;

    if (i == 0)
    {
        apply_face_state(
            u, v, w, p, T, smoke, fuel,
            index_tile_map,
            ref_temp,
            u_initial,
            v_initial,
            w_initial,
            i, j, k,
            1, j, k,
            0, 0,
            x_low_mode,
            x_low_u,
            x_low_v,
            x_low_w,
            x_low_temp,
            x_low_use_temp,
            tiles_y,
            tiles_z
        );
    }
    else if (i == nx - 1)
    {
        apply_face_state(
            u, v, w, p, T, smoke, fuel,
            index_tile_map,
            ref_temp,
            u_initial,
            v_initial,
            w_initial,
            i, j, k,
            nx - 2, j, k,
            0, 1,
            x_high_mode,
            x_high_u,
            x_high_v,
            x_high_w,
            x_high_temp,
            x_high_use_temp,
            tiles_y,
            tiles_z
        );
    }

    if (j == 0)
    {
        apply_face_state(
            u, v, w, p, T, smoke, fuel,
            index_tile_map,
            ref_temp,
            u_initial,
            v_initial,
            w_initial,
            i, j, k,
            i, 1, k,
            1, 0,
            y_low_mode,
            y_low_u,
            y_low_v,
            y_low_w,
            y_low_temp,
            y_low_use_temp,
            tiles_y,
            tiles_z
        );
    }
    else if (j == ny - 1)
    {
        apply_face_state(
            u, v, w, p, T, smoke, fuel,
            index_tile_map,
            ref_temp,
            u_initial,
            v_initial,
            w_initial,
            i, j, k,
            i, ny - 2, k,
            1, 1,
            y_high_mode,
            y_high_u,
            y_high_v,
            y_high_w,
            y_high_temp,
            y_high_use_temp,
            tiles_y,
            tiles_z
        );
    }

    if (k == 0)
    {
        apply_face_state(
            u, v, w, p, T, smoke, fuel,
            index_tile_map,
            ref_temp,
            u_initial,
            v_initial,
            w_initial,
            i, j, k,
            i, j, 1,
            2, 0,
            z_low_mode,
            z_low_u,
            z_low_v,
            z_low_w,
            z_low_temp,
            z_low_use_temp,
            tiles_y,
            tiles_z
        );
    }
    else if (k == nz - 1)
    {
        apply_face_state(
            u, v, w, p, T, smoke, fuel,
            index_tile_map,
            ref_temp,
            u_initial,
            v_initial,
            w_initial,
            i, j, k,
            i, j, nz - 2,
            2, 1,
            z_high_mode,
            z_high_u,
            z_high_v,
            z_high_w,
            z_high_temp,
            z_high_use_temp,
            tiles_y,
            tiles_z
        );
    }
}
