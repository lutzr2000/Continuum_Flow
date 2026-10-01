#ifndef TILE_SIZE
#define TILE_SIZE 4
#endif

#include "sparse_managment.cl"

__kernel void build_coarse_tile_level(
    __global const int *fine_tile_map,
    __global int *coarse_tile_map,
    __global int *coarse_active_tiles,
    __global int *coarse_active_tile_count,
    const int coarse_active_tile_capacity,
    const int fine_tiles_x,
    const int fine_tiles_y,
    const int fine_tiles_z,
    const int coarse_tiles_x,
    const int coarse_tiles_y,
    const int coarse_tiles_z
)
{
    const int coarse_i = get_global_id(0);
    const int coarse_j = get_global_id(1);
    const int coarse_k = get_global_id(2);

    if (
        coarse_i >= coarse_tiles_x ||
        coarse_j >= coarse_tiles_y ||
        coarse_k >= coarse_tiles_z
    )
        return;

    const int fine_i_start = coarse_i * 2;
    const int fine_j_start = coarse_j * 2;
    const int fine_k_start = coarse_k * 2;

    int is_active = 0;

    for (int offset_i = 0; offset_i < 2; ++offset_i)
    {
        const int fine_i = fine_i_start + offset_i;

        if (fine_i >= fine_tiles_x)
            continue;

        for (int offset_j = 0; offset_j < 2; ++offset_j)
        {
            const int fine_j = fine_j_start + offset_j;

            if (fine_j >= fine_tiles_y)
                continue;

            for (int offset_k = 0; offset_k < 2; ++offset_k)
            {
                const int fine_k = fine_k_start + offset_k;

                if (fine_k >= fine_tiles_z)
                    continue;

                const int fine_index =
                    (fine_i * fine_tiles_y + fine_j)
                    * fine_tiles_z + fine_k;

                if (fine_tile_map[fine_index] != -1)
                {
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

    const int pool_index =
        atomic_add(
            (volatile __global int *)
            coarse_active_tile_count,
            1
        );

    if (pool_index >= coarse_active_tile_capacity)
        return;

    const int coarse_index =
        (coarse_i * coarse_tiles_y + coarse_j)
        * coarse_tiles_z + coarse_k;

    coarse_tile_map[coarse_index] = pool_index;

    const int active_index = pool_index * 3;

    coarse_active_tiles[active_index + 0] = coarse_i;
    coarse_active_tiles[active_index + 1] = coarse_j;
    coarse_active_tiles[active_index + 2] = coarse_k;
}

inline float residual_sparse(
    __global const float *p,
    __global const float *b,
    const float inv_delta2,
    __global const int *tile_map,
    const int i,
    const int j,
    const int k,
    const int tiles_y,
    const int tiles_z,
    int *valid
)
{
    const int tile_i = i / TILE_SIZE;
    const int tile_j = j / TILE_SIZE;
    const int tile_k = k / TILE_SIZE;

    const int tile_map_index =
        (tile_i * tiles_y + tile_j)
        * tiles_z + tile_k;

    const int tile_index =
        tile_map[tile_map_index];

    if (tile_index == -1)
    {
        *valid = 0;
        return 0.0f;
    }

    const int local_i =
        i - tile_i * TILE_SIZE;

    const int local_j =
        j - tile_j * TILE_SIZE;

    const int local_k =
        k - tile_k * TILE_SIZE;

    const int index =
        ((tile_index * TILE_SIZE + local_i)
        * TILE_SIZE + local_j)
        * TILE_SIZE + local_k;

    const float laplace =
        (
            get_pool_value(
                p,
                tile_map,
                i + 1,
                j,
                k,
                0.0f,
                tiles_y,
                tiles_z
            )
            + get_pool_value(
                p,
                tile_map,
                i - 1,
                j,
                k,
                0.0f,
                tiles_y,
                tiles_z
            )
            + get_pool_value(
                p,
                tile_map,
                i,
                j + 1,
                k,
                0.0f,
                tiles_y,
                tiles_z
            )
            + get_pool_value(
                p,
                tile_map,
                i,
                j - 1,
                k,
                0.0f,
                tiles_y,
                tiles_z
            )
            + get_pool_value(
                p,
                tile_map,
                i,
                j,
                k + 1,
                0.0f,
                tiles_y,
                tiles_z
            )
            + get_pool_value(
                p,
                tile_map,
                i,
                j,
                k - 1,
                0.0f,
                tiles_y,
                tiles_z
            )
            - 6.0f * p[index]
        )
        * inv_delta2;

    const float rhs = b[index];

    *valid = 1;

    return rhs - laplace;
}


__kernel void restrict_residual_sparse(
    __global const float *fine_p,
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
    const int coarse_tiles_z
)
{
    const int coarse_pool_index =
        get_group_id(0);

    if (coarse_pool_index >= coarse_active_tile_count[0])
        return;

    const int local_i = get_local_id(0);
    const int local_j = get_local_id(1);
    const int local_k = get_local_id(2);

    const int active_index =
        coarse_pool_index * 3;

    const int coarse_tile_i =
        coarse_active_tiles[active_index + 0];

    const int coarse_tile_j =
        coarse_active_tiles[active_index + 1];

    const int coarse_tile_k =
        coarse_active_tiles[active_index + 2];

    const int I =
        coarse_tile_i * TILE_SIZE + local_i;

    const int J =
        coarse_tile_j * TILE_SIZE + local_j;

    const int K =
        coarse_tile_k * TILE_SIZE + local_k;

    if (
        I >= coarse_nx ||
        J >= coarse_ny ||
        K >= coarse_nz
    )
        return;

    const int coarse_tile_map_index =
        (coarse_tile_i * coarse_tiles_y + coarse_tile_j)
        * coarse_tiles_z + coarse_tile_k;

    const int coarse_tile_index =
        coarse_tile_map[coarse_tile_map_index];

    if (coarse_tile_index == -1)
        return;

    const float inv_delta2 =
        1.0f / (fine_delta * fine_delta);

    const int fine_i_start = 2 * I;
    const int fine_j_start = 2 * J;
    const int fine_k_start = 2 * K;

    float residual_sum = 0.0f;
    float residual_count = 0.0f;

    for (int offset_i = 0; offset_i < 2; ++offset_i)
    {
        for (int offset_j = 0; offset_j < 2; ++offset_j)
        {
            for (int offset_k = 0; offset_k < 2; ++offset_k)
            {
                const int i =
                    fine_i_start + offset_i;

                const int j =
                    fine_j_start + offset_j;

                const int k =
                    fine_k_start + offset_k;

                if (
                    i < 1 ||
                    j < 1 ||
                    k < 1 ||
                    i >= fine_nx - 1 ||
                    j >= fine_ny - 1 ||
                    k >= fine_nz - 1
                )
                    continue;

                int valid;

                const float residual_value =
                    residual_sparse(
                        fine_p,
                        fine_b,
                        inv_delta2,
                        fine_tile_map,
                        i,
                        j,
                        k,
                        fine_tiles_y,
                        fine_tiles_z,
                        &valid
                    );

                if (!valid)
                    continue;

                residual_sum += residual_value;
                residual_count += 1.0f;
            }
        }
    }

    const int index =
        ((coarse_tile_index * TILE_SIZE + local_i)
        * TILE_SIZE + local_j)
        * TILE_SIZE + local_k;

    coarse_p[index] = 0.0f;

    if (residual_count > 0.0f)
    {
        coarse_b[index] =
            residual_sum / residual_count;
    }
    else
    {
        coarse_b[index] = 0.0f;
    }
}


__kernel void prolongate_add_nearest_sparse(
    __global const float *coarse_e,
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
    const int fine_tiles_z
)
{
    const int coarse_pool_index =
        get_group_id(0);

    if (coarse_pool_index >= coarse_active_tile_count[0])
        return;

    const int local_i = get_local_id(0);
    const int local_j = get_local_id(1);
    const int local_k = get_local_id(2);

    const int active_index =
        coarse_pool_index * 3;

    const int coarse_tile_i =
        coarse_active_tiles[active_index + 0];

    const int coarse_tile_j =
        coarse_active_tiles[active_index + 1];

    const int coarse_tile_k =
        coarse_active_tiles[active_index + 2];

    const int I =
        coarse_tile_i * TILE_SIZE + local_i;

    const int J =
        coarse_tile_j * TILE_SIZE + local_j;

    const int K =
        coarse_tile_k * TILE_SIZE + local_k;

    if (
        I >= coarse_nx ||
        J >= coarse_ny ||
        K >= coarse_nz
    )
        return;

    const int coarse_tile_map_index =
        (coarse_tile_i * coarse_tiles_y + coarse_tile_j)
        * coarse_tiles_z + coarse_tile_k;

    const int coarse_tile_index =
        coarse_tile_map[coarse_tile_map_index];

    if (coarse_tile_index == -1)
        return;

    const int coarse_index =
        ((coarse_tile_index * TILE_SIZE + local_i)
        * TILE_SIZE + local_j)
        * TILE_SIZE + local_k;

    const float error =
        0.25f * coarse_e[coarse_index];

    const int fine_i_start = 2 * I;
    const int fine_j_start = 2 * J;
    const int fine_k_start = 2 * K;

    for (int offset_i = 0; offset_i < 2; ++offset_i)
    {
        for (int offset_j = 0; offset_j < 2; ++offset_j)
        {
            for (int offset_k = 0; offset_k < 2; ++offset_k)
            {
                const int i =
                    fine_i_start + offset_i;

                const int j =
                    fine_j_start + offset_j;

                const int k =
                    fine_k_start + offset_k;

                if (
                    i >= fine_nx ||
                    j >= fine_ny ||
                    k >= fine_nz
                )
                    continue;

                const int fine_tile_i =
                    i / TILE_SIZE;

                const int fine_tile_j =
                    j / TILE_SIZE;

                const int fine_tile_k =
                    k / TILE_SIZE;

                const int fine_tile_map_index =
                    (fine_tile_i * fine_tiles_y + fine_tile_j)
                    * fine_tiles_z + fine_tile_k;

                const int fine_tile_index =
                    fine_tile_map[fine_tile_map_index];

                if (fine_tile_index == -1)
                    continue;

                const int fine_local_i =
                    i - fine_tile_i * TILE_SIZE;

                const int fine_local_j =
                    j - fine_tile_j * TILE_SIZE;

                const int fine_local_k =
                    k - fine_tile_k * TILE_SIZE;

                const int fine_index =
                    ((fine_tile_index * TILE_SIZE + fine_local_i)
                    * TILE_SIZE + fine_local_j)
                    * TILE_SIZE + fine_local_k;

                fine_p[fine_index] += error;
            }
        }
    }
}


__kernel void rbgs_step_sparse(
    __global float *p,
    __global const float *b,
    const float delta,
    const int parity,
    __global const int *tile_map,
    __global const int *active_tiles,
    __global const int *active_tile_count,
    const int nx,
    const int ny,
    const int nz,
    const int tiles_y,
    const int tiles_z
)
{
    const int active_index = get_group_id(0);

    if (active_index >= active_tile_count[0])
        return;

    const int local_i = get_local_id(0);
    const int local_j = get_local_id(1);
    const int local_k = get_local_id(2);

    const int active_tile_index =
        active_index * 3;

    const int tile_i =
        active_tiles[active_tile_index + 0];

    const int tile_j =
        active_tiles[active_tile_index + 1];

    const int tile_k =
        active_tiles[active_tile_index + 2];

    const int tile_map_index =
        (tile_i * tiles_y + tile_j)
        * tiles_z + tile_k;

    const int tile_index =
        tile_map[tile_map_index];

    if (tile_index == -1)
        return;

    const int i =
        tile_i * TILE_SIZE + local_i;

    const int j =
        tile_j * TILE_SIZE + local_j;

    const int k =
        tile_k * TILE_SIZE + local_k;

    if (
        i < 1 ||
        j < 1 ||
        k < 1 ||
        i >= nx - 1 ||
        j >= ny - 1 ||
        k >= nz - 1
    )
        return;

    if (((i + j + k) & 1) != parity)
        return;

    const float delta2 =
        delta * delta;

    const int index =
        ((tile_index * TILE_SIZE + local_i)
        * TILE_SIZE + local_j)
        * TILE_SIZE + local_k;

    p[index] =
        (
            get_pool_value(
                p, tile_map,
                i + 1, j, k,
                0.0f,
                tiles_y, tiles_z
            )
            + get_pool_value(
                p, tile_map,
                i - 1, j, k,
                0.0f,
                tiles_y, tiles_z
            )
            + get_pool_value(
                p, tile_map,
                i, j + 1, k,
                0.0f,
                tiles_y, tiles_z
            )
            + get_pool_value(
                p, tile_map,
                i, j - 1, k,
                0.0f,
                tiles_y, tiles_z
            )
            + get_pool_value(
                p, tile_map,
                i, j, k + 1,
                0.0f,
                tiles_y, tiles_z
            )
            + get_pool_value(
                p, tile_map,
                i, j, k - 1,
                0.0f,
                tiles_y, tiles_z
            )
            - delta2 * b[index]
        )
        / 6.0f;
}


__kernel void rbgs_step_level_0(
    __global float *p,
    __global const float *b,
    const float delta,
    const int parity,
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

    const int local_i = get_local_id(0);
    const int local_j = get_local_id(1);
    const int local_k = get_local_id(2);

    if (
        tile_i >= tiles_x ||
        tile_j >= tiles_y ||
        tile_k >= tiles_z
    )
        return;

    const int tile_map_index =
        (tile_i * tiles_y + tile_j)
        * tiles_z + tile_k;

    const int tile_index =
        tile_map[tile_map_index];

    if (tile_index == -1)
        return;

    const int i =
        tile_i * TILE_SIZE + local_i;

    const int j =
        tile_j * TILE_SIZE + local_j;

    const int k =
        tile_k * TILE_SIZE + local_k;

    if (
        i < 1 ||
        j < 1 ||
        k < 1 ||
        i >= nx - 1 ||
        j >= ny - 1 ||
        k >= nz - 1 ||
        ((i + j + k) & 1) != parity
    )
        return;

    const float delta2 =
        delta * delta;

    const int index =
        ((tile_index * TILE_SIZE + local_i)
        * TILE_SIZE + local_j)
        * TILE_SIZE + local_k;

    const float center =
        (
            get_pool_value(
                p, tile_map,
                i + 1, j, k,
                0.0f,
                tiles_y, tiles_z
            )
            + get_pool_value(
                p, tile_map,
                i - 1, j, k,
                0.0f,
                tiles_y, tiles_z
            )
            + get_pool_value(
                p, tile_map,
                i, j + 1, k,
                0.0f,
                tiles_y, tiles_z
            )
            + get_pool_value(
                p, tile_map,
                i, j - 1, k,
                0.0f,
                tiles_y, tiles_z
            )
            + get_pool_value(
                p, tile_map,
                i, j, k + 1,
                0.0f,
                tiles_y, tiles_z
            )
            + get_pool_value(
                p, tile_map,
                i, j, k - 1,
                0.0f,
                tiles_y, tiles_z
            )
            - delta2
            * get_pool_value(
                b, tile_map,
                i, j, k,
                0.0f,
                tiles_y, tiles_z
            )
        )
        / 6.0f;

    p[index] = center;
}

__kernel void pressure_poisson_neumann_x(
    __global float *p,
    __global const int *tile_map,
    const int nx,
    const int ny,
    const int nz,
    const int tiles_y,
    const int tiles_z)
{
    const int j = get_global_id(0);
    const int k = get_global_id(1);

    if (j >= ny || k >= nz)
        return;

    {
        const int i = 0;
        const int tile_i = 0;
        const int tile_j = j / TILE_SIZE;
        const int tile_k = k / TILE_SIZE;

        const int tile_map_index =
            (tile_i * tiles_y + tile_j) * tiles_z + tile_k;

        const int tile_index = tile_map[tile_map_index];

        if (tile_index != -1)
        {
            const int local_i = 0;
            const int local_j = j - tile_j * TILE_SIZE;
            const int local_k = k - tile_k * TILE_SIZE;

            const int index =
                ((tile_index * TILE_SIZE + local_i)
                 * TILE_SIZE + local_j)
                 * TILE_SIZE + local_k;

            p[index] = get_pool_value(
                p, tile_map,
                1, j, k,
                0.0f,
                tiles_y, tiles_z);
        }
    }

    {
        const int i = nx - 1;
        const int tile_i = i / TILE_SIZE;
        const int tile_j = j / TILE_SIZE;
        const int tile_k = k / TILE_SIZE;

        const int tile_map_index =
            (tile_i * tiles_y + tile_j) * tiles_z + tile_k;

        const int tile_index = tile_map[tile_map_index];

        if (tile_index != -1)
        {
            const int local_i = i - tile_i * TILE_SIZE;
            const int local_j = j - tile_j * TILE_SIZE;
            const int local_k = k - tile_k * TILE_SIZE;

            const int index =
                ((tile_index * TILE_SIZE + local_i)
                 * TILE_SIZE + local_j)
                 * TILE_SIZE + local_k;

            p[index] = get_pool_value(
                p, tile_map,
                nx - 2, j, k,
                0.0f,
                tiles_y, tiles_z);
        }
    }
}


__kernel void pressure_poisson_neumann_y(
    __global float *p,
    __global const int *tile_map,
    const int nx,
    const int ny,
    const int nz,
    const int tiles_y,
    const int tiles_z)
{
    const int i = get_global_id(0);
    const int k = get_global_id(1);

    if (i >= nx || k >= nz)
        return;

    {
        const int j = 0;
        const int tile_i = i / TILE_SIZE;
        const int tile_j = 0;
        const int tile_k = k / TILE_SIZE;

        const int tile_map_index =
            (tile_i * tiles_y + tile_j) * tiles_z + tile_k;

        const int tile_index = tile_map[tile_map_index];

        if (tile_index != -1)
        {
            const int local_i = i - tile_i * TILE_SIZE;
            const int local_j = 0;
            const int local_k = k - tile_k * TILE_SIZE;

            const int index =
                ((tile_index * TILE_SIZE + local_i)
                 * TILE_SIZE + local_j)
                 * TILE_SIZE + local_k;

            p[index] = get_pool_value(
                p, tile_map,
                i, 1, k,
                0.0f,
                tiles_y, tiles_z);
        }
    }

    {
        const int j = ny - 1;
        const int tile_i = i / TILE_SIZE;
        const int tile_j = j / TILE_SIZE;
        const int tile_k = k / TILE_SIZE;

        const int tile_map_index =
            (tile_i * tiles_y + tile_j) * tiles_z + tile_k;

        const int tile_index = tile_map[tile_map_index];

        if (tile_index != -1)
        {
            const int local_i = i - tile_i * TILE_SIZE;
            const int local_j = j - tile_j * TILE_SIZE;
            const int local_k = k - tile_k * TILE_SIZE;

            const int index =
                ((tile_index * TILE_SIZE + local_i)
                 * TILE_SIZE + local_j)
                 * TILE_SIZE + local_k;

            p[index] = get_pool_value(
                p, tile_map,
                i, ny - 2, k,
                0.0f,
                tiles_y, tiles_z);
        }
    }
}


__kernel void pressure_poisson_neumann_z(
    __global float *p,
    __global const int *tile_map,
    const int nx,
    const int ny,
    const int nz,
    const int tiles_y,
    const int tiles_z)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);

    if (i >= nx || j >= ny)
        return;

    {
        const int k = 0;
        const int tile_i = i / TILE_SIZE;
        const int tile_j = j / TILE_SIZE;
        const int tile_k = 0;

        const int tile_map_index =
            (tile_i * tiles_y + tile_j) * tiles_z + tile_k;

        const int tile_index = tile_map[tile_map_index];

        if (tile_index != -1)
        {
            const int local_i = i - tile_i * TILE_SIZE;
            const int local_j = j - tile_j * TILE_SIZE;
            const int local_k = 0;

            const int index =
                ((tile_index * TILE_SIZE + local_i)
                 * TILE_SIZE + local_j)
                 * TILE_SIZE + local_k;

            p[index] = get_pool_value(
                p, tile_map,
                i, j, 1,
                0.0f,
                tiles_y, tiles_z);
        }
    }

    {
        const int k = nz - 1;
        const int tile_i = i / TILE_SIZE;
        const int tile_j = j / TILE_SIZE;
        const int tile_k = k / TILE_SIZE;

        const int tile_map_index =
            (tile_i * tiles_y + tile_j) * tiles_z + tile_k;

        const int tile_index = tile_map[tile_map_index];

        if (tile_index != -1)
        {
            const int local_i = i - tile_i * TILE_SIZE;
            const int local_j = j - tile_j * TILE_SIZE;
            const int local_k = k - tile_k * TILE_SIZE;

            const int index =
                ((tile_index * TILE_SIZE + local_i)
                 * TILE_SIZE + local_j)
                 * TILE_SIZE + local_k;

            p[index] = get_pool_value(
                p, tile_map,
                i, j, nz - 2,
                0.0f,
                tiles_y, tiles_z);
        }
    }
}