#ifndef TILE_SIZE
#define TILE_SIZE 4
#endif

#include "sparse_managment.cl"
#include "noise.cl"

__kernel void project_velocity_kernel(
    __global float *u,
    __global float *v,
    __global float *w,
    __global const float *p,
    __global const uchar *obstacle_mask,
    const float dt,
    const float delta,
    const float rho,
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

    const int index =
        ((tile_index * TILE_SIZE + local_i)
        * TILE_SIZE + local_j)
        * TILE_SIZE + local_k;

    if (obstacle_mask[index])
        return;

    const float pressure_coeff =
        dt / (2.0f * rho * delta);

    const float px1 =
        get_pool_value(
            p,
            tile_map,
            i + 1,
            j,
            k,
            0.0f,
            tiles_y,
            tiles_z
        );

    const float px0 =
        get_pool_value(
            p,
            tile_map,
            i - 1,
            j,
            k,
            0.0f,
            tiles_y,
            tiles_z
        );

    const float py1 =
        get_pool_value(
            p,
            tile_map,
            i,
            j + 1,
            k,
            0.0f,
            tiles_y,
            tiles_z
        );

    const float py0 =
        get_pool_value(
            p,
            tile_map,
            i,
            j - 1,
            k,
            0.0f,
            tiles_y,
            tiles_z
        );

    const float pz1 =
        get_pool_value(
            p,
            tile_map,
            i,
            j,
            k + 1,
            0.0f,
            tiles_y,
            tiles_z
        );

    const float pz0 =
        get_pool_value(
            p,
            tile_map,
            i,
            j,
            k - 1,
            0.0f,
            tiles_y,
            tiles_z
        );

    u[index] -=
        pressure_coeff * (px1 - px0);

    v[index] -=
        pressure_coeff * (py1 - py0);

    w[index] -=
        pressure_coeff * (pz1 - pz0);
}

__kernel void pressure_equation_right_side(
    __global const float *u,
    __global const float *v,
    __global const float *w,
    __global float *b,
    const float dt,
    const float delta,
    const float rho,
    __global const int *tile_map,
    const float u_initial,
    const float v_initial,
    const float w_initial,
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
        b[index] = 0.0f;
        return;
    }

    const float half_inv_delta =
        0.5f / delta;

    const float rho_over_dt =
        rho / dt;

    const float du_dx =
        (
            get_pool_value(
                u,
                tile_map,
                i + 1,
                j,
                k,
                u_initial,
                tiles_y,
                tiles_z
            )
            -
            get_pool_value(
                u,
                tile_map,
                i - 1,
                j,
                k,
                u_initial,
                tiles_y,
                tiles_z
            )
        )
        * half_inv_delta;

    const float dv_dy =
        (
            get_pool_value(
                v,
                tile_map,
                i,
                j + 1,
                k,
                v_initial,
                tiles_y,
                tiles_z
            )
            -
            get_pool_value(
                v,
                tile_map,
                i,
                j - 1,
                k,
                v_initial,
                tiles_y,
                tiles_z
            )
        )
        * half_inv_delta;

    const float dw_dz =
        (
            get_pool_value(
                w,
                tile_map,
                i,
                j,
                k + 1,
                w_initial,
                tiles_y,
                tiles_z
            )
            -
            get_pool_value(
                w,
                tile_map,
                i,
                j,
                k - 1,
                w_initial,
                tiles_y,
                tiles_z
            )
        )
        * half_inv_delta;

    b[index] =
        rho_over_dt
        * (
            du_dx
            + dv_dy
            + dw_dz
        );
}


__kernel void reset_inactive_pressure(
    __global float *p,
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
    {
        const int index =
            ((tile_index * TILE_SIZE + local_i)
            * TILE_SIZE + local_j)
            * TILE_SIZE + local_k;

        p[index] = 0.0f;
    }
}


#ifndef TILE_SIZE
#define TILE_SIZE 4
#endif

#ifndef REDUCTION_THREADS_PER_BLOCK
#define REDUCTION_THREADS_PER_BLOCK 256
#endif


__kernel void rhs_sum_count_partial_kernel(
    __global const float *b,
    __global const int *tile_map,
    __global float *partial_sums,
    __global float *partial_counts,
    const int nx,
    const int ny,
    const int nz,
    const int tiles_y,
    const int tiles_z
)
{
    const int interior_nx = nx - 2;
    const int interior_ny = ny - 2;
    const int interior_nz = nz - 2;

    const int interior_cell_count =
        interior_nx
        * interior_ny
        * interior_nz;

    const int tid =
        get_local_id(0);

    const int global_idx =
        get_global_id(0);

    const int stride =
        get_global_size(0);

    __local float shared_sums[
        REDUCTION_THREADS_PER_BLOCK
    ];

    __local float shared_counts[
        REDUCTION_THREADS_PER_BLOCK
    ];

    float local_sum = 0.0f;
    float local_count = 0.0f;

    const int plane_size =
        interior_ny * interior_nz;

    int flat_idx = global_idx;

    while (flat_idx < interior_cell_count)
    {
        const int i =
            flat_idx / plane_size + 1;

        const int remainder =
            flat_idx % plane_size;

        const int j =
            remainder / interior_nz + 1;

        const int k =
            remainder % interior_nz + 1;

        const int tile_i =
            i / TILE_SIZE;

        const int tile_j =
            j / TILE_SIZE;

        const int tile_k =
            k / TILE_SIZE;

        const int tile_map_index =
            (tile_i * tiles_y + tile_j)
            * tiles_z + tile_k;

        const int tile_index =
            tile_map[tile_map_index];

        if (tile_index != -1)
        {
            const int local_i =
                i % TILE_SIZE;

            const int local_j =
                j % TILE_SIZE;

            const int local_k =
                k % TILE_SIZE;

            const int index =
                ((tile_index * TILE_SIZE + local_i)
                * TILE_SIZE + local_j)
                * TILE_SIZE + local_k;

            local_sum += b[index];
            local_count += 1.0f;
        }

        flat_idx += stride;
    }

    shared_sums[tid] = local_sum;
    shared_counts[tid] = local_count;

    barrier(CLK_LOCAL_MEM_FENCE);

    int offset =
        get_local_size(0) >> 1;

    while (offset > 0)
    {
        if (tid < offset)
        {
            shared_sums[tid] +=
                shared_sums[tid + offset];

            shared_counts[tid] +=
                shared_counts[tid + offset];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        offset >>= 1;
    }

    if (tid == 0)
    {
        const int group_idx =
            get_group_id(0);

        partial_sums[group_idx] =
            shared_sums[0];

        partial_counts[group_idx] =
            shared_counts[0];
    }
}


__kernel void rhs_mean_kernel(
    __global const float *partial_sums,
    __global const float *partial_counts,
    const int partial_count,
    __global float *rhs_mean
)
{
    const int tid =
        get_local_id(0);

    const int local_size =
        get_local_size(0);

    __local float shared_sums[
        REDUCTION_THREADS_PER_BLOCK
    ];

    __local float shared_counts[
        REDUCTION_THREADS_PER_BLOCK
    ];

    float total_sum = 0.0f;
    float total_count = 0.0f;

    int idx = tid;

    while (idx < partial_count)
    {
        total_sum +=
            partial_sums[idx];

        total_count +=
            partial_counts[idx];

        idx += local_size;
    }

    shared_sums[tid] =
        total_sum;

    shared_counts[tid] =
        total_count;

    barrier(CLK_LOCAL_MEM_FENCE);

    int offset =
        local_size >> 1;

    while (offset > 0)
    {
        if (tid < offset)
        {
            shared_sums[tid] +=
                shared_sums[tid + offset];

            shared_counts[tid] +=
                shared_counts[tid + offset];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        offset >>= 1;
    }

    if (tid == 0)
    {
        if (shared_counts[0] > 0.0f)
        {
            rhs_mean[0] =
                shared_sums[0]
                / shared_counts[0];
        }
        else
        {
            rhs_mean[0] = 0.0f;
        }
    }
}


__kernel void subtract_rhs_mean_kernel(
    __global float *b,
    __global const float *rhs_mean,
    __global const int *tile_map,
    const int nx,
    const int ny,
    const int nz,
    const int tiles_x,
    const int tiles_y,
    const int tiles_z
)
{
    const int tile_i =
        get_group_id(0);

    const int tile_j =
        get_group_id(1);

    const int tile_k =
        get_group_id(2);

    const int local_k =
        get_local_id(0);

    const int local_j =
        get_local_id(1);

    const int local_i =
        get_local_id(2);

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
        k >= nz - 1
    )
        return;

    const int index =
        ((tile_index * TILE_SIZE + local_i)
        * TILE_SIZE + local_j)
        * TILE_SIZE + local_k;

    b[index] -= rhs_mean[0];
}


__kernel void add_artifical_divergence(
    __global const float *T,
    __global const uchar *source_mask,
    const float source_extra_pressure,
    const float noise_scale,
    const float noise_amplitude,
    const float noise_seed,
    const float expansion_rate,
    const float t_reference,
    __global float *b,
    __global const int *tile_map,
    const float rho,
    const float delta,
    const int nx,
    const int ny,
    const int nz,
    const float dt,
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
        k >= nz - 1
    )
        return;

    const int index =
        ((tile_index * TILE_SIZE + local_i)
        * TILE_SIZE + local_j)
        * TILE_SIZE + local_k;

    const float thermal_divergence =
        expansion_rate
        * (T[index] - t_reference);

    float extra_pressure_term = 0.0f;

    if (source_mask[index])
    {
        float scalar_multiplier = 1.0f;

        if (noise_amplitude != 0.0f)
        {
            const float scale =
                fmax(noise_scale, 1.0e-6f);

            const float noise_value =
                value_noise_3d(
                    (float)i / scale,
                    (float)j / scale,
                    (float)k / scale,
                    noise_seed
                );

            scalar_multiplier =
                fmax(
                    1.0f
                    + noise_value * noise_amplitude,
                    0.0f
                );
        }

        extra_pressure_term =
            (1.0f / dt)
            * source_extra_pressure
            * scalar_multiplier;
    }

    b[index] -=
        (rho / delta)
        * (
            thermal_divergence
            + extra_pressure_term
        );
}