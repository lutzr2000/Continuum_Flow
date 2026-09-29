#ifndef TILE_SIZE
#define TILE_SIZE 4
#endif

__kernel void velocity_maxima_partial(
    __global const float *u,
    __global const float *v,
    __global const float *w,
    __global const int *tile_map,
    __global float *partial_maxima,
    const int total_tile_count,
    const int tiles_y,
    const int tiles_z,
    __local float *s_u,
    __local float *s_v,
    __local float *s_w
)
{
    const int tid = get_local_id(0);
    const int local_size = get_local_size(0);

    int idx = get_global_id(0);
    const int stride = get_global_size(0);

    const int cells_per_tile =
        TILE_SIZE * TILE_SIZE * TILE_SIZE;

    const int tiles_per_yz =
        tiles_y * tiles_z;

    float max_u = 0.0f;
    float max_v = 0.0f;
    float max_w = 0.0f;

    while (idx < total_tile_count)
    {
        const int tile_i =
            idx / tiles_per_yz;

        const int remainder =
            idx % tiles_per_yz;

        const int tile_j =
            remainder / tiles_z;

        const int tile_k =
            remainder % tiles_z;

        const int tile_map_index =
            (tile_i * tiles_y + tile_j) * tiles_z + tile_k;

        const int tile_index =
            tile_map[tile_map_index];

        if (tile_index != -1)
        {
            const int tile_offset =
                tile_index * cells_per_tile;

            for (
                int local_flat = 0;
                local_flat < cells_per_tile;
                ++local_flat
            )
            {
                const int index =
                    tile_offset + local_flat;

                max_u = fmax(
                    max_u,
                    fabs(u[index])
                );

                max_v = fmax(
                    max_v,
                    fabs(v[index])
                );

                max_w = fmax(
                    max_w,
                    fabs(w[index])
                );
            }
        }

        idx += stride;
    }

    s_u[tid] = max_u;
    s_v[tid] = max_v;
    s_w[tid] = max_w;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (
        int offset = local_size / 2;
        offset > 0;
        offset >>= 1
    )
    {
        if (tid < offset)
        {
            s_u[tid] = fmax(
                s_u[tid],
                s_u[tid + offset]
            );

            s_v[tid] = fmax(
                s_v[tid],
                s_v[tid + offset]
            );

            s_w[tid] = fmax(
                s_w[tid],
                s_w[tid + offset]
            );
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid == 0)
    {
        const int group = get_group_id(0);
        const int output = group * 3;

        partial_maxima[output] = s_u[0];
        partial_maxima[output + 1] = s_v[0];
        partial_maxima[output + 2] = s_w[0];
    }
}


__kernel void velocity_maxima_final(
    __global const float *partial_maxima,
    __global float *maxima,
    const int partial_count,
    __local float *s_u,
    __local float *s_v,
    __local float *s_w
)
{
    const int tid = get_local_id(0);
    const int local_size = get_local_size(0);

    float max_u = 0.0f;
    float max_v = 0.0f;
    float max_w = 0.0f;

    for (
        int index = tid;
        index < partial_count;
        index += local_size
    )
    {
        const int offset = index * 3;

        max_u = fmax(
            max_u,
            partial_maxima[offset]
        );

        max_v = fmax(
            max_v,
            partial_maxima[offset + 1]
        );

        max_w = fmax(
            max_w,
            partial_maxima[offset + 2]
        );
    }

    s_u[tid] = max_u;
    s_v[tid] = max_v;
    s_w[tid] = max_w;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (
        int offset = local_size / 2;
        offset > 0;
        offset >>= 1
    )
    {
        if (tid < offset)
        {
            s_u[tid] = fmax(
                s_u[tid],
                s_u[tid + offset]
            );

            s_v[tid] = fmax(
                s_v[tid],
                s_v[tid + offset]
            );

            s_w[tid] = fmax(
                s_w[tid],
                s_w[tid + offset]
            );
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid == 0)
    {
        maxima[0] = s_u[0];
        maxima[1] = s_v[0];
        maxima[2] = s_w[0];
    }
}