#ifndef SPARSE_MANAGMENT_CL
#define SPARSE_MANAGMENT_CL

#ifndef TILE_SIZE
#define TILE_SIZE 4
#endif

inline float get_pool_value(
    __global const float *field,
    __global const int *index_tile_map,
    const int i,
    const int j,
    const int k,
    const float default_value,
    const int tiles_y,
    const int tiles_z
)
{
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
        index_tile_map[tile_map_index];

    if (tile_index == -1)
        return default_value;

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

    return field[index];
}

inline float get_pool_value_uint8(
    __global const uchar *field,
    __global const int *index_tile_map,
    const int i,
    const int j,
    const int k,
    const float default_value,
    const int tiles_y,
    const int tiles_z
)
{
    const int tile_i = i / TILE_SIZE;
    const int tile_j = j / TILE_SIZE;
    const int tile_k = k / TILE_SIZE;
    const int tile_map_index = (tile_i * tiles_y + tile_j) * tiles_z + tile_k;
    const int tile_index = index_tile_map[tile_map_index];

    if (tile_index == -1)
        return default_value;

    const int local_i = i - tile_i * TILE_SIZE;
    const int local_j = j - tile_j * TILE_SIZE;
    const int local_k = k - tile_k * TILE_SIZE;
    const int index = ((tile_index * TILE_SIZE + local_i) * TILE_SIZE + local_j)
        * TILE_SIZE + local_k;

    return (float)field[index];
}

__kernel void build_activity_mask(
    __global const float *smoke,
    __global const uchar *fuel,
    __global const float *flame,
    __global const int *index_tile_map,
    __global const uchar *source_tile_mask,
    __global uchar *activity_tile_map,
    const float threshold,
    const int nx,
    const int ny,
    const int nz,
    const int tile_count_x,
    const int tile_count_y,
    const int tile_count_z
)
{
    const int tile_i = get_global_id(0);
    const int tile_j = get_global_id(1);
    const int tile_k = get_global_id(2);

    if (
        tile_i >= tile_count_x ||
        tile_j >= tile_count_y ||
        tile_k >= tile_count_z
    )
        return;

    const int tile_map_index =
        (tile_i * tile_count_y + tile_j) * tile_count_z + tile_k;

    activity_tile_map[tile_map_index] = (uchar)0;

    // Sources activate a tile regardless of existing sparse allocation.
    if (source_tile_mask[tile_map_index])
    {
        activity_tile_map[tile_map_index] = (uchar)1;
        return;
    }

    const int tile_index = index_tile_map[tile_map_index];

    if (tile_index == -1)
        return;

    const int cell_i_start = tile_i * TILE_SIZE;
    const int cell_j_start = tile_j * TILE_SIZE;
    const int cell_k_start = tile_k * TILE_SIZE;

    for (int local_i = 0; local_i < TILE_SIZE; ++local_i)
    {
        const int i = cell_i_start + local_i;

        if (i >= nx)
            break;

        for (int local_j = 0; local_j < TILE_SIZE; ++local_j)
        {
            const int j = cell_j_start + local_j;

            if (j >= ny)
                break;

            for (int local_k = 0; local_k < TILE_SIZE; ++local_k)
            {
                const int k = cell_k_start + local_k;

                if (k >= nz)
                    break;

                const int index =
                    ((tile_index * TILE_SIZE + local_i)
                    * TILE_SIZE + local_j)
                    * TILE_SIZE + local_k;

                if (
                    smoke[index] >= threshold ||
                    (float)fuel[index] >= threshold ||
                    flame[index] >= threshold
                )
                {
                    activity_tile_map[tile_map_index] = (uchar)1;
                    return;
                }
            }
        }
    }
}


__kernel void dilate_activity_x(
    __global const uchar *src,
    __global uchar *dst,
    const int margin,
    const int tiles_x,
    const int tiles_y,
    const int tiles_z
)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int z = get_global_id(2);

    if (x >= tiles_x || y >= tiles_y || z >= tiles_z)
        return;

    int active = 0;
    for (int sample_x = max(x - margin, 0);
         sample_x <= min(x + margin, tiles_x - 1);
         ++sample_x)
    {
        const int index = (sample_x * tiles_y + y) * tiles_z + z;
        if (src[index])
        {
            active = 1;
            break;
        }
    }

    dst[(x * tiles_y + y) * tiles_z + z] = (uchar)active;
}


__kernel void dilate_activity_y(
    __global const uchar *src,
    __global uchar *dst,
    const int margin,
    const int tiles_x,
    const int tiles_y,
    const int tiles_z
)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int z = get_global_id(2);

    if (x >= tiles_x || y >= tiles_y || z >= tiles_z)
        return;

    int active = 0;
    for (int sample_y = max(y - margin, 0);
         sample_y <= min(y + margin, tiles_y - 1);
         ++sample_y)
    {
        const int index = (x * tiles_y + sample_y) * tiles_z + z;
        if (src[index])
        {
            active = 1;
            break;
        }
    }

    dst[(x * tiles_y + y) * tiles_z + z] = (uchar)active;
}


__kernel void dilate_activity_z(
    __global const uchar *src,
    __global uchar *dst,
    const int margin,
    const int tiles_x,
    const int tiles_y,
    const int tiles_z
)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int z = get_global_id(2);

    if (x >= tiles_x || y >= tiles_y || z >= tiles_z)
        return;

    int active = 0;
    for (int sample_z = max(z - margin, 0);
         sample_z <= min(z + margin, tiles_z - 1);
         ++sample_z)
    {
        const int index = (x * tiles_y + y) * tiles_z + sample_z;
        if (src[index])
        {
            active = 1;
            break;
        }
    }

    dst[(x * tiles_y + y) * tiles_z + z] = (uchar)active;
}


__kernel void activate_tiles_with_reuse(
    __global const uchar *activity_map,
    __global int *index_tile_map,
    __global const int *free_slot_list,
    volatile __global int *free_slot_count,
    __global int *reused_slot_list,
    volatile __global int *reused_slot_count,
    volatile __global int *next_pool_slot_counter,
    volatile __global int *active_tile_counter,
    const int tiles_x,
    const int tiles_y,
    const int tiles_z
)
{
    const int tile_i = get_global_id(0);
    const int tile_j = get_global_id(1);
    const int tile_k = get_global_id(2);

    if (
        tile_i >= tiles_x ||
        tile_j >= tiles_y ||
        tile_k >= tiles_z
    )
        return;

    const int tile_index =
        (tile_i * tiles_y + tile_j) * tiles_z + tile_k;

    if (!activity_map[tile_index])
        return;

    atomic_add(
        active_tile_counter,
        1
    );

    if (index_tile_map[tile_index] != -1)
        return;

    const int previous_free_count = atomic_add(
        free_slot_count,
        -1
    );

    if (previous_free_count > 0)
    {
        const int slot_index =
            free_slot_list[previous_free_count - 1];

        index_tile_map[tile_index] = slot_index;

        const int reused_index = atomic_add(
            reused_slot_count,
            1
        );

        reused_slot_list[reused_index] = slot_index;

        return;
    }

    atomic_add(
        free_slot_count,
        1
    );

    index_tile_map[tile_index] = atomic_add(
        next_pool_slot_counter,
        1
    );
}


__kernel void fill_sparse_tile_slots(
    __global float *pool,
    __global const int *slot_indices,
    const int slot_count,
    const float fill_value
)
{
    const int flat_index = get_global_id(0);

    const int cells_per_tile =
        TILE_SIZE * TILE_SIZE * TILE_SIZE;

    const int total_cell_count =
        slot_count * cells_per_tile;

    if (flat_index >= total_cell_count)
        return;

    const int slot_offset =
        flat_index / cells_per_tile;

    const int local_flat_index =
        flat_index % cells_per_tile;

    const int tile_index =
        slot_indices[slot_offset];

    const int local_i =
        local_flat_index / (TILE_SIZE * TILE_SIZE);

    const int remainder =
        local_flat_index % (TILE_SIZE * TILE_SIZE);

    const int local_j =
        remainder / TILE_SIZE;

    const int local_k =
        remainder % TILE_SIZE;

    const int index =
        ((tile_index * TILE_SIZE + local_i)
        * TILE_SIZE + local_j)
        * TILE_SIZE + local_k;

    pool[index] = fill_value;
}


__kernel void fill_sparse_tile_slots_uchar(
    __global uchar *pool,
    __global const int *slot_indices,
    const int slot_count,
    const uchar fill_value
)
{
    const int flat_index = get_global_id(0);

    const int cells_per_tile =
        TILE_SIZE * TILE_SIZE * TILE_SIZE;

    const int total_cell_count =
        slot_count * cells_per_tile;

    if (flat_index >= total_cell_count)
        return;

    const int slot_offset =
        flat_index / cells_per_tile;

    const int local_flat_index =
        flat_index % cells_per_tile;

    const int tile_index =
        slot_indices[slot_offset];

    const int local_i =
        local_flat_index / (TILE_SIZE * TILE_SIZE);

    const int remainder =
        local_flat_index % (TILE_SIZE * TILE_SIZE);

    const int local_j =
        remainder / TILE_SIZE;

    const int local_k =
        remainder % TILE_SIZE;

    const int index =
        ((tile_index * TILE_SIZE + local_i)
        * TILE_SIZE + local_j)
        * TILE_SIZE + local_k;

    pool[index] = fill_value;
}


__kernel void release_inactive_tile_slots(
    __global const uchar *activity_map,
    __global int *index_tile_map,
    __global int *free_slot_list,
    volatile __global int *free_slot_count,
    const int tiles_x,
    const int tiles_y,
    const int tiles_z
)
{
    const int tile_i = get_global_id(0);
    const int tile_j = get_global_id(1);
    const int tile_k = get_global_id(2);

    if (
        tile_i >= tiles_x ||
        tile_j >= tiles_y ||
        tile_k >= tiles_z
    )
        return;

    const int tile_index =
        (tile_i * tiles_y + tile_j) * tiles_z + tile_k;

    if (index_tile_map[tile_index] == -1)
        return;

    if (activity_map[tile_index])
        return;

    const int released_slot =
        index_tile_map[tile_index];

    index_tile_map[tile_index] = -1;

    const int list_index = atomic_add(
        free_slot_count,
        1
    );

    free_slot_list[list_index] =
        released_slot;
}
#endif // SPARSE_MANAGMENT_CL

