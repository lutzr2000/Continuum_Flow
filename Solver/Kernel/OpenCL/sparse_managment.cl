#ifndef TILE_SIZE
#define TILE_SIZE 4
#endif

inline float get_pool_value(
    __global const float *field,
    __global const int *tile_map,
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
        tile_map[tile_map_index];

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

__kernel void build_activity_mask(
    __global const float *smoke,
    __global const float *fuel,
    __global const float *flame,
    __global const int *tile_map,
    __global const uchar *source_tile_mask,
    __global int *base_tile_map,
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

    base_tile_map[tile_map_index] = -1;

    // Sources activate a tile regardless of existing sparse allocation.
    if (source_tile_mask[tile_map_index])
    {
        base_tile_map[tile_map_index] = 1;
        return;
    }

    const int tile_index = tile_map[tile_map_index];

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
                    fuel[index] >= threshold ||
                    flame[index] >= threshold
                )
                {
                    base_tile_map[tile_map_index] = 1;
                    return;
                }
            }
        }
    }
}


inline int tile_is_active_in_margin(
    __global const int *base_tile_map,
    const int tile_i,
    const int tile_j,
    const int tile_k,
    const int margin,
    const int tiles_x,
    const int tiles_y,
    const int tiles_z
)
{
    const int min_i = max(tile_i - margin, 0);
    const int min_j = max(tile_j - margin, 0);
    const int min_k = max(tile_k - margin, 0);

    const int max_i = min(tile_i + margin, tiles_x - 1);
    const int max_j = min(tile_j + margin, tiles_y - 1);
    const int max_k = min(tile_k + margin, tiles_z - 1);

    for (int i = min_i; i <= max_i; ++i)
    {
        for (int j = min_j; j <= max_j; ++j)
        {
            for (int k = min_k; k <= max_k; ++k)
            {
                const int index =
                    (i * tiles_y + j) * tiles_z + k;

                if (base_tile_map[index] != -1)
                    return 1;
            }
        }
    }

    return 0;
}


__kernel void activate_tiles_with_reuse(
    __global const int *base_tile_map,
    __global int *tile_map,
    const int margin,
    __global const int *free_slot_stack,
    volatile __global int *free_slot_count,
    __global int *reused_slot_stack,
    volatile __global int *reused_slot_count,
    volatile __global int *next_tile_index_counter,
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

    if (
        !tile_is_active_in_margin(
            base_tile_map,
            tile_i,
            tile_j,
            tile_k,
            margin,
            tiles_x,
            tiles_y,
            tiles_z
        )
    )
        return;

    atomic_add(
        active_tile_counter,
        1
    );

    const int tile_index =
        (tile_i * tiles_y + tile_j) * tiles_z + tile_k;

    if (tile_map[tile_index] != -1)
        return;

    const int previous_free_count = atomic_add(
        free_slot_count,
        -1
    );

    if (previous_free_count > 0)
    {
        const int slot_index =
            free_slot_stack[previous_free_count - 1];

        tile_map[tile_index] = slot_index;

        const int reused_index = atomic_add(
            reused_slot_count,
            1
        );

        reused_slot_stack[reused_index] = slot_index;

        return;
    }

    atomic_add(
        free_slot_count,
        1
    );

    tile_map[tile_index] = atomic_add(
        next_tile_index_counter,
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
    __global const int *base_tile_map,
    __global int *tile_map,
    const int margin,
    __global int *free_slot_stack,
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

    if (tile_map[tile_index] == -1)
        return;

    if (
        tile_is_active_in_margin(
            base_tile_map,
            tile_i,
            tile_j,
            tile_k,
            margin,
            tiles_x,
            tiles_y,
            tiles_z
        )
    )
        return;

    const int released_slot =
        tile_map[tile_index];

    tile_map[tile_index] = -1;

    const int stack_index = atomic_add(
        free_slot_count,
        1
    );

    free_slot_stack[stack_index] =
        released_slot;
}
