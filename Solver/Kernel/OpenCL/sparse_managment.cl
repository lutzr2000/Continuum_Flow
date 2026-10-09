#ifndef SPARSE_MANAGMENT_CL
#define SPARSE_MANAGMENT_CL

typedef struct {
    int i;
    int j;
    int k;
    int tile_index;
    int cell_index;
    int valid;
} SparseCell;

inline SparseCell get_sparse_cell_at(
    __global const int *index_tile_map, const int i, const int j, const int k, const int tiles_y, const int tiles_z) {
    /*
    Resolve explicit grid coordinates to their sparse tile and pool indices.
    */
    SparseCell cell;
    cell.valid = 0;

    if (i < 0 || j < 0 || k < 0)
        return cell;

    const int tile_i = i / TILE_SIZE;
    const int tile_j = j / TILE_SIZE;
    const int tile_k = k / TILE_SIZE;

    const int local_i = i - tile_i * TILE_SIZE;
    const int local_j = j - tile_j * TILE_SIZE;
    const int local_k = k - tile_k * TILE_SIZE;

    const int tile_map_index = (tile_i * tiles_y + tile_j) * tiles_z + tile_k;
    cell.tile_index = index_tile_map[tile_map_index];

    if (cell.tile_index == -1)
        return cell;

    cell.i = i;
    cell.j = j;
    cell.k = k;
    cell.cell_index = ((cell.tile_index * TILE_SIZE + local_i) * TILE_SIZE + local_j) * TILE_SIZE + local_k;
    cell.valid = 1;

    return cell;
}

inline SparseCell
get_sparse_cell(__global const int *index_tile_map, const int tiles_x, const int tiles_y, const int tiles_z) {
    /*
    Resolve the current work-item to its cell coordinates and sparse pool indices.
    */
    SparseCell cell;
    cell.valid = 0;

    const int tile_i = get_group_id(0);
    const int tile_j = get_group_id(1);
    const int tile_k = get_group_id(2);

    const int local_k = get_local_id(0);
    const int local_j = get_local_id(1);
    const int local_i = get_local_id(2);

    if (tile_i >= tiles_x || tile_j >= tiles_y || tile_k >= tiles_z || local_i >= TILE_SIZE || local_j >= TILE_SIZE ||
        local_k >= TILE_SIZE)
        return cell;

    const int i = tile_i * TILE_SIZE + local_i;
    const int j = tile_j * TILE_SIZE + local_j;
    const int k = tile_k * TILE_SIZE + local_k;

    return get_sparse_cell_at(index_tile_map, i, j, k, tiles_y, tiles_z);
}

inline float get_pool_value(__global const float *field,
                            __global const int *index_tile_map,
                            const int i,
                            const int j,
                            const int k,
                            const float default_value,
                            const int tiles_y,
                            const int tiles_z) {
    /*
    Helper device kernel for accessing a cells value in the tile pool
    */
    const SparseCell cell = get_sparse_cell_at(index_tile_map, i, j, k, tiles_y, tiles_z);

    if (!cell.valid)
        return default_value;

    return field[cell.cell_index];
}

__kernel void build_activity_mask(__global const float *smoke,
                                  __global const float *fuel,
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
                                  const int tile_count_z) {
    /*
    Determine activity based on fuel, smoke and flame.
    Sources activate the flow too.
    */
    const int tile_i = get_global_id(0);
    const int tile_j = get_global_id(1);
    const int tile_k = get_global_id(2);

    if (tile_i >= tile_count_x || tile_j >= tile_count_y || tile_k >= tile_count_z)
        return;

    const int tile_map_index = (tile_i * tile_count_y + tile_j) * tile_count_z + tile_k;

    activity_tile_map[tile_map_index] = (uchar)0;

    // Sources activate a tile regardless of existing sparse allocation.
    if (source_tile_mask[tile_map_index]) {
        activity_tile_map[tile_map_index] = (uchar)1;
        return;
    }

    const int tile_index = index_tile_map[tile_map_index];

    if (tile_index == -1)
        return;

    const int cell_i_start = tile_i * TILE_SIZE;
    const int cell_j_start = tile_j * TILE_SIZE;
    const int cell_k_start = tile_k * TILE_SIZE;

    for (int local_i = 0; local_i < TILE_SIZE; ++local_i) {
        const int i = cell_i_start + local_i;

        if (i >= nx)
            break;

        for (int local_j = 0; local_j < TILE_SIZE; ++local_j) {
            const int j = cell_j_start + local_j;

            if (j >= ny)
                break;

            for (int local_k = 0; local_k < TILE_SIZE; ++local_k) {
                const int k = cell_k_start + local_k;

                if (k >= nz)
                    break;

                const int index = ((tile_index * TILE_SIZE + local_i) * TILE_SIZE + local_j) * TILE_SIZE + local_k;

                if (smoke[index] >= threshold || fuel[index] >= threshold || flame[index] >= threshold) {
                    activity_tile_map[tile_map_index] = (uchar)1;
                    return;
                }
            }
        }
    }
}

__kernel void dilate_activity_x(__global const uchar *src,
                                __global uchar *dst,
                                const int margin,
                                const int tiles_x,
                                const int tiles_y,
                                const int tiles_z) {
    /*
    Since the flow needs to be able to move out of the active region
    the activity map needs to be dilated. This kernel dilates in the
    x-direciton.
    */
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int z = get_global_id(2);

    if (x >= tiles_x || y >= tiles_y || z >= tiles_z)
        return;

    int active = 0;
    for (int sample_x = max(x - margin, 0); sample_x <= min(x + margin, tiles_x - 1); ++sample_x) {
        const int index = (sample_x * tiles_y + y) * tiles_z + z;
        if (src[index]) {
            active = 1;
            break;
        }
    }

    dst[(x * tiles_y + y) * tiles_z + z] = (uchar)active;
}

__kernel void dilate_activity_y(__global const uchar *src,
                                __global uchar *dst,
                                const int margin,
                                const int tiles_x,
                                const int tiles_y,
                                const int tiles_z) {
    /*
    Same as dilate_activity_x but in the y-direction
    */
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int z = get_global_id(2);

    if (x >= tiles_x || y >= tiles_y || z >= tiles_z)
        return;

    int active = 0;
    for (int sample_y = max(y - margin, 0); sample_y <= min(y + margin, tiles_y - 1); ++sample_y) {
        const int index = (x * tiles_y + sample_y) * tiles_z + z;
        if (src[index]) {
            active = 1;
            break;
        }
    }

    dst[(x * tiles_y + y) * tiles_z + z] = (uchar)active;
}

__kernel void dilate_activity_z(__global const uchar *src,
                                __global uchar *dst,
                                const int margin,
                                const int tiles_x,
                                const int tiles_y,
                                const int tiles_z) {
    /*
    Same as dilate_activity_x but in the z-direction
    */
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int z = get_global_id(2);

    if (x >= tiles_x || y >= tiles_y || z >= tiles_z)
        return;

    int active = 0;
    for (int sample_z = max(z - margin, 0); sample_z <= min(z + margin, tiles_z - 1); ++sample_z) {
        const int index = (x * tiles_y + y) * tiles_z + sample_z;
        if (src[index]) {
            active = 1;
            break;
        }
    }

    dst[(x * tiles_y + y) * tiles_z + z] = (uchar)active;
}

__kernel void activate_tiles_with_reuse(__global const uchar *activity_map,
                                        __global int *index_tile_map,
                                        __global const int *free_slot_list,
                                        volatile __global int *free_slot_count,
                                        __global int *reused_slot_list,
                                        volatile __global int *reused_slot_count,
                                        volatile __global int *next_pool_slot_counter,
                                        volatile __global int *active_tile_counter,
                                        const int tiles_x,
                                        const int tiles_y,
                                        const int tiles_z) {
    /*
    Allocate pool slots for active tiles. Reuse released slots before growing the tile pool.
    */
    const int tile_i = get_global_id(0);
    const int tile_j = get_global_id(1);
    const int tile_k = get_global_id(2);

    if (tile_i >= tiles_x || tile_j >= tiles_y || tile_k >= tiles_z)
        return;

    const int tile_index = (tile_i * tiles_y + tile_j) * tiles_z + tile_k;

    if (!activity_map[tile_index])
        return;

    atomic_add(active_tile_counter, 1);

    if (index_tile_map[tile_index] != -1)
        return;

    const int previous_free_count = atomic_add(free_slot_count, -1);

    if (previous_free_count > 0) {
        const int slot_index = free_slot_list[previous_free_count - 1];

        index_tile_map[tile_index] = slot_index;

        const int reused_index = atomic_add(reused_slot_count, 1);

        reused_slot_list[reused_index] = slot_index;

        return;
    }

    atomic_add(free_slot_count, 1);

    index_tile_map[tile_index] = atomic_add(next_pool_slot_counter, 1);
}

__kernel void fill_sparse_tile_slots(__global float *pool,
                                     __global const int *slot_indices,
                                     const int slot_count,
                                     const float fill_value) {
    /*
    Initialize every cell in the specified floating-point tile pool slots with a constant value.
    */
    const int flat_index = get_global_id(0);

    const int cells_per_tile = TILE_SIZE * TILE_SIZE * TILE_SIZE;

    const int total_cell_count = slot_count * cells_per_tile;

    if (flat_index >= total_cell_count)
        return;

    const int slot_offset = flat_index / cells_per_tile;

    const int local_flat_index = flat_index % cells_per_tile;

    const int tile_index = slot_indices[slot_offset];

    const int local_i = local_flat_index / (TILE_SIZE * TILE_SIZE);

    const int remainder = local_flat_index % (TILE_SIZE * TILE_SIZE);

    const int local_j = remainder / TILE_SIZE;

    const int local_k = remainder % TILE_SIZE;

    const int index = ((tile_index * TILE_SIZE + local_i) * TILE_SIZE + local_j) * TILE_SIZE + local_k;

    pool[index] = fill_value;
}

__kernel void fill_sparse_tile_slots_uchar(__global uchar *pool,
                                           __global const int *slot_indices,
                                           const int slot_count,
                                           const uchar fill_value) {
    /*
    Initialize every cell in the specified unsigned-byte tile pool slots with a constant value.
    */
    const int flat_index = get_global_id(0);

    const int cells_per_tile = TILE_SIZE * TILE_SIZE * TILE_SIZE;

    const int total_cell_count = slot_count * cells_per_tile;

    if (flat_index >= total_cell_count)
        return;

    const int slot_offset = flat_index / cells_per_tile;

    const int local_flat_index = flat_index % cells_per_tile;

    const int tile_index = slot_indices[slot_offset];

    const int local_i = local_flat_index / (TILE_SIZE * TILE_SIZE);

    const int remainder = local_flat_index % (TILE_SIZE * TILE_SIZE);

    const int local_j = remainder / TILE_SIZE;

    const int local_k = remainder % TILE_SIZE;

    const int index = ((tile_index * TILE_SIZE + local_i) * TILE_SIZE + local_j) * TILE_SIZE + local_k;

    pool[index] = fill_value;
}

__kernel void release_inactive_tile_slots(__global const uchar *activity_map,
                                          __global int *index_tile_map,
                                          __global int *free_slot_list,
                                          volatile __global int *free_slot_count,
                                          const int tiles_x,
                                          const int tiles_y,
                                          const int tiles_z) {
    /*
    Release pool slots belonging to inactive tiles and add them to the reusable free-slot list.
    */
    const int tile_i = get_global_id(0);
    const int tile_j = get_global_id(1);
    const int tile_k = get_global_id(2);

    if (tile_i >= tiles_x || tile_j >= tiles_y || tile_k >= tiles_z)
        return;

    const int tile_index = (tile_i * tiles_y + tile_j) * tiles_z + tile_k;

    if (index_tile_map[tile_index] == -1)
        return;

    if (activity_map[tile_index])
        return;

    const int released_slot = index_tile_map[tile_index];

    index_tile_map[tile_index] = -1;

    const int list_index = atomic_add(free_slot_count, 1);

    free_slot_list[list_index] = released_slot;
}
#endif // SPARSE_MANAGMENT_CL
