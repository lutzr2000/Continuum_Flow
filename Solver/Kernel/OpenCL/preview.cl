// Pack active smoke and flame tiles into an atlas owned by the preview.
__kernel void pack_preview(__global const int *index_tile_map,
                           __global const float *smoke,
                           __global const float *flame,
                           __global float *preview_tile_lookup,
                           __global float *preview_fields,
                           volatile __global int *packed_tile_counter,
                           const int active_tile_count,
                           const int tiles_x,
                           const int tiles_y,
                           const int tiles_z,
                           const int atlas_tiles_x,
                           const int atlas_tiles_y,
                           const int tile_size) {
    const int tile_x = get_global_id(0);
    const int tile_y = get_global_id(1);
    const int tile_z = get_global_id(2);

    if (tile_x >= tiles_x || tile_y >= tiles_y || tile_z >= tiles_z)
        return;

    const int solver_map_index = (tile_x * tiles_y + tile_y) * tiles_z + tile_z;
    const int preview_map_index = (tile_z * tiles_y + tile_y) * tiles_x + tile_x;
    const int source_slot = index_tile_map[solver_map_index];

    if (source_slot < 0) {
        preview_tile_lookup[preview_map_index] = -1.0f;
        return;
    }

    const int packed_slot = atomic_add(packed_tile_counter, 1);

    if (packed_slot >= active_tile_count) {
        preview_tile_lookup[preview_map_index] = -1.0f;
        return;
    }

    preview_tile_lookup[preview_map_index] = (float)packed_slot;

    const int atlas_tile_x = packed_slot % atlas_tiles_x;
    const int atlas_tile_y = (packed_slot / atlas_tiles_x) % atlas_tiles_y;
    const int atlas_tile_z = packed_slot / (atlas_tiles_x * atlas_tiles_y);
    const int atlas_width = atlas_tiles_x * tile_size;
    const int atlas_height = atlas_tiles_y * tile_size;

    for (int local_x = 0; local_x < tile_size; ++local_x)
        for (int local_y = 0; local_y < tile_size; ++local_y)
            for (int local_z = 0; local_z < tile_size; ++local_z) {
                const int source_index =
                    ((source_slot * tile_size + local_x) * tile_size + local_y) * tile_size + local_z;
                const int atlas_x = atlas_tile_x * tile_size + local_x;
                const int atlas_y = atlas_tile_y * tile_size + local_y;
                const int atlas_z = atlas_tile_z * tile_size + local_z;
                const int atlas_index = (atlas_z * atlas_height + atlas_y) * atlas_width + atlas_x;
                preview_fields[atlas_index * 2] = smoke[source_index];
                preview_fields[atlas_index * 2 + 1] = flame[source_index];
            }
}
