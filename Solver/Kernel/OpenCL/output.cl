__kernel void write_metadata(
    __global const int *tile_map,
    __global int *active_tile_meta,
    __global int *active_tile_count,
    const int tile_size,
    const int tiles_x,
    const int tiles_y,
    const int tiles_z
)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int z = get_global_id(2);

    if (
        x >= tiles_x ||
        y >= tiles_y ||
        z >= tiles_z
    )
        return;

    const int tile_map_index =
        (x * tiles_y + y)
        * tiles_z + z;

    const int tile_idx =
        tile_map[tile_map_index];

    if (tile_idx >= 0)
    {
        const int active_idx =
            atomic_inc(
                (volatile __global unsigned int *)
                active_tile_count
            );

        const int meta_index =
            active_idx * 4;

        active_tile_meta[meta_index + 0] =
            tile_idx;

        active_tile_meta[meta_index + 1] =
            x * tile_size;

        active_tile_meta[meta_index + 2] =
            y * tile_size;

        active_tile_meta[meta_index + 3] =
            z * tile_size;
    }
}