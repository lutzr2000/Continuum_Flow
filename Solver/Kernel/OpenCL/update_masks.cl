__kernel void mark_source_tiles(
    __global uchar *source_tile_mask,
    const int size_x,
    const int size_y,
    const int size_z,
    const int offset_i,
    const int offset_j,
    const int offset_k
)
{
    int i = get_global_id(0) + offset_i;
    int j = get_global_id(1) + offset_j;
    int k = get_global_id(2) + offset_k;

    if (
        i < size_x &&
        j < size_y &&
        k < size_z
    )
    {
        int index =
            (i * size_y + j) * size_z + k;

        source_tile_mask[index] = 1;
    }
}