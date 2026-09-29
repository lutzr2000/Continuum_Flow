#ifndef TILE_SIZE
#define TILE_SIZE 4
#endif

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

__kernel void update_source_masks(
    __global uchar *mask,
    __global const int *tile_map,
    __global const uchar *local_mask,
    const float c0,
    const float c1,
    const float c2,
    const float a00,
    const float a01,
    const float a02,
    const float a10,
    const float a11,
    const float a12,
    const float a20,
    const float a21,
    const float a22,
    const int offset_i,
    const int offset_j,
    const int offset_k,
    const int tiles_x,
    const int tiles_y,
    const int tiles_z,
    const int local_size_x,
    const int local_size_y,
    const int local_size_z
)
{
    const int ti = get_group_id(0) + offset_i;
    const int tj = get_group_id(1) + offset_j;
    const int tk = get_group_id(2) + offset_k;

    const int k = get_local_id(0);
    const int j = get_local_id(1);
    const int i = get_local_id(2);

    if (
        ti >= tiles_x ||
        tj >= tiles_y ||
        tk >= tiles_z
    )
        return;

    const int tile_map_index =
        (ti * tiles_y + tj) * tiles_z + tk;

    const int tile = tile_map[tile_map_index];

    if (tile < 0)
        return;

    if (
        i >= TILE_SIZE ||
        j >= TILE_SIZE ||
        k >= TILE_SIZE
    )
        return;

    const int gi = ti * TILE_SIZE + i;
    const int gj = tj * TILE_SIZE + j;
    const int gk = tk * TILE_SIZE + k;

    const float fi =
        a00 * gi +
        a01 * gj +
        a02 * gk +
        c0;

    const float fj =
        a10 * gi +
        a11 * gj +
        a12 * gk +
        c1;

    const float fk =
        a20 * gi +
        a21 * gj +
        a22 * gk +
        c2;

    const int bi = (int)floor(fi + 0.5f);
    const int bj = (int)floor(fj + 0.5f);
    const int bk = (int)floor(fk + 0.5f);

    if (
        bi >= 0 && bi < local_size_x &&
        bj >= 0 && bj < local_size_y &&
        bk >= 0 && bk < local_size_z
    )
    {
        const int local_mask_index =
            (bi * local_size_y + bj) * local_size_z + bk;

        if (local_mask[local_mask_index])
        {
            const int mask_index =
                ((tile * TILE_SIZE + i)
                * TILE_SIZE + j)
                * TILE_SIZE + k;

            mask[mask_index] = 1;
        }
    }
}