__kernel void mark_source_tiles(__global uchar *source_tile_mask,
                                const int size_x,
                                const int size_y,
                                const int size_z,
                                const int offset_i,
                                const int offset_j,
                                const int offset_k) {
    /*
    Kernel used for marking tiles as active which contain sources.
    */
    int i = get_global_id(0) + offset_i;
    int j = get_global_id(1) + offset_j;
    int k = get_global_id(2) + offset_k;

    if (i < size_x && j < size_y && k < size_z) {
        int index = (i * size_y + j) * size_z + k;

        source_tile_mask[index] = 1;
    }
}

__kernel void update_source_masks(__global uchar *mask,
                                  __global const int *index_tile_map,
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
                                  const int local_size_z) {
    /*
    This kernel updates the global source mask using a local source mask.
    Each global cell is transformed into the local coordinate system of
    the source. If the corresponding local mask cell is active, the
    global cell is marked as active.
    */
    const int ti = get_group_id(0) + offset_i;
    const int tj = get_group_id(1) + offset_j;
    const int tk = get_group_id(2) + offset_k;

    const int k = get_local_id(0);
    const int j = get_local_id(1);
    const int i = get_local_id(2);

    if (ti >= tiles_x || tj >= tiles_y || tk >= tiles_z)
        return;

    const int tile_map_index = (ti * tiles_y + tj) * tiles_z + tk;

    const int tile = index_tile_map[tile_map_index];

    if (tile < 0)
        return;

    if (i >= TILE_SIZE || j >= TILE_SIZE || k >= TILE_SIZE)
        return;

    const int gi = ti * TILE_SIZE + i;
    const int gj = tj * TILE_SIZE + j;
    const int gk = tk * TILE_SIZE + k;

    const float fi = a00 * gi + a01 * gj + a02 * gk + c0;
    const float fj = a10 * gi + a11 * gj + a12 * gk + c1;
    const float fk = a20 * gi + a21 * gj + a22 * gk + c2;

    const int bi = (int)floor(fi + 0.5f);
    const int bj = (int)floor(fj + 0.5f);
    const int bk = (int)floor(fk + 0.5f);

    if (bi >= 0 && bi < local_size_x && bj >= 0 && bj < local_size_y && bk >= 0 && bk < local_size_z) {
        const int local_mask_index = (bi * local_size_y + bj) * local_size_z + bk;

        if (local_mask[local_mask_index]) {
            const int mask_index = ((tile * TILE_SIZE + i) * TILE_SIZE + j) * TILE_SIZE + k;

            mask[mask_index] = 1;
        }
    }
}

__kernel void update_source_velocity(__global float *velocity_x,
                                     __global float *velocity_y,
                                     __global float *velocity_z,
                                     __global const int *index_tile_map,
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
                                     const float source_u,
                                     const float source_v,
                                     const float source_w,
                                     const int offset_i,
                                     const int offset_j,
                                     const int offset_k,
                                     const int tiles_x,
                                     const int tiles_y,
                                     const int tiles_z,
                                     const int local_size_x,
                                     const int local_size_y,
                                     const int local_size_z) {
    /*
    This kernel is used to set the source velocity into scratch arrays.
    */
    const int ti = get_group_id(0) + offset_i;
    const int tj = get_group_id(1) + offset_j;
    const int tk = get_group_id(2) + offset_k;

    const int k = get_local_id(0);
    const int j = get_local_id(1);
    const int i = get_local_id(2);

    if (ti >= tiles_x || tj >= tiles_y || tk >= tiles_z)
        return;

    const int tile_map_index = (ti * tiles_y + tj) * tiles_z + tk;

    const int tile = index_tile_map[tile_map_index];

    if (tile < 0)
        return;

    if (i >= TILE_SIZE || j >= TILE_SIZE || k >= TILE_SIZE)
        return;

    const int gi = ti * TILE_SIZE + i;
    const int gj = tj * TILE_SIZE + j;
    const int gk = tk * TILE_SIZE + k;

    const int bi = (int)floor(a00 * gi + a01 * gj + a02 * gk + c0 + 0.5f);
    const int bj = (int)floor(a10 * gi + a11 * gj + a12 * gk + c1 + 0.5f);
    const int bk = (int)floor(a20 * gi + a21 * gj + a22 * gk + c2 + 0.5f);

    if (bi >= 0 && bi < local_size_x && bj >= 0 && bj < local_size_y && bk >= 0 && bk < local_size_z) {
        const int local_mask_index = (bi * local_size_y + bj) * local_size_z + bk;

        if (local_mask[local_mask_index]) {
            const int index = ((tile * TILE_SIZE + i) * TILE_SIZE + j) * TILE_SIZE + k;

            velocity_x[index] = source_u;
            velocity_y[index] = source_v;
            velocity_z[index] = source_w;
        }
    }
}

__kernel void update_obstacle_mask(__global uchar *mask,
                                   __global float *velocity_x,
                                   __global float *velocity_y,
                                   __global float *velocity_z,
                                   __global const int *index_tile_map,
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
                                   const float vx_i,
                                   const float vx_j,
                                   const float vx_k,
                                   const float vx_c,
                                   const float vy_i,
                                   const float vy_j,
                                   const float vy_k,
                                   const float vy_c,
                                   const float vz_i,
                                   const float vz_j,
                                   const float vz_k,
                                   const float vz_c,
                                   const int offset_i,
                                   const int offset_j,
                                   const int offset_k,
                                   const int tiles_x,
                                   const int tiles_y,
                                   const int tiles_z,
                                   const int local_size_x,
                                   const int local_size_y,
                                   const int local_size_z) {
    /*
    Similar to update_source_masks.
    */
    const int ti = get_group_id(0) + offset_i;
    const int tj = get_group_id(1) + offset_j;
    const int tk = get_group_id(2) + offset_k;

    const int k = get_local_id(0);
    const int j = get_local_id(1);
    const int i = get_local_id(2);

    if (ti >= tiles_x || tj >= tiles_y || tk >= tiles_z)
        return;

    const int tile_map_index = (ti * tiles_y + tj) * tiles_z + tk;

    const int tile = index_tile_map[tile_map_index];

    if (tile < 0)
        return;

    if (i >= TILE_SIZE || j >= TILE_SIZE || k >= TILE_SIZE)
        return;

    const int gi = ti * TILE_SIZE + i;
    const int gj = tj * TILE_SIZE + j;
    const int gk = tk * TILE_SIZE + k;

    const float fi = a00 * gi + a01 * gj + a02 * gk + c0;
    const float fj = a10 * gi + a11 * gj + a12 * gk + c1;
    const float fk = a20 * gi + a21 * gj + a22 * gk + c2;

    const int bi = (int)floor(fi + 0.5f);
    const int bj = (int)floor(fj + 0.5f);
    const int bk = (int)floor(fk + 0.5f);

    if (bi < 0 || bi >= local_size_x || bj < 0 || bj >= local_size_y || bk < 0 || bk >= local_size_z)
        return;

    const int local_mask_index = (bi * local_size_y + bj) * local_size_z + bk;

    if (!local_mask[local_mask_index])
        return;

    const int index = ((tile * TILE_SIZE + i) * TILE_SIZE + j) * TILE_SIZE + k;

    mask[index] = 1;

    velocity_x[index] = vx_i * gi + vx_j * gj + vx_k * gk + vx_c;
    velocity_y[index] = vy_i * gi + vy_j * gj + vy_k * gk + vy_c;
    velocity_z[index] = vz_i * gi + vz_j * gj + vz_k * gk + vz_c;
}