inline float3 interpolated_particle_vector(__global const float *current_values,
                                           __global const float *next_values,
                                           const int sample_index,
                                           const int next_count,
                                           const float alpha) {
    /*
    Interpolates a particle vector between two frames to avoid abrupt changes.
    */
    int index = sample_index * 3;

    float px = current_values[index + 0];
    float py = current_values[index + 1];
    float pz = current_values[index + 2];

    if (sample_index < next_count) {
        px += (next_values[index + 0] - px) * alpha;
        py += (next_values[index + 1] - py) * alpha;
        pz += (next_values[index + 2] - pz) * alpha;
    }

    return (float3)(px, py, pz);
}

inline float point_segment_distance_squared(
    float px, float py, float pz, float ax, float ay, float az, float bx, float by, float bz) {
    /*
    Computes the squared shortest distance from a point to a finite line segment.
    */
    float abx = bx - ax;
    float aby = by - ay;
    float abz = bz - az;

    float apx = px - ax;
    float apy = py - ay;
    float apz = pz - az;

    float length_squared = abx * abx + aby * aby + abz * abz;

    float t = 0.0f;

    if (length_squared > 0.0f) {
        t = (apx * abx + apy * aby + apz * abz) / length_squared;

        t = clamp(t, 0.0f, 1.0f);
    }

    float dx = px - (ax + t * abx);
    float dy = py - (ay + t * aby);
    float dz = pz - (az + t * abz);

    return dx * dx + dy * dy + dz * dz;
}

inline void particle_grid_bounds(float px,
                                 float py,
                                 float pz,
                                 float radius,
                                 float spacing,
                                 float origin_x,
                                 float origin_y,
                                 float origin_z,
                                 int size_i,
                                 int size_j,
                                 int size_k,
                                 int *min_i,
                                 int *min_j,
                                 int *min_k,
                                 int *max_i,
                                 int *max_j,
                                 int *max_k) {
    /*
    Computes the clamped grid bounds covered by a particle and its radius.
    */
    *min_i = max((int)floor((px - radius - origin_x) / spacing), 0);
    *min_j = max((int)floor((py - radius - origin_y) / spacing), 0);
    *min_k = max((int)floor((pz - radius - origin_z) / spacing), 0);
    *max_i = min((int)floor((px + radius - origin_x) / spacing), size_i - 1);
    *max_j = min((int)floor((py + radius - origin_y) / spacing), size_j - 1);
    *max_k = min((int)floor((pz + radius - origin_z) / spacing), size_k - 1);
}

inline void linear_to_grid_index(
    int linear_index, int min_i, int min_j, int min_k, int count_j, int count_k, int *i, int *j, int *k) {
    /*
    Converts a linear index within a bounded grid region to three-dimensional grid coordinates.
    */
    int entries_per_i = count_j * count_k;
    int local_i = linear_index / entries_per_i;
    int remainder = linear_index - local_i * entries_per_i;
    int local_j = remainder / count_k;
    int local_k = remainder - local_j * count_k;

    *i = min_i + local_i;
    *j = min_j + local_j;
    *k = min_k + local_k;
}

inline void atomic_add_float(volatile __global float *address, float value) {
    /*
    Atomically adds a floating-point value by using an integer compare-and-swap loop.
    */
    union {
        unsigned int int_value;
        float float_value;
    } old_value;

    union {
        unsigned int int_value;
        float float_value;
    } new_value;

    volatile __global unsigned int *address_uint = (volatile __global unsigned int *)address;

    old_value.int_value = *address_uint;

    while (1) {
        new_value.float_value = old_value.float_value + value;

        unsigned int previous = atomic_cmpxchg(address_uint, old_value.int_value, new_value.int_value);

        if (previous == old_value.int_value)
            break;

        old_value.int_value = previous;
    }
}

__kernel void sample_interpolated_vectors(__global float *output,
                                          __global const float *current_values,
                                          __global const float *next_values,
                                          const int count,
                                          const int next_count,
                                          const float alpha) {
    /*
    Samples interpolated particle vectors for a position between the current and next frame.
    */
    int sample_index = get_global_id(0);

    if (sample_index >= count)
        return;

    float3 value = interpolated_particle_vector(current_values, next_values, sample_index, next_count, alpha);

    int index = sample_index * 3;

    output[index + 0] = value.x;
    output[index + 1] = value.y;
    output[index + 2] = value.z;
}

__kernel void mark_particle_tiles(__global uchar *tile_mask,
                                  __global const float *previous_positions,
                                  __global const float *current_positions,
                                  const int count,
                                  const int previous_count,
                                  const float radius,
                                  const float delta,
                                  const float origin_x,
                                  const float origin_y,
                                  const float origin_z,
                                  const int size_x,
                                  const int size_y,
                                  const int size_z) {
    /*
    Marks every grid tile touched by a particle's swept path between two frames.
    */
    int sample_index = get_group_id(0);

    if (sample_index >= count)
        return;

    int position_index = sample_index * 3;

    float px = current_positions[position_index + 0];
    float py = current_positions[position_index + 1];
    float pz = current_positions[position_index + 2];

    float previous_px = px;
    float previous_py = py;
    float previous_pz = pz;

    if (sample_index < previous_count) {
        previous_px = previous_positions[position_index + 0];
        previous_py = previous_positions[position_index + 1];
        previous_pz = previous_positions[position_index + 2];
    }

    float tile_world_size = delta * TILE_SIZE;

    int min_ti, min_tj, min_tk;
    int max_ti, max_tj, max_tk;

    particle_grid_bounds(fmin(px, previous_px), fmin(py, previous_py), fmin(pz, previous_pz), radius, tile_world_size,
                         origin_x, origin_y, origin_z, size_x, size_y, size_z, &min_ti, &min_tj, &min_tk, &max_ti,
                         &max_tj, &max_tk);

    int path_min_i, path_min_j, path_min_k;
    int path_max_i, path_max_j, path_max_k;

    particle_grid_bounds(fmax(px, previous_px), fmax(py, previous_py), fmax(pz, previous_pz), radius, tile_world_size,
                         origin_x, origin_y, origin_z, size_x, size_y, size_z, &path_min_i, &path_min_j, &path_min_k,
                         &path_max_i, &path_max_j, &path_max_k);

    max_ti = path_max_i;
    max_tj = path_max_j;
    max_tk = path_max_k;

    if (min_ti > max_ti || min_tj > max_tj || min_tk > max_tk)
        return;

    int count_i = max_ti - min_ti + 1;
    int count_j = max_tj - min_tj + 1;
    int count_k = max_tk - min_tk + 1;

    int tile_count = count_i * count_j * count_k;

    int linear_index = get_local_id(0);

    while (linear_index < tile_count) {
        int ti, tj, tk;

        linear_to_grid_index(linear_index, min_ti, min_tj, min_tk, count_j, count_k, &ti, &tj, &tk);

        float tile_center_x = origin_x + ((float)ti + 0.5f) * tile_world_size;
        float tile_center_y = origin_y + ((float)tj + 0.5f) * tile_world_size;
        float tile_center_z = origin_z + ((float)tk + 0.5f) * tile_world_size;

        float tile_reach = radius + 0.8660254037844386f * tile_world_size;

        if (point_segment_distance_squared(tile_center_x, tile_center_y, tile_center_z, previous_px, previous_py,
                                           previous_pz, px, py, pz) <= tile_reach * tile_reach) {
            int index = (ti * size_y + tj) * size_z + tk;

            tile_mask[index] = 1;
        }

        linear_index += get_local_size(0);
    }
}

__kernel void rasterize_particle_spheres(__global uchar *source_mask,
                                         __global const int *index_tile_map,
                                         __global const float *previous_positions,
                                         __global const float *current_positions,
                                         const int count,
                                         const int previous_count,
                                         const float radius,
                                         const float delta,
                                         const float origin_x,
                                         const float origin_y,
                                         const float origin_z,
                                         const int tile_count_x,
                                         const int tile_count_y,
                                         const int tile_count_z) {
    /*
    Rasterizes each particle's swept sphere into the cells of the allocated grid tiles.
    */
    int sample_index = get_group_id(0);

    if (sample_index >= count)
        return;

    int position_index = sample_index * 3;

    float px = current_positions[position_index + 0];
    float py = current_positions[position_index + 1];
    float pz = current_positions[position_index + 2];

    float previous_px = px;
    float previous_py = py;
    float previous_pz = pz;

    if (sample_index < previous_count) {
        previous_px = previous_positions[position_index + 0];
        previous_py = previous_positions[position_index + 1];
        previous_pz = previous_positions[position_index + 2];
    }

    int min_i, min_j, min_k;
    int max_i, max_j, max_k;

    particle_grid_bounds(fmin(px, previous_px), fmin(py, previous_py), fmin(pz, previous_pz), radius, delta, origin_x,
                         origin_y, origin_z, tile_count_x * TILE_SIZE, tile_count_y * TILE_SIZE,
                         tile_count_z * TILE_SIZE, &min_i, &min_j, &min_k, &max_i, &max_j, &max_k);

    int path_min_i, path_min_j, path_min_k;
    int path_max_i, path_max_j, path_max_k;

    particle_grid_bounds(fmax(px, previous_px), fmax(py, previous_py), fmax(pz, previous_pz), radius, delta, origin_x,
                         origin_y, origin_z, tile_count_x * TILE_SIZE, tile_count_y * TILE_SIZE,
                         tile_count_z * TILE_SIZE, &path_min_i, &path_min_j, &path_min_k, &path_max_i, &path_max_j,
                         &path_max_k);

    max_i = path_max_i;
    max_j = path_max_j;
    max_k = path_max_k;

    if (min_i > max_i || min_j > max_j || min_k > max_k)
        return;

    float radius_squared = radius * radius;

    int count_i = max_i - min_i + 1;
    int count_j = max_j - min_j + 1;
    int count_k = max_k - min_k + 1;

    int cell_count = count_i * count_j * count_k;

    int linear_index = get_local_id(0);

    while (linear_index < cell_count) {
        int i, j, k;

        linear_to_grid_index(linear_index, min_i, min_j, min_k, count_j, count_k, &i, &j, &k);

        float cell_x = origin_x + ((float)i + 0.5f) * delta;
        float cell_y = origin_y + ((float)j + 0.5f) * delta;
        float cell_z = origin_z + ((float)k + 0.5f) * delta;

        if (point_segment_distance_squared(cell_x, cell_y, cell_z, previous_px, previous_py, previous_pz, px, py, pz) <=
            radius_squared) {
            int ti = i / TILE_SIZE;
            int tj = j / TILE_SIZE;
            int tk = k / TILE_SIZE;

            int tile_map_index = (ti * tile_count_y + tj) * tile_count_z + tk;

            int pool_index = index_tile_map[tile_map_index];

            if (pool_index >= 0) {
                int local_i = i - ti * TILE_SIZE;
                int local_j = j - tj * TILE_SIZE;
                int local_k = k - tk * TILE_SIZE;

                int index = ((pool_index * TILE_SIZE + local_i) * TILE_SIZE + local_j) * TILE_SIZE + local_k;

                source_mask[index] = 1;
            }
        }

        linear_index += get_local_size(0);
    }
}

__kernel void reset_particle_velocity_kernel(__global float *u,
                                             __global float *v,
                                             __global float *w,
                                             __global const int *index_tile_map,
                                             __global const float *current_positions,
                                             __global const float *next_positions,
                                             const int count,
                                             const int next_count,
                                             const float alpha,
                                             const float radius,
                                             const float delta,
                                             const float origin_x,
                                             const float origin_y,
                                             const float origin_z,
                                             const int tile_count_x,
                                             const int tile_count_y,
                                             const int tile_count_z) {
    /*
    Clears the grid velocity components inside each interpolated particle sphere.
    */
    int sample_index = get_group_id(0);

    if (sample_index >= count)
        return;

    float3 position = interpolated_particle_vector(current_positions, next_positions, sample_index, next_count, alpha);

    float px = position.x;
    float py = position.y;
    float pz = position.z;

    int min_i, min_j, min_k;
    int max_i, max_j, max_k;

    particle_grid_bounds(px, py, pz, radius, delta, origin_x, origin_y, origin_z, tile_count_x * TILE_SIZE,
                         tile_count_y * TILE_SIZE, tile_count_z * TILE_SIZE, &min_i, &min_j, &min_k, &max_i, &max_j,
                         &max_k);

    if (min_i > max_i || min_j > max_j || min_k > max_k)
        return;

    float radius_squared = radius * radius;

    int count_j = max_j - min_j + 1;
    int count_k = max_k - min_k + 1;

    int cell_count = (max_i - min_i + 1) * count_j * count_k;

    int linear_index = get_local_id(0);

    while (linear_index < cell_count) {
        int i, j, k;

        linear_to_grid_index(linear_index, min_i, min_j, min_k, count_j, count_k, &i, &j, &k);

        float dx = origin_x + ((float)i + 0.5f) * delta - px;
        float dy = origin_y + ((float)j + 0.5f) * delta - py;
        float dz = origin_z + ((float)k + 0.5f) * delta - pz;

        if (dx * dx + dy * dy + dz * dz <= radius_squared) {
            int ti = i / TILE_SIZE;
            int tj = j / TILE_SIZE;
            int tk = k / TILE_SIZE;

            int tile_map_index = (ti * tile_count_y + tj) * tile_count_z + tk;

            int pool_index = index_tile_map[tile_map_index];

            if (pool_index >= 0) {
                int local_i = i - ti * TILE_SIZE;
                int local_j = j - tj * TILE_SIZE;
                int local_k = k - tk * TILE_SIZE;

                int index = ((pool_index * TILE_SIZE + local_i) * TILE_SIZE + local_j) * TILE_SIZE + local_k;

                u[index] = 0.0f;
                v[index] = 0.0f;
                w[index] = 0.0f;
            }
        }

        linear_index += get_local_size(0);
    }
}

__kernel void transfer_particle_velocities(__global float *u,
                                           __global float *v,
                                           __global float *w,
                                           __global const int *index_tile_map,
                                           __global const float *current_positions,
                                           __global const float *next_positions,
                                           __global const float *current_velocities,
                                           __global const float *next_velocities,
                                           const int count,
                                           const int next_count,
                                           const float alpha,
                                           const float radius,
                                           const float velocity_transfer,
                                           const float delta,
                                           const float origin_x,
                                           const float origin_y,
                                           const float origin_z,
                                           const int tile_count_x,
                                           const int tile_count_y,
                                           const int tile_count_z) {
    /*
    Adds interpolated particle velocities to the grid cells inside each particle sphere.
    */
    int sample_index = get_group_id(0);

    if (sample_index >= count)
        return;

    float3 position = interpolated_particle_vector(current_positions, next_positions, sample_index, next_count, alpha);

    float3 velocity =
        interpolated_particle_vector(current_velocities, next_velocities, sample_index, next_count, alpha);

    float px = position.x;
    float py = position.y;
    float pz = position.z;

    float vx = velocity.x * velocity_transfer;
    float vy = velocity.y * velocity_transfer;
    float vz = velocity.z * velocity_transfer;

    int min_i, min_j, min_k;
    int max_i, max_j, max_k;

    particle_grid_bounds(px, py, pz, radius, delta, origin_x, origin_y, origin_z, tile_count_x * TILE_SIZE,
                         tile_count_y * TILE_SIZE, tile_count_z * TILE_SIZE, &min_i, &min_j, &min_k, &max_i, &max_j,
                         &max_k);

    if (min_i > max_i || min_j > max_j || min_k > max_k)
        return;

    float radius_squared = radius * radius;

    int count_j = max_j - min_j + 1;
    int count_k = max_k - min_k + 1;

    int cell_count = (max_i - min_i + 1) * count_j * count_k;

    int linear_index = get_local_id(0);

    while (linear_index < cell_count) {
        int i, j, k;

        linear_to_grid_index(linear_index, min_i, min_j, min_k, count_j, count_k, &i, &j, &k);

        float dx = origin_x + ((float)i + 0.5f) * delta - px;
        float dy = origin_y + ((float)j + 0.5f) * delta - py;
        float dz = origin_z + ((float)k + 0.5f) * delta - pz;

        if (dx * dx + dy * dy + dz * dz <= radius_squared) {
            int ti = i / TILE_SIZE;
            int tj = j / TILE_SIZE;
            int tk = k / TILE_SIZE;

            int tile_map_index = (ti * tile_count_y + tj) * tile_count_z + tk;

            int pool_index = index_tile_map[tile_map_index];

            if (pool_index >= 0) {
                int local_i = i - ti * TILE_SIZE;
                int local_j = j - tj * TILE_SIZE;
                int local_k = k - tk * TILE_SIZE;

                int index = ((pool_index * TILE_SIZE + local_i) * TILE_SIZE + local_j) * TILE_SIZE + local_k;

                atomic_add_float(&u[index], vx);
                atomic_add_float(&v[index], vy);
                atomic_add_float(&w[index], vz);
            }
        }

        linear_index += get_local_size(0);
    }
}
