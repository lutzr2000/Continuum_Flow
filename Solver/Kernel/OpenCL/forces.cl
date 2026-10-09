#include "noise.cl"
#include "sparse_managment.cl"

inline float buoyancy(__global const float *T,
                      __global const int *index_tile_map,
                      const int i,
                      const int j,
                      const int k,
                      const float buoyancy_factor,
                      const float t_reference,
                      const int tiles_y,
                      const int tiles_z) {
    /*
    Compute the buoyancy effect of the flow
    */
    const float temperature = get_pool_value(T, index_tile_map, i, j, k, t_reference, tiles_y, tiles_z);

    return buoyancy_factor * (temperature - t_reference);
}

inline void apply_swirl_forces(__global const float *swirl_config,
                               const int swirl_count,
                               const int i,
                               const int j,
                               const int k,
                               const float delta,
                               const float origin_x,
                               const float origin_y,
                               const float origin_z,
                               float *ax,
                               float *ay,
                               float *az) {
    /*
    Compute the acceleration due to the swirl force
    */
    *ax = 0.0f;
    *ay = 0.0f;
    *az = 0.0f;

    const float px = origin_x + (float)i * delta;
    const float py = origin_y + (float)j * delta;
    const float pz = origin_z + (float)k * delta;

    for (int swirl_idx = 0; swirl_idx < swirl_count; ++swirl_idx) {
        const int offset = swirl_idx * 8;

        const float strength = swirl_config[offset + 0];

        const float ox = swirl_config[offset + 1];
        const float oy = swirl_config[offset + 2];
        const float oz = swirl_config[offset + 3];

        float axis_x = swirl_config[offset + 4];
        float axis_y = swirl_config[offset + 5];
        float axis_z = swirl_config[offset + 6];

        const float radius = swirl_config[offset + 7];

        if (radius <= 0.0f || strength == 0.0f)
            continue;

        const float axis_len = sqrt(axis_x * axis_x + axis_y * axis_y + axis_z * axis_z);

        if (axis_len <= 1.0e-8f)
            continue;

        axis_x /= axis_len;
        axis_y /= axis_len;
        axis_z /= axis_len;

        const float rx = px - ox;
        const float ry = py - oy;
        const float rz = pz - oz;

        const float projection = rx * axis_x + ry * axis_y + rz * axis_z;

        const float closest_x = ox + projection * axis_x;
        const float closest_y = oy + projection * axis_y;
        const float closest_z = oz + projection * axis_z;

        const float radial_x = px - closest_x;
        const float radial_y = py - closest_y;
        const float radial_z = pz - closest_z;

        const float dist_sq = radial_x * radial_x + radial_y * radial_y + radial_z * radial_z;

        const float radius_sq = radius * radius;

        if (dist_sq > radius_sq || dist_sq <= 1.0e-12f)
            continue;

        float tx = axis_y * radial_z - axis_z * radial_y;
        float ty = axis_z * radial_x - axis_x * radial_z;
        float tz = axis_x * radial_y - axis_y * radial_x;

        const float t_len = sqrt(tx * tx + ty * ty + tz * tz);

        if (t_len <= 1.0e-8f)
            continue;

        tx /= t_len;
        ty /= t_len;
        tz /= t_len;

        const float dist = sqrt(dist_sq);

        const float falloff = 1.0f - dist / radius;

        *ax += strength * falloff * tx;
        *ay += strength * falloff * ty;
        *az += strength * falloff * tz;
    }
}

inline void apply_turbulence_forces(__global const float *turbulence_config,
                                    const int turbulence_count,
                                    const int i,
                                    const int j,
                                    const int k,
                                    const float delta,
                                    const float origin_x,
                                    const float origin_y,
                                    const float origin_z,
                                    float *ax,
                                    float *ay,
                                    float *az) {
    /*
    Compute the acceleration due to the turbulence force
    */
    float acceleration = 0.0f;

    for (int index = 0; index < turbulence_count; ++index) {
        const int offset = index * 4;
        const float amplitude = turbulence_config[offset];
        const float scale = turbulence_config[offset + 1];
        const int seed = convert_int_rte(turbulence_config[offset + 2]);
        const float frequency_factor = turbulence_config[offset + 3];
        const float noise = gradient_noise_3d(origin_x + (float)i * delta, origin_y + (float)j * delta,
                                              origin_z + (float)k * delta, seed, scale);

        acceleration = mad(amplitude * frequency_factor, noise, acceleration);
    }

    *ax = acceleration;
    *ay = acceleration;
    *az = acceleration;
}
