#ifndef TILE_SIZE
#define TILE_SIZE 4
#endif

#include "sparse_managment.cl"


inline float buoyancy_approximation(
    __global const float *T,
    __global const int *index_tile_map,
    const int i,
    const int j,
    const int k,
    const float buoyancy_factor,
    const float t_reference,
    const int tiles_y,
    const int tiles_z
)
{
    const float temperature =
        get_pool_value(
            T,
            index_tile_map,
            i,
            j,
            k,
            t_reference,
            tiles_y,
            tiles_z
        );

    return buoyancy_factor
        * (temperature - t_reference);
}


inline void apply_swirl_forces(
    __global const float *swirl_config,
    const int swirl_count,
    const int i,
    const int j,
    const int k,
    const float delta,
    const float origin_x,
    const float origin_y,
    const float origin_z,
    float *Fx,
    float *Fy,
    float *Fz
)
{
    *Fx = 0.0f;
    *Fy = 0.0f;
    *Fz = 0.0f;

    const float px =
        origin_x + (float)i * delta;

    const float py =
        origin_y + (float)j * delta;

    const float pz =
        origin_z + (float)k * delta;

    for (
        int swirl_idx = 0;
        swirl_idx < swirl_count;
        ++swirl_idx
    )
    {
        const int offset =
            swirl_idx * 8;

        const float strength =
            swirl_config[offset + 0];

        const float ox =
            swirl_config[offset + 1];

        const float oy =
            swirl_config[offset + 2];

        const float oz =
            swirl_config[offset + 3];

        float ax =
            swirl_config[offset + 4];

        float ay =
            swirl_config[offset + 5];

        float az =
            swirl_config[offset + 6];

        const float radius =
            swirl_config[offset + 7];

        if (
            radius <= 0.0f ||
            strength == 0.0f
        )
            continue;

        const float axis_len =
            sqrt(
                ax * ax +
                ay * ay +
                az * az
            );

        if (axis_len <= 1.0e-8f)
            continue;

        ax /= axis_len;
        ay /= axis_len;
        az /= axis_len;

        const float rx =
            px - ox;

        const float ry =
            py - oy;

        const float rz =
            pz - oz;

        const float projection =
            rx * ax +
            ry * ay +
            rz * az;

        const float closest_x =
            ox + projection * ax;

        const float closest_y =
            oy + projection * ay;

        const float closest_z =
            oz + projection * az;

        const float radial_x =
            px - closest_x;

        const float radial_y =
            py - closest_y;

        const float radial_z =
            pz - closest_z;

        const float dist_sq =
            radial_x * radial_x +
            radial_y * radial_y +
            radial_z * radial_z;

        const float radius_sq =
            radius * radius;

        if (
            dist_sq > radius_sq ||
            dist_sq <= 1.0e-12f
        )
            continue;

        float tx =
            ay * radial_z -
            az * radial_y;

        float ty =
            az * radial_x -
            ax * radial_z;

        float tz =
            ax * radial_y -
            ay * radial_x;

        const float t_len =
            sqrt(
                tx * tx +
                ty * ty +
                tz * tz
            );

        if (t_len <= 1.0e-8f)
            continue;

        tx /= t_len;
        ty /= t_len;
        tz /= t_len;

        const float dist =
            sqrt(dist_sq);

        const float falloff =
            1.0f - dist / radius;

        *Fx +=
            strength * falloff * tx;

        *Fy +=
            strength * falloff * ty;

        *Fz +=
            strength * falloff * tz;
    }
}
