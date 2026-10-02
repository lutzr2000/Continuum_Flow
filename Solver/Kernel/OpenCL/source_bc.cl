#ifndef TILE_SIZE
#define TILE_SIZE 4
#endif

#include "noise.cl"

__kernel void source_bc(
    __global float *u,
    __global float *v,
    __global float *w,
    __global float *T,
    __global float *smoke,
    __global uchar *fuel,
    __global const int *index_tile_map,
    __global const uchar *source_mask,
    const float temperature_value,
    const float smoke_value,
    const float fuel_value,
    const float velocity_x_value,
    const float velocity_y_value,
    const float velocity_z_value,
    const int velocity_local,
    __global const float *velocity_x_field,
    __global const float *velocity_y_field,
    __global const float *velocity_z_field,
    const float noise_scale,
    const float noise_amplitude,
    const int noise_seed,
    const float dt,
    const int tiles_x,
    const int tiles_y,
    const int tiles_z
)
{
    const int tile_i = get_group_id(0);
    const int tile_j = get_group_id(1);
    const int tile_k = get_group_id(2);

    const int local_k = get_local_id(0);
    const int local_j = get_local_id(1);
    const int local_i = get_local_id(2);

    if (
        tile_i >= tiles_x ||
        tile_j >= tiles_y ||
        tile_k >= tiles_z
    )
        return;

    if (
        local_i >= TILE_SIZE ||
        local_j >= TILE_SIZE ||
        local_k >= TILE_SIZE
    )
        return;

    const int tile_map_index =
        (tile_i * tiles_y + tile_j)
        * tiles_z + tile_k;

    const int tile_index =
        index_tile_map[tile_map_index];

    if (tile_index == -1)
        return;

    const int index =
        ((tile_index * TILE_SIZE + local_i)
        * TILE_SIZE + local_j)
        * TILE_SIZE + local_k;

    if (!source_mask[index])
        return;

    const int i =
        tile_i * TILE_SIZE + local_i;

    const int j =
        tile_j * TILE_SIZE + local_j;

    const int k =
        tile_k * TILE_SIZE + local_k;

    float source_u;
    float source_v;
    float source_w;

    if (velocity_local)
    {
        source_u = velocity_x_field[index];
        source_v = velocity_y_field[index];
        source_w = velocity_z_field[index];
    }
    else
    {
        source_u = velocity_x_value;
        source_v = velocity_y_value;
        source_w = velocity_z_value;
    }

    if (
        source_u != 0.0f ||
        source_v != 0.0f ||
        source_w != 0.0f
    )
    {
        u[index] = source_u;
        v[index] = source_v;
        w[index] = source_w;
    }

    float scalar_multiplier = 1.0f;

    if (noise_amplitude != 0.0f)
    {
        const float scale =
            fmax(noise_scale, 1.0e-6f);

        const float noise_value =
            value_noise_3d(
                (float)i / scale,
                (float)j / scale,
                (float)k / scale,
                noise_seed
            );

        scalar_multiplier = fmax(
            1.0f +
            noise_value * noise_amplitude,
            0.0f
        );
    }

    T[index] = fmax(
        temperature_value * scalar_multiplier,
        0.0f
    );

    smoke[index] = fmin(
        fmax(
            smoke[index] +
            smoke_value *
            scalar_multiplier *
            dt,
            0.0f
        ),
        100.0f
    );

    fuel[index] = convert_uchar_rte(
        clamp(
            (float)fuel[index] * (100.0f / 255.0f) +
            fuel_value * scalar_multiplier * dt,
            0.0f,
            100.0f
        ) * (255.0f / 100.0f)
    );
}
