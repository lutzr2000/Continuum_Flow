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
    __global float *fuel,
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
    const float randomness_scale,
    const int randomness_seed,
    const float temperature_randomness,
    const float smoke_randomness,
    const float fuel_randomness,
    const float velocity_randomness,
    const float delta,
    const float origin_x,
    const float origin_y,
    const float origin_z,
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

    const int i = tile_i * TILE_SIZE + local_i;
    const int j = tile_j * TILE_SIZE + local_j;
    const int k = tile_k * TILE_SIZE + local_k;
    float noise = 0.0f;
    if (
        temperature_randomness != 0.0f ||
        smoke_randomness != 0.0f ||
        fuel_randomness != 0.0f ||
        velocity_randomness != 0.0f
    )
    {
        noise = gradient_noise_3d(
            origin_x + (float)i * delta,
            origin_y + (float)j * delta,
            origin_z + (float)k * delta,
            randomness_seed,
            randomness_scale
        );
    }
    // Reuse this source's one procedural sample for every emitted field.
    const float temperature_multiplier = noise_amplitude_multiplier(
        noise,
        temperature_randomness
    );
    const float smoke_multiplier = noise_amplitude_multiplier(
        noise,
        smoke_randomness
    );
    const float fuel_multiplier = noise_amplitude_multiplier(
        noise,
        fuel_randomness
    );
    const float velocity_multiplier = noise_amplitude_multiplier(
        noise,
        velocity_randomness
    );

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

    source_u *= velocity_multiplier;
    source_v *= velocity_multiplier;
    source_w *= velocity_multiplier;

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

    T[index] = fmax(
        temperature_value * temperature_multiplier,
        0.0f
    );

    smoke[index] = fmin(
        fmax(
            smoke[index] +
            smoke_value * smoke_multiplier * dt,
            0.0f
        ),
        100.0f
    );

    fuel[index] = clamp(
        fuel[index] + fuel_value * fuel_multiplier * dt,
        0.0f,
        100.0f
    );
}
