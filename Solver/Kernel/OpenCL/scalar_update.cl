#ifndef TILE_SIZE
#define TILE_SIZE 4
#endif

#include "sparse_managment.cl"
#include "advection_schemes.cl"
#include "noise.cl"


__kernel void predict_scalar_fields_semi_lagrangian(
    __global const float *T,
    __global const float *smoke,
    __global const uchar *fuel,
    __global const float *u,
    __global const float *v,
    __global const float *w,
    const float dt,
    __global float *predictor_T,
    __global float *predictor_smoke,
    __global float *predictor_fuel,
    const float delta,
    const int n_substeps,
    const float t_reference,
    __global const int *index_tile_map,
    const float u_initial,
    const float v_initial,
    const float w_initial,
    const int nx,
    const int ny,
    const int nz,
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

    const int tile_map_index =
        (tile_i * tiles_y + tile_j)
        * tiles_z + tile_k;

    const int tile_index =
        index_tile_map[tile_map_index];

    if (tile_index == -1)
        return;

    const int i =
        tile_i * TILE_SIZE + local_i;

    const int j =
        tile_j * TILE_SIZE + local_j;

    const int k =
        tile_k * TILE_SIZE + local_k;

    float x_depart;
    float y_depart;
    float z_depart;

    backtrace_position_sparse(
        u,
        v,
        w,
        index_tile_map,
        (float)i,
        (float)j,
        (float)k,
        dt / delta,
        n_substeps,
        nx,
        ny,
        nz,
        u_initial,
        v_initial,
        w_initial,
        tiles_y,
        tiles_z,
        &x_depart,
        &y_depart,
        &z_depart
    );

    float sampled_T;
    float sampled_smoke;
    float sampled_fuel;

    int fuel_x0, fuel_y0, fuel_z0, fuel_x1, fuel_y1, fuel_z1;
    float fuel_tx, fuel_ty, fuel_tz;
    prepare_trilinear_coords(
        x_depart, y_depart, z_depart, nx, ny, nz,
        &fuel_x0, &fuel_y0, &fuel_z0,
        &fuel_x1, &fuel_y1, &fuel_z1,
        &fuel_tx, &fuel_ty, &fuel_tz
    );
    sampled_T = sample_trilinear_inner_sparse(
        T, index_tile_map,
        fuel_x0, fuel_y0, fuel_z0,
        fuel_x1, fuel_y1, fuel_z1,
        fuel_tx, fuel_ty, fuel_tz,
        t_reference, tiles_y, tiles_z
    );
    sampled_smoke = sample_trilinear_inner_sparse(
        smoke, index_tile_map,
        fuel_x0, fuel_y0, fuel_z0,
        fuel_x1, fuel_y1, fuel_z1,
        fuel_tx, fuel_ty, fuel_tz,
        0.0f, tiles_y, tiles_z
    );
    sampled_fuel = sample_trilinear_inner_sparse_uint8(
        fuel, index_tile_map,
        fuel_x0, fuel_y0, fuel_z0,
        fuel_x1, fuel_y1, fuel_z1,
        fuel_tx, fuel_ty, fuel_tz,
        0.0f, tiles_y, tiles_z
    );

    const int index =
        ((tile_index * TILE_SIZE + local_i)
        * TILE_SIZE + local_j)
        * TILE_SIZE + local_k;

    predictor_T[index] = sampled_T;
    predictor_smoke[index] = sampled_smoke;
    predictor_fuel[index] = sampled_fuel;
}


__kernel void update_scalar_fields_maccormack(
    __global const float *T,
    __global const float *smoke,
    __global const uchar *fuel,
    __global const float *predictor_T,
    __global const float *predictor_smoke,
    __global const float *predictor_fuel,
    __global const float *u,
    __global const float *v,
    __global const float *w,
    const float dt,
    __global float *T_out,
    __global float *smoke_out,
    __global uchar *fuel_out,
    __global float *flame_out,
    const float delta,
    const int n_substeps,
    const float temperature_dissipation_rate,
    const float temperature_production_rate,
    const float smoke_dissipation_rate,
    const float smoke_production_rate,
    const float fuel_dissipation_rate,
    const float fuel_burn_rate,
    const float fuel_ignition_temperature,
    const float burn_noise_scale,
    const float burn_noise_amplitude,
    const float t_reference,
    __global const int *index_tile_map,
    const float u_initial,
    const float v_initial,
    const float w_initial,
    const int nx,
    const int ny,
    const int nz,
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

    const int tile_map_index =
        (tile_i * tiles_y + tile_j)
        * tiles_z + tile_k;

    const int tile_index =
        index_tile_map[tile_map_index];

    if (tile_index == -1)
        return;

    const int i =
        tile_i * TILE_SIZE + local_i;

    const int j =
        tile_j * TILE_SIZE + local_j;

    const int k =
        tile_k * TILE_SIZE + local_k;

    if (
        i < 1 ||
        j < 1 ||
        k < 1 ||
        i >= nx - 1 ||
        j >= ny - 1 ||
        k >= nz - 1
    )
        return;

    const int index =
        ((tile_index * TILE_SIZE + local_i)
        * TILE_SIZE + local_j)
        * TILE_SIZE + local_k;

    const float dt_over_delta =
        dt / delta;


    // ---------------------------------------------------------
    // Backtrace
    // ---------------------------------------------------------

    float x_depart;
    float y_depart;
    float z_depart;

    backtrace_position_sparse(
        u,
        v,
        w,
        index_tile_map,
        (float)i,
        (float)j,
        (float)k,
        dt_over_delta,
        n_substeps,
        nx,
        ny,
        nz,
        u_initial,
        v_initial,
        w_initial,
        tiles_y,
        tiles_z,
        &x_depart,
        &y_depart,
        &z_depart
    );


    // ---------------------------------------------------------
    // Forward trace
    // ---------------------------------------------------------

    float x_forward;
    float y_forward;
    float z_forward;

    forward_trace_position_sparse(
        u,
        v,
        w,
        index_tile_map,
        x_depart,
        y_depart,
        z_depart,
        dt_over_delta,
        n_substeps,
        nx,
        ny,
        nz,
        u_initial,
        v_initial,
        w_initial,
        tiles_y,
        tiles_z,
        &x_forward,
        &y_forward,
        &z_forward
    );


    // ---------------------------------------------------------
    // Predictor
    // ---------------------------------------------------------

    const float T_advected =
        predictor_T[index];

    const float smoke_advected =
        predictor_smoke[index];

    const float fuel_advected =
        predictor_fuel[index];


    // ---------------------------------------------------------
    // Reverse sample
    // ---------------------------------------------------------

    float T_reverse;
    float smoke_reverse;
    float fuel_reverse;

    sample_trilinear_vec3_sparse(
        predictor_T,
        predictor_smoke,
        predictor_fuel,
        index_tile_map,
        x_forward,
        y_forward,
        z_forward,
        nx,
        ny,
        nz,
        t_reference,
        0.0f,
        0.0f,
        tiles_y,
        tiles_z,
        &T_reverse,
        &smoke_reverse,
        &fuel_reverse
    );


    // ---------------------------------------------------------
    // MacCormack correction
    // ---------------------------------------------------------

    float T_corrected =
        T_advected
        + 0.5f * (T[index] - T_reverse);

    float smoke_corrected =
        smoke_advected
        + 0.5f * (smoke[index] - smoke_reverse);

    float fuel_corrected =
        fuel_advected
        + 0.5f * ((float)fuel[index] * (100.0f / 255.0f) - fuel_reverse);


    // ---------------------------------------------------------
    // Departure cell
    // ---------------------------------------------------------

    int x0;
    int y0;
    int z0;

    int x1;
    int y1;
    int z1;

    float tx;
    float ty;
    float tz;

    prepare_trilinear_coords(
        x_depart,
        y_depart,
        z_depart,
        nx,
        ny,
        nz,
        &x0,
        &y0,
        &z0,
        &x1,
        &y1,
        &z1,
        &tx,
        &ty,
        &tz
    );


    // ---------------------------------------------------------
    // MacCormack limiter
    // ---------------------------------------------------------

    float T_lower;
    float T_upper;

    float smoke_lower;
    float smoke_upper;

    float fuel_lower;
    float fuel_upper;

    sample_cell_extrema_inner_sparse(
        T,
        index_tile_map,
        x0,
        y0,
        z0,
        x1,
        y1,
        z1,
        t_reference,
        tiles_y,
        tiles_z,
        &T_lower,
        &T_upper
    );

    sample_cell_extrema_inner_sparse(
        smoke,
        index_tile_map,
        x0,
        y0,
        z0,
        x1,
        y1,
        z1,
        0.0f,
        tiles_y,
        tiles_z,
        &smoke_lower,
        &smoke_upper
    );

    sample_cell_extrema_inner_sparse_uint8(
        fuel,
        index_tile_map,
        x0,
        y0,
        z0,
        x1,
        y1,
        z1,
        0.0f,
        tiles_y,
        tiles_z,
        &fuel_lower,
        &fuel_upper
    );

    T_corrected =
        clamp_value(
            T_corrected,
            T_lower,
            T_upper
        );

    smoke_corrected =
        clamp_value(
            smoke_corrected,
            smoke_lower,
            smoke_upper
        );

    fuel_corrected =
        clamp_value(
            fuel_corrected,
            fuel_lower,
            fuel_upper
        );


    // ---------------------------------------------------------
    // Combustion
    // ---------------------------------------------------------

    const float oxygen_center =
        clamp(
            (100.0f - smoke_corrected) / 100.0f,
            0.0f,
            1.0f
        );

    float temperature_burn_source = 0.0f;
    float smoke_burn_source = 0.0f;
    float fuel_burn_source = 0.0f;

    if (
        T_corrected > fuel_ignition_temperature &&
        fuel_corrected > 0.0f
    )
    {
        const float fuel_xp =
            get_pool_value_uint8(
                fuel,
                index_tile_map,
                i + 1,
                j,
                k,
                0.0f,
                tiles_y,
                tiles_z
            );

        const float fuel_xm =
            get_pool_value_uint8(
                fuel,
                index_tile_map,
                i - 1,
                j,
                k,
                0.0f,
                tiles_y,
                tiles_z
            );

        const float fuel_yp =
            get_pool_value_uint8(
                fuel,
                index_tile_map,
                i,
                j + 1,
                k,
                0.0f,
                tiles_y,
                tiles_z
            );

        const float fuel_ym =
            get_pool_value_uint8(
                fuel,
                index_tile_map,
                i,
                j - 1,
                k,
                0.0f,
                tiles_y,
                tiles_z
            );

        const float fuel_zp =
            get_pool_value_uint8(
                fuel,
                index_tile_map,
                i,
                j,
                k + 1,
                0.0f,
                tiles_y,
                tiles_z
            );

        const float fuel_zm =
            get_pool_value_uint8(
                fuel,
                index_tile_map,
                i,
                j,
                k - 1,
                0.0f,
                tiles_y,
                tiles_z
            );


        const float T_xp =
            get_pool_value(
                T,
                index_tile_map,
                i + 1,
                j,
                k,
                t_reference,
                tiles_y,
                tiles_z
            );

        const float T_xm =
            get_pool_value(
                T,
                index_tile_map,
                i - 1,
                j,
                k,
                t_reference,
                tiles_y,
                tiles_z
            );

        const float T_yp =
            get_pool_value(
                T,
                index_tile_map,
                i,
                j + 1,
                k,
                t_reference,
                tiles_y,
                tiles_z
            );

        const float T_ym =
            get_pool_value(
                T,
                index_tile_map,
                i,
                j - 1,
                k,
                t_reference,
                tiles_y,
                tiles_z
            );

        const float T_zp =
            get_pool_value(
                T,
                index_tile_map,
                i,
                j,
                k + 1,
                t_reference,
                tiles_y,
                tiles_z
            );

        const float T_zm =
            get_pool_value(
                T,
                index_tile_map,
                i,
                j,
                k - 1,
                t_reference,
                tiles_y,
                tiles_z
            );


        const float fuel_front =
            (
                fabs(fuel_xp - fuel_xm) +
                fabs(fuel_yp - fuel_ym) +
                fabs(fuel_zp - fuel_zm)
            )
            / 100.0f;

        const float temperature_front =
            (
                fabs(T_xp - T_xm) +
                fabs(T_yp - T_ym) +
                fabs(T_zp - T_zm)
            )
            / fmax(
                fuel_ignition_temperature,
                1.0f
            );

        const float front_factor =
            clamp(
                0.5f
                * (
                    fuel_front +
                    temperature_front
                ),
                0.0f,
                1.0f
            );

        const float n =
            value_noise_3d(
                (float)i * burn_noise_scale,
                (float)j * burn_noise_scale,
                (float)k * burn_noise_scale,
                0
            );

        float burn_noise =
            1.0f
            + burn_noise_amplitude * n;

        burn_noise =
            clamp(
                burn_noise,
                0.0f,
                2.0f
            );

        const float burn_front_weight =
            front_factor * front_factor;

        fuel_burn_source =
            -fuel_burn_rate
            * fuel_corrected
            * oxygen_center
            * burn_noise
            * burn_front_weight;

        temperature_burn_source =
            temperature_production_rate
            * -fuel_burn_source;

        smoke_burn_source =
            smoke_production_rate
            * -fuel_burn_source;
    }


    // ---------------------------------------------------------
    // Dissipation
    // ---------------------------------------------------------

    const float dT =
        T_corrected - t_reference;

    const float cool_factor =
        fabs(dT)
        / (fabs(dT) + 200.0f);

    const float temperature_dissipation =
        -temperature_dissipation_rate
        * dT
        * cool_factor;

    const float smoke_dissipation =
        -smoke_dissipation_rate
        * smoke_corrected;

    const float fuel_dissipation =
        -fuel_dissipation_rate
        * fuel_corrected;


    // ---------------------------------------------------------
    // Final update
    // ---------------------------------------------------------

    const float T_updated =
        T_corrected
        + dt * temperature_burn_source
        + dt * temperature_dissipation;

    const float smoke_updated =
        smoke_corrected
        + dt * smoke_burn_source
        + dt * smoke_dissipation;

    const float fuel_updated =
        fuel_corrected
        + dt * fuel_burn_source
        + dt * fuel_dissipation;


    T_out[index] =
        fmax(
            T_updated,
            0.0f
        );

    smoke_out[index] =
        clamp(
            smoke_updated,
            0.0f,
            100.0f
        );

    fuel_out[index] = convert_uchar_rte(
        clamp(fuel_updated, 0.0f, 100.0f) * (255.0f / 100.0f)
    );

    flame_out[index] =
        fmax(
            -fuel_burn_source,
            0.0f
        );
}
