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
    __global const uchar *oxygen,
    __global const float *u,
    __global const float *v,
    __global const float *w,
    const float dt,
    __global float *predictor_T,
    __global float *predictor_smoke,
    __global float *predictor_fuel,
    __global float *predictor_oxygen,
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
    float sampled_oxygen;

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
    sampled_oxygen = sample_trilinear_inner_sparse_uint8(
        oxygen, index_tile_map,
        fuel_x0, fuel_y0, fuel_z0,
        fuel_x1, fuel_y1, fuel_z1,
        fuel_tx, fuel_ty, fuel_tz,
        100.0f, tiles_y, tiles_z
    );

    const int index =
        ((tile_index * TILE_SIZE + local_i)
        * TILE_SIZE + local_j)
        * TILE_SIZE + local_k;

    predictor_T[index] = sampled_T;
    predictor_smoke[index] = sampled_smoke;
    predictor_fuel[index] = sampled_fuel;
    predictor_oxygen[index] = sampled_oxygen;
}


__kernel void update_scalar_fields_maccormack(
    __global const float *T,
    __global const float *smoke,
    __global const uchar *fuel,
    __global const uchar *oxygen,
    __global const float *predictor_T,
    __global const float *predictor_smoke,
    __global const float *predictor_fuel,
    __global const float *predictor_oxygen,
    __global const float *u,
    __global const float *v,
    __global const float *w,
    const float dt,
    __global float *T_out,
    __global float *smoke_out,
    __global uchar *fuel_out,
    __global uchar *oxygen_out,
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

    const float oxygen_advected =
        predictor_oxygen[index];


    // ---------------------------------------------------------
    // Reverse sample
    // ---------------------------------------------------------

    float T_reverse;
    float smoke_reverse;
    float fuel_reverse;
    float oxygen_reverse;

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

    int oxygen_x0, oxygen_y0, oxygen_z0;
    int oxygen_x1, oxygen_y1, oxygen_z1;
    float oxygen_tx, oxygen_ty, oxygen_tz;
    prepare_trilinear_coords(
        x_forward, y_forward, z_forward, nx, ny, nz,
        &oxygen_x0, &oxygen_y0, &oxygen_z0,
        &oxygen_x1, &oxygen_y1, &oxygen_z1,
        &oxygen_tx, &oxygen_ty, &oxygen_tz
    );
    oxygen_reverse = sample_trilinear_inner_sparse(
        predictor_oxygen, index_tile_map,
        oxygen_x0, oxygen_y0, oxygen_z0,
        oxygen_x1, oxygen_y1, oxygen_z1,
        oxygen_tx, oxygen_ty, oxygen_tz,
        100.0f, tiles_y, tiles_z
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

    float oxygen_corrected =
        oxygen_advected
        + 0.5f * ((float)oxygen[index] * (100.0f / 255.0f) - oxygen_reverse);


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

    float oxygen_lower;
    float oxygen_upper;

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

    sample_cell_extrema_inner_sparse_uint8(
        oxygen,
        index_tile_map,
        x0,
        y0,
        z0,
        x1,
        y1,
        z1,
        100.0f,
        tiles_y,
        tiles_z,
        &oxygen_lower,
        &oxygen_upper
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

    oxygen_corrected =
        clamp_value(
            oxygen_corrected,
            oxygen_lower,
            oxygen_upper
        );


    // ---------------------------------------------------------
    // Combustion
    // ---------------------------------------------------------

    // Work internally in normalized concentrations [0, 1].
    const float F =
        clamp(fuel_corrected * 0.01f, 0.0f, 1.0f);

    const float O =
        clamp(oxygen_corrected * 0.01f, 0.0f, 1.0f);


    // ---------------------------------------------------------
    // Artist / combustion constants
    // ---------------------------------------------------------

    // Width of the ignition transition in temperature units.
    const float ignition_width = 50.0f;

    // Oxygen required to burn one unit of fuel.
    // 1.0 = equal normalized quantities.
    const float oxygen_per_fuel = 1.0f;

    // Even well-mixed hot gas may continue burning.
    // Interface mixing adds extra combustion at the flame sheet.
    const float base_mixing = 0.20f;
    const float interface_gain = 2.0f;

    // Smoke yield for clean vs. oxygen-starved combustion.
    const float clean_smoke_yield = 0.10f;
    const float dirty_smoke_yield = 1.50f;


    // ---------------------------------------------------------
    // Ignition
    // ---------------------------------------------------------

    // Smooth ignition instead of a hard temperature threshold.
    //
    // 0 below ignition region
    // 1 above ignition region
    //
    const float ignition =
        smoothstep(
            fuel_ignition_temperature - ignition_width,
            fuel_ignition_temperature + ignition_width,
            T_corrected
        );


    // ---------------------------------------------------------
    // Fuel neighborhood
    // ---------------------------------------------------------

    const float F_xp =
        get_pool_value_uint8(
            fuel,
            index_tile_map,
            i + 1, j, k,
            0.0f,
            tiles_y,
            tiles_z
        ) * 0.01f;

    const float F_xm =
        get_pool_value_uint8(
            fuel,
            index_tile_map,
            i - 1, j, k,
            0.0f,
            tiles_y,
            tiles_z
        ) * 0.01f;

    const float F_yp =
        get_pool_value_uint8(
            fuel,
            index_tile_map,
            i, j + 1, k,
            0.0f,
            tiles_y,
            tiles_z
        ) * 0.01f;

    const float F_ym =
        get_pool_value_uint8(
            fuel,
            index_tile_map,
            i, j - 1, k,
            0.0f,
            tiles_y,
            tiles_z
        ) * 0.01f;

    const float F_zp =
        get_pool_value_uint8(
            fuel,
            index_tile_map,
            i, j, k + 1,
            0.0f,
            tiles_y,
            tiles_z
        ) * 0.01f;

    const float F_zm =
        get_pool_value_uint8(
            fuel,
            index_tile_map,
            i, j, k - 1,
            0.0f,
            tiles_y,
            tiles_z
        ) * 0.01f;


    // ---------------------------------------------------------
    // Oxygen neighborhood
    // ---------------------------------------------------------

    const float O_xp =
        get_pool_value_uint8(
            oxygen,
            index_tile_map,
            i + 1, j, k,
            100.0f,
            tiles_y,
            tiles_z
        ) * 0.01f;

    const float O_xm =
        get_pool_value_uint8(
            oxygen,
            index_tile_map,
            i - 1, j, k,
            100.0f,
            tiles_y,
            tiles_z
        ) * 0.01f;

    const float O_yp =
        get_pool_value_uint8(
            oxygen,
            index_tile_map,
            i, j + 1, k,
            100.0f,
            tiles_y,
            tiles_z
        ) * 0.01f;

    const float O_ym =
        get_pool_value_uint8(
            oxygen,
            index_tile_map,
            i, j - 1, k,
            100.0f,
            tiles_y,
            tiles_z
        ) * 0.01f;

    const float O_zp =
        get_pool_value_uint8(
            oxygen,
            index_tile_map,
            i, j, k + 1,
            100.0f,
            tiles_y,
            tiles_z
        ) * 0.01f;

    const float O_zm =
        get_pool_value_uint8(
            oxygen,
            index_tile_map,
            i, j, k - 1,
            100.0f,
            tiles_y,
            tiles_z
        ) * 0.01f;


    // ---------------------------------------------------------
    // Fuel / oxygen interface
    // ---------------------------------------------------------

    // Approximate gradient magnitudes.
    //
    // Strong gradients mean that fuel-rich and oxygen-rich gas
    // are meeting here. This approximates unresolved mixing at
    // the flame sheet.
    //
    const float grad_F =
        fabs(F_xp - F_xm)
        + fabs(F_yp - F_ym)
        + fabs(F_zp - F_zm);

    const float grad_O =
        fabs(O_xp - O_xm)
        + fabs(O_yp - O_ym)
        + fabs(O_zp - O_zm);


    // Geometric mean:
    // both gradients need to be present for a strong interface.
    const float interface_factor =
        clamp(
            sqrt(fmax(grad_F * grad_O, 0.0f)),
            0.0f,
            1.0f
        );


    // ---------------------------------------------------------
    // Mixing
    // ---------------------------------------------------------

    const float mixing =
        clamp(
            base_mixing
            + interface_gain * interface_factor,
            0.0f,
            1.0f
        );


    // ---------------------------------------------------------
    // Mixture / combustion quality
    // ---------------------------------------------------------

    // phi:
    //
    // < 1  -> oxygen rich
    // = 1  -> approximately stoichiometric
    // > 1  -> fuel rich
    //
    const float phi =
        F / fmax(O / oxygen_per_fuel, 1.0e-4f);


    // Combustion efficiency peaks around phi = 1.
    //
    // This is deliberately an artistic approximation rather
    // than detailed combustion chemistry.
    //
    const float log_phi =
        log(fmax(phi, 1.0e-4f));

    const float combustion_quality =
        exp(-1.5f * fabs(log_phi));


    // ---------------------------------------------------------
    // Burn noise
    // ---------------------------------------------------------

    const float n =
        value_noise_3d(
            (float)i * burn_noise_scale,
            (float)j * burn_noise_scale,
            (float)k * burn_noise_scale,
            0
        );

    // Keep noise subtle. It should perturb combustion rather
    // than determine where combustion exists.
    const float burn_noise =
        clamp(
            1.0f + 0.20f * burn_noise_amplitude * n,
            0.75f,
            1.25f
        );


    // ---------------------------------------------------------
    // Reaction rate
    // ---------------------------------------------------------

    // Both reactants must exist.
    //
    // sqrt(F * O) is intentionally used instead of F * O.
    // F * O tends to make low concentrations disappear too
    // aggressively and produces very thin / weak flames.
    //
    const float reactant_factor =
        sqrt(fmax(F * O, 0.0f));

    float reaction_rate =
        fuel_burn_rate
        * reactant_factor
        * ignition
        * mixing
        * burn_noise;


    // ---------------------------------------------------------
    // Stoichiometric limitation
    // ---------------------------------------------------------

    // reaction_rate is normalized fuel fraction / second.
    //
    // Determine how much fuel can actually burn during this
    // timestep.
    float burned_fuel =
        reaction_rate * dt;


    // Cannot consume more fuel than exists.
    burned_fuel =
        fmin(
            burned_fuel,
            F
        );


    // Cannot consume more oxygen than exists.
    burned_fuel =
        fmin(
            burned_fuel,
            O / oxygen_per_fuel
        );


    burned_fuel =
        fmax(
            burned_fuel,
            0.0f
        );


    // Convert back from normalized [0,1] concentration to the
    // solver's [0,100] concentration.
    //
    // These are actual changes over this timestep.
    const float fuel_consumed =
        burned_fuel * 100.0f;

    const float oxygen_consumed =
        burned_fuel
        * oxygen_per_fuel
        * 100.0f;


    // Convert timestep consumption into source rates so that
    // the existing:
    //
    //     field += dt * source
    //
    // update scheme can remain unchanged.
    const float inv_dt =
        1.0f / fmax(dt, 1.0e-6f);

    float fuel_burn_source =
        -fuel_consumed * inv_dt;

    float oxygen_burn_source =
        -oxygen_consumed * inv_dt;


    // ---------------------------------------------------------
    // Heat production
    // ---------------------------------------------------------

    // Clean / near-stoichiometric combustion releases the most
    // useful heat.
    //
    // Keep a minimum contribution so rich flames do not become
    // unnaturally cold immediately.
    const float heat_efficiency =
        mix(
            0.35f,
            1.0f,
            combustion_quality
        );

    float temperature_burn_source =
        temperature_production_rate
        * fuel_consumed
        * heat_efficiency
        * inv_dt;


    // ---------------------------------------------------------
    // Smoke / soot production
    // ---------------------------------------------------------

    // Fuel-rich combustion produces considerably more soot.
    //
    // Start increasing soot around phi ~= 1 and approach dirty
    // combustion for strongly fuel-rich mixtures.
    const float oxygen_starvation =
        smoothstep(
            0.9f,
            2.5f,
            phi
        );


    // A little smoke is produced even by clean combustion,
    // while oxygen-starved combustion becomes much dirtier.
    const float soot_yield =
        mix(
            clean_smoke_yield,
            dirty_smoke_yield,
            oxygen_starvation
        );

    float smoke_burn_source =
        smoke_production_rate
        * fuel_consumed
        * soot_yield
        * inv_dt;

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

    const float oxygen_updated =
        oxygen_corrected
        + dt * oxygen_burn_source;


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

    oxygen_out[index] = convert_uchar_rte(
        clamp(oxygen_updated, 0.0f, 100.0f) * (255.0f / 100.0f)
    );

    flame_out[index] =
        fmax(
            -fuel_burn_source,
            0.0f
        );
}
