#include "advection_schemes.cl"
#include "sparse_managment.cl"

inline void compute_combustion_sources(const float T,
                                       const float fuel_concentration,
                                       const float oxygen_concentration,
                                       const float dt,
                                       const float temperature_production_rate,
                                       const float smoke_production_rate,
                                       const float fuel_burn_rate,
                                       const float fuel_ignition_temperature,
                                       const float ignition_temperature_width,
                                       __private float *temperature_source,
                                       __private float *smoke_source,
                                       __private float *fuel_source,
                                       __private float *oxygen_source,
                                       __private float *flame_source) {
    /*
    This kernel computes the combustion behaviour of the solver. Combustion requieres
    oxygen, fuel and temperature larger than the threshold.
    */
    *temperature_source = 0.0f;
    *smoke_source = 0.0f;
    *fuel_source = 0.0f;
    *oxygen_source = 0.0f;
    *flame_source = 0.0f;

    const float oxygen_per_fuel = 1.0f;
    if (fuel_concentration <= 0.0f || oxygen_concentration <= 0.0f || fuel_burn_rate <= 0.0f) {
        return;
    }

    const float temperature_factor =
        ignition_temperature_width > 0.0f
            ? smoothstep(fuel_ignition_temperature, fuel_ignition_temperature + ignition_temperature_width, T)
            : step(fuel_ignition_temperature, T);

    const float available_reactant = fmin(fuel_concentration, oxygen_concentration / oxygen_per_fuel);
    const float effective_burn_rate = fuel_burn_rate * temperature_factor;
    const float burn_fraction = clamp(1.0f - exp(-effective_burn_rate * dt), 0.0f, 1.0f);
    const float burned_fuel = available_reactant * burn_fraction;
    const float reaction_rate = burned_fuel / dt;

    *fuel_source = -reaction_rate;
    *oxygen_source = -oxygen_per_fuel * reaction_rate;
    *smoke_source = smoke_production_rate * reaction_rate;
    *temperature_source = temperature_production_rate * reaction_rate;
    *flame_source = reaction_rate;
}

__kernel void predict_scalar_fields_semi_lagrangian(__global const float *T,
                                                    __global const float *smoke,
                                                    __global const float *fuel,
                                                    __global const float *oxygen,
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
                                                    const int tiles_z) {
    /*
    This kernel performs semi-lagrangian advection for the scalar fields fuel, oxygen,
    smoke and temeprature. It essentially asks: Going back by u*dt what quantity was at
    that position? The sampled quantity is then moved to the current cell.
    */
    const int tile_i = get_group_id(0);
    const int tile_j = get_group_id(1);
    const int tile_k = get_group_id(2);

    const int local_k = get_local_id(0);
    const int local_j = get_local_id(1);
    const int local_i = get_local_id(2);

    if (tile_i >= tiles_x || tile_j >= tiles_y || tile_k >= tiles_z)
        return;

    const int tile_map_index = (tile_i * tiles_y + tile_j) * tiles_z + tile_k;

    const int tile_index = index_tile_map[tile_map_index];

    if (tile_index == -1)
        return;

    const int i = tile_i * TILE_SIZE + local_i;
    const int j = tile_j * TILE_SIZE + local_j;
    const int k = tile_k * TILE_SIZE + local_k;

    // ---------------------------------------------------------
    // Backtrace
    // ---------------------------------------------------------

    float x_depart;
    float y_depart;
    float z_depart;

    backtrace_position_sparse(u, v, w, index_tile_map, (float)i, (float)j, (float)k, dt / delta, n_substeps, nx, ny, nz,
                              u_initial, v_initial, w_initial, tiles_y, tiles_z, &x_depart, &y_depart, &z_depart);

    float sampled_T;
    float sampled_smoke;
    float sampled_fuel;
    float sampled_oxygen;

    int fuel_x0, fuel_y0, fuel_z0, fuel_x1, fuel_y1, fuel_z1;
    float fuel_tx, fuel_ty, fuel_tz;

    // prepare data used for trilinear interpolation
    prepare_trilinear_coords(x_depart, y_depart, z_depart, nx, ny, nz, &fuel_x0, &fuel_y0, &fuel_z0, &fuel_x1, &fuel_y1,
                             &fuel_z1, &fuel_tx, &fuel_ty, &fuel_tz);

    // ---------------------------------------------------------
    // Trilinear sampling
    // ---------------------------------------------------------

    sampled_T = sample_trilinear_inner_sparse(T, index_tile_map, fuel_x0, fuel_y0, fuel_z0, fuel_x1, fuel_y1, fuel_z1,
                                              fuel_tx, fuel_ty, fuel_tz, t_reference, tiles_y, tiles_z);
    sampled_smoke = sample_trilinear_inner_sparse(smoke, index_tile_map, fuel_x0, fuel_y0, fuel_z0, fuel_x1, fuel_y1,
                                                  fuel_z1, fuel_tx, fuel_ty, fuel_tz, 0.0f, tiles_y, tiles_z);
    sampled_fuel = sample_trilinear_inner_sparse(fuel, index_tile_map, fuel_x0, fuel_y0, fuel_z0, fuel_x1, fuel_y1,
                                                 fuel_z1, fuel_tx, fuel_ty, fuel_tz, 0.0f, tiles_y, tiles_z);
    sampled_oxygen = sample_trilinear_inner_sparse(oxygen, index_tile_map, fuel_x0, fuel_y0, fuel_z0, fuel_x1, fuel_y1,
                                                   fuel_z1, fuel_tx, fuel_ty, fuel_tz, 100.0f, tiles_y, tiles_z);

    const int index = ((tile_index * TILE_SIZE + local_i) * TILE_SIZE + local_j) * TILE_SIZE + local_k;

    predictor_T[index] = sampled_T;
    predictor_smoke[index] = sampled_smoke;
    predictor_fuel[index] = sampled_fuel;
    predictor_oxygen[index] = sampled_oxygen;
}

__kernel void update_scalar_fields_maccormack(__global const float *T,
                                              __global const float *smoke,
                                              __global const float *fuel,
                                              __global const float *oxygen,
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
                                              __global float *fuel_out,
                                              __global float *oxygen_out,
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
                                              const float fuel_ignition_temperature_width,
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
                                              const int tiles_z) {
    /*
    This kernel performs the main update of the scalar fields. It receives the
    predicted values from predict_scalar_fields_semi_lagrangian and applies
    MacCormack's correction.

    Semi-Lagrangian advection introduces numerical diffusion, which smoothes
    out sharp features in the scalar fields. MacCormack reduces this error
    by advecting the predicted field backward in time and comparing the
    result with the original field. Half of this difference is then added
    to the predicted values.

    A limiter is applied to prevent overshooting and undershooting.
    */
    const int tile_i = get_group_id(0);
    const int tile_j = get_group_id(1);
    const int tile_k = get_group_id(2);

    const int local_k = get_local_id(0);
    const int local_j = get_local_id(1);
    const int local_i = get_local_id(2);

    if (tile_i >= tiles_x || tile_j >= tiles_y || tile_k >= tiles_z)
        return;

    const int tile_map_index = (tile_i * tiles_y + tile_j) * tiles_z + tile_k;

    const int tile_index = index_tile_map[tile_map_index];

    if (tile_index == -1)
        return;

    const int i = tile_i * TILE_SIZE + local_i;
    const int j = tile_j * TILE_SIZE + local_j;
    const int k = tile_k * TILE_SIZE + local_k;

    if (i < 1 || j < 1 || k < 1 || i >= nx - 1 || j >= ny - 1 || k >= nz - 1)
        return;

    const int index = ((tile_index * TILE_SIZE + local_i) * TILE_SIZE + local_j) * TILE_SIZE + local_k;

    const float dt_over_delta = dt / delta;

    // ---------------------------------------------------------
    // Backtrace
    // ---------------------------------------------------------

    float x_depart;
    float y_depart;
    float z_depart;

    backtrace_position_sparse(u, v, w, index_tile_map, (float)i, (float)j, (float)k, dt_over_delta, n_substeps, nx, ny,
                              nz, u_initial, v_initial, w_initial, tiles_y, tiles_z, &x_depart, &y_depart, &z_depart);

    // ---------------------------------------------------------
    // Forward trace
    // ---------------------------------------------------------

    float x_forward;
    float y_forward;
    float z_forward;

    forward_trace_position_sparse(u, v, w, index_tile_map, x_depart, y_depart, z_depart, dt_over_delta, n_substeps, nx,
                                  ny, nz, u_initial, v_initial, w_initial, tiles_y, tiles_z, &x_forward, &y_forward,
                                  &z_forward);

    // ---------------------------------------------------------
    // Reverse sample
    // ---------------------------------------------------------

    float T_reverse;
    float smoke_reverse;
    float fuel_reverse;
    float oxygen_reverse;

    int forward_x0;
    int forward_y0;
    int forward_z0;

    int forward_x1;
    int forward_y1;
    int forward_z1;

    float forward_tx;
    float forward_ty;
    float forward_tz;

    prepare_trilinear_coords(x_forward, y_forward, z_forward, nx, ny, nz, &forward_x0, &forward_y0, &forward_z0,
                             &forward_x1, &forward_y1, &forward_z1, &forward_tx, &forward_ty, &forward_tz);

    T_reverse = sample_trilinear_inner_sparse(predictor_T, index_tile_map, forward_x0, forward_y0, forward_z0,
                                              forward_x1, forward_y1, forward_z1, forward_tx, forward_ty, forward_tz,
                                              t_reference, tiles_y, tiles_z);

    smoke_reverse = sample_trilinear_inner_sparse(predictor_smoke, index_tile_map, forward_x0, forward_y0, forward_z0,
                                                  forward_x1, forward_y1, forward_z1, forward_tx, forward_ty,
                                                  forward_tz, 0.0f, tiles_y, tiles_z);

    fuel_reverse = sample_trilinear_inner_sparse(predictor_fuel, index_tile_map, forward_x0, forward_y0, forward_z0,
                                                 forward_x1, forward_y1, forward_z1, forward_tx, forward_ty, forward_tz,
                                                 0.0f, tiles_y, tiles_z);

    oxygen_reverse = sample_trilinear_inner_sparse(predictor_oxygen, index_tile_map, forward_x0, forward_y0, forward_z0,
                                                   forward_x1, forward_y1, forward_z1, forward_tx, forward_ty,
                                                   forward_tz, 100.0f, tiles_y, tiles_z);

    // ---------------------------------------------------------
    // MacCormack correction
    // ---------------------------------------------------------

    float T_corrected = predictor_T[index] + 0.5f * (T[index] - T_reverse);
    float smoke_corrected = predictor_smoke[index] + 0.5f * (smoke[index] - smoke_reverse);
    float fuel_corrected = predictor_fuel[index] + 0.5f * (fuel[index] - fuel_reverse);
    float oxygen_corrected = predictor_oxygen[index] + 0.5f * (oxygen[index] - oxygen_reverse);

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

    prepare_trilinear_coords(x_depart, y_depart, z_depart, nx, ny, nz, &x0, &y0, &z0, &x1, &y1, &z1, &tx, &ty, &tz);

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

    sample_cell_extrema_inner_sparse(T, index_tile_map, x0, y0, z0, x1, y1, z1, t_reference, tiles_y, tiles_z, &T_lower,
                                     &T_upper);

    sample_cell_extrema_inner_sparse(smoke, index_tile_map, x0, y0, z0, x1, y1, z1, 0.0f, tiles_y, tiles_z,
                                     &smoke_lower, &smoke_upper);

    sample_cell_extrema_inner_sparse(fuel, index_tile_map, x0, y0, z0, x1, y1, z1, 0.0f, tiles_y, tiles_z, &fuel_lower,
                                     &fuel_upper);

    sample_cell_extrema_inner_sparse(oxygen, index_tile_map, x0, y0, z0, x1, y1, z1, 100.0f, tiles_y, tiles_z,
                                     &oxygen_lower, &oxygen_upper);

    T_corrected = clamp_value(T_corrected, T_lower, T_upper);
    smoke_corrected = clamp_value(smoke_corrected, smoke_lower, smoke_upper);
    fuel_corrected = clamp_value(fuel_corrected, fuel_lower, fuel_upper);
    oxygen_corrected = clamp_value(oxygen_corrected, oxygen_lower, oxygen_upper);

    // ---------------------------------------------------------
    // Combustion
    // ---------------------------------------------------------

    float temperature_burn_source;
    float smoke_burn_source;
    float fuel_burn_source;
    float oxygen_burn_source;
    float flame_burn_source;

    compute_combustion_sources(T_corrected, fuel_corrected, oxygen_corrected, dt, temperature_production_rate,
                               smoke_production_rate, fuel_burn_rate, fuel_ignition_temperature,
                               fuel_ignition_temperature_width, &temperature_burn_source, &smoke_burn_source,
                               &fuel_burn_source, &oxygen_burn_source, &flame_burn_source);

    // ---------------------------------------------------------
    // Dissipation
    // ---------------------------------------------------------

    const float temperature_dissipation = -temperature_dissipation_rate * (T_corrected - t_reference);
    const float smoke_dissipation = -smoke_dissipation_rate * smoke_corrected;
    const float fuel_dissipation = -fuel_dissipation_rate * fuel_corrected;

    // ---------------------------------------------------------
    // Final update
    // ---------------------------------------------------------

    const float T_updated = T_corrected + dt * temperature_burn_source + dt * temperature_dissipation;
    const float smoke_updated = smoke_corrected + dt * smoke_burn_source + dt * smoke_dissipation;
    const float fuel_updated = fuel_corrected + dt * fuel_burn_source + dt * fuel_dissipation;
    const float oxygen_updated = oxygen_corrected + dt * oxygen_burn_source;

    T_out[index] = fmax(T_updated, 0.0f);

    smoke_out[index] = clamp(smoke_updated, 0.0f, 100.0f);
    fuel_out[index] = clamp(fuel_updated, 0.0f, 100.0f);
    oxygen_out[index] = clamp(oxygen_updated, 0.0f, 100.0f);
    flame_out[index] = fmax(flame_burn_source, 0.0f);
}
