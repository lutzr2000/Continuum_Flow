#ifndef TILE_SIZE
#define TILE_SIZE 4
#endif

#include "advection_schemes.cl"
#include "forces.cl"
#include "sparse_managment.cl"
#include "vorticity.cl"

__kernel void advect_velocity_semi_lagrangian(__global const float *u,
                                              __global const float *v,
                                              __global const float *w,
                                              __global float *advected_u,
                                              __global float *advected_v,
                                              __global float *advected_w,
                                              const float dt,
                                              const float delta,
                                              const int n_substeps,
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

    float x_depart;
    float y_depart;
    float z_depart;

    backtrace_position_sparse(u, v, w, index_tile_map, (float)i, (float)j, (float)k, dt / delta, n_substeps, nx, ny, nz,
                              u_initial, v_initial, w_initial, tiles_y, tiles_z, &x_depart, &y_depart, &z_depart);

    float sampled_u;
    float sampled_v;
    float sampled_w;

    sample_trilinear_vec3_sparse(u, v, w, index_tile_map, x_depart, y_depart, z_depart, nx, ny, nz, u_initial,
                                 v_initial, w_initial, tiles_y, tiles_z, &sampled_u, &sampled_v, &sampled_w);

    const int index = ((tile_index * TILE_SIZE + local_i) * TILE_SIZE + local_j) * TILE_SIZE + local_k;

    advected_u[index] = sampled_u;
    advected_v[index] = sampled_v;
    advected_w[index] = sampled_w;
}

inline float3 apply_force_acceleration(__global const float *u,
                           __global const float *v,
                           __global const float *w,
                           __global const uchar *obstacle_mask,
                           __global const float *vorticity_magnitude,
                           const float vorticity_strength,
                           __global const float *temperature,
                           const float buoyancy_factor,
                           const float t_reference,
                           const float gravity_x,
                           const float gravity_y,
                           const float gravity_z,
                           __global const int *index_tile_map,
                           const float fx_const,
                           const float fy_const,
                           const float fz_const,
                           const int has_swirl_nodes,
                           __global const float *swirl_config,
                           const int swirl_count,
                           const float origin_x,
                           const float origin_y,
                           const float origin_z,
                           const int has_turbulence_nodes,
                           __global const float *turbulence_config,
                           const int turbulence_count,
                           const int i,
                           const int j,
                           const int k,
                           const float delta,
                           const float u_initial,
                           const float v_initial,
                           const float w_initial,
                           const int nx,
                           const int ny,
                           const int nz,
                           const int tiles_y,
                           const int tiles_z) {
    float ax = 0.0f;
    float ay = 0.0f;
    float az = 0.0f;

    if (vorticity_strength > 0.0f) {
        apply_vorticity_confinement(u, v, w, obstacle_mask, vorticity_magnitude, i, j, k, delta, vorticity_strength,
                                    index_tile_map, u_initial, v_initial, w_initial, nx, ny, nz, tiles_y, tiles_z, &ax,
                                    &ay, &az);
    }

    if (has_swirl_nodes && swirl_count > 0) {
        float swirl_ax;
        float swirl_ay;
        float swirl_az;

        apply_swirl_forces(swirl_config, swirl_count, i, j, k, delta, origin_x, origin_y, origin_z, &swirl_ax,
                           &swirl_ay, &swirl_az);

        ax += swirl_ax;
        ay += swirl_ay;
        az += swirl_az;
    }

    if (has_turbulence_nodes && turbulence_count > 0) {
        float turbulence_ax;
        float turbulence_ay;
        float turbulence_az;

        apply_turbulence_forces(turbulence_config, turbulence_count, i, j, k, delta, origin_x, origin_y, origin_z,
                                &turbulence_ax, &turbulence_ay, &turbulence_az);

        ax += turbulence_ax;
        ay += turbulence_ay;
        az += turbulence_az;
    }

    ax += fx_const * 0.1f;
    ay += fy_const * 0.1f;
    az += fz_const * 0.1f;

    const float buoyancy_value =
        buoyancy(temperature, index_tile_map, i, j, k, buoyancy_factor, t_reference, tiles_y, tiles_z);

    ax += gravity_x * buoyancy_value;
    ay += gravity_y * buoyancy_value;
    az += gravity_z * buoyancy_value;

    return (float3)(ax, ay, az);
}

inline float3 diffusion(__global const float *u,
                        __global const float *v,
                        __global const float *w,
                        __global const int *index_tile_map,
                        const float3 rhs,
                        const int i,
                        const int j,
                        const int k,
                        const float diffusion_alpha,
                        const float diffusion_inv_diag,
                        const float u_initial,
                        const float v_initial,
                        const float w_initial,
                        const int tiles_y,
                        const int tiles_z) {
    const float u_neighbor_sum = get_pool_value(u, index_tile_map, i + 1, j, k, u_initial, tiles_y, tiles_z) +
                                 get_pool_value(u, index_tile_map, i - 1, j, k, u_initial, tiles_y, tiles_z) +
                                 get_pool_value(u, index_tile_map, i, j + 1, k, u_initial, tiles_y, tiles_z) +
                                 get_pool_value(u, index_tile_map, i, j - 1, k, u_initial, tiles_y, tiles_z) +
                                 get_pool_value(u, index_tile_map, i, j, k + 1, u_initial, tiles_y, tiles_z) +
                                 get_pool_value(u, index_tile_map, i, j, k - 1, u_initial, tiles_y, tiles_z);

    const float v_neighbor_sum = get_pool_value(v, index_tile_map, i + 1, j, k, v_initial, tiles_y, tiles_z) +
                                 get_pool_value(v, index_tile_map, i - 1, j, k, v_initial, tiles_y, tiles_z) +
                                 get_pool_value(v, index_tile_map, i, j + 1, k, v_initial, tiles_y, tiles_z) +
                                 get_pool_value(v, index_tile_map, i, j - 1, k, v_initial, tiles_y, tiles_z) +
                                 get_pool_value(v, index_tile_map, i, j, k + 1, v_initial, tiles_y, tiles_z) +
                                 get_pool_value(v, index_tile_map, i, j, k - 1, v_initial, tiles_y, tiles_z);

    const float w_neighbor_sum = get_pool_value(w, index_tile_map, i + 1, j, k, w_initial, tiles_y, tiles_z) +
                                 get_pool_value(w, index_tile_map, i - 1, j, k, w_initial, tiles_y, tiles_z) +
                                 get_pool_value(w, index_tile_map, i, j + 1, k, w_initial, tiles_y, tiles_z) +
                                 get_pool_value(w, index_tile_map, i, j - 1, k, w_initial, tiles_y, tiles_z) +
                                 get_pool_value(w, index_tile_map, i, j, k + 1, w_initial, tiles_y, tiles_z) +
                                 get_pool_value(w, index_tile_map, i, j, k - 1, w_initial, tiles_y, tiles_z);

    return (float3)((rhs.x + diffusion_alpha * u_neighbor_sum) * diffusion_inv_diag,
                    (rhs.y + diffusion_alpha * v_neighbor_sum) * diffusion_inv_diag,
                    (rhs.z + diffusion_alpha * w_neighbor_sum) * diffusion_inv_diag);
}

__kernel void update_velocity_maccormack(__global const float *u,
                                         __global const float *v,
                                         __global const float *w,
                                         __global const uchar *obstacle_mask,
                                         __global const float *predictor_u,
                                         __global const float *predictor_v,
                                         __global const float *predictor_w,
                                         const float dt,
                                         __global float *un,
                                         __global float *vn,
                                         __global float *wn,
                                         const float delta,
                                         const float rho,
                                         const int n_substeps,
                                         const float nu,
                                         __global const float *vorticity_magnitude,
                                         const float vorticity_strength,
                                         __global const float *temperature,
                                         const float buoyancy_factor,
                                         const float t_reference,
                                         const float gravity_x,
                                         const float gravity_y,
                                         const float gravity_z,
                                         __global const int *index_tile_map,
                                         const float fx_const,
                                         const float fy_const,
                                         const float fz_const,
                                         const int has_swirl_nodes,
                                         __global const float *swirl_config,
                                         const int swirl_count,
                                         const float origin_x,
                                         const float origin_y,
                                         const float origin_z,
                                         const int has_turbulence_nodes,
                                         __global const float *turbulence_config,
                                         const int turbulence_count,
                                         const float u_initial,
                                         const float v_initial,
                                         const float w_initial,
                                         const int nx,
                                         const int ny,
                                         const int nz,
                                         const int tiles_x,
                                         const int tiles_y,
                                         const int tiles_z) {
    const int tile_i = get_group_id(0);
    const int tile_j = get_group_id(1);
    const int tile_k = get_group_id(2);

    const int local_k = get_local_id(0);
    const int local_j = get_local_id(1);
    const int local_i = get_local_id(2);

    if (tile_i >= tiles_x || tile_j >= tiles_y || tile_k >= tiles_z)
        return;

    const int i = tile_i * TILE_SIZE + local_i;
    const int j = tile_j * TILE_SIZE + local_j;
    const int k = tile_k * TILE_SIZE + local_k;

    const int tile_map_index = (tile_i * tiles_y + tile_j) * tiles_z + tile_k;

    const int tile_index = index_tile_map[tile_map_index];

    if (tile_index == -1)
        return;

    if (i < 1 || j < 1 || k < 1 || i >= nx - 1 || j >= ny - 1 || k >= nz - 1)
        return;

    const int index = ((tile_index * TILE_SIZE + local_i) * TILE_SIZE + local_j) * TILE_SIZE + local_k;

    const float dt_over_delta = dt / delta;

    const float diffusion_alpha = nu * dt / (delta * delta);

    const float diffusion_inv_diag = 1.0f / (1.0f + 6.0f * diffusion_alpha);

    const float force_coeff = dt / rho;

    const float u_center = u[index];
    const float v_center = v[index];
    const float w_center = w[index];

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
    // Predictor
    // ---------------------------------------------------------

    const float advected_u = predictor_u[index];
    const float advected_v = predictor_v[index];
    const float advected_w = predictor_w[index];

    // ---------------------------------------------------------
    // Reverse sample
    // ---------------------------------------------------------

    float reverse_u;
    float reverse_v;
    float reverse_w;

    sample_trilinear_vec3_sparse(predictor_u, predictor_v, predictor_w, index_tile_map, x_forward, y_forward, z_forward,
                                 nx, ny, nz, u_initial, v_initial, w_initial, tiles_y, tiles_z, &reverse_u, &reverse_v,
                                 &reverse_w);

    // ---------------------------------------------------------
    // MacCormack correction
    // ---------------------------------------------------------

    float corrected_u = advected_u + 0.5f * (u_center - reverse_u);
    float corrected_v = advected_v + 0.5f * (v_center - reverse_v);
    float corrected_w = advected_w + 0.5f * (w_center - reverse_w);

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

    float u_lower;
    float u_upper;

    float v_lower;
    float v_upper;

    float w_lower;
    float w_upper;

    sample_cell_extrema_inner_sparse(u, index_tile_map, x0, y0, z0, x1, y1, z1, u_initial, tiles_y, tiles_z, &u_lower,
                                     &u_upper);

    sample_cell_extrema_inner_sparse(v, index_tile_map, x0, y0, z0, x1, y1, z1, v_initial, tiles_y, tiles_z, &v_lower,
                                     &v_upper);

    sample_cell_extrema_inner_sparse(w, index_tile_map, x0, y0, z0, x1, y1, z1, w_initial, tiles_y, tiles_z, &w_lower,
                                     &w_upper);

    corrected_u = clamp_value(corrected_u, u_lower, u_upper);
    corrected_v = clamp_value(corrected_v, v_lower, v_upper);
    corrected_w = clamp_value(corrected_w, w_lower, w_upper);

    // ---------------------------------------------------------
    // Forces
    // ---------------------------------------------------------

    const float3 acceleration = apply_force_acceleration(
        u, v, w, obstacle_mask, vorticity_magnitude, vorticity_strength, temperature, buoyancy_factor, t_reference,
        gravity_x, gravity_y, gravity_z, index_tile_map, fx_const, fy_const, fz_const, has_swirl_nodes, swirl_config,
        swirl_count, origin_x, origin_y, origin_z, has_turbulence_nodes, turbulence_config, turbulence_count, i, j, k,
        delta, u_initial, v_initial, w_initial, nx, ny, nz, tiles_y, tiles_z);

    const float3 rhs =
        (float3)(corrected_u + force_coeff * acceleration.x, corrected_v + force_coeff * acceleration.y,
                 corrected_w + force_coeff * acceleration.z);

    // ---------------------------------------------------------
    // Diffusion
    // ---------------------------------------------------------

    const float3 velocity = diffusion(u, v, w, index_tile_map, rhs, i, j, k, diffusion_alpha, diffusion_inv_diag,
                                      u_initial, v_initial, w_initial, tiles_y, tiles_z);

    un[index] = velocity.x;
    vn[index] = velocity.y;
    wn[index] = velocity.z;
}
