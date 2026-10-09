#include "helper.cl"

__kernel void compute_vorticity(__global const float *u,
                                __global const float *v,
                                __global const float *w,
                                const float u_initial,
                                const float v_initial,
                                const float w_initial,
                                __global const uchar *obstacle_mask,
                                __global float *vorticity_magnitude,
                                const float delta,
                                __global const int *index_tile_map,
                                const int nx,
                                const int ny,
                                const int nz,
                                const int tiles_x,
                                const int tiles_y,
                                const int tiles_z) {
    /*
    Compute the magnitude of vorticity (rotation of the velocity field) with central differences
    */
    const SparseCell cell = get_sparse_cell(index_tile_map, tiles_x, tiles_y, tiles_z);

    if (!cell.valid)
        return;

    const int i = cell.i;
    const int j = cell.j;
    const int k = cell.k;
    const int index = cell.cell_index;

    if (i < 1 || j < 1 || k < 1 || i >= nx - 1 || j >= ny - 1 || k >= nz - 1) {
        vorticity_magnitude[index] = 0.0f;
        return;
    }

    if (obstacle_mask[index]) {
        vorticity_magnitude[index] = 0.0f;
        return;
    }

    const float half_inv_delta = 0.5f / delta;

    const float3 omega = curl_sparse(u, v, w, index_tile_map, i, j, k, half_inv_delta, u_initial, v_initial, w_initial,
                                     tiles_y, tiles_z);

    vorticity_magnitude[index] = length(omega);
}

inline void apply_vorticity_confinement(__global const float *u,
                                        __global const float *v,
                                        __global const float *w,
                                        __global const uchar *obstacle_mask,
                                        __global const float *omega_magnitude,
                                        const int i,
                                        const int j,
                                        const int k,
                                        const float delta,
                                        const float vorticity_strength,
                                        __global const int *index_tile_map,
                                        const float u_initial,
                                        const float v_initial,
                                        const float w_initial,
                                        const int nx,
                                        const int ny,
                                        const int nz,
                                        const int tiles_y,
                                        const int tiles_z,
                                        float *ax,
                                        float *ay,
                                        float *az) {
    /*
    Numerical methods for computing flow can be more diffusive. To allow for
    more swirl in the flow and aritistical control a vorticity force is added.
    It is proportional to the rotation of the velocity field scaled by the magnitude
    of vorticity.
    */
    const SparseCell cell = get_sparse_cell_at(index_tile_map, i, j, k, tiles_y, tiles_z);

    if (!cell.valid) {
        *ax = 0.0f;
        *ay = 0.0f;
        *az = 0.0f;
        return;
    }

    const int index = cell.cell_index;

    if (i < 2 || j < 2 || k < 2 || i >= nx - 2 || j >= ny - 2 || k >= nz - 2 || obstacle_mask[index]) {
        *ax = 0.0f;
        *ay = 0.0f;
        *az = 0.0f;
        return;
    }

    const float half_inv_delta = 0.5f / delta;

    const float3 gradient =
        central_gradient_sparse(omega_magnitude, index_tile_map, i, j, k, half_inv_delta, 0.0f, tiles_y, tiles_z);

    const float grad_length = length(gradient);

    if (grad_length <= 1.0e-12f) {
        *ax = 0.0f;
        *ay = 0.0f;
        *az = 0.0f;
        return;
    }

    const float3 normal = gradient / grad_length;

    const float3 omega = curl_sparse(u, v, w, index_tile_map, i, j, k, half_inv_delta, u_initial, v_initial, w_initial,
                                     tiles_y, tiles_z);

    const float3 acceleration = vorticity_strength * cross(normal, omega);

    *ax = acceleration.x;
    *ay = acceleration.y;
    *az = acceleration.z;
}
