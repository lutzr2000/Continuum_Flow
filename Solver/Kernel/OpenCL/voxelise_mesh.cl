inline float edge_function(const float2 a, const float2 b, const float2 p) {
    return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
}

inline int conservative_edge_test(
    const float2 a, const float2 b, const float2 p, const float orientation, const float half_delta) {
    const float2 edge = b - a;
    const float support = half_delta * (fabs(edge.x) + fabs(edge.y));
    return orientation * edge_function(a, b, p) + support >= 0.0f;
}

inline void
mark_voxel(volatile __global uint *mask, const int i, const int j, const int k, const int size_y, const int size_z) {
    const int index = (i * size_y + j) * size_z + k;
    atomic_or(mask + (index >> 5), 1u << (index & 31));
}

__kernel void surface(__global const float *triangles,
                      volatile __global uint *mask,
                      const int triangle_count,
                      const int size_x,
                      const int size_y,
                      const int size_z,
                      const float delta,
                      const float ox,
                      const float oy,
                      const float oz) {
    /*
    Triangle-driven conservative surface voxelization. Each work-item handles
    one triangle and scans only its dominant-axis 2-D projection.
    */
    const int triangle_index = get_global_id(0);

    if (triangle_index >= triangle_count)
        return;

    const int base = triangle_index * 9;
    const float3 a = (float3)(triangles[base + 0], triangles[base + 1], triangles[base + 2]);
    const float3 b = (float3)(triangles[base + 3], triangles[base + 4], triangles[base + 5]);
    const float3 c = (float3)(triangles[base + 6], triangles[base + 7], triangles[base + 8]);
    const float3 normal = cross(b - a, c - a);
    const float3 abs_normal = fabs(normal);

    int axis;

    if (abs_normal.x >= abs_normal.y && abs_normal.x >= abs_normal.z)
        axis = 0;
    else if (abs_normal.y >= abs_normal.z)
        axis = 1;
    else
        axis = 2;

    if (normal[axis] == 0.0f)
        return;

    const int u_axis = (axis + 1) % 3;
    const int v_axis = (axis + 2) % 3;
    const float3 origin = (float3)(ox, oy, oz);
    const int3 sizes = (int3)(size_x, size_y, size_z);
    const float half_delta = 0.5f * delta;
    const float2 projected_a = (float2)(a[u_axis], a[v_axis]);
    const float2 projected_b = (float2)(b[u_axis], b[v_axis]);
    const float2 projected_c = (float2)(c[u_axis], c[v_axis]);
    const float orientation = edge_function(projected_a, projected_b, projected_c) >= 0.0f ? 1.0f : -1.0f;
    const float plane_grid_extent = 0.5f + 0.5f * (abs_normal[u_axis] + abs_normal[v_axis]) / abs_normal[axis];

    const float min_u = fmin(a[u_axis], fmin(b[u_axis], c[u_axis])) - half_delta;
    const float max_u = fmax(a[u_axis], fmax(b[u_axis], c[u_axis])) + half_delta;
    const float min_v = fmin(a[v_axis], fmin(b[v_axis], c[v_axis])) - half_delta;
    const float max_v = fmax(a[v_axis], fmax(b[v_axis], c[v_axis])) + half_delta;

    const int u_min = max(0, (int)floor((min_u - origin[u_axis]) / delta));
    const int u_max = min(sizes[u_axis] - 1, (int)ceil((max_u - origin[u_axis]) / delta));
    const int v_min = max(0, (int)floor((min_v - origin[v_axis]) / delta));
    const int v_max = min(sizes[v_axis] - 1, (int)ceil((max_v - origin[v_axis]) / delta));

    for (int u = u_min; u <= u_max; ++u) {
        const float position_u = origin[u_axis] + (float)u * delta;

        for (int v = v_min; v <= v_max; ++v) {
            const float position_v = origin[v_axis] + (float)v * delta;
            const float2 projected_position = (float2)(position_u, position_v);

            if (!conservative_edge_test(projected_a, projected_b, projected_position, orientation, half_delta) ||
                !conservative_edge_test(projected_b, projected_c, projected_position, orientation, half_delta) ||
                !conservative_edge_test(projected_c, projected_a, projected_position, orientation, half_delta))
                continue;

            const float plane_position =
                a[axis] -
                (normal[u_axis] * (position_u - a[u_axis]) + normal[v_axis] * (position_v - a[v_axis])) / normal[axis];

            const float grid_position = (plane_position - origin[axis]) / delta;
            const int w_min = max(0, (int)ceil(grid_position - plane_grid_extent - 1.0e-5f));
            const int w_max = min(sizes[axis] - 1, (int)floor(grid_position + plane_grid_extent + 1.0e-5f));

            for (int w = w_min; w <= w_max; ++w) {
                int3 grid_position_3d;
                grid_position_3d[axis] = w;
                grid_position_3d[u_axis] = u;
                grid_position_3d[v_axis] = v;

                mark_voxel(mask, grid_position_3d.x, grid_position_3d.y, grid_position_3d.z, size_y, size_z);

                // Preserve the old one-voxel inward dilation.
                int3 inward = grid_position_3d;
                inward[axis] += normal[axis] > 0.0f ? -1 : 1;

                if (inward[axis] >= 0 && inward[axis] < sizes[axis])
                    mark_voxel(mask, inward.x, inward.y, inward.z, size_y, size_z);
            }
        }
    }
}
