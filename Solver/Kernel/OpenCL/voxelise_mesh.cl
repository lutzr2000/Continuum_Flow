inline float point_triangle_distance_squared(const float3 p, const float3 a, const float3 b, const float3 c) {
    /*
    Closest-point regions from Real-Time Collision Detection.
    */
    const float3 ab = b - a;
    const float3 ac = c - a;
    const float3 ap = p - a;
    const float d1 = dot(ab, ap);
    const float d2 = dot(ac, ap);

    if (d1 <= 0.0f && d2 <= 0.0f)
        return dot(ap, ap);

    const float3 bp = p - b;
    const float d3 = dot(ab, bp);
    const float d4 = dot(ac, bp);

    if (d3 >= 0.0f && d4 <= d3)
        return dot(bp, bp);

    const float vc = d1 * d4 - d3 * d2;

    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        const float v = d1 / (d1 - d3);
        const float3 pq = p - (a + v * ab);
        return dot(pq, pq);
    }

    const float3 cp = p - c;
    const float d5 = dot(ab, cp);
    const float d6 = dot(ac, cp);

    if (d6 >= 0.0f && d5 <= d6)
        return dot(cp, cp);

    const float vb = d5 * d2 - d1 * d6;

    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        const float w = d2 / (d2 - d6);
        const float3 pq = p - (a + w * ac);
        return dot(pq, pq);
    }

    const float va = d3 * d6 - d5 * d4;

    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        const float3 pq = p - (b + w * (c - b));
        return dot(pq, pq);
    }

    const float inverse_sum = 1.0f / (va + vb + vc);
    const float3 q = a + ab * (vb * inverse_sum) + ac * (vc * inverse_sum);
    const float3 pq = p - q;
    return dot(pq, pq);
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
    const float radius = 0.866025404f * delta;
    const float radius_squared = radius * radius;

    const float min_u = fmin(a[u_axis], fmin(b[u_axis], c[u_axis])) - radius;
    const float max_u = fmax(a[u_axis], fmax(b[u_axis], c[u_axis])) + radius;
    const float min_v = fmin(a[v_axis], fmin(b[v_axis], c[v_axis])) - radius;
    const float max_v = fmax(a[v_axis], fmax(b[v_axis], c[v_axis])) + radius;

    const int u_min = max(0, (int)floor((min_u - origin[u_axis]) / delta));
    const int u_max = min(sizes[u_axis] - 1, (int)ceil((max_u - origin[u_axis]) / delta));
    const int v_min = max(0, (int)floor((min_v - origin[v_axis]) / delta));
    const int v_max = min(sizes[v_axis] - 1, (int)ceil((max_v - origin[v_axis]) / delta));

    for (int u = u_min; u <= u_max; ++u) {
        const float position_u = origin[u_axis] + (float)u * delta;

        for (int v = v_min; v <= v_max; ++v) {
            const float position_v = origin[v_axis] + (float)v * delta;
            const float plane_position =
                a[axis] -
                (normal[u_axis] * (position_u - a[u_axis]) + normal[v_axis] * (position_v - a[v_axis])) / normal[axis];
            const float grid_position = (plane_position - origin[axis]) / delta;
            const int w_min = max(0, (int)floor(grid_position) - 1);
            const int w_max = min(sizes[axis] - 1, (int)ceil(grid_position) + 1);

            for (int w = w_min; w <= w_max; ++w) {
                int3 grid_position_3d;
                grid_position_3d[axis] = w;
                grid_position_3d[u_axis] = u;
                grid_position_3d[v_axis] = v;

                const float3 p = origin + convert_float3(grid_position_3d) * delta;

                if (point_triangle_distance_squared(p, a, b, c) > radius_squared)
                    continue;

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
