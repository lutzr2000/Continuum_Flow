__kernel void surface(
    __global const float *triangles,
    __global uchar *mask,
    const int triangle_count,
    const int size_x,
    const int size_y,
    const int size_z,
    const float delta,
    const float ox,
    const float oy,
    const float oz
)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);
    const int k = get_global_id(2);

    if (i >= size_x || j >= size_y || k >= size_z)
        return;

    const float x = ox + (float)i * delta;
    const float y = oy + (float)j * delta;
    const float z = oz + (float)k * delta;

    for (int t = 0; t < triangle_count; ++t)
    {
        const int base = t * 9;

        const float x0 = triangles[base + 0];
        const float y0 = triangles[base + 1];
        const float z0 = triangles[base + 2];

        const float x1 = triangles[base + 3];
        const float y1 = triangles[base + 4];
        const float z1 = triangles[base + 5];

        const float x2 = triangles[base + 6];
        const float y2 = triangles[base + 7];
        const float z2 = triangles[base + 8];

        const float xmin = fmin(x0, fmin(x1, x2));
        const float xmax = fmax(x0, fmax(x1, x2));

        const float ymin = fmin(y0, fmin(y1, y2));
        const float ymax = fmax(y0, fmax(y1, y2));

        const float zmin = fmin(z0, fmin(z1, z2));
        const float zmax = fmax(z0, fmax(z1, z2));

        if (
            x < xmin - delta ||
            x > xmax + delta ||
            y < ymin - delta ||
            y > ymax + delta ||
            z < zmin - delta ||
            z > zmax + delta
        )
            continue;

        const float ax = x1 - x0;
        const float ay = y1 - y0;
        const float az = z1 - z0;

        const float bx = x2 - x0;
        const float by = y2 - y0;
        const float bz = z2 - z0;

        float nx = ay * bz - az * by;
        float ny = az * bx - ax * bz;
        float nz = ax * by - ay * bx;

        const float length = sqrt(
            nx * nx +
            ny * ny +
            nz * nz
        );

        if (length == 0.0f)
            continue;

        const float distance = fabs(
            nx * (x - x0) +
            ny * (y - y0) +
            nz * (z - z0)
        );

        if (distance > delta * 0.87f * length)
            continue;

        // Surface voxel.
        const int index =
            (i * size_y + j) * size_z + k;

        mask[index] = 1;

        // Normalize outward triangle normal.
        nx /= length;
        ny /= length;
        nz /= length;

        // Move one voxel inward.
        int ii = i;
        int jj = j;
        int kk = k;

        const float anx = fabs(nx);
        const float any_ = fabs(ny);
        const float anz = fabs(nz);

        if (anx >= any_ && anx >= anz)
        {
            if (nx > 0.0f)
                ii -= 1;
            else
                ii += 1;
        }
        else if (any_ >= anz)
        {
            if (ny > 0.0f)
                jj -= 1;
            else
                jj += 1;
        }
        else
        {
            if (nz > 0.0f)
                kk -= 1;
            else
                kk += 1;
        }

        if (
            ii >= 0 && ii < size_x &&
            jj >= 0 && jj < size_y &&
            kk >= 0 && kk < size_z
        )
        {
            const int inward_index =
                (ii * size_y + jj) * size_z + kk;

            mask[inward_index] = 1;
        }

        return;
    }
}