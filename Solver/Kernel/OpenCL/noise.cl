inline float smoothstep_noise(
    const float t
)
{
    return t * t * (3.0f - 2.0f * t);
}


inline float lerp_noise(
    const float a,
    const float b,
    const float t
)
{
    return a + t * (b - a);
}


inline int fast_floor(
    const float x
)
{
    const int i = (int)x;

    if (x < (float)i)
        return i - 1;

    return i;
}


inline float hash_noise_3d(
    const int ix,
    const int iy,
    const int iz,
    const int seed
)
{
    int n =
        ix * 15731 +
        iy * 789221 +
        iz * 1376312589 +
        seed * 1013;

    n = (n << 13) ^ n;

    int nn =
        n * (n * n * 15731 + 789221)
        + 1376312589;

    nn = nn & 0x7FFFFFFF;

    return
        (float)nn / 1073741824.0f
        - 1.0f;
}


inline float value_noise_3d(
    const float x,
    const float y,
    const float z,
    const int seed
)
{
    const int x0 = fast_floor(x);
    const int y0 = fast_floor(y);
    const int z0 = fast_floor(z);

    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const int z1 = z0 + 1;

    const float tx =
        smoothstep_noise(x - (float)x0);

    const float ty =
        smoothstep_noise(y - (float)y0);

    const float tz =
        smoothstep_noise(z - (float)z0);

    const float c000 =
        hash_noise_3d(x0, y0, z0, seed);

    const float c100 =
        hash_noise_3d(x1, y0, z0, seed);

    const float c010 =
        hash_noise_3d(x0, y1, z0, seed);

    const float c110 =
        hash_noise_3d(x1, y1, z0, seed);

    const float c001 =
        hash_noise_3d(x0, y0, z1, seed);

    const float c101 =
        hash_noise_3d(x1, y0, z1, seed);

    const float c011 =
        hash_noise_3d(x0, y1, z1, seed);

    const float c111 =
        hash_noise_3d(x1, y1, z1, seed);

    const float x00 =
        lerp_noise(c000, c100, tx);

    const float x10 =
        lerp_noise(c010, c110, tx);

    const float x01 =
        lerp_noise(c001, c101, tx);

    const float x11 =
        lerp_noise(c011, c111, tx);

    const float y0v =
        lerp_noise(x00, x10, ty);

    const float y1v =
        lerp_noise(x01, x11, ty);

    return lerp_noise(
        y0v,
        y1v,
        tz
    );
}