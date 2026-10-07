inline uint gradient_noise_hash_3d(
    const int x,
    const int y,
    const int z,
    const int seed
)
{
    uint h =
        (uint)x * 0x8da6b343u
        ^ (uint)y * 0xd8163841u
        ^ (uint)z * 0xcb1ab31fu
        ^ (uint)seed * 0x9e3779b9u;

    h ^= h >> 15;
    h *= 0x2c1b3c6du;
    h ^= h >> 12;

    return h;
}


inline float gradient_noise_dot(
    const uint hash,
    const float x,
    const float y,
    const float z
)
{
    const uint h = hash & 15u;
    const float u = h < 8u ? x : y;
    const float v =
        h < 4u
        ? y
        : (h == 12u || h == 14u ? x : z);

    return
        ((h & 1u) == 0u ? u : -u)
        + ((h & 2u) == 0u ? v : -v);
}


inline float gradient_noise_3d(
    const float x,
    const float y,
    const float z,
    const int seed,
    const float scale
)
{
    const float safe_scale = fmax(fabs(scale), 1.0e-6f);

    const float px = x / safe_scale;
    const float py = y / safe_scale;
    const float pz = z / safe_scale;

    const int x0 = convert_int_rtn(px);
    const int y0 = convert_int_rtn(py);
    const int z0 = convert_int_rtn(pz);

    const float dx = px - (float)x0;
    const float dy = py - (float)y0;
    const float dz = pz - (float)z0;

    const float u = dx * dx * dx * (dx * (dx * 6.0f - 15.0f) + 10.0f);
    const float v = dy * dy * dy * (dy * (dy * 6.0f - 15.0f) + 10.0f);
    const float w = dz * dz * dz * (dz * (dz * 6.0f - 15.0f) + 10.0f);

    const float n000 = gradient_noise_dot(
        gradient_noise_hash_3d(x0, y0, z0, seed),
        dx,
        dy,
        dz
    );
    const float n100 = gradient_noise_dot(
        gradient_noise_hash_3d(x0 + 1, y0, z0, seed),
        dx - 1.0f,
        dy,
        dz
    );
    const float n010 = gradient_noise_dot(
        gradient_noise_hash_3d(x0, y0 + 1, z0, seed),
        dx,
        dy - 1.0f,
        dz
    );
    const float n110 = gradient_noise_dot(
        gradient_noise_hash_3d(x0 + 1, y0 + 1, z0, seed),
        dx - 1.0f,
        dy - 1.0f,
        dz
    );
    const float n001 = gradient_noise_dot(
        gradient_noise_hash_3d(x0, y0, z0 + 1, seed),
        dx,
        dy,
        dz - 1.0f
    );
    const float n101 = gradient_noise_dot(
        gradient_noise_hash_3d(x0 + 1, y0, z0 + 1, seed),
        dx - 1.0f,
        dy,
        dz - 1.0f
    );
    const float n011 = gradient_noise_dot(
        gradient_noise_hash_3d(x0, y0 + 1, z0 + 1, seed),
        dx,
        dy - 1.0f,
        dz - 1.0f
    );
    const float n111 = gradient_noise_dot(
        gradient_noise_hash_3d(x0 + 1, y0 + 1, z0 + 1, seed),
        dx - 1.0f,
        dy - 1.0f,
        dz - 1.0f
    );

    const float nx00 = mad(u, n100 - n000, n000);
    const float nx10 = mad(u, n110 - n010, n010);
    const float nx01 = mad(u, n101 - n001, n001);
    const float nx11 = mad(u, n111 - n011, n011);

    const float nxy0 = mad(v, nx10 - nx00, nx00);
    const float nxy1 = mad(v, nx11 - nx01, nx01);

    return mad(w, nxy1 - nxy0, nxy0) * 0.70710678118f;
}


inline float noise_amplitude_multiplier(
    const float noise,
    const float amplitude
)
{
    // An amplitude of 0.5 varies the configured value by up to +/-50%.
    return fmax(1.0f + noise * amplitude, 0.0f);
}
