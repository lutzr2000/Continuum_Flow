uint gradient_noise_hash_3d(ivec3 p, int seed)
{
    uint h = uint(p.x) * 0x8da6b343u
        ^ uint(p.y) * 0xd8163841u
        ^ uint(p.z) * 0xcb1ab31fu
        ^ uint(seed) * 0x9e3779b9u;
    h ^= h >> 15u;
    h *= 0x2c1b3c6du;
    h ^= h >> 12u;
    return h;
}

float gradient_noise_dot(uint hash, vec3 p)
{
    uint h = hash & 15u;
    float u = h < 8u ? p.x : p.y;
    float v = h < 4u ? p.y : ((h == 12u || h == 14u) ? p.x : p.z);
    return ((h & 1u) == 0u ? u : -u) + ((h & 2u) == 0u ? v : -v);
}

float gradient_noise_3d(vec3 position, int seed, float scale)
{
    vec3 p = position / max(abs(scale), 1.0e-6);
    ivec3 p0 = ivec3(floor(p));
    vec3 d = p - vec3(p0);
    vec3 f = d * d * d * (d * (d * 6.0 - 15.0) + 10.0);

    float n000 = gradient_noise_dot(gradient_noise_hash_3d(p0, seed), d);
    float n100 = gradient_noise_dot(gradient_noise_hash_3d(p0 + ivec3(1, 0, 0), seed), d - vec3(1, 0, 0));
    float n010 = gradient_noise_dot(gradient_noise_hash_3d(p0 + ivec3(0, 1, 0), seed), d - vec3(0, 1, 0));
    float n110 = gradient_noise_dot(gradient_noise_hash_3d(p0 + ivec3(1, 1, 0), seed), d - vec3(1, 1, 0));
    float n001 = gradient_noise_dot(gradient_noise_hash_3d(p0 + ivec3(0, 0, 1), seed), d - vec3(0, 0, 1));
    float n101 = gradient_noise_dot(gradient_noise_hash_3d(p0 + ivec3(1, 0, 1), seed), d - vec3(1, 0, 1));
    float n011 = gradient_noise_dot(gradient_noise_hash_3d(p0 + ivec3(0, 1, 1), seed), d - vec3(0, 1, 1));
    float n111 = gradient_noise_dot(gradient_noise_hash_3d(p0 + ivec3(1, 1, 1), seed), d - vec3(1, 1, 1));

    float nx00 = mix(n000, n100, f.x);
    float nx10 = mix(n010, n110, f.x);
    float nx01 = mix(n001, n101, f.x);
    float nx11 = mix(n011, n111, f.x);
    return mix(mix(nx00, nx10, f.y), mix(nx01, nx11, f.y), f.z) * 0.70710678118;
}

void main()
{
    float noise = gradient_noise_3d(noise_position, seed, scale);
    float gray = 0.5 + 0.5 * noise * animation_factor;
    frag_color = vec4(vec3(clamp(gray, 0.0, 1.0)), 1.0);
}
