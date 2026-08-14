from numba import cuda

@cuda.jit(device=True, inline=True, cache=True)
def _smoothstep(t):
    return t * t * (3.0 - 2.0 * t)


@cuda.jit(device=True, inline=True, cache=True)
def _lerp(a, b, t):
    return a + t * (b - a)


@cuda.jit(device=True, inline=True, cache=True)
def _fast_floor(x):
    i = int(x)
    if x < float(i):
        return i - 1
    return i


@cuda.jit(device=True, inline=True, cache=True)
def _hash_noise_3d(ix, iy, iz, seed):
    n = ix * 15731 + iy * 789221 + iz * 1376312589 + seed * 1013
    n = (n << 13) ^ n
    nn = n * (n * n * 15731 + 789221) + 1376312589
    nn = nn & 0x7FFFFFFF
    return float(nn) / 1073741824.0 - 1.0  # [-1, 1]


@cuda.jit(device=True, inline=True, cache=True)
def _value_noise_3d(x, y, z, seed):
    x0 = _fast_floor(x)
    y0 = _fast_floor(y)
    z0 = _fast_floor(z)

    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    tx = _smoothstep(x - float(x0))
    ty = _smoothstep(y - float(y0))
    tz = _smoothstep(z - float(z0))

    c000 = _hash_noise_3d(x0, y0, z0, seed)
    c100 = _hash_noise_3d(x1, y0, z0, seed)
    c010 = _hash_noise_3d(x0, y1, z0, seed)
    c110 = _hash_noise_3d(x1, y1, z0, seed)

    c001 = _hash_noise_3d(x0, y0, z1, seed)
    c101 = _hash_noise_3d(x1, y0, z1, seed)
    c011 = _hash_noise_3d(x0, y1, z1, seed)
    c111 = _hash_noise_3d(x1, y1, z1, seed)

    x00 = _lerp(c000, c100, tx)
    x10 = _lerp(c010, c110, tx)
    x01 = _lerp(c001, c101, tx)
    x11 = _lerp(c011, c111, tx)

    y0v = _lerp(x00, x10, ty)
    y1v = _lerp(x01, x11, ty)

    return _lerp(y0v, y1v, tz)