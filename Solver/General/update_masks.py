from numba import njit, prange
import numpy as np

@njit(cache=True, parallel=True)
def _sample_mask_backwards(
    out, base, delta, ox, oy, oz, ix0, ix1, iy0, iy1, iz0, iz1, box, inv
):
    """Back-sample a local reference mask into the world grid."""
    base_ox, base_oy, base_oz = box
    bn_x, bn_y, bn_z = base.shape
    sx, sy, sz = ix1 - ix0 + 1, iy1 - iy0 + 1, iz1 - iz0 + 1

    for n in prange(sx * sy * sz):
        i = ix0 + n // (sy * sz)
        r = n % (sy * sz)
        j = iy0 + r // sz
        k = iz0 + r % sz

        x = np.float32(ox + i * delta)
        y = np.float32(oy + j * delta)
        z = np.float32(oz + k * delta)

        bx = inv[0, 0] * x + inv[0, 1] * y + inv[0, 2] * z + inv[0, 3]
        by = inv[1, 0] * x + inv[1, 1] * y + inv[1, 2] * z + inv[1, 3]
        bz = inv[2, 0] * x + inv[2, 1] * y + inv[2, 2] * z + inv[2, 3]

        bi = int(np.floor((bx - base_ox) / delta + 0.5))
        bj = int(np.floor((by - base_oy) / delta + 0.5))
        bk = int(np.floor((bz - base_oz) / delta + 0.5))

        if 0 <= bi < bn_x and 0 <= bj < bn_y and 0 <= bk < bn_z and base[bi, bj, bk]:
            out[i, j, k] = True