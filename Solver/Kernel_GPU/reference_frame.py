"""GPU velocity transfer from an animated simulation reference frame."""

from typing import Any

from numba import cuda

import Solver.Kernel_GPU.sparse_managment as sparse_managment


@cuda.jit(cache=True)
def transfer_velocity(
    u: Any,
    v: Any,
    w: Any,
    tile_map: Any,
    a00: float,
    a01: float,
    a02: float,
    a03: float,
    a10: float,
    a11: float,
    a12: float,
    a13: float,
    a20: float,
    a21: float,
    a22: float,
    a23: float,
    origin_x: float,
    origin_y: float,
    origin_z: float,
    delta: float,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    """Add the change in affine frame velocity to active fluid cells."""
    (
        tile_i,
        tile_j,
        tile_k,
        local_i,
        local_j,
        local_k,
        i,
        j,
        k,
    ) = sparse_managment.tile_to_index()

    if i < 1 or j < 1 or k < 1 or i >= nx - 1 or j >= ny - 1 or k >= nz - 1:
        return

    tile_index = tile_map[tile_i, tile_j, tile_k]
    if tile_index == -1:
        return

    x = origin_x + float(i) * delta
    y = origin_y + float(j) * delta
    z = origin_z + float(k) * delta

    u[tile_index, local_i, local_j, local_k] += a00 * x + a01 * y + a02 * z + a03
    v[tile_index, local_i, local_j, local_k] += a10 * x + a11 * y + a12 * z + a13
    w[tile_index, local_i, local_j, local_k] += a20 * x + a21 * y + a22 * z + a23
