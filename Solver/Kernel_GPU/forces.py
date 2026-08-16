from numba import cuda
from typing import Any
import Solver.Kernel_GPU.sparse_managment as sparse_managment
import Solver.Kernel_GPU.noise as noise

@cuda.jit(device=True, inline=True, cache=True)
def buoyancy_approximation(
    T: Any,
    tile_map: Any,
    i: int,
    j: int,
    k: int,
    buoyancy_factor: float,
    t_reference: float,
) -> float:
    r"""
    Compute the buoyancy acceleration contribution from temperature deviation.
    Assumes that g is aligned with the global z-axis.

    The implemented relation is

    .. math::

        b = g \, \beta \, (T - T_{\mathrm{ref}}).

    Parameters
    ----------
    T
        temperature field.
    tile_map
        Mapping used to access tiles in ``T``.
    i, j, k
        Cell indices.
    buoyancy_factor
        Thermal buoyancy coefficient.
    t_reference
        Reference temperature.

    Returns
    -------
    float
        Buoyancy acceleration contribution for the cell ``(i, j, k)``.
    """
    g = 9.81

    temperature = sparse_managment.get_pool_value(
        T, tile_map, i, j, k, t_reference
    )

    return g * buoyancy_factor * (temperature - t_reference)


@cuda.jit(device=True, inline=True, cache=True)
def apply_swirl_forces(
    swirl_config,
    i,
    j,
    k,
    delta,
    origin_x,
    origin_y,
    origin_z,
):
    Fx = 0.0
    Fy = 0.0
    Fz = 0.0

    px = origin_x + float(i) * delta
    py = origin_y + float(j) * delta
    pz = origin_z + float(k) * delta

    for swirl_idx in range(swirl_config.shape[0]):
        strength = swirl_config[swirl_idx, 0]

        ox = swirl_config[swirl_idx, 1]
        oy = swirl_config[swirl_idx, 2]
        oz = swirl_config[swirl_idx, 3]

        ax = swirl_config[swirl_idx, 4]
        ay = swirl_config[swirl_idx, 5]
        az = swirl_config[swirl_idx, 6]

        radius = swirl_config[swirl_idx, 7]

        if radius <= 0.0 or strength == 0.0:
            continue

        axis_len = (ax * ax + ay * ay + az * az) ** 0.5
        if axis_len <= 1e-8:
            continue

        ax /= axis_len
        ay /= axis_len
        az /= axis_len

        rx = px - ox
        ry = py - oy
        rz = pz - oz

        projection = rx * ax + ry * ay + rz * az

        closest_x = ox + projection * ax
        closest_y = oy + projection * ay
        closest_z = oz + projection * az

        radial_x = px - closest_x
        radial_y = py - closest_y
        radial_z = pz - closest_z

        dist_sq = radial_x * radial_x + radial_y * radial_y + radial_z * radial_z
        radius_sq = radius * radius

        if dist_sq > radius_sq or dist_sq <= 1e-12:
            continue

        tx = ay * radial_z - az * radial_y
        ty = az * radial_x - ax * radial_z
        tz = ax * radial_y - ay * radial_x

        t_len = (tx * tx + ty * ty + tz * tz) ** 0.5
        if t_len <= 1e-8:
            continue

        tx /= t_len
        ty /= t_len
        tz /= t_len

        dist = dist_sq**0.5
        falloff = 1.0 - dist / radius

        Fx += strength * falloff * tx
        Fy += strength * falloff * ty
        Fz += strength * falloff * tz

    return Fx, Fy, Fz


@cuda.jit(device=True, inline=True, cache=True)
def apply_turbulence_forces(
    turbulence_config,
    i,
    j,
    k,
    delta,
    origin_x,
    origin_y,
    origin_z,
    t,
):
    Fx = 0.0
    Fy = 0.0
    Fz = 0.0

    px = origin_x + float(i) * delta
    py = origin_y + float(j) * delta
    pz = origin_z + float(k) * delta

    for turb_idx in range(turbulence_config.shape[0]):
        amplitude = turbulence_config[turb_idx, 0]
        scale = turbulence_config[turb_idx, 1]
        frequency = turbulence_config[turb_idx, 2]
        seed = int(turbulence_config[turb_idx, 3])

        if amplitude == 0.0 or scale <= 1e-8:
            continue

        inv_scale = 1.0 / scale
        time_offset = t * frequency

        x = px * inv_scale
        y = py * inv_scale
        z = pz * inv_scale + time_offset

        noise_values = noise._value_noise_3d(x, y, z, seed)

        Fx += amplitude * noise_values
        Fy += amplitude * noise_values
        Fz += amplitude * noise_values

    return Fx, Fy, Fz
