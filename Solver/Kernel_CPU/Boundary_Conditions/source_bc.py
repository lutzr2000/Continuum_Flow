from typing import Any

from numba import njit, prange

import Solver.Kernel_CPU.kernel_config as kernel_config
import Solver.Kernel_CPU.sparse_managment as sparse_managment
import Solver.Kernel_CPU.noise as noise


@njit(cache=True, parallel=True)
def source_bc(
    u: Any,
    v: Any,
    w: Any,
    T: Any,
    smoke: Any,
    fuel: Any,
    tile_map: Any,
    source_mask: Any,
    temperature_value: Any,
    smoke_value: Any,
    fuel_value: Any,
    velocity_x_value: Any,
    velocity_y_value: Any,
    velocity_z_value: Any,
    noise_scale: float,
    noise_amplitude: Any,
    noise_seed: Any,
    dt: float,
) -> None:
    """
    Apply source values with procedural spatial noise.

    Source velocity is added directly. Temperature is assigned, while smoke
    and fuel are integrated over ``dt`` and clamped to their valid ranges. A
    seeded spatial noise sample can modulate all injected scalar values.
    """

    total_tiles = tile_map.shape[0] * tile_map.shape[1] * tile_map.shape[2]

    for tile_flat in prange(total_tiles):
        for local_i in range(kernel_config.TILE_SIZE):
            for local_j in range(kernel_config.TILE_SIZE):
                for local_k in range(kernel_config.TILE_SIZE):
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
                    ) = sparse_managment.tile_to_index(
                        tile_flat,
                        local_i,
                        local_j,
                        local_k,
                        tile_map.shape[0],
                        tile_map.shape[1],
                        tile_map.shape[2],
                    )

                    tile_index = tile_map[tile_i, tile_j, tile_k]

                    if tile_index == -1:
                        continue

                    if not source_mask[tile_index, local_i, local_j, local_k]:
                        continue

                    u[tile_index, local_i, local_j, local_k] += velocity_x_value
                    v[tile_index, local_i, local_j, local_k] += velocity_y_value
                    w[tile_index, local_i, local_j, local_k] += velocity_z_value

                    scalar_multiplier = 1.0
                    if noise_amplitude != 0.0:
                        scale = max(noise_scale, 1.0e-6)
                        noise_value = noise._value_noise_3d(
                            i / scale,
                            j / scale,
                            k / scale,
                            noise_seed,
                        )
                        scalar_multiplier = max(
                            1.0 + noise_value * noise_amplitude,
                            0.0,
                        )

                    T[tile_index, local_i, local_j, local_k] = max(
                        temperature_value * scalar_multiplier,
                        0.0,
                    )

                    smoke[tile_index, local_i, local_j, local_k] = min(
                        max(
                            smoke[tile_index, local_i, local_j, local_k]
                            + smoke_value * scalar_multiplier * dt,
                            0.0,
                        ),
                        100.0,
                    )

                    fuel[tile_index, local_i, local_j, local_k] = min(
                        max(
                            fuel[tile_index, local_i, local_j, local_k]
                            + fuel_value * scalar_multiplier * dt,
                            0.0,
                        ),
                        100.0,
                    )
