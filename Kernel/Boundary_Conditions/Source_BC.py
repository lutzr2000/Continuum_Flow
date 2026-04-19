import numpy as np
from numba import cuda

import Kernel.Boundary_Conditions.Obstacles as Obstacles

THREADS_PER_BLOCK_3D = (8, 8, 8)


def build_source_data(domain_cfg, source_entries):
    """
    build persistent source target fields from exported source nodes.

    Multiple source nodes are merged with a max operation per scalar field.
    Velocity targets are written directly so later overlapping sources override
    earlier ones component-wise.

    Args:
        domain_cfg (dict): exported domain configuration
        source_entries (list[dict]): exported source node configurations
    Returns:
        dict: source mask and persistent source target fields
    """
    nx = int(domain_cfg["grid"]["nx"])
    ny = int(domain_cfg["grid"]["ny"])
    nz = int(domain_cfg["grid"]["nz"])
    delta = float(domain_cfg["resolution"])
    origin_x = -0.5 * nx * delta
    origin_y = -0.5 * ny * delta
    origin_z = 0.0

    temperature_field = np.zeros((nx, ny, nz), dtype=np.float32)
    smoke_field = np.zeros((nx, ny, nz), dtype=np.float32)
    fuel_field = np.zeros((nx, ny, nz), dtype=np.float32)
    velocity_x_field = np.zeros((nx, ny, nz), dtype=np.float32)
    velocity_y_field = np.zeros((nx, ny, nz), dtype=np.float32)
    velocity_z_field = np.zeros((nx, ny, nz), dtype=np.float32)
    source_active_mask = np.zeros((nx, ny, nz), dtype=np.bool_)

    boundary_clip_layers = 4 # this is important, otherwise there can be conflict between Neumann BC and Sources

    for source_entry in source_entries:
        if source_entry.get("shape") != "mesh":
            continue

        mesh_cfg = source_entry.get("mesh", {})
        mesh_objects = mesh_cfg.get("objects", ())
        if not mesh_objects:
            continue

        source_mask = Obstacles.mesh(
            nx, ny, nz, delta, mesh_objects,
            origin_x=origin_x, origin_y=origin_y, origin_z=origin_z,
        )
        if boundary_clip_layers > 0:
            source_mask[:boundary_clip_layers, :, :] = False
            source_mask[-boundary_clip_layers:, :, :] = False
            source_mask[:, :boundary_clip_layers, :] = False
            source_mask[:, -boundary_clip_layers:, :] = False
            source_mask[:, :, :boundary_clip_layers] = False
            source_mask[:, :, -boundary_clip_layers:] = False

        if not np.any(source_mask):
            continue

        velocity = source_entry.get("velocity", (0.0, 0.0, 0.0))
        velocity_x = np.float32(velocity[0] if len(velocity) > 0 else 0.0)
        velocity_y = np.float32(velocity[1] if len(velocity) > 1 else 0.0)
        velocity_z = np.float32(velocity[2] if len(velocity) > 2 else 0.0)

        source_active_mask |= source_mask
        temperature_field[source_mask] = np.maximum(
            temperature_field[source_mask],
            np.float32(source_entry.get("temperature", 0.0)),
        )
        smoke_field[source_mask] = np.maximum(
            smoke_field[source_mask],
            np.float32(source_entry.get("smoke", 0.0)),
        )
        fuel_field[source_mask] = np.maximum(
            fuel_field[source_mask],
            np.float32(source_entry.get("fuel", 0.0)),
        )
        velocity_x_field[source_mask] = velocity_x
        velocity_y_field[source_mask] = velocity_y
        velocity_z_field[source_mask] = velocity_z

    return {
        "mask": source_active_mask,
        "temperature": temperature_field,
        "smoke": smoke_field,
        "fuel": fuel_field,
        "velocity_x": velocity_x_field,
        "velocity_y": velocity_y_field,
        "velocity_z": velocity_z_field,
    }


@cuda.jit
def _source_bc_kernel(
    u, v, w, T, smoke, fuel,
    source_mask,
    source_temperature, source_smoke, source_fuel,
    source_velocity_x, source_velocity_y, source_velocity_z,
):
    """
    clamps source regions to persistent source maxima on the GPU.

    Each thread checks one source cell, sets the configured source velocity,
    and raises temperature, smoke and fuel to the configured source values
    when the source mask is active there.

    Args:
        u (device array): x-velocity field
        v (device array): y-velocity field
        w (device array): z-velocity field
        T (device array): temperature field
        smoke (device array): smoke field
        fuel (device array): fuel field
        source_mask (device array): boolean source mask
        source_temperature (device array): source temperature targets
        source_smoke (device array): source smoke targets
        source_fuel (device array): source fuel targets
        source_velocity_x (device array): source x-velocity targets
        source_velocity_y (device array): source y-velocity targets
        source_velocity_z (device array): source z-velocity targets
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = source_mask.shape

    if i >= nx or j >= ny or k >= nz:
        return

    if not source_mask[i, j, k]:
        return

    source_temperature_value = source_temperature[i, j, k]
    source_smoke_value = source_smoke[i, j, k]
    source_fuel_value = source_fuel[i, j, k]
    u[i, j, k] = source_velocity_x[i, j, k]
    v[i, j, k] = source_velocity_y[i, j, k]
    w[i, j, k] = source_velocity_z[i, j, k]

    if T[i, j, k] < source_temperature_value:
        T[i, j, k] = source_temperature_value
    if smoke[i, j, k] < source_smoke_value:
        smoke[i, j, k] = source_smoke_value
    if fuel[i, j, k] < source_fuel_value:
        fuel[i, j, k] = source_fuel_value


def source_bc(
    u, v, w, T, smoke, fuel,
    source_mask,
    source_temperature, source_smoke, source_fuel,
    source_velocity_x, source_velocity_y, source_velocity_z,
    threadsperblock=None,
):
    """
    applies all source boundary conditions to the GPU field state.

    Velocity is imposed directly and temperature, smoke and fuel are clamped to
    their persistent source target values inside active source cells.

    Args:
        u (device array): x-velocity field
        v (device array): y-velocity field
        w (device array): z-velocity field
        T (device array): temperature field
        smoke (device array): smoke field
        fuel (device array): fuel field
        source_mask (device array): boolean source mask
        source_temperature (device array): source temperature targets
        source_smoke (device array): source smoke targets
        source_fuel (device array): source fuel targets
        source_velocity_x (device array): source x-velocity targets
        source_velocity_y (device array): source y-velocity targets
        source_velocity_z (device array): source z-velocity targets
        threadsperblock (tuple[int, int, int], optional): CUDA block shape for volume kernels
    Returns:
        tuple: updated velocity, temperature, smoke and fuel fields
    """
    if threadsperblock is None:
        threadsperblock = THREADS_PER_BLOCK_3D

    blockspergrid = (
        (source_mask.shape[0] + threadsperblock[0] - 1) // threadsperblock[0],
        (source_mask.shape[1] + threadsperblock[1] - 1) // threadsperblock[1],
        (source_mask.shape[2] + threadsperblock[2] - 1) // threadsperblock[2],
    )

    _source_bc_kernel[blockspergrid, threadsperblock](
        u, v, w, T, smoke, fuel,
        source_mask,
        source_temperature, source_smoke, source_fuel,
        source_velocity_x, source_velocity_y, source_velocity_z,
    )
    return u, v, w, T, smoke, fuel
