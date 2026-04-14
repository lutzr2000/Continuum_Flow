import numpy as np
from numba import cuda

import Obstacles as Obstacles

THREADS_PER_BLOCK_3D = (8, 8, 8)


def build_source_data(domain_cfg, source_entries):
    """
    build persistent source target fields from exported source nodes.

    Multiple source nodes are merged with a max operation per scalar field.

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
    source_active_mask = np.zeros((nx, ny, nz), dtype=np.bool_)

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
        if not np.any(source_mask):
            continue

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

    return {
        "mask": source_active_mask,
        "temperature": temperature_field,
        "smoke": smoke_field,
        "fuel": fuel_field,
    }


@cuda.jit
def _source_bc_kernel(T, smoke, fuel, source_mask, source_temperature, source_smoke, source_fuel):
    """
    clamps source regions to persistent source maxima on the GPU.

    Each thread checks one source cell and raises temperature, smoke and fuel
    to the configured source values when the source mask is active there.

    Args:
        T (device array): temperature field
        smoke (device array): smoke field
        fuel (device array): fuel field
        source_mask (device array): boolean source mask
        source_temperature (device array): source temperature targets
        source_smoke (device array): source smoke targets
        source_fuel (device array): source fuel targets
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

    if T[i, j, k] < source_temperature_value:
        T[i, j, k] = source_temperature_value
    if smoke[i, j, k] < source_smoke_value:
        smoke[i, j, k] = source_smoke_value
    if fuel[i, j, k] < source_fuel_value:
        fuel[i, j, k] = source_fuel_value


def source_bc(T, smoke, fuel, source_mask, source_temperature, source_smoke, source_fuel, threadsperblock=None):
    """
    applies all source boundary conditions to the GPU field state.

    Temperature, smoke and fuel are clamped to their persistent source target
    values inside active source cells.

    Args:
        T (device array): temperature field
        smoke (device array): smoke field
        fuel (device array): fuel field
        source_mask (device array): boolean source mask
        source_temperature (device array): source temperature targets
        source_smoke (device array): source smoke targets
        source_fuel (device array): source fuel targets
        threadsperblock (tuple[int, int, int], optional): CUDA block shape for volume kernels
    Returns:
        tuple: updated temperature, smoke and fuel fields
    """
    if threadsperblock is None:
        threadsperblock = THREADS_PER_BLOCK_3D

    blockspergrid = (
        (source_mask.shape[0] + threadsperblock[0] - 1) // threadsperblock[0],
        (source_mask.shape[1] + threadsperblock[1] - 1) // threadsperblock[1],
        (source_mask.shape[2] + threadsperblock[2] - 1) // threadsperblock[2],
    )

    _source_bc_kernel[blockspergrid, threadsperblock](
        T, smoke, fuel, source_mask, source_temperature, source_smoke, source_fuel
    )
    return T, smoke, fuel
