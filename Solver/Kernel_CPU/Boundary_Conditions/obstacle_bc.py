import numpy as np
from numba import njit, prange

import Solver.Kernel_CPU.Boundary_Conditions.obstacles as obstacles


def _combine_exported_obstacles(obstacle_entries):
    """Merge exported obstacle nodes into one kernel obstacle configuration."""
    mesh_objects = []
    for obstacle_entry in obstacle_entries:
        if obstacle_entry.get("shape") != "mesh":
            continue
        mesh_cfg = obstacle_entry.get("mesh", {})
        mesh_objects.extend(mesh_cfg.get("objects", ()))

    if mesh_objects:
        return {
            "shape": "mesh",
            "solid": True,
            "mesh": {
                "objects": mesh_objects,
            },
        }

    return {
        "shape": "empty",
        "solid": False,
        "mesh": {
            "objects": [],
        },
    }


def build_obstacle_data(domain_cfg, obstacle_entries):
    """
    Build the voxel obstacle mask from exported obstacle nodes for the CPU solver.

    The heavy mesh voxelization and dynamic obstacle sampling already exist as
    host-side helpers, so the CPU kernel reuses them directly instead of
    duplicating that geometry pipeline.
    """
    obstacle_cfg = _combine_exported_obstacles(obstacle_entries)
    nx = int(domain_cfg["grid"]["nx"])
    ny = int(domain_cfg["grid"]["ny"])
    nz = int(domain_cfg["grid"]["nz"])
    delta = float(domain_cfg["resolution"])
    origin_x = -0.5 * nx * delta
    origin_y = -0.5 * ny * delta
    origin_z = 0.0

    zero_velocity_x = np.zeros((nx, ny, nz), dtype=np.float32)
    zero_velocity_y = np.zeros((nx, ny, nz), dtype=np.float32)
    zero_velocity_z = np.zeros((nx, ny, nz), dtype=np.float32)

    if obstacle_cfg["shape"] == "mesh":
        mesh_cfg = obstacle_cfg.get("mesh", {})
        mesh_objects = mesh_cfg.get("objects", mesh_cfg if isinstance(mesh_cfg, list) else [])
        obstacle_runtime = obstacles.build_dynamic_runtime(
            nx,
            ny,
            nz,
            delta,
            mesh_objects,
            origin_x=origin_x,
            origin_y=origin_y,
            origin_z=origin_z,
        )
        obstacle_mask, obstacle_velocity_x, obstacle_velocity_y, obstacle_velocity_z = obstacles.update_dynamic_obstacle_data(
            obstacle_runtime,
            0.0,
        )
        return {
            "config": obstacle_cfg,
            "mask": obstacle_mask,
            "velocity_x": obstacle_velocity_x,
            "velocity_y": obstacle_velocity_y,
            "velocity_z": obstacle_velocity_z,
            "runtime": obstacle_runtime,
            "is_animated": bool(obstacle_runtime.get("is_animated", False)),
        }

    if obstacle_cfg["shape"] == "empty":
        return {
            "config": obstacle_cfg,
            "mask": np.zeros((nx, ny, nz), dtype=np.bool_),
            "velocity_x": zero_velocity_x,
            "velocity_y": zero_velocity_y,
            "velocity_z": zero_velocity_z,
            "runtime": None,
            "is_animated": False,
        }

    raise ValueError(f"Unsupported obstacle shape '{obstacle_cfg['shape']}'")


def update_obstacle_mask(obstacle_data, time_value):
    """Update the combined obstacle mask and wall velocities for the current simulation time."""
    runtime = obstacle_data.get("runtime")
    if runtime is None:
        return obstacle_data["mask"]

    updated_mask, updated_velocity_x, updated_velocity_y, updated_velocity_z = obstacles.update_dynamic_obstacle_data(
        runtime,
        time_value,
        out_mask=obstacle_data["mask"],
        out_velocity_x=obstacle_data["velocity_x"],
        out_velocity_y=obstacle_data["velocity_y"],
        out_velocity_z=obstacle_data["velocity_z"],
    )
    obstacle_data["mask"] = updated_mask
    obstacle_data["velocity_x"] = updated_velocity_x
    obstacle_data["velocity_y"] = updated_velocity_y
    obstacle_data["velocity_z"] = updated_velocity_z
    return updated_mask


@njit(cache=True, parallel=True)
def _obstacle_bc_kernel_cpu(
    u, v, w, smoke, fuel, flame,
    mask, obstacle_velocity_x, obstacle_velocity_y, obstacle_velocity_z,
):
    """
    Apply the obstacle overwrite rules over the whole volume on the CPU.

    Each worker scans a disjoint chunk of the flattened arrays and overwrites
    obstacle cells with obstacle wall velocities while clearing supported
    scalar fields.
    """
    total_size = mask.size
    u_flat = u.reshape(total_size)
    v_flat = v.reshape(total_size)
    w_flat = w.reshape(total_size)
    smoke_flat = smoke.reshape(total_size)
    fuel_flat = fuel.reshape(total_size)
    flame_flat = flame.reshape(total_size)
    mask_flat = mask.reshape(total_size)
    velocity_x_flat = obstacle_velocity_x.reshape(total_size)
    velocity_y_flat = obstacle_velocity_y.reshape(total_size)
    velocity_z_flat = obstacle_velocity_z.reshape(total_size)

    for idx in prange(total_size):
        if not mask_flat[idx]:
            continue

        u_flat[idx] = velocity_x_flat[idx]
        v_flat[idx] = velocity_y_flat[idx]
        w_flat[idx] = velocity_z_flat[idx]
        smoke_flat[idx] = 0.0
        fuel_flat[idx] = 0.0
        flame_flat[idx] = 0.0


def obstacle_bc(
    u, v, w, smoke, fuel, flame,
    obstacle_mask, obstacle_velocity_x, obstacle_velocity_y, obstacle_velocity_z,
):
    """
    Apply all obstacle boundary conditions to the CPU field state.

    Velocity and supported scalar fields are overwritten inside obstacle cells
    so obstacle regions stay empty and act as solid regions.
    """
    if obstacle_mask.size == 0 or not np.any(obstacle_mask):
        return u, v, w, smoke, fuel, flame

    _obstacle_bc_kernel_cpu(
        u, v, w, smoke, fuel, flame,
        obstacle_mask, obstacle_velocity_x, obstacle_velocity_y, obstacle_velocity_z,
    )

    return u, v, w, smoke, fuel, flame
