import Solver.General.voxelise_mesh as voxelise_mesh_module
import Solver.Kernel_GPU.kernel as gpu_kernel_module


def _collect_mesh_objects(entries):
    mesh_objects = []
    for entry in entries or ():
        mesh_cfg = entry.get("mesh", {})
        mesh_objects.extend(mesh_cfg.get("objects", ()))
    return mesh_objects


def main(config=None):

    simulations = config.get("simulations") or []

    simulation_cfg = simulations[0]
    domain_cfg = simulation_cfg.get("domain") or {}
    grid_cfg = domain_cfg.get("grid") or {}

    nx = int(grid_cfg.get("nx", 0))
    ny = int(grid_cfg.get("ny", 0))
    nz = int(grid_cfg.get("nz", 0))
    delta = float(domain_cfg.get("resolution", 0.0))
    origin_x = -0.5 * nx * delta
    origin_y = -0.5 * ny * delta
    origin_z = 0.0

    obstacle_mesh_objects = _collect_mesh_objects(simulation_cfg.get("obstacles"))
    source_mesh_objects = _collect_mesh_objects(simulation_cfg.get("sources"))

    obstacle_mask = voxelise_mesh_module.voxelise_mesh_all(
        nx,
        ny,
        nz,
        delta,
        obstacle_mesh_objects,
        origin_x=origin_x,
        origin_y=origin_y,
        origin_z=origin_z,
    )
    source_mask = voxelise_mesh_module.voxelise_mesh_all(
        nx,
        ny,
        nz,
        delta,
        source_mesh_objects,
        origin_x=origin_x,
        origin_y=origin_y,
        origin_z=origin_z,
    )

    return gpu_kernel_module.solver(config, obstacle_mask, source_mask)
