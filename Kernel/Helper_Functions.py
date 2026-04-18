import json
import sys

import numpy as np
from numba import cuda

import Kernel.Boundary_Conditions.Domain_BC as BC
import Kernel.Forcing as Forcing
import Kernel.Boundary_Conditions.Obstacle_BC as Obstacle_BC
import Kernel.Boundary_Conditions.Source_BC as Source_BC

THREADS_PER_BLOCK_3D = (8, 8, 8)
THREADS_PER_BLOCK_2D = (4, 4)
BOUNDARY_FACE_NAMES = ("x_low", "x_high", "y_low", "y_high", "z_low", "z_high")
OUTPUT_BUFFER_MULTIPLIER = 2
PROGRESS_EVENT_PREFIX = "__BLENDERCFD_PROGRESS__ "


def emit_progress(percent, time_value=None):
    """Emit a machine-readable progress event for Blender's bake progress bar."""
    payload = {
        "percent": max(0.0, min(100.0, float(percent))),
    }
    if time_value is not None:
        payload["time"] = float(time_value)
    sys.stdout.write(PROGRESS_EVENT_PREFIX + json.dumps(payload) + "\n")
    sys.stdout.flush()


def build_boundary_config(domain_cfg):
    """Build BC config and extract the first inflow velocity from boundary faces."""
    zero_velocity = (0.0, 0.0, 0.0)
    boundary_cfg = domain_cfg.get("boundary_conditions", {})
    bc_config = {}
    inflow_velocity = zero_velocity

    for face_name in BOUNDARY_FACE_NAMES:
        face_cfg = boundary_cfg.get(face_name, {})
        bc_type = face_cfg.get("type", "OUTFLOW")
        bc_config[face_name] = {
            "type": bc_type,
        }

        if bc_type == "INFLOW" and inflow_velocity == zero_velocity:
            velocity = face_cfg.get("velocity", zero_velocity)
            if len(velocity) >= 3:
                inflow_velocity = (
                    float(velocity[0]),
                    float(velocity[1]),
                    float(velocity[2]),
                )

    return bc_config, inflow_velocity


def collect_output_variables(output_cfg):
    """Translate exported output field toggles to kernel field identifiers."""
    field_mapping = {
        "u": "u",
        "v": "v",
        "w": "w",
        "p": "p",
        "t": "T",
        "smoke": "smoke",
        "fuel": "fuel",
        "flame": "flame",
    }

    enabled_fields = []
    for field_name, is_enabled in output_cfg.get("fields", {}).items():
        if is_enabled and field_name in field_mapping:
            enabled_fields.append(field_mapping[field_name])
    return enabled_fields


def build_output_performance_config(output_cfg):
    """Build output pipeline performance settings from the exported output node."""
    performance_cfg = output_cfg.get("performance", {})
    writer_processes = max(1, int(performance_cfg.get("writer_processes", 4)))

    return {
        "writer_processes": writer_processes,
        "buffer_count": writer_processes * OUTPUT_BUFFER_MULTIPLIER,
    }


def resolve_output_dtype(output_cfg):
    """Resolve the requested VDB storage precision from the exported output node."""
    precision_name = str(output_cfg.get("precision", "float16")).strip().lower()
    precision_mapping = {
        "float16": np.float16,
        "float32": np.float32,
    }
    return precision_mapping.get(precision_name, np.float16)


def apply_config(config):
    """Extract kernel settings and persistent data from the exported config."""
    simulations = config.get("simulations")
    simulation_cfg = simulations[0]
    domain_cfg = simulation_cfg.get("domain")
    physics_cfg = simulation_cfg.get("physics")
    output_cfg = simulation_cfg.get("outputs", [None])[0]
    obstacle_entries = simulation_cfg.get("obstacles", [])
    source_entries = simulation_cfg.get("sources", [])
    force_entries = simulation_cfg.get("forces", [])
    host_vdb_writer = config.get("_host_vdb_writer")

    bc_config, inflow_velocity = build_boundary_config(domain_cfg)
    obstacle_data = Obstacle_BC.build_obstacle_data(domain_cfg, obstacle_entries)
    source_data = Source_BC.build_source_data(domain_cfg, source_entries)
    force_data = Forcing.build_force_field_data(domain_cfg, force_entries, dtype=np.float32)
    source_temperature_max = float(np.max(source_data["temperature"]))
    output_performance = build_output_performance_config(output_cfg)
    output_dtype = resolve_output_dtype(output_cfg)

    BC.THREADS_PER_BLOCK_3D = THREADS_PER_BLOCK_3D
    BC.THREADS_PER_BLOCK_2D = THREADS_PER_BLOCK_2D
    Obstacle_BC.THREADS_PER_BLOCK_3D = THREADS_PER_BLOCK_3D
    Source_BC.THREADS_PER_BLOCK_3D = THREADS_PER_BLOCK_3D

    return {
        "RHO": float(physics_cfg["fluid"]["density"]),
        "NU": float(physics_cfg["fluid"]["viscosity"]),
        "NU_TEMPERATURE": float(physics_cfg["temperature"]["diffusion"]),
        "NU_SMOKE": float(physics_cfg["smoke"]["diffusion"]),
        "NU_FUEL": float(physics_cfg["fuel"]["diffusion"]),
        "TEMPERATURE_DISSIPATION_RATE": float(physics_cfg["temperature"]["dissipation"]),
        "TEMPERATURE_PRODUCTION_RATE": 1.0,
        "SMOKE_DISSIPATION_RATE": float(physics_cfg["smoke"]["dissipation"]),
        "SMOKE_PRODUCTION_RATE": 1.0,
        "FUEL_BURN_RATE": float(physics_cfg["fuel"]["burn_rate"]),
        "FUEL_IGNITION_TEMPERATURE": float(physics_cfg["fuel"]["ignition_temperature"]),
        "T_REFERENCE": float(physics_cfg["temperature"]["reference_temperature"]),
        "SOURCE_TEMPERATURE_MAX": source_temperature_max,
        "BUOANCY_FACTOR": float(physics_cfg["temperature"]["buoyancy"]),
        "EXPANSION_RATE": float(physics_cfg["temperature"]["expansion_rate"]),
        "VORTICITY": float(physics_cfg.get("extras", {}).get("vorticity", 0.0)),
        "T_MAX": float(simulation_cfg["settings"]["simulation_length"]),
        "CFL_MAX": float(simulation_cfg["settings"]["cfl"]),
        "MAX_ITER": int(simulation_cfg["settings"]["iterations"]),
        "PRECISION": np.float32,
        "CPU_COUNT": 1,
        "DELTA": float(domain_cfg["resolution"]),
        "NX": int(domain_cfg["grid"]["nx"]),
        "NY": int(domain_cfg["grid"]["ny"]),
        "NZ": int(domain_cfg["grid"]["nz"]),
        "OUTPUT_FPS": int(output_cfg["fps"]),
        "PRINT_FREQUENCY": 100,
        "OUTPUT_TIME_STEP": 1.0 / int(output_cfg["fps"]),
        "OUTPUT_STATUS": False,
        "OUTPUT_FORWARDER_COUNT": output_performance["writer_processes"],
        "WRITE_QUEUE_SIZE": output_performance["buffer_count"],
        "OUTPATH": output_cfg.get("output_path", ""),
        "OUTPUT_DTYPE": output_dtype,
        "OUTPUT_VARIABLES": collect_output_variables(output_cfg),
        "HOST_VDB_WRITER": host_vdb_writer,
        "BC_CONFIG": bc_config,
        "U_INFLOW": inflow_velocity[0],
        "V_INFLOW": inflow_velocity[1],
        "W_INFLOW": inflow_velocity[2],
        "obstacle_mask": obstacle_data["mask"],
        "source_field_data": source_data,
        "force_field_data": force_data,
    }


def upload_simulation_state_to_gpu(simulation_params):
    """
    Allocate the simulation fields on the host and upload persistent arrays to the GPU.
    """
    precision_dtype = np.dtype(simulation_params["PRECISION"])
    nx = simulation_params["NX"]
    ny = simulation_params["NY"]
    nz = simulation_params["NZ"]

    u = np.full((nx, ny, nz), simulation_params["U_INFLOW"], dtype=precision_dtype)
    v = np.full((nx, ny, nz), simulation_params["V_INFLOW"], dtype=precision_dtype)
    w = np.full((nx, ny, nz), simulation_params["W_INFLOW"], dtype=precision_dtype)

    p = np.zeros((nx, ny, nz), dtype=precision_dtype)
    T = np.full((nx, ny, nz), simulation_params["T_REFERENCE"], dtype=precision_dtype)
    smoke = np.zeros((nx, ny, nz), dtype=precision_dtype)
    fuel = np.zeros((nx, ny, nz), dtype=precision_dtype)
    flame = np.zeros((nx, ny, nz), dtype=precision_dtype)

    force_field_data = simulation_params["force_field_data"]
    Fx_base = np.asarray(force_field_data["Fx_base"], dtype=precision_dtype)
    Fy_base = np.asarray(force_field_data["Fy_base"], dtype=precision_dtype)
    Fz_base = np.asarray(force_field_data["Fz_base"], dtype=precision_dtype)
    Fx = np.asarray(force_field_data["Fx"], dtype=precision_dtype)
    Fy = np.asarray(force_field_data["Fy"], dtype=precision_dtype)
    Fz = np.asarray(force_field_data["Fz"], dtype=precision_dtype)
    point_divergence = np.asarray(force_field_data["point_divergence"], dtype=precision_dtype)
    turbulence_data = force_field_data["turbulence"]

    obstacle_mask_host = np.asarray(simulation_params["obstacle_mask"])
    source_field_data = simulation_params["source_field_data"]
    source_mask_host = np.asarray(source_field_data["mask"])
    source_temperature_host = np.asarray(source_field_data["temperature"], dtype=precision_dtype)
    source_smoke_host = np.asarray(source_field_data["smoke"], dtype=precision_dtype)
    source_fuel_host = np.asarray(source_field_data["fuel"], dtype=precision_dtype)

    T = np.maximum(T, source_temperature_host)
    smoke = np.maximum(smoke, source_smoke_host)
    fuel = np.maximum(fuel, source_fuel_host)

    device_state = {
        "u": cuda.to_device(u),
        "v": cuda.to_device(v),
        "w": cuda.to_device(w),
        "u_work": cuda.device_array((nx, ny, nz), dtype=precision_dtype),
        "v_work": cuda.device_array((nx, ny, nz), dtype=precision_dtype),
        "w_work": cuda.device_array((nx, ny, nz), dtype=precision_dtype),
        "p": cuda.to_device(p),
        "pressure_work": cuda.device_array((nx, ny, nz), dtype=precision_dtype),
        "pressure_rhs": cuda.device_array((nx, ny, nz), dtype=precision_dtype),
        "T": cuda.to_device(T),
        "temperature_work": cuda.device_array((nx, ny, nz), dtype=precision_dtype),
        "smoke": cuda.to_device(smoke),
        "smoke_work": cuda.device_array((nx, ny, nz), dtype=precision_dtype),
        "fuel": cuda.to_device(fuel),
        "fuel_work": cuda.device_array((nx, ny, nz), dtype=precision_dtype),
        "flame": cuda.to_device(flame),
        "flame_work": cuda.device_array((nx, ny, nz), dtype=precision_dtype),
        "vorticity_x": cuda.device_array((nx, ny, nz), dtype=precision_dtype),
        "vorticity_y": cuda.device_array((nx, ny, nz), dtype=precision_dtype),
        "vorticity_z": cuda.device_array((nx, ny, nz), dtype=precision_dtype),
        "vorticity_magnitude": cuda.device_array((nx, ny, nz), dtype=precision_dtype),
        "Fx": cuda.to_device(Fx),
        "Fy": cuda.to_device(Fy),
        "Fz": cuda.to_device(Fz),
        "point_divergence": cuda.to_device(point_divergence),
        "Fx_base": cuda.to_device(Fx_base),
        "Fy_base": cuda.to_device(Fy_base),
        "Fz_base": cuda.to_device(Fz_base),
        "turbulence_Fx_a": cuda.to_device(np.asarray(turbulence_data["Fx_a"], dtype=precision_dtype)),
        "turbulence_Fy_a": cuda.to_device(np.asarray(turbulence_data["Fy_a"], dtype=precision_dtype)),
        "turbulence_Fz_a": cuda.to_device(np.asarray(turbulence_data["Fz_a"], dtype=precision_dtype)),
        "turbulence_Fx_b": cuda.to_device(np.asarray(turbulence_data["Fx_b"], dtype=precision_dtype)),
        "turbulence_Fy_b": cuda.to_device(np.asarray(turbulence_data["Fy_b"], dtype=precision_dtype)),
        "turbulence_Fz_b": cuda.to_device(np.asarray(turbulence_data["Fz_b"], dtype=precision_dtype)),
        "turbulence_cos_coeffs": cuda.to_device(np.ones(len(turbulence_data["angular_frequencies"]), dtype=precision_dtype)),
        "turbulence_sin_coeffs": cuda.to_device(np.zeros(len(turbulence_data["angular_frequencies"]), dtype=precision_dtype)),
        "obstacle_mask": cuda.to_device(obstacle_mask_host),
        "source_mask": cuda.to_device(source_mask_host),
        "source_temperature": cuda.to_device(source_temperature_host),
        "source_smoke": cuda.to_device(source_smoke_host),
        "source_fuel": cuda.to_device(source_fuel_host),
    }

    gpu_constants = {
        "RHO": precision_dtype.type(simulation_params["RHO"]),
        "NU": precision_dtype.type(simulation_params["NU"]),
        "NU_TEMPERATURE": precision_dtype.type(simulation_params["NU_TEMPERATURE"]),
        "NU_SMOKE": precision_dtype.type(simulation_params["NU_SMOKE"]),
        "NU_FUEL": precision_dtype.type(simulation_params["NU_FUEL"]),
        "TEMPERATURE_DISSIPATION_RATE": precision_dtype.type(simulation_params["TEMPERATURE_DISSIPATION_RATE"]),
        "TEMPERATURE_PRODUCTION_RATE": precision_dtype.type(simulation_params["TEMPERATURE_PRODUCTION_RATE"]),
        "SMOKE_DISSIPATION_RATE": precision_dtype.type(simulation_params["SMOKE_DISSIPATION_RATE"]),
        "SMOKE_PRODUCTION_RATE": precision_dtype.type(simulation_params["SMOKE_PRODUCTION_RATE"]),
        "FUEL_BURN_RATE": precision_dtype.type(simulation_params["FUEL_BURN_RATE"]),
        "FUEL_IGNITION_TEMPERATURE": precision_dtype.type(simulation_params["FUEL_IGNITION_TEMPERATURE"]),
        "T_REFERENCE": precision_dtype.type(simulation_params["T_REFERENCE"]),
        "SOURCE_TEMPERATURE_MAX": precision_dtype.type(simulation_params["SOURCE_TEMPERATURE_MAX"]),
        "BUOANCY_FACTOR": precision_dtype.type(simulation_params["BUOANCY_FACTOR"]),
        "EXPANSION_RATE": precision_dtype.type(simulation_params["EXPANSION_RATE"]),
        "VORTICITY": precision_dtype.type(simulation_params["VORTICITY"]),
        "DELTA": precision_dtype.type(simulation_params["DELTA"]),
        "U_INFLOW": precision_dtype.type(simulation_params["U_INFLOW"]),
        "V_INFLOW": precision_dtype.type(simulation_params["V_INFLOW"]),
        "W_INFLOW": precision_dtype.type(simulation_params["W_INFLOW"]),
        "NX": simulation_params["NX"],
        "NY": simulation_params["NY"],
        "NZ": simulation_params["NZ"],
        "FORCE_X_MAX": precision_dtype.type(force_field_data["max_abs"][0]),
        "FORCE_Y_MAX": precision_dtype.type(force_field_data["max_abs"][1]),
        "FORCE_Z_MAX": precision_dtype.type(force_field_data["max_abs"][2]),
    }

    return device_state, gpu_constants


def select_fields(field_map, field_names):
    """Build a dictionary view containing only the requested field names."""
    return {name: field_map[name] for name in field_names}
