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
SPARSE_MASK_FIELDS = ("smoke", "flame")


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
    """Build per-face BC config from the exported domain settings."""
    zero_velocity = (0.0, 0.0, 0.0)
    boundary_cfg = domain_cfg.get("boundary_conditions", {})
    bc_config = {}

    for face_name in BOUNDARY_FACE_NAMES:
        face_cfg = boundary_cfg.get(face_name, {})
        bc_type = face_cfg.get("type", "OUTFLOW")
        face_bc = {
            "type": bc_type,
        }

        velocity = face_cfg.get("velocity", zero_velocity)
        if len(velocity) >= 3:
            face_bc["u"] = float(velocity[0])
            face_bc["v"] = float(velocity[1])
            face_bc["w"] = float(velocity[2])

        if "temperature" in face_cfg:
            face_bc["temperature"] = float(face_cfg["temperature"])
        if "T" in face_cfg:
            face_bc["T"] = float(face_cfg["T"])

        bc_config[face_name] = face_bc

    return bc_config


def initial_velocity_from_inflows(bc_config):
    """
    Build one uniform initial velocity vector from all configured inflow faces.

    A uniform start field avoids the old arbitrary "first inflow wins" behavior
    while still preventing a zero initialization when inflows are present.
    Opposing or multiple inflows are combined by averaging their prescribed
    velocity vectors.
    """
    inflow_count = 0
    initial_u = 0.0
    initial_v = 0.0
    initial_w = 0.0

    for bc in bc_config.values():
        if bc.get("type") != "INFLOW":
            continue
        inflow_count += 1
        initial_u += float(bc.get("u", 0.0))
        initial_v += float(bc.get("v", 0.0))
        initial_w += float(bc.get("w", 0.0))

    if inflow_count == 0:
        return 0.0, 0.0, 0.0

    inv_count = 1.0 / float(inflow_count)
    return (
        initial_u * inv_count,
        initial_v * inv_count,
        initial_w * inv_count,
    )


def _field_option_value(fields_cfg, key, legacy_key=None):
    """Read one output-field option from new nested config or legacy bool config."""
    entry = fields_cfg.get(key)
    if isinstance(entry, dict):
        return {
            "enabled": bool(entry.get("enabled", False)),
            "sparse": bool(entry.get("sparse", False)),
        }

    if legacy_key is None:
        legacy_key = key
    return {
        "enabled": bool(fields_cfg.get(legacy_key, False)),
        "sparse": False,
    }


def build_output_field_config(output_cfg):
    """Translate output settings to concrete kernel field export rules."""
    fields_cfg = output_cfg.get("fields", {})
    velocity_cfg = _field_option_value(fields_cfg, "velocity")
    if "velocity" not in fields_cfg:
        velocity_cfg = {
            "enabled": bool(fields_cfg.get("u", False) or fields_cfg.get("v", False) or fields_cfg.get("w", False)),
            "sparse": False,
        }
    pressure_cfg = _field_option_value(fields_cfg, "pressure", "p")
    temperature_cfg = _field_option_value(fields_cfg, "temperature", "t")
    smoke_cfg = _field_option_value(fields_cfg, "smoke")
    fuel_cfg = _field_option_value(fields_cfg, "fuel")
    flame_cfg = _field_option_value(fields_cfg, "flame")

    return {
        "u": {"export": velocity_cfg["enabled"], "sparse": velocity_cfg["sparse"]},
        "v": {"export": velocity_cfg["enabled"], "sparse": velocity_cfg["sparse"]},
        "w": {"export": velocity_cfg["enabled"], "sparse": velocity_cfg["sparse"]},
        "p": {"export": pressure_cfg["enabled"], "sparse": pressure_cfg["sparse"]},
        "T": {"export": temperature_cfg["enabled"], "sparse": temperature_cfg["sparse"]},
        "smoke": {"export": smoke_cfg["enabled"], "sparse": smoke_cfg["sparse"]},
        "fuel": {"export": fuel_cfg["enabled"], "sparse": fuel_cfg["sparse"]},
        "flame": {"export": flame_cfg["enabled"], "sparse": flame_cfg["sparse"]},
    }


def collect_output_variables(output_field_config):
    """Return the concrete field names that should be written to the VDB."""
    return [field_name for field_name, cfg in output_field_config.items() if cfg.get("export")]


def collect_buffer_variables(output_field_config):
    """Return all fields that must be copied to host buffers for one output frame."""
    buffer_fields = list(collect_output_variables(output_field_config))
    if any(cfg.get("export") and cfg.get("sparse") for cfg in output_field_config.values()):
        for mask_field in SPARSE_MASK_FIELDS:
            if mask_field not in buffer_fields:
                buffer_fields.append(mask_field)
    return buffer_fields


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

    bc_config = build_boundary_config(domain_cfg)
    initial_velocity = initial_velocity_from_inflows(bc_config)
    obstacle_data = Obstacle_BC.build_obstacle_data(domain_cfg, obstacle_entries)
    source_data = Source_BC.build_source_data(domain_cfg, source_entries)
    force_data = Forcing.build_force_field_data(domain_cfg, force_entries, dtype=np.float32)
    source_temperature_max = float(np.max(source_data["temperature"]))
    has_source = bool(np.any(source_data["mask"]))
    has_obstacle = bool(np.any(obstacle_data["mask"]))
    has_force = bool(
        np.any(force_data["Fx_base"]) or
        np.any(force_data["Fy_base"]) or
        np.any(force_data["Fz_base"]) or
        np.any(force_data["point_divergence"]) or
        force_data["turbulence"]["angular_frequencies"].size > 0
    )
    output_performance = build_output_performance_config(output_cfg)
    output_dtype = resolve_output_dtype(output_cfg)
    output_field_config = build_output_field_config(output_cfg)

    BC.THREADS_PER_BLOCK_3D = THREADS_PER_BLOCK_3D
    BC.THREADS_PER_BLOCK_2D = THREADS_PER_BLOCK_2D
    Obstacle_BC.THREADS_PER_BLOCK_3D = THREADS_PER_BLOCK_3D
    Source_BC.THREADS_PER_BLOCK_3D = THREADS_PER_BLOCK_3D

    return {
        "RHO": float(physics_cfg["fluid"]["density"]),
        "NU": float(physics_cfg["fluid"]["viscosity"]),
        "TEMPERATURE_DISSIPATION_RATE": float(physics_cfg["temperature"]["dissipation"]),
        "TEMPERATURE_PRODUCTION_RATE": 1.0,
        "SMOKE_DISSIPATION_RATE": float(physics_cfg["smoke"]["dissipation"]),
        "SMOKE_PRODUCTION_RATE": float(physics_cfg["smoke"].get("production_rate", 1.0)),
        "FUEL_BURN_RATE": float(physics_cfg["fuel"]["burn_rate"]),
        "FUEL_IGNITION_TEMPERATURE": float(physics_cfg["fuel"]["ignition_temperature"]),
        "T_REFERENCE": float(physics_cfg["temperature"]["reference_temperature"]),
        "SOURCE_TEMPERATURE_MAX": source_temperature_max,
        "BUOANCY_FACTOR": float(physics_cfg["temperature"]["buoyancy"]),
        "EXPANSION_RATE": float(physics_cfg["temperature"]["expansion_rate"]),
        "VORTICITY": float(physics_cfg.get("extras", {}).get("vorticity", 0.0)),
        "T_MAX": float(simulation_cfg["settings"]["simulation_length"]),
        "FRAME_START": int(simulation_cfg["settings"].get("start_frame", 0)),
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
        "OUTPUT_FIELD_CONFIG": output_field_config,
        "OUTPUT_VARIABLES": collect_output_variables(output_field_config),
        "OUTPUT_BUFFER_VARIABLES": collect_buffer_variables(output_field_config),
        "HOST_VDB_WRITER": host_vdb_writer,
        "HAS_SOURCE": has_source,
        "HAS_OBSTACLE": has_obstacle,
        "HAS_FORCE": has_force,
        "BC_CONFIG": bc_config,
        "INITIAL_U": initial_velocity[0],
        "INITIAL_V": initial_velocity[1],
        "INITIAL_W": initial_velocity[2],
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

    u = np.full((nx, ny, nz), simulation_params["INITIAL_U"], dtype=precision_dtype)
    v = np.full((nx, ny, nz), simulation_params["INITIAL_V"], dtype=precision_dtype)
    w = np.full((nx, ny, nz), simulation_params["INITIAL_W"], dtype=precision_dtype)

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
    source_velocity_x_host = np.asarray(source_field_data["velocity_x"], dtype=precision_dtype)
    source_velocity_y_host = np.asarray(source_field_data["velocity_y"], dtype=precision_dtype)
    source_velocity_z_host = np.asarray(source_field_data["velocity_z"], dtype=precision_dtype)

    u[source_mask_host] = source_velocity_x_host[source_mask_host]
    v[source_mask_host] = source_velocity_y_host[source_mask_host]
    w[source_mask_host] = source_velocity_z_host[source_mask_host]

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
        "source_velocity_x": cuda.to_device(source_velocity_x_host),
        "source_velocity_y": cuda.to_device(source_velocity_y_host),
        "source_velocity_z": cuda.to_device(source_velocity_z_host),
        "velocity_maxima": cuda.device_array(3, dtype=np.float32),
    }

    gpu_constants = {
        "RHO": precision_dtype.type(simulation_params["RHO"]),
        "NU": precision_dtype.type(simulation_params["NU"]),
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
        "NX": simulation_params["NX"],
        "NY": simulation_params["NY"],
        "NZ": simulation_params["NZ"],
        "FORCE_X_MAX": precision_dtype.type(force_field_data["max_abs"][0]),
        "FORCE_Y_MAX": precision_dtype.type(force_field_data["max_abs"][1]),
        "FORCE_Z_MAX": precision_dtype.type(force_field_data["max_abs"][2]),
        "HAS_SOURCE": simulation_params["HAS_SOURCE"],
        "HAS_OBSTACLE": simulation_params["HAS_OBSTACLE"],
        "HAS_FORCE": simulation_params["HAS_FORCE"],
    }

    return device_state, gpu_constants


def select_fields(field_map, field_names):
    """Build a dictionary view containing only the requested field names."""
    return {name: field_map[name] for name in field_names}
