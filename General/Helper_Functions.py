import json
import sys

import numpy as np

import General.Forcing as Forcing
import Kernel_GPU.Boundary_Conditions.Obstacle_BC as Obstacle_BC
import Kernel_GPU.Boundary_Conditions.Source_BC as Source_BC
import Kernel_GPU.Kernel_Config as Kernel_Config

BOUNDARY_FACE_NAMES = ("x_low", "x_high", "y_low", "y_high", "z_low", "z_high")
OUTPUT_BUFFER_MULTIPLIER = 2
PROGRESS_EVENT_PREFIX = "__BLENDERCFD_PROGRESS__ "
SPARSE_MASK_FIELDS = ("smoke", "flame")
PHYSICS_ANIMATION_TO_GPU_CONSTANT = {
    "temperature_dissipation": "TEMPERATURE_DISSIPATION_RATE",
    "reference_temperature": "T_REFERENCE",
    "buoyancy": "BUOANCY_FACTOR",
    "expansion_rate": "EXPANSION_RATE",
    "smoke_dissipation": "SMOKE_DISSIPATION_RATE",
    "smoke_production_rate": "SMOKE_PRODUCTION_RATE",
    "fuel_burn_rate": "FUEL_BURN_RATE",
    "fuel_ignition_temperature": "FUEL_IGNITION_TEMPERATURE",
    "vorticity": "VORTICITY",
}


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


def _animation_series_to_arrays(animation_entry, dtype):
    """Convert one exported animation series to contiguous numpy arrays."""
    if not animation_entry:
        return None

    times = np.asarray(animation_entry.get("times", ()), dtype=np.float32)
    values = np.asarray(animation_entry.get("values", ()), dtype=dtype)
    if times.size == 0 or values.size == 0:
        return None

    return {
        "times": np.ascontiguousarray(times),
        "values": np.ascontiguousarray(values),
        "cursor": 0,
    }


def build_animation_state(simulation_cfg, dtype=np.float32):
    """Build compact runtime animation data for lightweight kernel updates."""
    dtype = np.dtype(dtype)
    animation_state = {
        "constants": {},
        "constant_force": {},
        "enabled": False,
    }

    physics_cfg = simulation_cfg.get("physics") or {}
    physics_animations = physics_cfg.get("animations", {})
    for property_name, constant_name in PHYSICS_ANIMATION_TO_GPU_CONSTANT.items():
        series = _animation_series_to_arrays(physics_animations.get(property_name), dtype)
        if series is None:
            continue
        animation_state["constants"][constant_name] = series
        animation_state["enabled"] = True

    combined_force_times = None
    combined_force_values = {
        "x": None,
        "y": None,
        "z": None,
    }
    force_property_to_axis = {
        "fx": "x",
        "fy": "y",
        "fz": "z",
    }
    for force_cfg in simulation_cfg.get("forces", ()):
        if force_cfg.get("node_type") != "BLENDERCFD_FORCE_CONSTANT_NODE":
            continue
        animations = force_cfg.get("animations", {})
        for property_name, axis_name in force_property_to_axis.items():
            series = _animation_series_to_arrays(animations.get(property_name), dtype)
            if series is None:
                continue
            if combined_force_times is None:
                combined_force_times = series["times"].copy()
            if combined_force_values[axis_name] is None:
                combined_force_values[axis_name] = np.zeros_like(series["values"], dtype=dtype)
            combined_force_values[axis_name] += series["values"]
            animation_state["enabled"] = True

    if combined_force_times is not None:
        for axis_name in ("x", "y", "z"):
            values = combined_force_values[axis_name]
            if values is None:
                values = np.zeros(combined_force_times.shape, dtype=dtype)
            animation_state["constant_force"][axis_name] = {
                "times": np.ascontiguousarray(combined_force_times.copy()),
                "values": np.ascontiguousarray(values),
                "cursor": 0,
            }

    return animation_state


def _interpolate_animation_series(series, time_value):
    """Interpolate one monotonic time series with a rolling cursor."""
    if not series:
        return 0.0

    times = series["times"]
    values = series["values"]
    if times.size == 0:
        return 0.0
    if times.size == 1 or time_value <= float(times[0]):
        return values[0]

    cursor = int(series.get("cursor", 0))
    last_segment = int(times.size - 2)
    if cursor > last_segment:
        cursor = last_segment

    while cursor < last_segment and time_value >= float(times[cursor + 1]):
        cursor += 1
    series["cursor"] = cursor

    if cursor >= last_segment and time_value >= float(times[-1]):
        return values[-1]

    t0 = float(times[cursor])
    t1 = float(times[cursor + 1])
    if t1 <= t0:
        return values[cursor]

    alpha = (float(time_value) - t0) / (t1 - t0)
    return values[cursor] * (1.0 - alpha) + values[cursor + 1] * alpha


def update_animated_gpu_constants(animation_state, gpu_constants, time_value):
    """Update animated scalar kernel constants and uniform constant-force offsets."""
    animated_force = {
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
    }
    if not animation_state or not animation_state.get("enabled"):
        return animated_force

    for constant_name, series in animation_state.get("constants", {}).items():
        gpu_constants[constant_name] = np.float32(_interpolate_animation_series(series, time_value))

    for axis_name, series in animation_state.get("constant_force", {}).items():
        animated_force[axis_name] = float(_interpolate_animation_series(series, time_value))

    return animated_force


def _series_max_abs(series):
    """Return the maximum absolute value of one sampled linear animation series."""
    if not series:
        return 0.0

    values = series.get("values")
    if values is None or values.size == 0:
        return 0.0
    return float(np.max(np.abs(values)))


def estimate_theoretical_force_maxima(gpu_constants, animation_state):
    """Estimate one conservative force bound for the whole simulation run."""
    g = 9.81
    constant_force_animation = animation_state.get("constant_force", {}) if animation_state else {}
    fx_max = float(gpu_constants["FORCE_X_MAX"]) + _series_max_abs(constant_force_animation.get("x"))
    fy_max = float(gpu_constants["FORCE_Y_MAX"]) + _series_max_abs(constant_force_animation.get("y"))

    reference_temperature_series = (
        animation_state.get("constants", {}).get("T_REFERENCE")
        if animation_state
        else None
    )
    buoyancy_factor_series = (
        animation_state.get("constants", {}).get("BUOANCY_FACTOR")
        if animation_state
        else None
    )
    reference_temperature_min = float(gpu_constants["T_REFERENCE"])
    if reference_temperature_series and reference_temperature_series["values"].size > 0:
        reference_temperature_min = min(
            reference_temperature_min,
            float(np.min(reference_temperature_series["values"])),
        )

    buoyancy_factor_max_abs = abs(float(gpu_constants["BUOANCY_FACTOR"]))
    if buoyancy_factor_series and buoyancy_factor_series["values"].size > 0:
        buoyancy_factor_max_abs = max(
            buoyancy_factor_max_abs,
            float(np.max(np.abs(buoyancy_factor_series["values"]))),
        )

    source_temperature_delta = max(
        0.0,
        float(gpu_constants["SOURCE_TEMPERATURE_MAX"] - reference_temperature_min),
    )
    fz_buoyancy_max = g * buoyancy_factor_max_abs * source_temperature_delta * 1.5
    fz_max = (
        float(gpu_constants["FORCE_Z_MAX"]) +
        _series_max_abs(constant_force_animation.get("z")) +
        float(fz_buoyancy_max)
    )
    return fx_max, fy_max, fz_max


def apply_config(config):
    """Extract kernel settings and persistent data from the exported config."""
    simulations = config.get("simulations")
    simulation_cfg = simulations[0]
    meta_cfg = config.get("meta") or {}
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
    for runtime_entry in source_data.get("runtime_entries", ()):
        source_temperature_max = max(
            source_temperature_max,
            float(runtime_entry.get("temperature", 0.0)),
        )
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
    animation_state = build_animation_state(simulation_cfg, dtype=np.float32)
    if animation_state["constant_force"]:
        has_force = True
    temperature_dissipation_rate = float(physics_cfg["temperature"]["dissipation"])
    smoke_dissipation_rate = float(physics_cfg["smoke"]["dissipation"])
    smoke_production_rate = float(physics_cfg["smoke"].get("production_rate", 1.0))
    fuel_burn_rate = float(physics_cfg["fuel"]["burn_rate"])
    fuel_ignition_temperature = float(physics_cfg["fuel"]["ignition_temperature"])
    t_reference = float(physics_cfg["temperature"]["reference_temperature"])
    buoyancy_factor = float(physics_cfg["temperature"]["buoyancy"])
    expansion_rate = float(physics_cfg["temperature"]["expansion_rate"])
    vorticity = float(physics_cfg.get("extras", {}).get("vorticity", 0.0))

    for constant_name, series in animation_state["constants"].items():
        initial_value = float(series["values"][0])
        if constant_name == "TEMPERATURE_DISSIPATION_RATE":
            temperature_dissipation_rate = initial_value
        elif constant_name == "SMOKE_DISSIPATION_RATE":
            smoke_dissipation_rate = initial_value
        elif constant_name == "SMOKE_PRODUCTION_RATE":
            smoke_production_rate = initial_value
        elif constant_name == "FUEL_BURN_RATE":
            fuel_burn_rate = initial_value
        elif constant_name == "FUEL_IGNITION_TEMPERATURE":
            fuel_ignition_temperature = initial_value
        elif constant_name == "T_REFERENCE":
            t_reference = initial_value
        elif constant_name == "BUOANCY_FACTOR":
            buoyancy_factor = initial_value
        elif constant_name == "EXPANSION_RATE":
            expansion_rate = initial_value
        elif constant_name == "VORTICITY":
            vorticity = initial_value

    return {
        "RHO": float(physics_cfg["fluid"]["density"]),
        "NU": float(physics_cfg["fluid"]["viscosity"]),
        "TEMPERATURE_DISSIPATION_RATE": temperature_dissipation_rate,
        "TEMPERATURE_PRODUCTION_RATE": 1.0,
        "SMOKE_DISSIPATION_RATE": smoke_dissipation_rate,
        "SMOKE_PRODUCTION_RATE": smoke_production_rate,
        "FUEL_BURN_RATE": fuel_burn_rate,
        "FUEL_IGNITION_TEMPERATURE": fuel_ignition_temperature,
        "T_REFERENCE": t_reference,
        "SOURCE_TEMPERATURE_MAX": source_temperature_max,
        "BUOANCY_FACTOR": buoyancy_factor,
        "EXPANSION_RATE": expansion_rate,
        "VORTICITY": vorticity,
        "T_MAX": float(simulation_cfg["settings"]["simulation_length"]),
        "FRAME_START": int(simulation_cfg["settings"].get("start_frame", 0)),
        "CFL_MAX": float(simulation_cfg["settings"]["cfl"]),
        "MAX_ITER": int(simulation_cfg["settings"]["iterations"]),
        "VELOCITY_ADVECTION_SCHEME": str(
            simulation_cfg["settings"].get(
                "velocity_advection_scheme",
                "SECOND_ORDER_UPWIND",
            )
        ),
        "PRESSURE_SOLVER": str(
            simulation_cfg["settings"].get("pressure_solver", "jacobi")
        ),
        "MAX_VELOCITY_INCREMENT_FACTOR": float(
            simulation_cfg["settings"].get(
                "max_velocity_increment_factor",
                Kernel_Config.MAX_VELOCITY_INCREMENT_FACTOR,
            )
        ),
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
        "meta": meta_cfg,
        "HAS_SOURCE": has_source,
        "HAS_OBSTACLE": has_obstacle,
        "HAS_FORCE": has_force,
        "BC_CONFIG": bc_config,
        "ANIMATION_STATE": animation_state,
        "INITIAL_U": initial_velocity[0],
        "INITIAL_V": initial_velocity[1],
        "INITIAL_W": initial_velocity[2],
        "obstacle_data": obstacle_data,
        "obstacle_mask": obstacle_data["mask"],
        "source_field_data": source_data,
        "force_field_data": force_data,
        "HAS_DYNAMIC_BOUNDARIES": bool(
            obstacle_data.get("is_animated", False) or
            source_data.get("is_animated", False)
        ),
    }


def select_fields(field_map, field_names):
    """Build a dictionary view containing only the requested field names."""
    return {name: field_map[name] for name in field_names}
