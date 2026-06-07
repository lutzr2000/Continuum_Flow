import ctypes
import json
import os
import sys
from ctypes import wintypes
from time import perf_counter

import numpy as np

try:
    from numba import cuda
except ImportError:
    cuda = None

import Solver.General.forcing as forcing
import Solver.General.obstacles as general_obstacles
import Solver.Kernel_GPU.Boundary_Conditions.obstacle_bc as obstacle_bc
import Solver.Kernel_GPU.Boundary_Conditions.obstacles as gpu_obstacles
import Solver.Kernel_GPU.Boundary_Conditions.source_bc as source_bc
import Solver.Kernel_GPU.kernel_config as kernel_config

try:
    import resource
except ImportError:
    resource = None

BOUNDARY_FACE_NAMES = ("x_low", "x_high", "y_low", "y_high", "z_low", "z_high")
OUTPUT_BUFFER_MULTIPLIER = 2
PROGRESS_EVENT_PREFIX = "__CONTINUUM_FLOW_PROGRESS__ "
SPARSE_MASK_FIELDS = ("smoke", "flame")
PHYSICS_ANIMATION_TO_GPU_CONSTANT = {
    "temperature_dissipation": "TEMPERATURE_DISSIPATION_RATE",
    "reference_temperature": "T_REFERENCE",
    "buoyancy": "BUOANCY_FACTOR",
    "expansion_rate": "EXPANSION_RATE",
    "smoke_dissipation": "SMOKE_DISSIPATION_RATE",
    "smoke_production_rate": "SMOKE_PRODUCTION_RATE",
    "fuel_dissipation": "FUEL_DISSIPATION_RATE",
    "fuel_burn_rate": "FUEL_BURN_RATE",
    "fuel_ignition_temperature": "FUEL_IGNITION_TEMPERATURE",
    "minimum_oxygen_concentration": "MINIMUM_OXYGEN_CONCENTRATION",
    "vorticity": "VORTICITY",
}


class MemoryUsageTracker:
    """
    Track sampled solver memory usage and report average and peak values.
    """

    def __init__(self, label, sample_func):
        """
        Initialize one tracker with a display label and memory sampling callback.
        """
        self.label = label
        self._sample_func = sample_func
        self._sample_count = 0
        self._usage_sum = 0
        self._peak_usage = 0
        self._capacity_bytes = None

    def sample(self):
        """
        Record one memory usage sample if the backend can provide it.
        """
        usage_bytes, capacity_bytes = self._sample_func()
        if usage_bytes is None:
            return
        self._sample_count += 1
        self._usage_sum += int(usage_bytes)
        self._peak_usage = max(self._peak_usage, int(usage_bytes))
        if capacity_bytes is not None:
            self._capacity_bytes = int(capacity_bytes)

    def print_summary(self):
        """
        Print average and peak memory usage for all recorded samples.
        """
        if self._sample_count <= 0:
            print(f"{self.label} usage: unavailable")
            return
        average_usage = self._usage_sum / self._sample_count
        print(
            f"Average {self.label} usage: "
            f"{_format_bytes(average_usage)}{_format_percentage(average_usage, self._capacity_bytes)}"
        )
        print(
            f"Max {self.label} usage: "
            f"{_format_bytes(self._peak_usage)}{_format_percentage(self._peak_usage, self._capacity_bytes)}"
        )


def _record_timing(stats, name, elapsed):
    """
    Accumulate wall-clock timings for one named solver section.
    """
    entry = stats.get(name)
    if entry is None:
        stats[name] = {"total": float(elapsed), "count": 1}
        return
    entry["total"] += float(elapsed)
    entry["count"] += 1


def _print_timing_summary(stats, total_runtime, step_count, output_frame_count):
    """
    Print a compact timing table for the recorded solver sections.
    """
    print("Timing summary:")
    print(f"  Sim steps: {int(step_count)}")
    print(f"  Output frames enqueued: {int(output_frame_count)}")
    print(f"  Solver wall time: {total_runtime:.3f} s")

    if not stats:
        print("  No timing sections were recorded.")
        return

    for name, entry in sorted(
        stats.items(), key=lambda item: item[1]["total"], reverse=True
    ):
        total = entry["total"]
        count = entry["count"]
        avg_ms = (1000.0 * total / count) if count > 0 else 0.0
        share = (100.0 * total / total_runtime) if total_runtime > 0.0 else 0.0
        print(
            f"  {name:<28} total={total:8.3f} s  "
            f"calls={count:6d}  avg={avg_ms:8.3f} ms  share={share:6.2f}%"
        )


def _format_bytes(byte_count):
    """
    Format a byte count as a human-readable mebibyte string.
    """
    return f"{float(byte_count) / (1024.0 * 1024.0):.2f} MiB"


def _format_percentage(usage_bytes, capacity_bytes):
    """
    Format one usage value as a percentage of the known total capacity.
    """
    if not capacity_bytes:
        return ""
    return f" ({100.0 * float(usage_bytes) / float(capacity_bytes):.1f}% of total)"


def _sample_gpu_memory_usage():
    """
    Sample current CUDA memory usage and total device capacity when available.
    """
    if cuda is None:
        return None, None
    try:
        free_bytes, total_bytes = cuda.current_context().get_memory_info()
    except Exception:
        return None, None
    return int(total_bytes - free_bytes), int(total_bytes)


def _sample_process_memory_usage():
    """
    Sample the current solver process memory usage on the active platform.
    """
    if os.name == "nt":
        counters = _get_windows_process_memory_counters()
        if counters is not None:
            usage_bytes, capacity_bytes = counters
            usage_bytes = int(usage_bytes)
            capacity_bytes = None if capacity_bytes is None else int(capacity_bytes)
            return usage_bytes, capacity_bytes
        return None, None

    if resource is None:
        return None, None

    rss_value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return rss_value, None
    return rss_value * 1024, None


def _get_windows_process_memory_counters():
    """
    Read working-set and total physical memory information on Windows.
    """
    class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    psapi = ctypes.WinDLL("psapi")
    kernel32 = ctypes.WinDLL("kernel32")
    get_process_memory_info = psapi.GetProcessMemoryInfo
    get_process_memory_info.argtypes = [
        wintypes.HANDLE,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    get_process_memory_info.restype = wintypes.BOOL

    get_current_process = kernel32.GetCurrentProcess
    get_current_process.argtypes = []
    get_current_process.restype = wintypes.HANDLE

    counters = PROCESS_MEMORY_COUNTERS_EX()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
    process_handle = get_current_process()
    success = get_process_memory_info(
        process_handle, ctypes.byref(counters), counters.cb
    )
    if not success:
        return None

    memory_status = ctypes.c_ulonglong()
    get_physical_memory = getattr(kernel32, "GetPhysicallyInstalledSystemMemory", None)
    if get_physical_memory is None:
        total_bytes = None
    else:
        get_physical_memory.argtypes = [ctypes.POINTER(ctypes.c_ulonglong)]
        get_physical_memory.restype = wintypes.BOOL
        success = get_physical_memory(ctypes.byref(memory_status))
        total_bytes = int(memory_status.value) * 1024 if success else None

    return int(counters.WorkingSetSize), total_bytes


def emit_progress(percent, time_value=None):
    """
    Emit a machine-readable progress event for Blender's bake progress bar.
    """
    payload = {
        "percent": max(0.0, min(100.0, float(percent))),
    }
    if time_value is not None:
        payload["time"] = float(time_value)
    sys.stdout.write(PROGRESS_EVENT_PREFIX + json.dumps(payload) + "\n")
    sys.stdout.flush()


_ANIMATED_CONSTANT_DEFAULTS = {
    "TEMPERATURE_DISSIPATION_RATE": lambda physics_cfg: float(
        physics_cfg["temperature"]["dissipation"]
    ),
    "SMOKE_DISSIPATION_RATE": lambda physics_cfg: float(
        physics_cfg["smoke"]["dissipation"]
    ),
    "SMOKE_PRODUCTION_RATE": lambda physics_cfg: float(
        physics_cfg["smoke"].get("production_rate", 1.0)
    ),
    "FUEL_DISSIPATION_RATE": lambda physics_cfg: float(
        physics_cfg["fuel"].get("dissipation", 0.0)
    ),
    "FUEL_BURN_RATE": lambda physics_cfg: float(physics_cfg["fuel"]["burn_rate"]),
    "FUEL_IGNITION_TEMPERATURE": lambda physics_cfg: float(
        physics_cfg["fuel"]["ignition_temperature"]
    ),
    "MINIMUM_OXYGEN_CONCENTRATION": lambda physics_cfg: float(
        physics_cfg["fuel"].get("minimum_oxygen_concentration", 0.0)
    ),
    "T_REFERENCE": lambda physics_cfg: float(
        physics_cfg["temperature"]["reference_temperature"]
    ),
    "BUOANCY_FACTOR": lambda physics_cfg: float(physics_cfg["temperature"]["buoyancy"]),
    "EXPANSION_RATE": lambda physics_cfg: float(
        physics_cfg["temperature"]["expansion_rate"]
    ),
    "VORTICITY": lambda physics_cfg: float(
        physics_cfg.get("extras", {}).get("vorticity", 0.0)
    ),
}


def _build_runtime_data(domain_cfg, obstacle_entries, source_entries, force_entries):
    """
    Build persistent obstacle, source, and force data structures for the solver.
    """
    obstacle_data = general_obstacles.build_obstacle_data(
        domain_cfg,
        obstacle_entries,
        gpu_obstacles,
    )
    source_data = source_bc.build_source_data(domain_cfg, source_entries)
    force_start = perf_counter()
    force_data = forcing.build_force_field_data(
        domain_cfg, force_entries, dtype=np.float32
    )
    runtime_timings = {
        "init_forces": perf_counter() - force_start,
    }
    return obstacle_data, source_data, force_data, runtime_timings


def _source_temperature_max(source_data):
    """
    Return the maximum source temperature across static and animated sources.
    """
    source_temperature_max = 0.0
    for runtime_entry in source_data.get("runtime_entries", ()):
        source_temperature_max = max(
            source_temperature_max,
            float(runtime_entry.get("temperature", 0.0)),
        )
    return source_temperature_max


def _has_force_data(force_data):
    """
    Return whether the exported force data contains any active force contribution.
    """
    return bool(
        np.any(force_data["Fx_base"])
        or np.any(force_data["Fy_base"])
        or np.any(force_data["Fz_base"])
        or np.any(force_data["point_divergence"])
        or force_data.get("point_force_entries")
        or force_data["turbulence"]["angular_frequencies"].size > 0
    )


def _build_physics_constants(physics_cfg, animation_state):
    """
    Build kernel physics constants and apply animated initial overrides.
    """
    constants = {
        constant_name: default_builder(physics_cfg)
        for constant_name, default_builder in _ANIMATED_CONSTANT_DEFAULTS.items()
    }

    for constant_name, series in animation_state["constants"].items():
        constants[constant_name] = float(series["values"][0])

    return constants


def build_boundary_config(domain_cfg):
    """
    Build per-face BC config from the exported domain settings.
    """
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
    """
    Read one output-field option from new nested config or legacy bool config.
    """
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
    """
    Translate output settings to concrete kernel field export rules.
    """
    fields_cfg = output_cfg.get("fields", {})
    velocity_cfg = _field_option_value(fields_cfg, "velocity")
    if "velocity" not in fields_cfg:
        velocity_cfg = {
            "enabled": bool(
                fields_cfg.get("u", False)
                or fields_cfg.get("v", False)
                or fields_cfg.get("w", False)
            ),
            "sparse": False,
        }
    pressure_cfg = _field_option_value(fields_cfg, "pressure", "p")
    temperature_cfg = _field_option_value(fields_cfg, "temperature", "t")
    smoke_cfg = _field_option_value(fields_cfg, "smoke")
    fuel_cfg = _field_option_value(fields_cfg, "fuel")
    flame_cfg = _field_option_value(fields_cfg, "flame")

    output_field_config = {
        "u": {"export": velocity_cfg["enabled"], "sparse": velocity_cfg["sparse"]},
        "v": {"export": velocity_cfg["enabled"], "sparse": velocity_cfg["sparse"]},
        "w": {"export": velocity_cfg["enabled"], "sparse": velocity_cfg["sparse"]},
        "p": {"export": pressure_cfg["enabled"], "sparse": pressure_cfg["sparse"]},
        "T": {
            "export": temperature_cfg["enabled"],
            "sparse": temperature_cfg["sparse"],
        },
        "smoke": {"export": smoke_cfg["enabled"], "sparse": smoke_cfg["sparse"]},
        "fuel": {"export": fuel_cfg["enabled"], "sparse": fuel_cfg["sparse"]},
        "flame": {"export": flame_cfg["enabled"], "sparse": flame_cfg["sparse"]},
    }
    return output_field_config


def collect_output_variables(output_field_config):
    """
    Return the concrete field names that should be written to the VDB.
    """
    return [
        field_name
        for field_name, cfg in output_field_config.items()
        if cfg.get("export")
    ]


def collect_buffer_variables(output_field_config):
    """
    Return all fields that must be copied to host buffers for one output frame.
    """
    buffer_fields = list(collect_output_variables(output_field_config))
    if any(
        cfg.get("export") and cfg.get("sparse") for cfg in output_field_config.values()
    ):
        for mask_field in SPARSE_MASK_FIELDS:
            if mask_field not in buffer_fields:
                buffer_fields.append(mask_field)
    return buffer_fields


def build_output_performance_config(output_cfg):
    """
    Build output pipeline performance settings from the exported output node.
    """
    performance_cfg = output_cfg.get("performance", {})
    writer_processes = max(1, int(performance_cfg.get("writer_processes", 4)))

    return {
        "writer_processes": writer_processes,
        "buffer_count": writer_processes * OUTPUT_BUFFER_MULTIPLIER,
    }


def resolve_output_dtype(output_cfg):
    """
    Resolve the requested VDB storage precision from the exported output node.
    """
    precision_name = str(output_cfg.get("precision", "float16")).strip().lower()
    precision_mapping = {
        "float16": np.float16,
        "float32": np.float32,
    }
    return precision_mapping.get(precision_name, np.float16)


def _animation_series_to_arrays(animation_entry, dtype):
    """
    Convert one exported animation series to contiguous numpy arrays.
    """
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


def _merge_scalar_animation_series(
    existing_times, existing_values, new_times, new_values, dtype
):
    """
    Merge two scalar animation series onto one shared monotonic time axis.
    """
    dtype = np.dtype(dtype)
    existing_times = np.asarray(existing_times, dtype=np.float32)
    existing_values = np.asarray(existing_values, dtype=dtype)
    new_times = np.asarray(new_times, dtype=np.float32)
    new_values = np.asarray(new_values, dtype=dtype)

    if existing_times.size == 0:
        return (
            np.ascontiguousarray(new_times),
            np.ascontiguousarray(new_values, dtype=dtype),
        )
    if new_times.size == 0:
        return (
            np.ascontiguousarray(existing_times),
            np.ascontiguousarray(existing_values, dtype=dtype),
        )

    merged_times = np.unique(
        np.concatenate((existing_times, new_times)).astype(np.float32, copy=False)
    )
    merged_existing = np.interp(merged_times, existing_times, existing_values)
    merged_new = np.interp(merged_times, new_times, new_values)
    merged_values = np.asarray(merged_existing + merged_new, dtype=dtype)
    return np.ascontiguousarray(merged_times), np.ascontiguousarray(
        merged_values, dtype=dtype
    )


def _cached_animation_series(container, property_name, dtype):
    """
    Return one cached runtime animation series from an exported animation entry.
    """
    cache = container.setdefault("_animation_series_cache", {})
    if property_name in cache:
        return cache[property_name]

    series = _animation_series_to_arrays(
        (container.get("animations") or {}).get(property_name),
        dtype,
    )
    cache[property_name] = series
    return series


def build_animation_state(simulation_cfg, dtype=np.float32):
    """
    Build compact runtime animation data for lightweight kernel updates.
    """
    dtype = np.dtype(dtype)
    animation_state = {
        "constants": {},
        "constant_force": {},
        "enabled": False,
    }

    physics_cfg = simulation_cfg.get("physics") or {}
    physics_animations = physics_cfg.get("animations", {})
    for property_name, constant_name in PHYSICS_ANIMATION_TO_GPU_CONSTANT.items():
        series = _animation_series_to_arrays(
            physics_animations.get(property_name), dtype
        )
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
        if force_cfg.get("node_type") not in {
            "CONTINUUM_FLOW_FORCE_CONSTANT_NODE",
        }:
            continue
        animations = force_cfg.get("animations", {})
        for property_name, axis_name in force_property_to_axis.items():
            series = _animation_series_to_arrays(animations.get(property_name), dtype)
            if series is None:
                continue
            if combined_force_times is None:
                combined_force_times = series["times"].copy()
            if combined_force_values[axis_name] is None:
                combined_force_values[axis_name] = np.zeros_like(
                    series["values"], dtype=dtype
                )
                combined_force_values[axis_name] = np.ascontiguousarray(
                    combined_force_values[axis_name]
                )
            if combined_force_times.shape != series[
                "times"
            ].shape or not np.array_equal(combined_force_times, series["times"]):
                merged_times, merged_values = _merge_scalar_animation_series(
                    combined_force_times,
                    combined_force_values[axis_name],
                    series["times"],
                    series["values"],
                    dtype,
                )
                if (
                    merged_times.shape != combined_force_times.shape
                    or not np.array_equal(merged_times, combined_force_times)
                ):
                    for (
                        existing_axis_name,
                        existing_values,
                    ) in combined_force_values.items():
                        if existing_values is None or existing_axis_name == axis_name:
                            continue
                        combined_force_values[existing_axis_name] = np.asarray(
                            np.interp(
                                merged_times, combined_force_times, existing_values
                            ),
                            dtype=dtype,
                        )
                    combined_force_times = merged_times
                combined_force_values[axis_name] = merged_values
            else:
                combined_force_values[axis_name] = np.asarray(
                    combined_force_values[axis_name] + series["values"],
                    dtype=dtype,
                )
            animation_state["enabled"] = True

    for force_cfg in simulation_cfg.get("forces", ()):
        if force_cfg.get("node_type") not in forcing.POINT_FORCE_NODE_TYPES:
            continue
        animations = force_cfg.get("animations", {})
        if any(name in animations for name in ("strength", "origin", "radius")):
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
    """
    Interpolate one monotonic time series with a rolling cursor.
    """
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


def _series_max_abs(series):
    """
    Return the maximum absolute value of one sampled linear animation series.
    """
    if not series:
        return 0.0

    values = series.get("values")
    if values is None or values.size == 0:
        return 0.0
    return float(np.max(np.abs(values)))


def estimate_theoretical_force_maxima(gpu_constants, animation_state):
    """
    Estimate one conservative force bound for the whole simulation run.
    """
    g = 9.81
    constant_force_animation = (
        animation_state.get("constant_force", {}) if animation_state else {}
    )
    fx_max = float(gpu_constants["FORCE_X_MAX"]) + _series_max_abs(
        constant_force_animation.get("x")
    )
    fy_max = float(gpu_constants["FORCE_Y_MAX"]) + _series_max_abs(
        constant_force_animation.get("y")
    )

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
        float(gpu_constants["FORCE_Z_MAX"])
        + _series_max_abs(constant_force_animation.get("z"))
        + float(fz_buoyancy_max)
    )
    return fx_max, fy_max, fz_max


def apply_config(config):
    """
    Extract kernel settings and persistent data from the exported config.
    """
    # Read the exported simulation blocks we need for one solver run.
    simulations = config.get("simulations")
    simulation_cfg = simulations[0]
    meta_cfg = config.get("meta") or {}
    settings_cfg = simulation_cfg["settings"]
    domain_cfg = simulation_cfg["domain"]
    physics_cfg = simulation_cfg["physics"]
    output_cfg = simulation_cfg.get("outputs", [None])[0]
    obstacle_entries = simulation_cfg.get("obstacles", [])
    source_entries = simulation_cfg.get("sources", [])
    force_entries = simulation_cfg.get("forces", [])
    host_vdb_writer = config.get("_host_vdb_writer")

    # Build persistent runtime data derived from the exported node config.
    bc_config = build_boundary_config(domain_cfg)
    initial_velocity = initial_velocity_from_inflows(bc_config)
    obstacle_data, source_data, force_data, runtime_timings = _build_runtime_data(
        domain_cfg,
        obstacle_entries,
        source_entries,
        force_entries,
    )
    source_temperature_max = _source_temperature_max(source_data)

    # Resolve boolean feature flags and output/export settings.
    has_source = bool(np.any(source_data["mask"]))
    has_obstacle = bool(np.any(obstacle_data["mask"]))
    has_force = _has_force_data(force_data)
    output_performance = build_output_performance_config(output_cfg)
    output_dtype = resolve_output_dtype(output_cfg)
    output_field_config = build_output_field_config(output_cfg)

    # Prepare animated constant overrides before the kernel starts stepping.
    animation_state = build_animation_state(simulation_cfg, dtype=np.float32)
    if animation_state["constant_force"]:
        has_force = True
    if force_data.get("point_force_entries"):
        has_force = True
    physics_constants = _build_physics_constants(physics_cfg, animation_state)

    # Flatten everything into the solver-facing parameter dictionary.
    physics_params = {
        "RHO": float(physics_cfg["fluid"]["density"]),
        "NU": float(physics_cfg["fluid"]["viscosity"]),
        "TEMPERATURE_DISSIPATION_RATE": physics_constants[
            "TEMPERATURE_DISSIPATION_RATE"
        ],
        "TEMPERATURE_PRODUCTION_RATE": 1.0,
        "SMOKE_DISSIPATION_RATE": physics_constants["SMOKE_DISSIPATION_RATE"],
        "SMOKE_PRODUCTION_RATE": physics_constants["SMOKE_PRODUCTION_RATE"],
        "FUEL_DISSIPATION_RATE": physics_constants["FUEL_DISSIPATION_RATE"],
        "FUEL_BURN_RATE": physics_constants["FUEL_BURN_RATE"],
        "FUEL_IGNITION_TEMPERATURE": physics_constants["FUEL_IGNITION_TEMPERATURE"],
        "MINIMUM_OXYGEN_CONCENTRATION": physics_constants[
            "MINIMUM_OXYGEN_CONCENTRATION"
        ],
        "T_REFERENCE": physics_constants["T_REFERENCE"],
        "SOURCE_TEMPERATURE_MAX": source_temperature_max,
        "BUOANCY_FACTOR": physics_constants["BUOANCY_FACTOR"],
        "EXPANSION_RATE": physics_constants["EXPANSION_RATE"],
        "VORTICITY": physics_constants["VORTICITY"],
    }

    simulation_params = {
        "T_MAX": float(settings_cfg["simulation_length"]),
        "FRAME_START": int(settings_cfg.get("start_frame", 0)),
        "CFL_MAX": float(settings_cfg["cfl"]),
        "MAX_ITER": int(settings_cfg["iterations"]),
        "MACCORMACK_FACTOR": float(settings_cfg.get("maccormack_factor", 0.25)),
        "SIMULATE_SPARSELY": bool(settings_cfg.get("simulate_sparsely", True)),
        "ADAPTIVE_DOMAIN_THRESHOLD": float(
            settings_cfg.get("adaptive_domain_threshold", 0.001)
        ),
        "MAX_VELOCITY_INCREMENT_FACTOR": float(
            settings_cfg.get(
                "max_velocity_increment_factor",
                kernel_config.MAX_VELOCITY_INCREMENT_FACTOR,
            )
        ),
        "PRECISION": np.float32,
    }

    domain_params = {
        "DELTA": float(domain_cfg["resolution"]),
        "NX": int(domain_cfg["grid"]["nx"]),
        "NY": int(domain_cfg["grid"]["ny"]),
        "NZ": int(domain_cfg["grid"]["nz"]),
        "BC_CONFIG": bc_config,
        "INITIAL_U": initial_velocity[0],
        "INITIAL_V": initial_velocity[1],
        "INITIAL_W": initial_velocity[2],
    }

    output_params = {
        "OUTPUT_FPS": int(output_cfg["fps"]),
        "PRINT_FREQUENCY": 100,
        "OUTPUT_TIME_STEP": 1.0 / int(output_cfg["fps"]),
        "OUTPUT_STATUS": False,
        "OUTPUT_FORWARDER_COUNT": output_performance["writer_processes"],
        "WRITE_QUEUE_SIZE": output_performance["buffer_count"],
        "OUTPATH": output_cfg.get("output_path", ""),
        "OUTPUT_DTYPE": output_dtype,
        "OUTPUT_FIELD_CONFIG": output_field_config,
        "OUTPUT_SPARSE_THRESHOLD": float(
            settings_cfg.get("adaptive_domain_threshold", 0.001)
        ),
        "OUTPUT_VARIABLES": collect_output_variables(output_field_config),
        "OUTPUT_BUFFER_VARIABLES": collect_buffer_variables(output_field_config),
        "HOST_VDB_WRITER": host_vdb_writer,
    }

    runtime_params = {
        "meta": meta_cfg,
        "HAS_SOURCE": has_source,
        "HAS_OBSTACLE": has_obstacle,
        "HAS_FORCE": has_force,
        "ANIMATION_STATE": animation_state,
        "obstacle_data": obstacle_data,
        "obstacle_mask": obstacle_data["mask"],
        "source_field_data": source_data,
        "force_field_data": force_data,
        "INIT_FORCE_BUILD_TIME": float(runtime_timings.get("init_forces", 0.0)),
        "HAS_DYNAMIC_BOUNDARIES": bool(
            obstacle_data.get("is_animated", False)
            or source_data.get("is_animated", False)
        ),
    }

    return {
        **physics_params,
        **simulation_params,
        **domain_params,
        **output_params,
        **runtime_params,
    }


def select_fields(field_map, field_names):
    """
    Build a dictionary view containing only the requested field names.
    """
    return {name: field_map[name] for name in field_names}
