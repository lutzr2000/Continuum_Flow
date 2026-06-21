import json
import sys

import numpy as np


OUTPUT_BUFFER_MULTIPLIER = 2
PROGRESS_EVENT_PREFIX = "__CONTINUUM_FLOW_PROGRESS__ "
SPARSE_MASK_FIELDS = ("smoke", "flame")


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


def select_fields(field_map, field_names):
    """
    Build a dictionary view containing only the requested field names.
    """
    return {name: field_map[name] for name in field_names}
