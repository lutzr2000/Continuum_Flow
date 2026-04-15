import hashlib

import numpy as np


FORCE_TURBULENCE_NODE = "BLENDERCFD_FORCE_TURBULENCE_NODE"
TWO_PI = 2.0 * np.pi
TURBULENCE_WAVE_COUNT = 5


def _stable_seed(*parts):
    """Build a deterministic RNG seed from force-node data."""
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return int.from_bytes(digest.digest()[:8], byteorder="little", signed=False)


def _normalize_vector_field(fx, fy, fz, amplitude):
    """Scale a zero-mean vector field to the requested RMS magnitude."""
    fx -= np.mean(fx, dtype=np.float64)
    fy -= np.mean(fy, dtype=np.float64)
    fz -= np.mean(fz, dtype=np.float64)

    rms = np.sqrt(np.mean(fx * fx + fy * fy + fz * fz, dtype=np.float64))
    if rms <= 1e-12:
        return fx, fy, fz

    scale = amplitude / rms
    return fx * scale, fy * scale, fz * scale


def _build_turbulence_force_basis(shape, delta, force_cfg, dtype, seed_suffix):
    """Build one smooth turbulence basis field from a few broad sine waves."""
    amplitude = float(force_cfg.get("amplitude", 0.0))
    if amplitude == 0.0:
        return tuple(np.zeros(shape, dtype=dtype) for _ in range(3))

    spatial_scale = max(float(force_cfg.get("scale", 1.0)), delta * 2.0, 1e-6)

    seed = _stable_seed(
        force_cfg.get("node_name", ""),
        spatial_scale,
        amplitude,
        shape,
        delta,
        seed_suffix,
    )
    rng = np.random.default_rng(seed)

    x = np.arange(shape[0], dtype=np.float32)[:, None, None] * delta
    y = np.arange(shape[1], dtype=np.float32)[None, :, None] * delta
    z = np.arange(shape[2], dtype=np.float32)[None, None, :] * delta

    fx = np.zeros(shape, dtype=np.float32)
    fy = np.zeros(shape, dtype=np.float32)
    fz = np.zeros(shape, dtype=np.float32)

    for _ in range(TURBULENCE_WAVE_COUNT):
        direction = rng.normal(size=3)
        direction_norm = np.linalg.norm(direction)
        if direction_norm <= 1e-12:
            continue
        direction /= direction_norm

        wavelength = spatial_scale * rng.uniform(0.75, 1.75)
        phase = rng.uniform(0.0, TWO_PI)
        wave = np.sin(
            TWO_PI * (
                direction[0] * x +
                direction[1] * y +
                direction[2] * z
            ) / wavelength + phase
        )

        fx += rng.uniform(-1.0, 1.0) * wave
        fy += rng.uniform(-1.0, 1.0) * wave
        fz += rng.uniform(-1.0, 1.0) * wave

    fx, fy, fz = _normalize_vector_field(fx, fy, fz, amplitude)

    return (
        fx.astype(dtype, copy=False),
        fy.astype(dtype, copy=False),
        fz.astype(dtype, copy=False),
    )


def _empty_turbulence_array(component_arrays, shape, dtype):
    if not component_arrays:
        return np.zeros((0,) + shape, dtype=dtype)
    return np.stack(component_arrays).astype(dtype, copy=False)


def build_force_field_data(domain_cfg, force_entries, dtype=np.float32):
    """Build cheap animated turbulence basis fields from turbulence force nodes."""
    shape = (
        int(domain_cfg["grid"]["nx"]),
        int(domain_cfg["grid"]["ny"]),
        int(domain_cfg["grid"]["nz"]),
    )
    delta = float(domain_cfg["resolution"])
    dtype = np.dtype(dtype)

    fx_base = np.zeros(shape, dtype=dtype)
    fy_base = np.zeros(shape, dtype=dtype)
    fz_base = np.zeros(shape, dtype=dtype)
    turbulence_fx_a = []
    turbulence_fy_a = []
    turbulence_fz_a = []
    turbulence_fx_b = []
    turbulence_fy_b = []
    turbulence_fz_b = []
    angular_frequencies = []

    for force_cfg in force_entries or ():
        if force_cfg.get("node_type", "") != FORCE_TURBULENCE_NODE:
            continue

        basis_a = _build_turbulence_force_basis(shape, delta, force_cfg, dtype, "a")
        basis_b = _build_turbulence_force_basis(shape, delta, force_cfg, dtype, "b")
        turbulence_fx_a.append(basis_a[0])
        turbulence_fy_a.append(basis_a[1])
        turbulence_fz_a.append(basis_a[2])
        turbulence_fx_b.append(basis_b[0])
        turbulence_fy_b.append(basis_b[1])
        turbulence_fz_b.append(basis_b[2])
        angular_frequencies.append(TWO_PI * max(float(force_cfg.get("frequency", 0.0)), 0.0))

    turbulence = {
        "Fx_a": _empty_turbulence_array(turbulence_fx_a, shape, dtype),
        "Fy_a": _empty_turbulence_array(turbulence_fy_a, shape, dtype),
        "Fz_a": _empty_turbulence_array(turbulence_fz_a, shape, dtype),
        "Fx_b": _empty_turbulence_array(turbulence_fx_b, shape, dtype),
        "Fy_b": _empty_turbulence_array(turbulence_fy_b, shape, dtype),
        "Fz_b": _empty_turbulence_array(turbulence_fz_b, shape, dtype),
        "angular_frequencies": np.asarray(angular_frequencies, dtype=dtype),
    }

    fx_initial = fx_base.copy()
    fy_initial = fy_base.copy()
    fz_initial = fz_base.copy()
    for index in range(len(angular_frequencies)):
        fx_initial += turbulence["Fx_a"][index]
        fy_initial += turbulence["Fy_a"][index]
        fz_initial += turbulence["Fz_a"][index]

    fx_dynamic_max = 0.0
    fy_dynamic_max = 0.0
    fz_dynamic_max = 0.0
    for index in range(len(angular_frequencies)):
        fx_dynamic_max += float(np.max(np.hypot(turbulence["Fx_a"][index], turbulence["Fx_b"][index])))
        fy_dynamic_max += float(np.max(np.hypot(turbulence["Fy_a"][index], turbulence["Fy_b"][index])))
        fz_dynamic_max += float(np.max(np.hypot(turbulence["Fz_a"][index], turbulence["Fz_b"][index])))

    return {
        "Fx_base": fx_base,
        "Fy_base": fy_base,
        "Fz_base": fz_base,
        "Fx": fx_initial,
        "Fy": fy_initial,
        "Fz": fz_initial,
        "turbulence": turbulence,
        "max_abs": (
            fx_dynamic_max,
            fy_dynamic_max,
            fz_dynamic_max,
        ),
    }
