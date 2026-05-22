import numpy as np

POINT_FORCE_NODE_TYPES = {
    "BLENDERCFD_FORCE_POINT_NODE",
    "CONTINUUM_FLOW_FORCE_POINT_NODE",
}


def _animation_series_max_abs(animation_entry):
    """Return the maximum absolute sampled value from one exported animation entry."""
    if not animation_entry:
        return 0.0

    values = np.asarray(animation_entry.get("values", ()), dtype=np.float32)
    if values.size == 0:
        return 0.0
    return float(np.max(np.abs(values)))


def _normalise_vector_field(components, amplitude, dtype):
    """Center one vector field and scale its RMS magnitude to the target amplitude."""
    centered_components = []
    for component in components:
        centered = np.asarray(component, dtype=np.float32).copy()
        centered -= np.mean(centered, dtype=np.float64)
        centered_components.append(centered)

    rms = np.sqrt(
        sum(
            np.mean(component * component, dtype=np.float64)
            for component in centered_components
        )
    )
    if rms > 1.0e-12:
        scale = float(amplitude) / rms
        for component in centered_components:
            component *= scale
    else:
        for component in centered_components:
            component.fill(0.0)

    return tuple(np.asarray(component, dtype=dtype) for component in centered_components)


def _upsample_repeat_to_shape(field, shape):
    """Expand one coarse field to the target shape with cheap nearest-neighbour repeats."""
    result = np.asarray(field, dtype=np.float32)
    for axis, target_size in enumerate(shape):
        source_size = int(result.shape[axis])
        repeat_count = max(1, int(np.ceil(float(target_size) / float(source_size))))
        result = np.repeat(result, repeat_count, axis=axis)

        slicer = [slice(None)] * result.ndim
        slicer[axis] = slice(0, target_size)
        result = result[tuple(slicer)]

    return np.asarray(result, dtype=np.float32)


def build_divergence_free_noise_field(shape, delta, scale, amplitude, seed, dtype=np.float32):
    """
    Build one smooth, divergence-free random force field from a random vector potential.

    A random vector potential is low-pass filtered in Fourier space and converted
    to a velocity-like field by taking its curl. To reduce startup cost, the
    field is built on a coarse FFT grid first and then upsampled to the solver
    resolution.
    """
    dtype = np.dtype(dtype)
    amplitude = float(amplitude)
    if abs(amplitude) <= 1.0e-12:
        return tuple(np.zeros(shape, dtype=dtype) for _ in range(3))

    nx, ny, nz = (int(shape[0]), int(shape[1]), int(shape[2]))
    scale = max(float(scale), 2.0 * float(delta), 1.0e-6)
    rng = np.random.default_rng(int(seed))

    coarse_factor = max(2, int(np.ceil(scale / delta)))
    coarse_shape = (
        max(2, int(np.ceil(nx / coarse_factor))),
        max(2, int(np.ceil(ny / coarse_factor))),
        max(2, int(np.ceil(nz / coarse_factor))),
    )
    coarse_delta = float(delta) * float(coarse_factor)

    potential_x = rng.standard_normal(coarse_shape, dtype=np.float32)
    potential_y = rng.standard_normal(coarse_shape, dtype=np.float32)
    potential_z = rng.standard_normal(coarse_shape, dtype=np.float32)

    kx = (2.0 * np.pi * np.fft.fftfreq(coarse_shape[0], d=coarse_delta)).astype(np.float32)[:, None, None]
    ky = (2.0 * np.pi * np.fft.fftfreq(coarse_shape[1], d=coarse_delta)).astype(np.float32)[None, :, None]
    kz = (2.0 * np.pi * np.fft.rfftfreq(coarse_shape[2], d=coarse_delta)).astype(np.float32)[None, None, :]

    gaussian_width = scale / (2.0 * np.pi)
    filter_kernel = np.exp(
        -0.5 * (
            (kx * gaussian_width) * (kx * gaussian_width) +
            (ky * gaussian_width) * (ky * gaussian_width) +
            (kz * gaussian_width) * (kz * gaussian_width)
        )
    ).astype(np.float32)

    potential_x_hat = np.fft.rfftn(potential_x).astype(np.complex64, copy=False) * filter_kernel
    potential_y_hat = np.fft.rfftn(potential_y).astype(np.complex64, copy=False) * filter_kernel
    potential_z_hat = np.fft.rfftn(potential_z).astype(np.complex64, copy=False) * filter_kernel

    curl_x_hat = 1j * (ky * potential_z_hat - kz * potential_y_hat)
    curl_y_hat = 1j * (kz * potential_x_hat - kx * potential_z_hat)
    curl_z_hat = 1j * (kx * potential_y_hat - ky * potential_x_hat)

    field_x = _upsample_repeat_to_shape(
        np.fft.irfftn(curl_x_hat, s=coarse_shape).real.astype(np.float32, copy=False),
        shape,
    )
    field_y = _upsample_repeat_to_shape(
        np.fft.irfftn(curl_y_hat, s=coarse_shape).real.astype(np.float32, copy=False),
        shape,
    )
    field_z = _upsample_repeat_to_shape(
        np.fft.irfftn(curl_z_hat, s=coarse_shape).real.astype(np.float32, copy=False),
        shape,
    )

    return _normalise_vector_field(
        (field_x, field_y, field_z),
        amplitude,
        dtype,
    )


def _point_divergence_weight(shape, delta, origin, radius):
    """Return the normalized Gaussian support weight for one point divergence."""
    radius = max(float(radius), 1.0e-6)
    origin_x = -0.5 * shape[0] * delta
    origin_y = -0.5 * shape[1] * delta
    origin_z = 0.0
    x = origin_x + np.arange(shape[0], dtype=np.float32)[:, None, None] * delta
    y = origin_y + np.arange(shape[1], dtype=np.float32)[None, :, None] * delta
    z = origin_z + np.arange(shape[2], dtype=np.float32)[None, None, :] * delta

    dx = x - float(origin[0])
    dy = y - float(origin[1])
    dz = z - float(origin[2])
    distance2 = dx * dx + dy * dy + dz * dz
    weight = np.exp(-distance2 / (radius * radius))
    weight_integral = float(np.sum(weight, dtype=np.float64)) * (delta ** 3)
    return weight, weight_integral


def build_point_divergence_field(shape, delta, strength, origin, radius, dtype, out=None):
    """Build or overwrite one point-divergence field from one point-force setup."""
    dtype = np.dtype(dtype)
    if out is None:
        out = np.zeros(shape, dtype=dtype)
    else:
        out.fill(0.0)

    strength = float(strength)
    if abs(strength) <= 1.0e-12:
        return out

    weight, weight_integral = _point_divergence_weight(shape, delta, origin, radius)
    if weight_integral <= 1.0e-12:
        return out

    divergence_rate = (strength / weight_integral) * weight
    out[...] = divergence_rate.astype(dtype, copy=False)
    return out


def forcing_constant(force_state, force_cfg, dtype):
    """
    Add one spatially uniform constant force node to the static base field.

    The exported config stores the force vector directly as x/y/z components.
    Multiple constant nodes accumulate linearly.
    """
    if force_cfg.get("animations"):
        return

    force_vector = force_cfg.get("force", {})
    force_state["Fx_base"] += np.asarray(float(force_vector.get("x", 0.0)), dtype=dtype)
    force_state["Fy_base"] += np.asarray(float(force_vector.get("y", 0.0)), dtype=dtype)
    force_state["Fz_base"] += np.asarray(float(force_vector.get("z", 0.0)), dtype=dtype)


def forcing_point_divergence(force_state, shape, delta, force_cfg, dtype):
    """
    Add one smoothed point divergence source to the pressure RHS helper field.

    The source is a scalar divergence profile centered at `origin` and shaped
    by a Gaussian radius so it stays smooth around the source point.
    The profile is normalised over the discrete cell volumes so `strength`
    acts like an integrated source rate and remains stable when the grid
    resolution changes. Positive strength expands locally, negative strength
    contracts locally.
    """
    build_point_divergence_field(
        shape,
        delta,
        force_cfg.get("strength", 0.0),
        force_cfg.get("origin", (0.0, 0.0, 0.0)),
        force_cfg.get("radius", 1.0),
        dtype,
        out=force_state.setdefault("_point_divergence_work", np.zeros(shape, dtype=np.dtype(dtype))),
    )
    force_state["point_divergence"] += force_state["_point_divergence_work"]


def _has_point_force_animation(force_cfg):
    """Return whether one point force has animated values that affect divergence."""
    animations = force_cfg.get("animations") or {}
    return any(name in animations for name in ("strength", "origin", "radius"))


def forcing_swirl(force_state, shape, delta, force_cfg, dtype):
    """
    Add one swirl force field to the static base field.

    The swirl rotates around the configured axis through the configured origin.
    The force is tangential to circles around that axis, zero on the axis
    itself, grows linearly with perpendicular distance, and is clipped outside
    the configured radius.
    """
    strength = float(force_cfg.get("strength", 0.0))
    origin = force_cfg.get("origin", (0.0, 0.0, 0.0))
    axis = force_cfg.get("axis", (0.0, 0.0, 1.0))
    radius = max(float(force_cfg.get("radius", 0.0)), 0.0)
    if abs(strength) <= 1e-12 or radius <= 1e-12:
        return

    axis_vector = np.asarray(axis, dtype=np.float32)
    axis_norm = float(np.linalg.norm(axis_vector))
    if axis_norm <= 1e-12:
        return
    axis_unit = axis_vector / axis_norm

    origin_x = -0.5 * shape[0] * delta
    origin_y = -0.5 * shape[1] * delta
    origin_z = 0.0
    x = origin_x + np.arange(shape[0], dtype=np.float32)[:, None, None] * delta
    y = origin_y + np.arange(shape[1], dtype=np.float32)[None, :, None] * delta
    z = origin_z + np.arange(shape[2], dtype=np.float32)[None, None, :] * delta

    dx = x - float(origin[0])
    dy = y - float(origin[1])
    dz = z - float(origin[2])

    axial_projection = (
        dx * axis_unit[0] +
        dy * axis_unit[1] +
        dz * axis_unit[2]
    )
    radial_x = dx - axial_projection * axis_unit[0]
    radial_y = dy - axial_projection * axis_unit[1]
    radial_z = dz - axial_projection * axis_unit[2]
    radial_distance = np.sqrt(
        radial_x * radial_x +
        radial_y * radial_y +
        radial_z * radial_z
    )

    tangential_x = axis_unit[1] * radial_z - axis_unit[2] * radial_y
    tangential_y = axis_unit[2] * radial_x - axis_unit[0] * radial_z
    tangential_z = axis_unit[0] * radial_y - axis_unit[1] * radial_x

    active_mask = (radial_distance > 1e-12) & (radial_distance <= radius)

    fx = np.where(active_mask, strength * tangential_x, 0.0)
    fy = np.where(active_mask, strength * tangential_y, 0.0)
    fz = np.where(active_mask, strength * tangential_z, 0.0)

    force_state["Fx_base"] += fx.astype(dtype, copy=False)
    force_state["Fy_base"] += fy.astype(dtype, copy=False)
    force_state["Fz_base"] += fz.astype(dtype, copy=False)


def forcing_turbulence(force_state, shape, delta, force_cfg, dtype):
    """
    Build exactly two precomputed turbulence force fields for one force node.

    The GPU kernel later blends the two fields over time with one sinus-based
    mix factor, so the expensive random smooth field generation happens only
    once before the first timestep.
    """
    amplitude = float(force_cfg.get("amplitude", 0.0))
    spatial_scale = max(float(force_cfg.get("scale", 1.0)), delta, 1.0e-6)
    seed_base = int(force_cfg.get("seed", 0))
    animations = force_cfg.get("animations") or {}
    amplitude_animation = animations.get("amplitude")
    amplitude_max = max(abs(amplitude), _animation_series_max_abs(amplitude_animation))
    if amplitude_max <= 1.0e-12:
        return
    components = ("x", "y", "z")
    turbulence_index = len(force_state["turbulence"]["angular_frequencies"])

    basis_by_suffix = {
        "a": build_divergence_free_noise_field(
            shape, delta, spatial_scale, 1.0, seed_base, dtype=dtype,
        ),
        "b": build_divergence_free_noise_field(
            shape, delta, spatial_scale, 1.0, seed_base + 1, dtype=dtype,
        ),
    }

    for suffix in ("a", "b"):
        basis = basis_by_suffix[suffix]
        for index, component in enumerate(components):
            force_state["turbulence"][f"F{component}_{suffix}"].append(basis[index])
    force_state["turbulence"]["angular_frequencies"].append(
        2.0 * np.pi * max(float(force_cfg.get("frequency", 0.0)), 0.0)
    )
    force_state["turbulence"]["amplitudes"].append(np.float32(amplitude))
    force_state["turbulence"]["max_amplitudes"].append(np.float32(amplitude_max))
    if amplitude_animation:
        force_state["turbulence_runtime_entries"].append(
            {
                "index": int(turbulence_index),
                "amplitude": np.float32(amplitude),
                "animations": dict(animations),
            }
        )


def build_force_field_data(domain_cfg, force_entries, dtype=np.float32):
    """
    Build all force-field data on the CPU before the simulation starts.

    Static components are stored directly, while turbulence is represented as
    two precomputed force fields plus one angular frequency per node.
    """
    #------------Grid-------------------
    shape = (
        int(domain_cfg["grid"]["nx"]),
        int(domain_cfg["grid"]["ny"]),
        int(domain_cfg["grid"]["nz"]),
    )
    delta = float(domain_cfg["resolution"])
    dtype = np.dtype(dtype)
    components = ("x", "y", "z")

    #------------Initialise-------------------
    force_state = {
        "Fx_base": np.zeros(shape, dtype=dtype),
        "Fy_base": np.zeros(shape, dtype=dtype),
        "Fz_base": np.zeros(shape, dtype=dtype),
        "point_divergence": np.zeros(shape, dtype=dtype),
        "_point_divergence_work": np.zeros(shape, dtype=dtype),
        "point_force_entries": [],
        "turbulence": {
            "Fx_a": [],
            "Fy_a": [],
            "Fz_a": [],
            "Fx_b": [],
            "Fy_b": [],
            "Fz_b": [],
            "angular_frequencies": [],
            "amplitudes": [],
            "max_amplitudes": [],
        },
        "turbulence_runtime_entries": [],
    }

    #------------Dispatch force nodes-------------------
    for force_cfg in force_entries or ():
        node_type = force_cfg.get("node_type", "")
        if node_type in {"BLENDERCFD_FORCE_TURBULENCE_NODE", "CONTINUUM_FLOW_FORCE_TURBULENCE_NODE"}:
            forcing_turbulence(force_state, shape, delta, force_cfg, dtype)
        elif node_type in {"BLENDERCFD_FORCE_CONSTANT_NODE", "CONTINUUM_FLOW_FORCE_CONSTANT_NODE"}:
            forcing_constant(force_state, force_cfg, dtype)
        elif node_type in {"BLENDERCFD_FORCE_SWIRL_NODE", "CONTINUUM_FLOW_FORCE_SWIRL_NODE"}:
            forcing_swirl(force_state, shape, delta, force_cfg, dtype)
        elif node_type in POINT_FORCE_NODE_TYPES:
            if _has_point_force_animation(force_cfg):
                force_state["point_force_entries"].append(
                    {
                        "node_name": force_cfg.get("node_name", ""),
                        "node_type": node_type,
                        "strength": np.float32(force_cfg.get("strength", 0.0)),
                        "origin": np.asarray(force_cfg.get("origin", (0.0, 0.0, 0.0)), dtype=np.float32),
                        "radius": np.float32(force_cfg.get("radius", 1.0)),
                        "animations": dict(force_cfg.get("animations", {})),
                    }
                )
            else:
                forcing_point_divergence(force_state, shape, delta, force_cfg, dtype)

    #------------Pack turbulence arrays-------------------
    turbulence = {}
    for suffix in ("a", "b"):
        for component in components:
            key = f"F{component}_{suffix}"
            if force_state["turbulence"][key]:
                turbulence[key] = np.stack(force_state["turbulence"][key]).astype(dtype, copy=False)
            else:
                turbulence[key] = np.zeros((0,) + shape, dtype=dtype)
    turbulence["angular_frequencies"] = np.asarray(
        force_state["turbulence"]["angular_frequencies"], dtype=dtype
    )
    turbulence["amplitudes"] = np.asarray(
        force_state["turbulence"]["amplitudes"], dtype=dtype
    )
    turbulence["max_amplitudes"] = np.asarray(
        force_state["turbulence"]["max_amplitudes"], dtype=dtype
    )

    #------------Initial force state-------------------
    fx_initial = force_state["Fx_base"].copy()
    fy_initial = force_state["Fy_base"].copy()
    fz_initial = force_state["Fz_base"].copy()
    initial_components = {
        "x": fx_initial,
        "y": fy_initial,
        "z": fz_initial,
    }
    for index in range(len(force_state["turbulence"]["angular_frequencies"])):
        amplitude = float(turbulence["amplitudes"][index])
        for component in components:
            initial_components[component] += amplitude * turbulence[f"F{component}_b"][index]

    #------------Force bounds-------------------
    dynamic_max = {component: 0.0 for component in components}
    for index in range(len(force_state["turbulence"]["angular_frequencies"])):
        amplitude = float(turbulence["max_amplitudes"][index])
        for component in components:
            dynamic_max[component] += float(
                amplitude * max(
                    np.max(np.abs(turbulence[f"F{component}_a"][index])),
                    np.max(
                        np.abs(
                            2.0 * turbulence[f"F{component}_b"][index] -
                            turbulence[f"F{component}_a"][index]
                        )
                    ),
                )
            )
    static_max = {
        "x": float(np.max(np.abs(force_state["Fx_base"]))),
        "y": float(np.max(np.abs(force_state["Fy_base"]))),
        "z": float(np.max(np.abs(force_state["Fz_base"]))),
    }

    #------------Return-------------------
    return {
        "Fx_base": force_state["Fx_base"],
        "Fy_base": force_state["Fy_base"],
        "Fz_base": force_state["Fz_base"],
        "Fx": fx_initial,
        "Fy": fy_initial,
        "Fz": fz_initial,
        "point_divergence": force_state["point_divergence"],
        "point_divergence_base": force_state["point_divergence"].copy(),
        "point_force_entries": force_state["point_force_entries"],
        "turbulence_runtime_entries": force_state["turbulence_runtime_entries"],
        "turbulence": turbulence,
        "max_abs": (
            static_max["x"] + dynamic_max["x"],
            static_max["y"] + dynamic_max["y"],
            static_max["z"] + dynamic_max["z"],
        ),
    }
