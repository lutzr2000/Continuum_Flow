import hashlib
import numpy as np


def forcing_constant(force_state, force_cfg, dtype):
    """
    Add one spatially uniform constant force node to the static base field.

    The exported config stores the force vector directly as x/y/z components.
    Multiple constant nodes accumulate linearly.
    """
    force_vector = force_cfg.get("force", {})
    force_state["Fx_base"] += np.asarray(float(force_vector.get("x", 0.0)), dtype=dtype)
    force_state["Fy_base"] += np.asarray(float(force_vector.get("y", 0.0)), dtype=dtype)
    force_state["Fz_base"] += np.asarray(float(force_vector.get("z", 0.0)), dtype=dtype)


def forcing_point_divergence(force_state, shape, delta, force_cfg, dtype):
    """
    Add one smoothed point divergence source to the pressure RHS helper field.

    The source is a scalar divergence profile centered at `origin` and shaped
    by a Gaussian falloff length so it stays smooth around the source point.
    The profile is normalised over the discrete cell volumes so `strength`
    acts like an integrated source rate and remains stable when the grid
    resolution changes. Positive strength expands locally, negative strength
    contracts locally.
    """
    strength = float(force_cfg.get("strength", 0.0))
    origin = force_cfg.get("origin", (0.0, 0.0, 0.0))
    falloff = max(float(force_cfg.get("falloff", 1.0)), 1.0e-6)
    if abs(strength) <= 1.0e-12:
        return

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
    weight = np.exp(-distance2 / (falloff * falloff))
    weight_integral = float(np.sum(weight, dtype=np.float64)) * (delta ** 3)
    if weight_integral <= 1.0e-12:
        return

    # Convert the user strength into a resolution-independent divergence rate density.
    divergence_rate = (strength / weight_integral) * weight
    force_state["point_divergence"] += divergence_rate.astype(dtype, copy=False)


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
    Build the precomputed turbulence basis fields for one force node.

    The time-dependent animation is applied later by the GPU kernel using
    cosine and sine coefficients for the two stored basis fields.
    """
    #------------Parameters-------------------
    amplitude = float(force_cfg.get("amplitude", 0.0))
    spatial_scale = max(float(force_cfg.get("scale", 1.0)), delta * 2.0, 1e-6)
    seed_base = int(force_cfg.get("seed", 0))
    components = ("x", "y", "z")

    #------------Basis fields-------------------
    if amplitude == 0.0:
        basis_by_suffix = {
            suffix: tuple(np.zeros(shape, dtype=dtype) for _ in components)
            for suffix in ("a", "b")
        }
    else:
        x = np.arange(shape[0], dtype=np.float32)[:, None, None] * delta
        y = np.arange(shape[1], dtype=np.float32)[None, :, None] * delta
        z = np.arange(shape[2], dtype=np.float32)[None, None, :] * delta

        def build_basis(seed_suffix):
            digest = hashlib.sha256()
            for part in (
                seed_base,
                force_cfg.get("node_name", ""),
                spatial_scale,
                amplitude,
                shape,
                delta,
                seed_suffix,
            ):
                digest.update(str(part).encode("utf-8"))
                digest.update(b"\0")
            seed = int.from_bytes(digest.digest()[:8], byteorder="little", signed=False)
            rng = np.random.default_rng(seed)

            field = {component: np.zeros(shape, dtype=np.float32) for component in components}

            #------------Waves-------------------
            for _ in range(5):
                direction = rng.normal(size=3)
                direction_norm = np.linalg.norm(direction)
                if direction_norm <= 1e-12:
                    continue
                direction /= direction_norm

                wavelength = spatial_scale * rng.uniform(0.75, 1.75)
                phase = rng.uniform(0.0, 2.0 * np.pi)
                wave = np.sin(
                    2.0 * np.pi * (
                        direction[0] * x +
                        direction[1] * y +
                        direction[2] * z
                    ) / wavelength + phase
                )

                for component in components:
                    field[component] += rng.uniform(-1.0, 1.0) * wave

            #------------Normalise-------------------
            for component in components:
                field[component] -= np.mean(field[component], dtype=np.float64)
            rms = np.sqrt(
                sum(
                    np.mean(field[component] * field[component], dtype=np.float64)
                    for component in components
                )
            )
            if rms > 1e-12:
                scale = amplitude / rms
                for component in components:
                    field[component] *= scale

            return tuple(field[component].astype(dtype, copy=False) for component in components)

        basis_by_suffix = {suffix: build_basis(suffix) for suffix in ("a", "b")}

    #------------Store-------------------
    for suffix in ("a", "b"):
        basis = basis_by_suffix[suffix]
        for index, component in enumerate(components):
            force_state["turbulence"][f"F{component}_{suffix}"].append(basis[index])
    force_state["turbulence"]["angular_frequencies"].append(
        2.0 * np.pi * max(float(force_cfg.get("frequency", 0.0)), 0.0)
    )


def build_force_field_data(domain_cfg, force_entries, dtype=np.float32):
    """
    Build all force-field data on the CPU before the simulation starts.

    Static components are stored directly, while turbulence is represented as
    precomputed basis fields plus angular frequencies.
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
        "turbulence": {
            "Fx_a": [],
            "Fy_a": [],
            "Fz_a": [],
            "Fx_b": [],
            "Fy_b": [],
            "Fz_b": [],
            "angular_frequencies": [],
        },
    }

    #------------Dispatch force nodes-------------------
    for force_cfg in force_entries or ():
        node_type = force_cfg.get("node_type", "")
        if node_type == "BLENDERCFD_FORCE_TURBULENCE_NODE":
            forcing_turbulence(force_state, shape, delta, force_cfg, dtype)
        elif node_type == "BLENDERCFD_FORCE_CONSTANT_NODE":
            forcing_constant(force_state, force_cfg, dtype)
        elif node_type == "BLENDERCFD_FORCE_SWIRL_NODE":
            forcing_swirl(force_state, shape, delta, force_cfg, dtype)
        elif node_type == "BLENDERCFD_FORCE_POINT_NODE":
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
        for component in components:
            initial_components[component] += turbulence[f"F{component}_a"][index]

    #------------Force bounds-------------------
    dynamic_max = {component: 0.0 for component in components}
    for index in range(len(force_state["turbulence"]["angular_frequencies"])):
        for component in components:
            dynamic_max[component] += float(
                np.max(
                    np.hypot(
                        turbulence[f"F{component}_a"][index],
                        turbulence[f"F{component}_b"][index],
                    )
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
        "turbulence": turbulence,
        "max_abs": (
            static_max["x"] + dynamic_max["x"],
            static_max["y"] + dynamic_max["y"],
            static_max["z"] + dynamic_max["z"],
        ),
    }
