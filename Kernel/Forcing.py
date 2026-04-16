import hashlib
import numpy as np


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
            pass
        elif node_type == "BLENDERCFD_FORCE_POINT_NODE":
            pass

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

    #------------Return-------------------
    return {
        "Fx_base": force_state["Fx_base"],
        "Fy_base": force_state["Fy_base"],
        "Fz_base": force_state["Fz_base"],
        "Fx": fx_initial,
        "Fy": fy_initial,
        "Fz": fz_initial,
        "turbulence": turbulence,
        "max_abs": (
            dynamic_max["x"],
            dynamic_max["y"],
            dynamic_max["z"],
        ),
    }
