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

    #------------Basis fields-------------------
    if amplitude == 0.0:
        basis_a = tuple(np.zeros(shape, dtype=dtype) for _ in range(3))
        basis_b = tuple(np.zeros(shape, dtype=dtype) for _ in range(3))
    else:
        x = np.arange(shape[0], dtype=np.float32)[:, None, None] * delta
        y = np.arange(shape[1], dtype=np.float32)[None, :, None] * delta
        z = np.arange(shape[2], dtype=np.float32)[None, None, :] * delta

        basis_components = []
        for seed_suffix in ("a", "b"):
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

            fx = np.zeros(shape, dtype=np.float32)
            fy = np.zeros(shape, dtype=np.float32)
            fz = np.zeros(shape, dtype=np.float32)

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

                fx += rng.uniform(-1.0, 1.0) * wave
                fy += rng.uniform(-1.0, 1.0) * wave
                fz += rng.uniform(-1.0, 1.0) * wave

            #------------Normalise-------------------
            fx -= np.mean(fx, dtype=np.float64)
            fy -= np.mean(fy, dtype=np.float64)
            fz -= np.mean(fz, dtype=np.float64)
            rms = np.sqrt(np.mean(fx * fx + fy * fy + fz * fz, dtype=np.float64))
            if rms > 1e-12:
                scale = amplitude / rms
                fx *= scale
                fy *= scale
                fz *= scale

            basis_components.append((
                fx.astype(dtype, copy=False),
                fy.astype(dtype, copy=False),
                fz.astype(dtype, copy=False),
            ))

        basis_a, basis_b = basis_components

    #------------Store-------------------
    force_state["turbulence_fx_a"].append(basis_a[0])
    force_state["turbulence_fy_a"].append(basis_a[1])
    force_state["turbulence_fz_a"].append(basis_a[2])
    force_state["turbulence_fx_b"].append(basis_b[0])
    force_state["turbulence_fy_b"].append(basis_b[1])
    force_state["turbulence_fz_b"].append(basis_b[2])
    force_state["turbulence_angular_frequencies"].append(
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

    #------------Initialise-------------------
    force_state = {
        "Fx_base": np.zeros(shape, dtype=dtype),
        "Fy_base": np.zeros(shape, dtype=dtype),
        "Fz_base": np.zeros(shape, dtype=dtype),
        "turbulence_fx_a": [],
        "turbulence_fy_a": [],
        "turbulence_fz_a": [],
        "turbulence_fx_b": [],
        "turbulence_fy_b": [],
        "turbulence_fz_b": [],
        "turbulence_angular_frequencies": [],
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
    if force_state["turbulence_fx_a"]:
        turbulence_fx_a = np.stack(force_state["turbulence_fx_a"]).astype(dtype, copy=False)
        turbulence_fy_a = np.stack(force_state["turbulence_fy_a"]).astype(dtype, copy=False)
        turbulence_fz_a = np.stack(force_state["turbulence_fz_a"]).astype(dtype, copy=False)
        turbulence_fx_b = np.stack(force_state["turbulence_fx_b"]).astype(dtype, copy=False)
        turbulence_fy_b = np.stack(force_state["turbulence_fy_b"]).astype(dtype, copy=False)
        turbulence_fz_b = np.stack(force_state["turbulence_fz_b"]).astype(dtype, copy=False)
    else:
        turbulence_fx_a = np.zeros((0,) + shape, dtype=dtype)
        turbulence_fy_a = np.zeros((0,) + shape, dtype=dtype)
        turbulence_fz_a = np.zeros((0,) + shape, dtype=dtype)
        turbulence_fx_b = np.zeros((0,) + shape, dtype=dtype)
        turbulence_fy_b = np.zeros((0,) + shape, dtype=dtype)
        turbulence_fz_b = np.zeros((0,) + shape, dtype=dtype)

    turbulence = {
        "Fx_a": turbulence_fx_a,
        "Fy_a": turbulence_fy_a,
        "Fz_a": turbulence_fz_a,
        "Fx_b": turbulence_fx_b,
        "Fy_b": turbulence_fy_b,
        "Fz_b": turbulence_fz_b,
        "angular_frequencies": np.asarray(
            force_state["turbulence_angular_frequencies"], dtype=dtype
        ),
    }

    #------------Initial force state-------------------
    fx_initial = force_state["Fx_base"].copy()
    fy_initial = force_state["Fy_base"].copy()
    fz_initial = force_state["Fz_base"].copy()
    for index in range(len(force_state["turbulence_angular_frequencies"])):
        fx_initial += turbulence["Fx_a"][index]
        fy_initial += turbulence["Fy_a"][index]
        fz_initial += turbulence["Fz_a"][index]

    #------------Force bounds-------------------
    fx_dynamic_max = 0.0
    fy_dynamic_max = 0.0
    fz_dynamic_max = 0.0
    for index in range(len(force_state["turbulence_angular_frequencies"])):
        fx_dynamic_max += float(np.max(np.hypot(turbulence["Fx_a"][index], turbulence["Fx_b"][index])))
        fy_dynamic_max += float(np.max(np.hypot(turbulence["Fy_a"][index], turbulence["Fy_b"][index])))
        fz_dynamic_max += float(np.max(np.hypot(turbulence["Fz_a"][index], turbulence["Fz_b"][index])))

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
            fx_dynamic_max,
            fy_dynamic_max,
            fz_dynamic_max,
        ),
    }
