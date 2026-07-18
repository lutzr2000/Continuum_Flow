import numpy as np

def _get_animation_times(container):
    timeline = container.get("animation_timeline") or {}
    return timeline.get("times") or ()


def get_animated_node_value(node, var_name, t, default=0.0, animation_times=()):
    value = node.get(var_name, default)

    animation_entry = (node.get("animations") or {}).get(var_name) or {}
    values = animation_entry.get("values") or ()
    times = animation_times or animation_entry.get("times") or ()
    sample_count = min(len(times), len(values))

    if sample_count > 0:
        nearest_idx = min(
            range(sample_count),
            key=lambda i: abs(float(times[i]) - float(t)),
        )
        value = values[nearest_idx]

    return default if value is None else value


def constant_force(simulation, t):
    fx = 0.0
    fy = 0.0
    fz = 0.0
    animation_times = _get_animation_times(simulation)

    for force_node in simulation.get("forces", []):
        if force_node.get("node_type") == "CONTINUUM_FLOW_FORCE_CONSTANT_NODE":
            force = force_node.get("force") or {}

            fx += get_animated_node_value(force_node, "fx", t, force.get("x", 0.0), animation_times)
            fy += get_animated_node_value(force_node, "fy", t, force.get("y", 0.0), animation_times)
            fz += get_animated_node_value(force_node, "fz", t, force.get("z", 0.0), animation_times)

    return fx, fy, fz


def swirl_force(simulation, t):
    swirl_nodes = []
    animation_times = _get_animation_times(simulation)

    for node in simulation.get("forces", []):
        if node.get("node_type") != "CONTINUUM_FLOW_FORCE_SWIRL_NODE":
            continue

        strength = get_animated_node_value(node, "strength", t, 0.0, animation_times)
        origin = get_animated_node_value(node, "origin", t, [0.0, 0.0, 0.0], animation_times)
        axis = get_animated_node_value(node, "axis", t, [0.0, 0.0, 1.0], animation_times)
        radius = get_animated_node_value(node, "radius", t, 0.0, animation_times)

        origin = np.asarray(origin, dtype=np.float32)
        axis = np.asarray(axis, dtype=np.float32)

        swirl_nodes.append([
            strength,
            origin[0], origin[1], origin[2],
            axis[0], axis[1], axis[2],
            radius,
        ])

    swirl_nodes = np.asarray(swirl_nodes, dtype=np.float32).reshape((-1, 8))

    return swirl_nodes, swirl_nodes.shape[0] > 0


def turbulence_force(simulation, t):
    turbulence_nodes = []
    animation_times = _get_animation_times(simulation)

    for node in simulation.get("forces", []):
        if node.get("node_type") != "CONTINUUM_FLOW_FORCE_TURBULENCE_NODE":
            continue

        amplitude = get_animated_node_value(node, "amplitude", t, 0.0, animation_times)
        scale = node.get("scale", 1.0)
        frequency = node.get("frequency", 1.0)
        seed = node.get("seed", 0)

        turbulence_nodes.append([
            amplitude,
            scale,
            frequency,
            seed,
        ])

    turbulence_nodes = np.asarray(turbulence_nodes, dtype=np.float32).reshape((-1, 4))

    return turbulence_nodes, turbulence_nodes.shape[0] > 0