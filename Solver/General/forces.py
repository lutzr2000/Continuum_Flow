import numpy as np

def get_animated_node_value(node, var_name, t, default=0.0):
    value = node.get(var_name, default)

    animation_entry = (node.get("animations") or {}).get(var_name) or {}
    times = animation_entry.get("times") or ()
    values = animation_entry.get("values") or ()

    if times and values:
        nearest_idx = min(
            range(len(times)),
            key=lambda i: abs(float(times[i]) - float(t)),
        )
        value = values[nearest_idx]

    return default if value is None else value


def constant_force(simulation, t):
    fx = 0.0
    fy = 0.0
    fz = 0.0

    for force_node in simulation.get("forces", []):
        if force_node.get("node_type") == "CONTINUUM_FLOW_FORCE_CONSTANT_NODE":
            force = force_node.get("force") or {}

            fx += get_animated_node_value(force_node, "fx", t, force.get("x", 0.0))
            fy += get_animated_node_value(force_node, "fy", t, force.get("y", 0.0))
            fz += get_animated_node_value(force_node, "fz", t, force.get("z", 0.0))

    return fx, fy, fz


def swirl_force(simulation, t):
    swirl_nodes = []

    for node in simulation.get("forces", []):
        if node.get("node_type") != "CONTINUUM_FLOW_FORCE_SWIRL_NODE":
            continue

        strength = get_animated_node_value(node, "strength", t, 0.0)
        origin = get_animated_node_value(node, "origin", t, [0.0, 0.0, 0.0])
        axis = get_animated_node_value(node, "axis", t, [0.0, 0.0, 1.0])
        radius = get_animated_node_value(node, "radius", t, 0.0)

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

    for node in simulation.get("forces", []):
        if node.get("node_type") != "CONTINUUM_FLOW_FORCE_TURBULENCE_NODE":
            continue

        amplitude = get_animated_node_value(node, "amplitude", t, 0.0)
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