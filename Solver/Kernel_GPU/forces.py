

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
        return values[nearest_idx]

    return value if value is not None else default


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
