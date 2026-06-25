from numba import cuda

@cuda.jit(device=True, inline=True, cache=True)
def buoyancy_approximation(
    T,
    i,
    j,
    k,
    buoyancy_factor,
    t_reference,
):
    """
    computes the buoyancy force in z-direction with the Boussinesq approximation on the GPU.
    """
    g = 9.81
    return g * buoyancy_factor * (T[i, j, k] - t_reference)


def constant_force(simulation):
    fx = 0
    fy = 0
    fz = 0
    for force_node in simulation.get("forces", []):
        if force_node.get("node_type") == "CONTINUUM_FLOW_FORCE_CONSTANT_NODE":
            fx += force_node.get("force", {}).get("x", 0.0)
            fy += force_node.get("force", {}).get("y", 0.0)
            fz += force_node.get("force", {}).get("z", 0.0)

    return fx,fy,fz
