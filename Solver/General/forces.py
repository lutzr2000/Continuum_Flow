from typing import Any
import math

import numpy as np

from ..Kernel.helper import nearest_sample_index


def _get_animation_times(container: Any) -> Any:
    """
    Read the shared animation sample times from a simulation container.

    An empty tuple is returned when no animation timeline is configured.
    """
    timeline = container.get("animation_timeline") or {}
    return timeline.get("times") or ()


def get_animated_node_value(
    node: Any,
    var_name: str,
    t: float,
    default: Any = 0.0,
    animation_times: Any = (),
    nearest_indices: dict[tuple[int, int], int] | None = None,
) -> Any:
    """
    Resolve a node property at the animation sample nearest to ``t``.

    Static node data provides the fallback. Animation values are considered
    only where both a time and a value sample exist, and ``None`` resolves to
    the supplied default.
    """
    value = node.get(var_name, default)

    animation_entry = (node.get("animations") or {}).get(var_name) or {}
    values = animation_entry.get("values") or ()
    times = animation_times or animation_entry.get("times") or ()
    sample_count = min(len(times), len(values))

    if sample_count > 0:
        timeline_key = (id(times), sample_count)
        if nearest_indices is not None and timeline_key in nearest_indices:
            nearest_idx = nearest_indices[timeline_key]
        else:
            nearest_idx = nearest_sample_index(times, t, sample_count)
            if nearest_indices is not None:
                nearest_indices[timeline_key] = nearest_idx
        value = values[nearest_idx]

    return default if value is None else value


def constant_force(simulation: dict[str, Any], t: float) -> Any:
    """
    Sum all configured constant-force nodes at simulation time ``t``.

    Animated force components are sampled individually and accumulated into a
    single x-, y-, and z-force vector.
    """
    fx = 0.0
    fy = 0.0
    fz = 0.0
    animation_times = _get_animation_times(simulation)
    nearest_indices: dict[tuple[int, int], int] = {}

    for force_node in simulation.get("forces", []):
        if force_node.get("node_type") == "CONTINUUM_FLOW_FORCE_CONSTANT_NODE":
            force = force_node.get("force") or {}

            fx += get_animated_node_value(
                force_node,
                "fx",
                t,
                force.get("x", 0.0),
                animation_times,
                nearest_indices,
            )
            fy += get_animated_node_value(
                force_node,
                "fy",
                t,
                force.get("y", 0.0),
                animation_times,
                nearest_indices,
            )
            fz += get_animated_node_value(
                force_node,
                "fz",
                t,
                force.get("z", 0.0),
                animation_times,
                nearest_indices,
            )

    return fx, fy, fz


def swirl_force(simulation: dict[str, Any], t: float) -> Any:
    """
    Pack active swirl-force nodes into the GPU input layout.

    Each row stores strength, world-space origin, normalized-axis input, and
    radius. The accompanying Boolean indicates whether any swirl nodes exist.
    """
    swirl_nodes = []
    animation_times = _get_animation_times(simulation)
    nearest_indices: dict[tuple[int, int], int] = {}

    for node in simulation.get("forces", []):
        if node.get("node_type") != "CONTINUUM_FLOW_FORCE_SWIRL_NODE":
            continue

        strength = get_animated_node_value(
            node, "strength", t, 0.0, animation_times, nearest_indices
        )
        origin = get_animated_node_value(
            node, "origin", t, [0.0, 0.0, 0.0], animation_times, nearest_indices
        )
        axis = get_animated_node_value(
            node, "axis", t, [0.0, 0.0, 1.0], animation_times, nearest_indices
        )
        radius = get_animated_node_value(
            node, "radius", t, 0.0, animation_times, nearest_indices
        )

        origin = np.asarray(origin, dtype=np.float32)
        axis = np.asarray(axis, dtype=np.float32)

        swirl_nodes.append(
            [
                strength,
                origin[0],
                origin[1],
                origin[2],
                axis[0],
                axis[1],
                axis[2],
                radius,
            ]
        )

    swirl_nodes = np.asarray(swirl_nodes, dtype=np.float32).reshape((-1, 8))

    return swirl_nodes, swirl_nodes.shape[0] > 0


def turbulence_force(simulation: dict[str, Any], t: float) -> Any:
    turbulence_nodes = []
    animation_times = _get_animation_times(simulation)
    nearest_indices: dict[tuple[int, int], int] = {}

    for node in simulation.get("forces", []):
        if node.get("node_type") != "CONTINUUM_FLOW_FORCE_TURBULENCE_NODE":
            continue

        amplitude = get_animated_node_value(
            node, "amplitude", t, 0.0, animation_times, nearest_indices
        )
        frequency = max(abs(float(node.get("frequency", 1.0))), 1.0e-6)

        turbulence_nodes.append(
            [
                amplitude,
                node.get("scale", 1.0),
                node.get("seed", 0),
                math.sin(float(t) * frequency),
            ]
        )

    turbulence_nodes = np.asarray(turbulence_nodes, dtype=np.float32).reshape((-1, 4))

    return turbulence_nodes, turbulence_nodes.shape[0] > 0
