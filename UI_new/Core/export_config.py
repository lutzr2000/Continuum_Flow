from pathlib import Path
import bpy
import json
from datetime import datetime, timezone
from . import export_geometry
from ..Interface.node_tree import ContinuumFlowNodeTree

NODE_TREE_ID = "CONTINUUM_FLOW_NODE_TREE"
_SYNC_DEBUGGED_ACTIONS = set()
PERCENTAGE_MAPPED_PROPERTY_RANGES = {
    "temperature_dissipation": (0.0, 0.5),
    "temperature_production_rate": (0.0, 0.05),
    "buoyancy": (0.0, 0.01),
    "expansion_rate": (0.0, 0.01),
    "smoke_dissipation": (0.0, 1.0),
    "smoke_production_rate": (0.0, 0.05),
    "fuel_dissipation": (0.0, 1.0),
    "fuel_burn_rate": (0.0, 1.0),
    "vorticity": (0.0, 1.0),
}


def _serialize_animation_payload(node, start_frame, end_frame, fps):
    props = _animated_node_property_names(node)
    if not props:
        return {}, {}

    sampled = _sample_node_property_series(node, props, start_frame, end_frame)
    times = _animation_times(start_frame, end_frame, fps)

    animations = {
        name: {"times": times, "values": sampled[name]}
        for name in props
    }
    return animations, sampled


def _mapped_percentage_to_actual(property_name, value):
    """
    Convert one UI percentage value back into its solver-facing numeric range.
    """
    if property_name not in PERCENTAGE_MAPPED_PROPERTY_RANGES:
        return value

    minimum, maximum = PERCENTAGE_MAPPED_PROPERTY_RANGES[property_name]
    return minimum + ((float(value) / 100.0) * (maximum - minimum))


def _serialize_node_property_value(node, property_name, default=None):
    """
    Serialize one scalar node property, applying UI-to-solver remapping when needed.
    """
    value = getattr(node, property_name, default)
    if property_name in PERCENTAGE_MAPPED_PROPERTY_RANGES:
        return float(_mapped_percentage_to_actual(property_name, value))
    return float(value)


def _serialize_animation_value_for_property(property_name, value):
    """
    Serialize one sampled animation value, applying UI-to-solver remapping when needed.
    """
    if isinstance(value, (str, bytes)):
        return value
    if hasattr(value, "__len__") and not isinstance(value, (int, float, bool)):
        return _safe_float_vector(value)
    if property_name in PERCENTAGE_MAPPED_PROPERTY_RANGES:
        return float(_mapped_percentage_to_actual(property_name, value))
    return float(value)


def _safe_float_vector(value):
    """
    Convert Blender float vectors to plain Python float lists.
    """
    return [float(component) for component in value]


def _safe_float_matrix(matrix):
    """
    Convert a Blender matrix into nested Python float lists.
    """
    return [[float(component) for component in row] for row in matrix]


def _animation_times(start_frame, end_frame, fps):
    """
    Return the shared simulation time axis derived from the Blender frame range.
    """
    return [
        float(frame - int(start_frame)) / float(fps)
        for frame in range(int(start_frame), int(end_frame) + 1)
    ]


def _animated_property_names(node):
    """
    Return node property names that are likely intended for animation export.
    """
    property_names = []
    seen_names = set()

    # Prefer property names that the custom node classes already expose in the UI.
    for group in getattr(node, "property_groups", ()):
        if len(group) < 2:
            continue
        for property_name in group[1]:
            if property_name in seen_names or not hasattr(node, property_name):
                continue
            property_names.append(property_name)
            seen_names.add(property_name)

    for attr_name in ("scalar_property_names", "draw_property_names"):
        for property_name in getattr(node, attr_name, ()):
            if property_name in seen_names or not hasattr(node, property_name):
                continue
            property_names.append(property_name)
            seen_names.add(property_name)

    # Fall back to Blender RNA metadata when available.
    for prop in node.bl_rna.properties:
        if prop.identifier == "rna_type" or prop.is_readonly:
            continue
        is_animatable = bool(getattr(prop, "is_animatable", False))
        if not is_animatable and "ANIMATABLE" not in getattr(prop, "options", set()):
            continue
        if prop.identifier in seen_names or not hasattr(node, prop.identifier):
            continue
        property_names.append(prop.identifier)
        seen_names.add(prop.identifier)
    return property_names


def _animated_node_property_names(node):
    """
    Return animatable node property names that have matching F-curves or drivers.
    """
    property_names = []
    for property_name in _animated_property_names(node):
        if _node_property_is_animated(node, property_name):
            property_names.append(property_name)
    return property_names


def _iter_action_fcurves(action):
    """
    Yield F-curves from legacy or layered Blender actions.
    """
    if action is None:
        return

    legacy_fcurves = getattr(action, "fcurves", None)
    if legacy_fcurves is not None:
        for fcurve in legacy_fcurves:
            yield fcurve
        return

    layers = getattr(action, "layers", None)
    if layers is None:
        return

    for layer in layers:
        for strip in getattr(layer, "strips", ()):
            channelbags = getattr(strip, "channelbags", None)
            if channelbags is None:
                continue
            for channelbag in channelbags:
                for fcurve in getattr(channelbag, "fcurves", ()):
                    yield fcurve


def _node_property_is_animated(node, property_name):
    """
    Return whether the node property is driven by F-curves or drivers.
    """
    node_tree = getattr(node, "id_data", None)
    animation_data = getattr(node_tree, "animation_data", None)
    if animation_data is None:
        return False

    property_path = node.path_from_id(property_name)
    action = getattr(animation_data, "action", None)
    if action is not None:
        for fcurve in _iter_action_fcurves(action):
            if getattr(fcurve, "data_path", "") == property_path:
                return True

    drivers = getattr(animation_data, "drivers", None)
    if drivers is not None:
        for fcurve in drivers:
            if getattr(fcurve, "data_path", "") == property_path:
                return True

    return False


def _set_node_property_component(node, property_name, value, array_index):
    """
    Assign one scalar F-curve value to a scalar or vector node property.
    """
    current_value = getattr(node, property_name)
    is_vector_like = hasattr(current_value, "__len__") and not isinstance(
        current_value, (str, bytes)
    )

    if array_index < 0 or not is_vector_like:
        setattr(node, property_name, value)
        return

    current_value = list(current_value)
    if array_index >= len(current_value):
        return
    current_value[array_index] = value
    setattr(node, property_name, current_value)


def _iter_keyframeable_node_properties(node_tree):
    """
    Yield writable Continuum Flow node properties that expose a valid RNA path.
    """
    for node in getattr(node_tree, "nodes", ()):
        for prop in node.bl_rna.properties:
            if prop.identifier == "rna_type" or prop.is_readonly:
                continue
            yield node, prop.identifier


def _sync_node_tree_animation(node_tree, frame_value):
    """
    Evaluate one Continuum Flow node-tree action and push values onto node properties.
    """
    animation_data = getattr(node_tree, "animation_data", None)
    action = getattr(animation_data, "action", None)
    if action is None:
        return

    fcurves = list(_iter_action_fcurves(action))
    if not fcurves:
        return

    property_path_map = {}
    for node, property_name in _iter_keyframeable_node_properties(node_tree):
        try:
            property_path_map[node.path_from_id(property_name)] = (node, property_name)
        except Exception:
            continue

    matched_any_curve = False
    for fcurve in fcurves:
        property_target = property_path_map.get(getattr(fcurve, "data_path", ""))
        if property_target is None:
            continue

        node, property_name = property_target
        evaluated_value = fcurve.evaluate(frame_value)
        _set_node_property_component(
            node,
            property_name,
            evaluated_value,
            int(getattr(fcurve, "array_index", -1)),
        )
        matched_any_curve = True

    action_key = str(getattr(action, "name_full", getattr(action, "name", "")))
    if not matched_any_curve and action_key not in _SYNC_DEBUGGED_ACTIONS:
        _SYNC_DEBUGGED_ACTIONS.add(action_key)
        print("Continuum Flow animation sync: no matching node property paths found.")
        print(f"  Node tree: {getattr(node_tree, 'name', '<unnamed>')}")
        print("  Known animatable paths:")
        for known_path in sorted(property_path_map.keys()):
            print(f"    {known_path}")
        print("  Action F-Curve paths:")
        for fcurve in fcurves:
            print(
                f"    {getattr(fcurve, 'data_path', '')} [{int(getattr(fcurve, 'array_index', -1))}]"
            )


def sync_all_continuum_flow_node_animations(scene=None):
    """
    Evaluate all Continuum Flow node-tree animations for the current frame.
    """
    scene = getattr(bpy.context, "scene", None)

    frame_value = float(getattr(scene, "frame_current", 0))
    for node_tree in _iter_continuum_flow_node_trees():
        _sync_node_tree_animation(node_tree, frame_value)


def _iter_continuum_flow_node_trees():
    """
    Yield all Continuum Flow node trees in the current file.
    """
    for node_tree in bpy.data.node_groups:
        if getattr(node_tree, "bl_idname", "") == ContinuumFlowNodeTree.bl_idname:
            yield node_tree


def _sync_sampled_custom_node_animations(scene):
    """
    Force Continuum Flow custom node properties to follow their F-curves before sampling.
    """
    sync_all_continuum_flow_node_animations(scene)

    view_layer = getattr(bpy.context, "view_layer", None)
    if view_layer is not None and hasattr(view_layer, "update"):
        try:
            view_layer.update()
        except Exception:
            pass


def _sample_node_property_series(
    node, property_names, start_frame, end_frame):
    """
    Sample one set of node properties once per Blender frame.
    """
    if not property_names:
        return {}

    scene = getattr(bpy.context, "scene", None)
    current_frame = int(getattr(scene, "frame_current", start_frame))
    frame_numbers = list(range(int(start_frame), int(end_frame) + 1))
    sampled_values = {property_name: [] for property_name in property_names}

    try:
        for frame in frame_numbers:
            scene.frame_set(frame)
            _sync_sampled_custom_node_animations(scene)
            for property_name in property_names:
                sampled_values[property_name].append(
                    _serialize_animation_value_for_property(
                        property_name, getattr(node, property_name)
                    )
                )
    finally:
        scene.frame_set(current_frame)
        _sync_sampled_custom_node_animations(scene)

    return sampled_values


def _linked_socket_nodes(
    socket_collection, socket_name, linked_node_attr, expected_idname=None
):
    """
    Return linked nodes from one socket collection and link direction.
    """
    socket = socket_collection.get(socket_name)
    if socket is None or not socket.is_linked:
        return []

    linked_nodes = []
    for link in socket.links:
        linked_node = getattr(link, linked_node_attr, None)
        if linked_node is None:
            continue
        if (
            expected_idname is not None
            and getattr(linked_node, "bl_idname", "") != expected_idname
        ):
            continue
        linked_nodes.append(linked_node)
    return linked_nodes


def _linked_input_nodes(node, socket_name, expected_idname=None):
    """
    Return upstream nodes connected to the given input socket.
    """
    return _linked_socket_nodes(
        node.inputs, socket_name, "from_node", expected_idname=expected_idname
    )


def _linked_output_nodes(node, socket_name, expected_idname=None):
    """
    Return downstream nodes connected to the given output socket.
    """
    return _linked_socket_nodes(
        node.outputs, socket_name, "to_node", expected_idname=expected_idname
    )


def _resolve_simulation_output_fps(simulation_node):
    """
    Resolve the FPS that defines frame-to-time conversion for one simulation.
    """
    output_nodes = _linked_output_nodes(
        simulation_node, "Result", "CONTINUUM_FLOW_OUTPUT_NODE"
    )
    if output_nodes:
        return max(1, int(getattr(output_nodes[0], "fps", 24)))

    scene = getattr(bpy.context, "scene", None)
    render = getattr(scene, "render", None)
    return max(1, int(getattr(render, "fps", 24)))


def _simulation_length_from_frames(start_frame, end_frame, fps):
    """
    Convert an inclusive exported frame range to kernel simulation seconds.
    """
    if end_frame <= start_frame:
        raise ValueError("Simulation end frame must be greater than the start frame.")
    return float((end_frame - start_frame) + 1) / float(fps)


def _linked_geometry_nodes(node):
    """
    Return linked geometry nodes for source and obstacle nodes.
    """
    return _linked_input_nodes(node, "Geometry", "CONTINUUM_FLOW_GEOMETRY_NODE")


def _sample_geometry_object_transforms(
    geometry_nodes, start_frame, end_frame, fps
):
    """
    Sample evaluated world transforms for linked geometry objects once per Blender frame.
    """
    scene = getattr(bpy.context, "scene", None)

    current_frame = int(getattr(scene, "frame_current", start_frame))
    frame_numbers = list(range(int(start_frame), int(end_frame) + 1))
    transform_samples = {}

    try:
        for frame in frame_numbers:
            scene.frame_set(frame)
            depsgraph = bpy.context.evaluated_depsgraph_get()

            for geometry_node in geometry_nodes:
                source_object = getattr(geometry_node, "source_object", None)
                if source_object is None:
                    continue

                object_eval = source_object.evaluated_get(depsgraph)
                matrix_world = object_eval.matrix_world.copy()
                object_name = source_object.name
                object_samples = transform_samples.setdefault(
                    object_name,
                    {
                        "times": [
                            float(sample_frame - int(start_frame)) / float(fps)
                            for sample_frame in frame_numbers
                        ],
                        "matrices_world": [],
                    },
                )
                object_samples["matrices_world"].append(
                    _safe_float_matrix(matrix_world)
                )
    finally:
        scene.frame_set(current_frame)

    return transform_samples


_DOMAIN_BOUNDARY_AXES = ("x_low", "x_high", "y_low", "y_high", "z_low", "z_high")
_PHYSICS_SECTION_FIELDS = {
    "fluid": (
        ("density", "fluid_density", float, 0.0),
        ("viscosity", "fluid_viscosity", float, 0.0),
    ),
    "temperature": (
        ("dissipation", "temperature_dissipation", float, 0.0),
        ("production_rate", "temperature_production_rate", float, 1.0),
        ("reference_temperature", "reference_temperature", float, 0.0),
        ("buoyancy", "buoyancy", float, 0.0),
        ("expansion_rate", "expansion_rate", float, 0.0),
    ),
    "smoke": (
        ("dissipation", "smoke_dissipation", float, 0.0),
        ("production_rate", "smoke_production_rate", float, 1.0),
    ),
    "fuel": (
        ("dissipation", "fuel_dissipation", float, 0.0),
        ("burn_rate", "fuel_burn_rate", float, 0.0),
        ("ignition_temperature", "fuel_ignition_temperature", float, 0.0),
        ("minimum_oxygen_concentration", "minimum_oxygen_concentration", float, 0.0),
    ),
    "extras": (("vorticity", "vorticity", float, 0.0),),
}
_FORCE_NODE_FIELDS = {
    "CONTINUUM_FLOW_FORCE_CONSTANT_NODE": (
        ("force", (("x", "fx"), ("y", "fy"), ("z", "fz"))),
    ),
    "CONTINUUM_FLOW_FORCE_SWIRL_NODE": (
        ("strength", "strength", float, 0.0),
        ("origin", "origin", _safe_float_vector, None),
        ("axis", "axis", _safe_float_vector, None),
        ("radius", "radius", float, 0.0),
    ),
    "CONTINUUM_FLOW_FORCE_TURBULENCE_NODE": (
        ("scale", "scale", float, 0.0),
        ("frequency", "frequency", float, 0.0),
        ("amplitude", "amplitude", float, 0.0),
        ("seed", "seed", int, 0),
    ),
}
_OUTPUT_FIELD_SPECS = (
    ("velocity", "export_velocity", "sparse_velocity", True),
    ("pressure", "export_p", "sparse_p", False),
    ("temperature", "export_t", "sparse_t", False),
    ("smoke", "export_smoke", "sparse_smoke", False),
    ("fuel", "export_fuel", "sparse_fuel", False),
    ("flame", "export_flame", "sparse_flame", False),
)


def _serialize_named_fields(node, field_specs):
    """
    Serialize a flat set of named fields from one node via declarative specs.
    """
    return {
        export_name: (
            _serialize_node_property_value(node, property_name, default)
            if converter is float
            else converter(getattr(node, property_name, default))
        )
        for export_name, property_name, converter, default in field_specs
    }


def _serialize_nested_vector(node, component_names):
    """
    Serialize one nested vector object from multiple scalar node properties.
    """
    return {
        component_name: float(getattr(node, property_name))
        for component_name, property_name in component_names
    }


def _serialize_domain_node(node):
    """
    Serialize one domain node.
    """
    return {
        "node_name": node.name,
        "resolution": float(node.resolution),
        "grid": {
            "nx": int(node.nx),
            "ny": int(node.ny),
            "nz": int(node.nz),
        },
        "boundary_conditions": {
            axis: {
                "type": getattr(node, f"{axis}_bc"),
                "velocity": _safe_float_vector(getattr(node, f"{axis}_velocity")),
            }
            for axis in _DOMAIN_BOUNDARY_AXES
        },
    }


def _serialize_physics_node(node, start_frame, end_frame, fps):
    """
    Serialize one physics node.
    """
    return {
        "node_name": node.name,
        **{
            section_name: _serialize_named_fields(node, field_specs)
            for section_name, field_specs in _PHYSICS_SECTION_FIELDS.items()
        },
        "animations": _serialize_animation_payload(
            node,
            start_frame,
            end_frame,
            fps,
        ),
        "animated_values": _serialize_animation_payload(
            node,
            start_frame,
            end_frame,
            fps,
        ),
    }


def _build_geometry_payload(
    geometry_nodes, start_frame, end_frame, fps
):
    """
    Build shared geometry export data for source and obstacle nodes.
    """
    geometry_exports = export_geometry.export_geometry_nodes(
        geometry_nodes,
    )
    transform_samples = _sample_geometry_object_transforms(
        geometry_nodes,
        start_frame,
        end_frame,
        fps,
    )
    for geometry_export in geometry_exports:
        object_name = geometry_export.get("object_name")
        geometry_export["transform_animation"] = transform_samples.get(object_name, {})

    return {
        "geometry_inputs": _linked_geometry_names_from_nodes(geometry_nodes),
        "shape": "mesh" if geometry_exports else "empty",
        "mesh": {
            "objects": geometry_exports,
        },
    }


def _linked_geometry_names_from_nodes(geometry_nodes):
    """
    Return node/object name pairs for already resolved geometry nodes.
    """
    return [
        {
            "node_name": geometry_node.name,
            "object_name": getattr(
                getattr(geometry_node, "source_object", None), "name", None
            ),
        }
        for geometry_node in geometry_nodes
    ]


def _serialize_source_node(
    node, start_frame, end_frame, fps
):
    """
    Serialize one source node, including linked geometry names.
    """
    geometry_nodes = _linked_geometry_nodes(node)
    geometry_payload = _build_geometry_payload(
        geometry_nodes,
        start_frame,
        end_frame,
        fps,
    )

    return {
        "node_name": node.name,
        "fuel": float(node.fuel),
        "smoke": float(node.smoke),
        "temperature": float(node.temperature),
        "extra_pressure": float(getattr(node, "extra_pressure", 0.0)),
        "velocity_space": str(getattr(node, "velocity_space", "WORLD")),
        "velocity": _safe_float_vector(node.velocity),
        "animations": _serialize_animation_payload(
            node,
            start_frame,
            end_frame,
            fps,
        ),
        "animated_values": _serialize_animation_payload(
            node,
            start_frame,
            end_frame,
            fps,
        ),
        **geometry_payload,
    }


def _serialize_obstacle_node(
    node, start_frame, end_frame, fps
):
    """
    Serialize one obstacle node, including linked geometry names.
    """
    geometry_nodes = _linked_geometry_nodes(node)
    geometry_payload = _build_geometry_payload(
        geometry_nodes,
        start_frame,
        end_frame,
        fps,
    )

    return {
        "node_name": node.name,
        **geometry_payload,
    }


def _serialize_force_node(node, start_frame, end_frame, fps):
    """
    Serialize one force node by its concrete subtype.
    """
    base_data = {
        "node_name": node.name,
        "node_type": node.bl_idname,
        "animations": _serialize_animation_payload(
            node,
            start_frame,
            end_frame,
            fps,
        ),
        "animated_values": _serialize_animation_payload(
            node,
            start_frame,
            end_frame,
            fps,
        ),
    }
    for field_spec in _FORCE_NODE_FIELDS.get(node.bl_idname, ()):
        if len(field_spec) == 2:
            field_name, component_names = field_spec
            base_data[field_name] = _serialize_nested_vector(node, component_names)
            continue

        field_name, property_name, converter, default = field_spec
        base_data[field_name] = converter(getattr(node, property_name, default))

    return base_data


def _serialize_output_node(node):
    """
    Serialize one output node.
    """
    output_path = bpy.path.abspath(node.output_path) if node.output_path else ""
    return {
        "node_name": node.name,
        "fps": int(node.fps),
        "precision": str(getattr(node, "output_precision", "float16")),
        "fields": {
            field_name: {
                "enabled": bool(getattr(node, enabled_property, enabled_default)),
                "sparse": bool(getattr(node, sparse_property, False)),
            }
            for field_name, enabled_property, sparse_property, enabled_default in _OUTPUT_FIELD_SPECS
        },
        "performance": {
            "writer_processes": int(getattr(node, "writer_processes", 4)),
        },
        "output_path": output_path,
    }


def _serialize_viewer_node(node):
    """
    Serialize one viewer node.
    """
    return {
        "node_name": node.name,
        "live_preview": bool(getattr(node, "live_preview", True)),
        "debug": bool(getattr(node, "debug", False)),
    }


def _build_simulation_entry(simulation_node):
    """
    Build a grouped config entry for one simulation node.
    """
    domain_nodes = _linked_input_nodes(
        simulation_node, "Domain", "CONTINUUM_FLOW_DOMAIN_NODE"
    )
    physics_nodes = _linked_input_nodes(
        simulation_node, "Physics", "CONTINUUM_FLOW_PHYSICS_NODE"
    )
    source_nodes = _linked_input_nodes(
        simulation_node, "Source", "CONTINUUM_FLOW_SOURCE_NODE"
    )
    obstacle_nodes = _linked_input_nodes(
        simulation_node, "Obstacles", "CONTINUUM_FLOW_OBSTACLE_NODE"
    )
    force_nodes = _linked_input_nodes(simulation_node, "Forces")

    output_nodes = _linked_output_nodes(
        simulation_node, "Result", "CONTINUUM_FLOW_OUTPUT_NODE"
    )
    viewer_nodes = _linked_output_nodes(
        simulation_node, "Result", "CONTINUUM_FLOW_VIEWER_NODE"
    )
    start_frame = int(getattr(simulation_node, "start_frame", 1))
    end_frame = int(getattr(simulation_node, "end_frame", start_frame + 1))
    simulation_fps = _resolve_simulation_output_fps(simulation_node)

    return {
        "node_name": simulation_node.name,
        "settings": {
            "solver_backend": str(getattr(simulation_node, "solver_backend", "GPU")),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "simulation_length": _simulation_length_from_frames(
                start_frame, end_frame, simulation_fps
            ),
            "cfl": float(getattr(simulation_node, "cfl", 10.0)),
            "iterations": int(simulation_node.iterations),
            "simulate_sparsely": bool(
                getattr(simulation_node, "simulate_sparsely", True)
            ),
            "adaptive_domain_threshold": float(
                getattr(simulation_node, "adaptive_domain_threshold", 0.001)
            ),
        },
        "animation_timeline": {
            "fps": simulation_fps,
            "times": _animation_times(start_frame, end_frame, simulation_fps),
        },
        "domain": _serialize_domain_node(domain_nodes[0]) if domain_nodes else None,
        "physics": (
            _serialize_physics_node(
                physics_nodes[0],
                start_frame,
                end_frame,
                simulation_fps,
            )
            if physics_nodes
            else None
        ),
        "sources": [
            _serialize_source_node(
                node,
                start_frame,
                end_frame,
                simulation_fps,
            )
            for node in source_nodes
        ],
        "obstacles": [
            _serialize_obstacle_node(
                node,
                start_frame,
                end_frame,
                simulation_fps,
            )
            for node in obstacle_nodes
        ],
        "forces": [
            _serialize_force_node(
                node,
                start_frame,
                end_frame,
                simulation_fps,
            )
            for node in force_nodes
        ],
        "outputs": [_serialize_output_node(node) for node in output_nodes],
        "viewers": [_serialize_viewer_node(node) for node in viewer_nodes],
    }


def _collect_tree_geometry_nodes(node_tree):
    """
    Collect all geometry nodes for visibility in the exported JSON.
    """
    geometry_entries = []
    for node in node_tree.nodes:
        if getattr(node, "bl_idname", "") != "CONTINUUM_FLOW_GEOMETRY_NODE":
            continue
        source_object = getattr(node, "source_object", None)
        geometry_entries.append(
            {
                "node_name": node.name,
                "object_name": (
                    source_object.name if source_object is not None else None
                ),
            }
        )
    return geometry_entries


def _resolve_node_tree():
    """
    Resolve the active Continuum Flow node tree or fall back to the first one.
    """
    for node_group in bpy.data.node_groups:
        if getattr(node_group, "bl_idname", "") == NODE_TREE_ID:
            return node_group


def build_config_dict():
    """
    Evaluate the Continuum Flow node tree and return a grouped config dict.
    """

    node_tree = _resolve_node_tree()

    simulation_nodes = [
        node
        for node in node_tree.nodes
        if getattr(node, "bl_idname", "") == "CONTINUUM_FLOW_SIMULATION_NODE"
    ]

    config_dict = {
        "meta": {
            "node_tree_name": node_tree.name,
            "exported_at_utc": datetime.now(timezone.utc).isoformat(),
            "simulation_count": len(simulation_nodes),
        },
        "simulations": [
            _build_simulation_entry(
                node,
            )
            for node in simulation_nodes
        ],
        "geometry_nodes": _collect_tree_geometry_nodes(node_tree),
    }
    return config_dict


##################################################
# Only necessary for debug
##################################################


def _resolve_export_directory(simulation_entries):
    """
    Choose a directory for the exported JSON.
    """
    for simulation_entry in simulation_entries:
        for output_entry in simulation_entry["outputs"]:
            output_path = output_entry.get("output_path", "")
            if output_path:
                return Path(output_path)

    blend_directory = bpy.path.abspath("//")
    if blend_directory:
        return Path(blend_directory)
    return Path.cwd()


def export_config_dict(config_dict):
    """
    Evaluate the node tree, write a JSON file, and return both path and dict.
    """
    export_directory = _resolve_export_directory(config_dict["simulations"])
    export_directory.mkdir(parents=True, exist_ok=True)

    file_name = f"{config_dict['meta']['node_tree_name']}_config_export.json"
    file_path = export_directory / file_name
    file_path.write_text(json.dumps(config_dict, indent=2), encoding="utf-8")
    return file_path, config_dict