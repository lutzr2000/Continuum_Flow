import json
import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

import bpy

try:
    from . import geometry as GeometryExport
except ImportError:
    try:
        import geometry as GeometryExport
    except ImportError:
        if "__file__" in globals():
            geometry_export_path = Path(__file__).resolve().with_name("geometry.py")
        else:
            geometry_export_path = (
                Path.cwd() / "UI" / "Export" / "geometry.py"
            ).resolve()

        spec = importlib.util.spec_from_file_location(
            "continuum_flow_geometry_export", geometry_export_path
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["continuum_flow_geometry_export"] = module
        spec.loader.exec_module(module)
        GeometryExport = module

try:
    from ..Nodes import general as GeneralNodes
except ImportError:
    try:
        import general as GeneralNodes
    except ImportError:
        if "__file__" in globals():
            general_nodes_path = (
                Path(__file__).resolve().parents[1] / "Nodes" / "general.py"
            )
        else:
            general_nodes_path = (Path.cwd() / "UI" / "Nodes" / "general.py").resolve()

        spec = importlib.util.spec_from_file_location(
            "continuum_flow_general_nodes_export", general_nodes_path
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["continuum_flow_general_nodes_export"] = module
        spec.loader.exec_module(module)
        GeneralNodes = module


NODE_TREE_ID = "CONTINUUM_FLOW_NODE_TREE"


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


def _safe_animation_value(value):
    """
    Convert one sampled Blender property value to a JSON-friendly shape.
    """
    if isinstance(value, (str, bytes)):
        return value
    if hasattr(value, "__len__") and not isinstance(value, (int, float, bool)):
        return _safe_float_vector(value)
    return float(value)


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
        for fcurve in GeneralNodes._iter_action_fcurves(action):
            if getattr(fcurve, "data_path", "") == property_path:
                return True

    drivers = getattr(animation_data, "drivers", None)
    if drivers is not None:
        for fcurve in drivers:
            if getattr(fcurve, "data_path", "") == property_path:
                return True

    return False


def _sync_sampled_custom_node_animations(scene, context=None):
    """
    Force Continuum Flow custom node properties to follow their F-curves before sampling.
    """
    sync_fn = getattr(
        GeneralNodes,
        "sync_all_continuum_flow_node_animations",
        None,
    )
    if callable(sync_fn):
        sync_fn(scene)

    view_layer = getattr(context, "view_layer", None) if context is not None else None
    if view_layer is None:
        view_layer = getattr(bpy.context, "view_layer", None)
    if view_layer is not None and hasattr(view_layer, "update"):
        try:
            view_layer.update()
        except Exception:
            pass


def _sample_node_property_series(
    node, property_names, start_frame, end_frame, context=None
):
    """
    Sample one set of node properties once per Blender frame.
    """
    if not property_names:
        return {}

    scene = getattr(context, "scene", None) if context is not None else None
    if scene is None:
        scene = getattr(bpy.context, "scene", None)
    if scene is None:
        return {}

    current_frame = int(getattr(scene, "frame_current", start_frame))
    frame_numbers = list(range(int(start_frame), int(end_frame) + 1))
    sampled_values = {property_name: [] for property_name in property_names}

    try:
        for frame in frame_numbers:
            scene.frame_set(frame)
            _sync_sampled_custom_node_animations(scene, context=context)
            for property_name in property_names:
                sampled_values[property_name].append(
                    _safe_animation_value(getattr(node, property_name))
                )
    finally:
        scene.frame_set(current_frame)
        _sync_sampled_custom_node_animations(scene, context=context)

    return sampled_values


def _serialize_node_animations(node, start_frame, end_frame, fps, context=None):
    """
    Sample animated node properties once per Blender frame for kernel playback.
    """
    animated_properties = _animated_node_property_names(node)
    if not animated_properties:
        return {}
    sampled_values = _sample_node_property_series(
        node,
        animated_properties,
        start_frame,
        end_frame,
        context=context,
    )

    return {
        property_name: {
            "times": _animation_times(start_frame, end_frame, fps),
            "values": sampled_values[property_name],
        }
        for property_name in animated_properties
    }


def _serialize_animatable_value_series(node, start_frame, end_frame, context=None):
    """
    Sample only animatable UI properties that are actually animated.
    """
    animated_properties = _animated_node_property_names(node)
    if not animated_properties:
        return {}
    return _sample_node_property_series(
        node,
        animated_properties,
        start_frame,
        end_frame,
        context=context,
    )


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


def _resolve_simulation_output_fps(simulation_node, context=None):
    """
    Resolve the FPS that defines frame-to-time conversion for one simulation.
    """
    output_nodes = _linked_output_nodes(
        simulation_node, "Result", "CONTINUUM_FLOW_OUTPUT_NODE"
    )
    if output_nodes:
        return max(1, int(getattr(output_nodes[0], "fps", 24)))

    scene = getattr(context, "scene", None) if context is not None else None
    if scene is None:
        scene = getattr(bpy.context, "scene", None)
    render = getattr(scene, "render", None)
    if render is not None:
        return max(1, int(getattr(render, "fps", 24)))
    return 24


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
    geometry_nodes, start_frame, end_frame, fps, context=None
):
    """
    Sample evaluated world transforms for linked geometry objects once per Blender frame.
    """
    scene = getattr(context, "scene", None) if context is not None else None
    if scene is None:
        scene = getattr(bpy.context, "scene", None)
    if scene is None:
        return {}

    current_frame = int(getattr(scene, "frame_current", start_frame))
    frame_numbers = list(range(int(start_frame), int(end_frame) + 1))
    transform_samples = {}

    try:
        for frame in frame_numbers:
            scene.frame_set(frame)
            depsgraph = (
                context.evaluated_depsgraph_get()
                if context is not None
                else bpy.context.evaluated_depsgraph_get()
            )
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
    "CONTINUUM_FLOW_FORCE_POINT_NODE": (
        ("strength", "strength", float, 0.0),
        ("origin", "origin", _safe_float_vector, None),
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
        export_name: converter(getattr(node, property_name, default))
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


def _serialize_physics_node(node, start_frame, end_frame, fps, context=None):
    """
    Serialize one physics node.
    """
    return {
        "node_name": node.name,
        **{
            section_name: _serialize_named_fields(node, field_specs)
            for section_name, field_specs in _PHYSICS_SECTION_FIELDS.items()
        },
        "animations": _serialize_node_animations(
            node,
            start_frame,
            end_frame,
            fps,
            context=context,
        ),
        "animated_values": _serialize_animatable_value_series(
            node,
            start_frame,
            end_frame,
            context=context,
        ),
    }


def _build_geometry_payload(
    geometry_nodes, start_frame, end_frame, fps, context=None, geometry_storage_dir=None
):
    """
    Build shared geometry export data for source and obstacle nodes.
    """
    depsgraph = context.evaluated_depsgraph_get() if context is not None else None
    geometry_exports = GeometryExport.export_geometry_nodes(
        geometry_nodes,
        depsgraph=depsgraph,
        storage_dir=geometry_storage_dir,
    )
    transform_samples = _sample_geometry_object_transforms(
        geometry_nodes,
        start_frame,
        end_frame,
        fps,
        context=context,
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
    node, start_frame, end_frame, fps, context=None, geometry_storage_dir=None
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
        context=context,
        geometry_storage_dir=geometry_storage_dir,
    )

    return {
        "node_name": node.name,
        "fuel": float(node.fuel),
        "smoke": float(node.smoke),
        "temperature": float(node.temperature),
        "extra_pressure": float(getattr(node, "extra_pressure", 0.0)),
        "velocity_space": str(getattr(node, "velocity_space", "WORLD")),
        "velocity": _safe_float_vector(node.velocity),
        "animations": _serialize_node_animations(
            node,
            start_frame,
            end_frame,
            fps,
            context=context,
        ),
        "animated_values": _serialize_animatable_value_series(
            node,
            start_frame,
            end_frame,
            context=context,
        ),
        **geometry_payload,
    }


def _serialize_obstacle_node(
    node, start_frame, end_frame, fps, context=None, geometry_storage_dir=None
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
        context=context,
        geometry_storage_dir=geometry_storage_dir,
    )

    return {
        "node_name": node.name,
        **geometry_payload,
    }


def _serialize_force_node(node, start_frame, end_frame, fps, context=None):
    """
    Serialize one force node by its concrete subtype.
    """
    base_data = {
        "node_name": node.name,
        "node_type": node.bl_idname,
        "animations": _serialize_node_animations(
            node,
            start_frame,
            end_frame,
            fps,
            context=context,
        ),
        "animated_values": _serialize_animatable_value_series(
            node,
            start_frame,
            end_frame,
            context=context,
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
    }


def _build_simulation_entry(simulation_node, context=None, geometry_storage_dir=None):
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
    simulation_fps = _resolve_simulation_output_fps(simulation_node, context=context)

    return {
        "node_name": simulation_node.name,
        "settings": {
            "solver_backend": str(getattr(simulation_node, "solver_backend", "GPU")),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "simulation_length": _simulation_length_from_frames(
                start_frame, end_frame, simulation_fps
            ),
            "cfl": float(simulation_node.cfl),
            "iterations": int(simulation_node.iterations),
            "maccormack_factor": float(
                getattr(simulation_node, "maccormack_factor", 0.25)
            ),
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
                context=context,
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
                context=context,
                geometry_storage_dir=geometry_storage_dir,
            )
            for node in source_nodes
        ],
        "obstacles": [
            _serialize_obstacle_node(
                node,
                start_frame,
                end_frame,
                simulation_fps,
                context=context,
                geometry_storage_dir=geometry_storage_dir,
            )
            for node in obstacle_nodes
        ],
        "forces": [
            _serialize_force_node(
                node,
                start_frame,
                end_frame,
                simulation_fps,
                context=context,
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


def _resolve_node_tree(context=None):
    """
    Resolve the active Continuum Flow node tree or fall back to the first one.
    """
    if context is not None:
        space_data = getattr(context, "space_data", None)
        edit_tree = getattr(space_data, "edit_tree", None)
        if (
            edit_tree is not None
            and getattr(edit_tree, "bl_idname", "") == NODE_TREE_ID
        ):
            return edit_tree

    for node_group in bpy.data.node_groups:
        if getattr(node_group, "bl_idname", "") == NODE_TREE_ID:
            return node_group
    return None


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


def build_config_dict(context=None, geometry_storage_dir=None):
    """
    Evaluate the Continuum Flow node tree and return a grouped config dict.
    """
    node_tree = _resolve_node_tree(context)
    if node_tree is None:
        raise RuntimeError("No Continuum Flow node tree found.")

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
                context=context,
                geometry_storage_dir=geometry_storage_dir,
            )
            for node in simulation_nodes
        ],
        "geometry_nodes": _collect_tree_geometry_nodes(node_tree),
    }
    return config_dict


def export_config_dict(context=None):
    """
    Evaluate the node tree, write a JSON file, and return both path and dict.
    """
    config_dict = build_config_dict(context)
    export_directory = _resolve_export_directory(config_dict["simulations"])
    export_directory.mkdir(parents=True, exist_ok=True)

    file_name = f"{config_dict['meta']['node_tree_name']}_config_export.json"
    file_path = export_directory / file_name
    file_path.write_text(json.dumps(config_dict, indent=2), encoding="utf-8")
    return file_path, config_dict


class ContinuumFlow_OT_export_config_dict(bpy.types.Operator):
    """
    Export the evaluated Continuum Flow node tree to JSON.
    """

    bl_idname = "continuum_flow.export_config_dict"
    bl_label = "Export Config Dict"
    bl_description = (
        "Evaluate the Continuum Flow node tree and write a JSON config file"
    )

    def execute(self, context):
        try:
            file_path, config_dict = export_config_dict(context)
        except Exception as exc:
            self.report({"ERROR"}, f"Config export failed: {exc}")
            return {"CANCELLED"}

        self.report(
            {"INFO"},
            f"Exported {config_dict['meta']['simulation_count']} simulation config(s) to {file_path}",
        )
        return {"FINISHED"}


classes = (ContinuumFlow_OT_export_config_dict,)
