from pathlib import Path
import re
import bpy
import json
from datetime import datetime, timezone
from . import export_geometry

NODE_TREE_ID = "CONTINUUM_FLOW_NODE_TREE"
ANIMATABLE_PROPERTIES = {
    "CONTINUUM_FLOW_PHYSICS_NODE": (
        "fluid_density",
        "fluid_viscosity",
        "temperature_dissipation",
        "temperature_production_rate",
        "reference_temperature",
        "buoyancy",
        "expansion_rate",
        "smoke_dissipation",
        "smoke_production_rate",
        "fuel_dissipation",
        "fuel_burn_rate",
        "fuel_ignition_temperature",
        "burn_noise_scale",
        "burn_noise_amplitude",
        "vorticity",
    ),

    "CONTINUUM_FLOW_SOURCE_NODE": (
        "fuel",
        "smoke",
        "temperature",
        "extra_pressure",
        "noise_scale",
        "noise_amplitude",
        "velocity",
    ),

    "CONTINUUM_FLOW_FORCE_SWIRL_NODE": (
        "strength",
        "origin",
        "axis",
        "radius",
    ),
}

PERCENTAGE_MAPPING = {
    "temperature_dissipation": (0.0, 10),
    "temperature_production_rate": (0.0, 1),
    "buoyancy": (0.0, 0.01),
    "expansion_rate": (0.0, 0.1),
    "smoke_dissipation": (0.0, 10),
    "smoke_production_rate": (0.0, 1),
    "fuel_dissipation": (0.0, 10),
    "fuel_burn_rate": (0.0, 20.0),
    "burn_noise_amplitude": (0.0, 1.0),
    "vorticity": (0.0, 1.0),
}


#-------------- export ----------------
def export_config_dict(config_dict):
    """
    Prepare a fresh bake subfolder and export the STL geometry assets into it.
    """
    export_root_directory = Path(config_dict["simulation"][0]["outputs"][0]["output_path"])
    export_root_directory.mkdir(parents=True, exist_ok=True)

    export_directory = create_bake_subdirectory(export_root_directory, config_dict)
    set_subdirectory_paths(config_dict, export_directory)
    export_geomtry_stls(config_dict, export_directory)

    return export_directory, config_dict


def create_bake_subdirectory(base_directory, config_dict):
    """
    Create a fresh subfolder for one bake run inside the configured output root.
    """
    node_tree_name = (config_dict.get("meta") or {}).get("node_tree_name")

    node_tree_name = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        str(node_tree_name or "bake").strip(),
    )
    node_tree_name = node_tree_name.strip("._-") or "bake"

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    bake_directory = (
        Path(base_directory)
        / f"{node_tree_name}_bake_{timestamp}"
    )

    bake_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    return bake_directory


def set_subdirectory_paths(config_dict, bake_directory):
    """
    Point every simulation output at the concrete bake subfolder for this run.
    """
    bake_directory = str(Path(bake_directory).resolve())
    for simulation_entry in config_dict.get("simulations", []):
        for output_entry in simulation_entry.get("outputs", []):
            output_entry["output_path"] = bake_directory


def export_geomtry_stls(config_dict, export_directory):
    geometry_dir = Path(export_directory) / "geometry"
    geometry_dir.mkdir(parents=True, exist_ok=True)

    depsgraph = bpy.context.evaluated_depsgraph_get()
    simulation = config_dict["simulation"][0]

    seen = set()

    for group_name in ("sources", "obstacles"):
        for entry in simulation.get(group_name, []):
            for geometry_input in entry.get("geometry_inputs", []):
                object_name = geometry_input.get("object_name")

                if not object_name or object_name in seen:
                    continue

                seen.add(object_name)

                source_object = bpy.data.objects.get(object_name)
                if source_object is None:
                    continue

                export_geometry.export_object_as_local_stl(
                    source_object,
                    geometry_dir,
                    depsgraph=depsgraph,
                )


#-------------- build ----------------
def build_config_dict(context, simulation_node):
    """
    Build the genetal simulation config 
    """
    node_tree = getattr(simulation_node, "id_data")

    config_dict = {
        "meta": {
            "node_tree_name": node_tree.name,
            "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "simulation": [build_entries(simulation_node)],
        "geometry_nodes": get_geometry_nodes(node_tree),
    }
    return config_dict


def build_entries(simulation_node):
    """
    Build a grouped config for the connected nodes
    """
    domain_node = linked_node(
        simulation_node,
        "Domain",
        "input",
    )

    physics_node = linked_node(
        simulation_node,
        "Physics",
        "input",
    )

    source_nodes = multiple_linked_nodes(
        simulation_node,
        "Source",
        "input",
    )

    obstacle_nodes = multiple_linked_nodes(
        simulation_node,
        "Obstacles",
        "input",
    )

    force_nodes = multiple_linked_nodes(
        simulation_node,
        "Forces",
        "input",
    )
    result_nodes = multiple_linked_nodes(
        simulation_node,
        "Result",
        "output",
    )

    output_node = next(
        (
            node
            for node in result_nodes
            if node.bl_idname == "CONTINUUM_FLOW_OUTPUT_NODE"
        ),
    )

    viewer_node = next(
        (
            node
            for node in result_nodes
            if node.bl_idname == "CONTINUUM_FLOW_VIEWER_NODE"
        ),
        None,
    )

    start_frame = int(getattr(simulation_node, "start_frame", 1))
    end_frame = int(getattr(simulation_node, "end_frame", start_frame + 1))
    simulation_fps = max(1, int(getattr(output_node, "fps", 24)))
    simulation_length = float(end_frame - start_frame) / float(simulation_fps)
    simulation_times = [
        float(frame - int(start_frame)) / float(simulation_fps)
        for frame in range(int(start_frame), int(end_frame))
    ]

    return {
        "node_name": simulation_node.name,
        "settings": {
            "solver_backend": str(getattr(simulation_node, "solver_backend", "CPU")),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "simulation_length":simulation_length,
            "cfl": float(getattr(simulation_node, "cfl", 10.0)),
            "iterations": int(simulation_node.iterations),
            "simulate_sparsely": bool(getattr(simulation_node, "simulate_sparsely", True)),
            "adaptive_domain_threshold": float(getattr(simulation_node, "adaptive_domain_threshold", 0.001)),
        },
        "animation_timeline": {
            "fps": simulation_fps,
            "times": simulation_times,
        },
        "domain": build_domain_node_entries(domain_node),
        "physics": (build_physics_node_entries(physics_node,start_frame,end_frame,simulation_fps,)),
        "sources": [build_source_node_entries(node,start_frame,end_frame,simulation_fps,)
            for node in source_nodes
        ],
        "obstacles": [build_obstacle_node_entries(node,start_frame,end_frame,simulation_fps,)
            for node in obstacle_nodes
        ],
        "forces": [build_force_entries(node,start_frame,end_frame,simulation_fps,)
            for node in force_nodes
        ],
        "outputs": [build_output_node_entries(output_node)],
        "viewers": [build_viewer_node_entries(viewer_node)],
    }


def build_domain_node_entries(node):
    """
    Serialize one domain node.
    """
    return {
        "resolution": float(node.resolution),
        "grid": {
            "nx": int(node.nx),
            "ny": int(node.ny),
            "nz": int(node.nz),
        },
        "boundary_conditions": {
            axis: {
                "type": getattr(node, f"{axis}_bc"),
                "velocity": safe_float_vector(getattr(node, f"{axis}_velocity")),
            }
            for axis in ("x_low", "x_high", "y_low", "y_high", "z_low", "z_high")
        },
    }


def build_physics_node_entries(node, start_frame, end_frame, fps):
    animations, animated_values = build_animations(
        node,
        start_frame,
        end_frame,
    )

    return {
        "fluid": {
            "density": physics_value(node, "fluid_density"),
            "viscosity": physics_value(node, "fluid_viscosity"),
        },

        "temperature": {
            "dissipation": physics_value(node, "temperature_dissipation"),
            "production_rate": physics_value(
                node, "temperature_production_rate"
            ),
            "reference_temperature": physics_value(
                node, "reference_temperature"
            ),
            "buoyancy": physics_value(node, "buoyancy"),
            "expansion_rate": physics_value(node, "expansion_rate"),
        },

        "smoke": {
            "dissipation": physics_value(node, "smoke_dissipation"),
            "production_rate": physics_value(
                node, "smoke_production_rate"
            ),
        },

        "fuel": {
            "dissipation": physics_value(node, "fuel_dissipation"),
            "burn_rate": physics_value(node, "fuel_burn_rate"),
            "ignition_temperature": physics_value(
                node, "fuel_ignition_temperature"
            ),
        },

        "burning": {
            "scale": physics_value(node, "burn_noise_scale"),
            "amplitude": physics_value(node, "burn_noise_amplitude"),
        },

        "extras": {
            "vorticity": physics_value(node, "vorticity"),
        },

        "animations": animations,
        "animated_values": animated_values,
    }


def physics_value(node, property_name):
    value = float(getattr(node, property_name))

    if property_name in PERCENTAGE_MAPPING:
        minimum, maximum = PERCENTAGE_MAPPING[property_name]
        value = minimum + (
            (value / 100.0) * (maximum - minimum)
        )

    return value


def build_source_node_entries(node, start_frame, end_frame, fps):
    """
    Serialize one source node, including linked geometry names.
    """
    animations, animated_values = build_animations(
        node,
        start_frame,
        end_frame,
    )

    geometry_nodes = multiple_linked_nodes(
        node,
        "Geometry",
        direction="input",
    )

    return {
        "node_name": node.name,
        "fuel": float(node.fuel),
        "smoke": float(node.smoke),
        "temperature": float(node.temperature),
        "extra_pressure": float(getattr(node, "extra_pressure", 0.0)),
        "source_noise": bool(getattr(node, "source_noise", False)),
        "noise_scale": float(getattr(node, "noise_scale", 1.0)),
        "noise_seed": int(getattr(node, "noise_seed", 0)),
        "noise_amplitude": float(getattr(node, "noise_amplitude", 0.0)),
        "velocity": safe_float_vector(node.velocity),
        "animations": animations,
        "animated_values": animated_values,
        **build_geometry_entries(
            geometry_nodes,
            start_frame,
            end_frame,
            fps,
        ),
    }


def build_obstacle_node_entries(node, start_frame, end_frame, fps):
    """
    Serialize one obstacle node, including linked geometry names.
    """
    geometry_nodes = multiple_linked_nodes(
        node,
        "Geometry",
        direction="input",
    )

    return {
        "node_name": node.name,
        **build_geometry_entries(
            geometry_nodes,
            start_frame,
            end_frame,
            fps,
        ),
    }


def build_geometry_entries(geometry_nodes, start_frame, end_frame, fps):
    transform_samples = get_geometry_transforms(
        geometry_nodes,
        start_frame,
        end_frame,
    )

    geometry_inputs = [
        {
            "node_name": geometry_node.name,
            "object_name": getattr(
                getattr(geometry_node, "source_object", None),
                "name",
                None,
            ),
        }
        for geometry_node in geometry_nodes
    ]

    for geometry_input in geometry_inputs:
        object_name = geometry_input.get("object_name")
        geometry_input["mesh_file"] = (
            f"geometry/{object_name}.stl" if object_name else None
        )
        geometry_input["mesh_format"] = "stl"
        geometry_input["transform_animation"] = transform_samples.get(object_name, {})

    return {
        "geometry_inputs": geometry_inputs,
        "shape": "mesh" if geometry_inputs else "empty",
    }


def build_force_entries(node, start_frame, end_frame, fps):
    """
    Serialize one force node by its concrete subtype.
    """
    animations, animated_values = build_animations(
        node,
        start_frame,
        end_frame,
    )

    data = {
        "node_name": node.name,
        "node_type": node.bl_idname,
        "animations": animations,
        "animated_values": animated_values,
    }

    if node.bl_idname == "CONTINUUM_FLOW_FORCE_CONSTANT_NODE":
        data["force"] = {
            "x": float(node.fx),
            "y": float(node.fy),
            "z": float(node.fz),
        }

    elif node.bl_idname == "CONTINUUM_FLOW_FORCE_SWIRL_NODE":
        data.update({
            "strength": float(node.strength),
            "origin": safe_float_vector(node.origin),
            "axis": safe_float_vector(node.axis),
            "radius": float(node.radius),
        })

    elif node.bl_idname == "CONTINUUM_FLOW_FORCE_TURBULENCE_NODE":
        data.update({
            "scale": float(node.scale),
            "frequency": float(node.frequency),
            "amplitude": float(node.amplitude),
            "seed": int(node.seed),
        })

    return data


def build_output_node_entries(node):
    """
    Serialize one output node.
    """
    output_path = bpy.path.abspath(node.output_path) if node.output_path else ""
    return {
        "fps": int(node.fps),
        "precision": str(getattr(node, "output_precision", "float16")),
        "fields": {
            "velocity": {"enabled": bool(getattr(node, "export_velocity", True))},
            "pressure": {"enabled": bool(getattr(node, "export_p", False))},
            "temperature": {"enabled": bool(getattr(node, "export_t", False))},
            "smoke": {"enabled": bool(getattr(node, "export_smoke", False))},
            "fuel": {"enabled": bool(getattr(node, "export_fuel", False))},
            "flame": {"enabled": bool(getattr(node, "export_flame", False))},
        },
        "performance": {
            "writer_processes": int(getattr(node, "writer_processes", 4)),
        },
        "output_path": output_path,
    }


def build_viewer_node_entries(node):
    """
    Serialize one viewer node.
    """
    return {
        "live_preview": bool(getattr(node, "live_preview", True)),
    }


#-------------- helper ----------------
def multiple_linked_nodes(node, socket_name, direction="input"):
    """
    Return all nodes connected to the socket.
    """
    if direction == "input":
        socket = node.inputs.get(socket_name)
        linked_node_attr = "from_node"
    elif direction == "output":
        socket = node.outputs.get(socket_name)
        linked_node_attr = "to_node"
    else:
        raise ValueError("direction must be 'input' or 'output'")

    if socket is None or not socket.is_linked:
        return []

    return [
        linked_node
        for link in socket.links
        if (linked_node := getattr(link, linked_node_attr, None)) is not None
    ]


def linked_node(node, socket_name, direction="input"):
    """
    Return the first node connected to the socket.
    """
    if direction == "input":
        socket = node.inputs.get(socket_name)
        linked_node_attr = "from_node"
    elif direction == "output":
        socket = node.outputs.get(socket_name)
        linked_node_attr = "to_node"
    else:
        raise ValueError("direction must be 'input' or 'output'")

    if socket is None or not socket.is_linked:
        return None

    for link in socket.links:
        linked_node = getattr(link, linked_node_attr, None)
        if linked_node is not None:
            return linked_node

    return None


def safe_float_vector(value):
    """
    Convert Blender float vectors to plain Python float lists.
    """
    return [float(component) for component in value]


def get_geometry_transforms(
    geometry_nodes, start_frame, end_frame
):
    """
    Sample evaluated world transforms for linked geometry objects once per Blender frame.
    """
    scene = getattr(bpy.context, "scene", None)

    current_frame = int(getattr(scene, "frame_current", start_frame))
    frame_numbers = list(range(int(start_frame), int(end_frame)))
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
                        "matrices_world": [],
                    },
                )

                object_samples["matrices_world"].append(
                    [
                        [float(component) for component in row]
                        for row in matrix_world
                    ]
                )

    finally:
        scene.frame_set(current_frame)

    return transform_samples


def get_geometry_nodes(node_tree):
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


#-------------- animation data ----------------
def build_animations(node, start_frame, end_frame):
    property_names = ANIMATABLE_PROPERTIES.get(
        node.bl_idname,
        (),
    )

    scene = bpy.context.scene
    current_frame = scene.frame_current

    sampled = {
        name: []
        for name in property_names
        if node_property_is_animated(node, name)
    }

    try:
        for frame in range(start_frame, end_frame):
            scene.frame_set(frame)

            for name in sampled:
                sampled[name].append(
                    sample_animated_value(
                        name,
                        getattr(node, name),
                    )
                )

    finally:
        scene.frame_set(current_frame)

    animations = {
        name: {"values": values}
        for name, values in sampled.items()
    }
    animated_values = {
        name: sample_animated_value(
            name,
            getattr(node, name),
        )
        for name in sampled
    }

    return animations, animated_values


def iter_action_curves(action):
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


def sample_animated_value(property_name, value):
    """
    Convert a sampled Blender property into JSON-friendly data.
    """
    if isinstance(value, (str, bytes)):
        return value

    if hasattr(value, "__len__") and not isinstance(
        value,
        (int, float, bool),
    ):
        return [
            float(component)
            for component in value
        ]

    if property_name in PERCENTAGE_MAPPING:
        minimum, maximum = PERCENTAGE_MAPPING[property_name]

        return minimum + (
            (float(value) / 100.0)
            * (maximum - minimum)
        )

    return float(value)


def node_property_is_animated(node, property_name):
    try:
        data_path = node.path_from_id(property_name)
    except Exception:
        return False

    animation_data = getattr(node.id_data, "animation_data", None)
    if animation_data is None:
        return False

    for fcurve in iter_action_curves(
        getattr(animation_data, "action", None)
    ):
        if getattr(fcurve, "data_path", None) == data_path:
            return True

    for fcurve in getattr(animation_data, "drivers", ()):
        if getattr(fcurve, "data_path", None) == data_path:
            return True

    return False


def sync_node_tree_animations(scene=None):
    """
    Touch animated node properties so Blender evaluates them consistently before UI redraws.
    """
    del scene

    node_groups = getattr(bpy.data, "node_groups", None)
    if node_groups is None:
        return

    for node_tree in node_groups:
        if getattr(node_tree, "bl_idname", "") != NODE_TREE_ID:
            continue

        for node in getattr(node_tree, "nodes", ()):
            for property_name in ANIMATABLE_PROPERTIES.get(
                getattr(node, "bl_idname", ""),
                (),
            ):
                if not node_property_is_animated(node, property_name):
                    continue

                try:
                    getattr(node, property_name)
                except Exception:
                    continue