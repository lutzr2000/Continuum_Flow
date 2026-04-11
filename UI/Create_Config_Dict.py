import json
from datetime import datetime, timezone
from pathlib import Path

import bpy


NODE_TREE_ID = "BLENDERCFD_NODE_TREE"


def _safe_float_vector(value):
    """Convert Blender float vectors to plain Python float lists."""
    return [float(component) for component in value]


def _linked_input_nodes(node, socket_name, expected_idname=None):
    """Return upstream nodes connected to the given input socket."""
    socket = node.inputs.get(socket_name)
    if socket is None or not socket.is_linked:
        return []

    linked_nodes = []
    for link in socket.links:
        from_node = getattr(link, "from_node", None)
        if from_node is None:
            continue
        if expected_idname is not None and getattr(from_node, "bl_idname", "") != expected_idname:
            continue
        linked_nodes.append(from_node)
    return linked_nodes


def _linked_output_nodes(node, socket_name, expected_idname=None):
    """Return downstream nodes connected to the given output socket."""
    socket = node.outputs.get(socket_name)
    if socket is None or not socket.is_linked:
        return []

    linked_nodes = []
    for link in socket.links:
        to_node = getattr(link, "to_node", None)
        if to_node is None:
            continue
        if expected_idname is not None and getattr(to_node, "bl_idname", "") != expected_idname:
            continue
        linked_nodes.append(to_node)
    return linked_nodes


def _linked_geometry_names(node):
    """Return the linked geometry object names for a source or obstacle node."""
    geometry_nodes = _linked_input_nodes(node, "Geometry", "BLENDERCFD_GEOMETRY_NODE")
    geometry_entries = []
    for geometry_node in geometry_nodes:
        source_object = getattr(geometry_node, "source_object", None)
        geometry_entries.append(
            {
                "node_name": geometry_node.name,
                "object_name": source_object.name if source_object is not None else None,
            }
        )
    return geometry_entries


def _serialize_domain_node(node):
    """Serialize one domain node."""
    return {
        "node_name": node.name,
        "resolution": float(node.resolution),
        "grid": {
            "nx": int(node.nx),
            "ny": int(node.ny),
            "nz": int(node.nz),
        },
        "boundary_conditions": {
            "x_low": {"type": node.x_low_bc, "velocity": _safe_float_vector(node.x_low_velocity)},
            "x_high": {"type": node.x_high_bc, "velocity": _safe_float_vector(node.x_high_velocity)},
            "y_low": {"type": node.y_low_bc, "velocity": _safe_float_vector(node.y_low_velocity)},
            "y_high": {"type": node.y_high_bc, "velocity": _safe_float_vector(node.y_high_velocity)},
            "z_low": {"type": node.z_low_bc, "velocity": _safe_float_vector(node.z_low_velocity)},
            "z_high": {"type": node.z_high_bc, "velocity": _safe_float_vector(node.z_high_velocity)},
        },
    }


def _serialize_physics_node(node):
    """Serialize one physics node."""
    return {
        "node_name": node.name,
        "fluid": {
            "density": float(node.fluid_density),
            "viscosity": float(node.fluid_viscosity),
        },
        "temperature": {
            "diffusion": float(node.temperature_diffusion),
            "dissipation": float(node.temperature_dissipation),
            "reference_temperature": float(node.reference_temperature),
            "buoyancy": float(node.buoyancy),
            "expansion_rate": float(node.expansion_rate),
        },
        "smoke": {
            "diffusion": float(node.smoke_diffusion),
            "dissipation": float(node.smoke_dissipation),
        },
        "fuel": {
            "diffusion": float(node.fuel_diffusion),
            "dissipation": float(node.fuel_dissipation),
            "burn_rate": float(node.fuel_burn_rate),
            "ignition_temperature": float(node.fuel_ignition_temperature),
        },
    }


def _serialize_reference_frame_node(node):
    """Serialize one reference-frame node."""
    source_object = getattr(node, "source_object", None)
    return {
        "node_name": node.name,
        "object_name": source_object.name if source_object is not None else None,
    }


def _serialize_source_node(node):
    """Serialize one source node, including linked geometry names."""
    return {
        "node_name": node.name,
        "fuel": float(node.fuel),
        "smoke": float(node.smoke),
        "temperature": float(node.temperature),
        "geometry_inputs": _linked_geometry_names(node),
    }


def _serialize_obstacle_node(node):
    """Serialize one obstacle node, including linked geometry names."""
    return {
        "node_name": node.name,
        "geometry_inputs": _linked_geometry_names(node),
    }


def _serialize_force_node(node):
    """Serialize one force node by its concrete subtype."""
    base_data = {
        "node_name": node.name,
        "node_type": node.bl_idname,
    }

    if node.bl_idname == "BLENDERCFD_FORCE_CONSTANT_NODE":
        base_data["force"] = {
            "x": float(node.fx),
            "y": float(node.fy),
            "z": float(node.fz),
        }
    elif node.bl_idname == "BLENDERCFD_FORCE_POINT_NODE":
        base_data["strength"] = float(node.strength)
        base_data["position"] = _safe_float_vector(node.position)
    elif node.bl_idname == "BLENDERCFD_FORCE_TURBULENCE_NODE":
        base_data["scale"] = float(node.scale)
        base_data["frequency"] = float(node.frequency)
        base_data["amplitude"] = float(node.amplitude)

    return base_data


def _serialize_output_node(node):
    """Serialize one output node."""
    output_path = bpy.path.abspath(node.output_path) if node.output_path else ""
    return {
        "node_name": node.name,
        "fps": int(node.fps),
        "fields": {
            "u": bool(node.export_u),
            "v": bool(node.export_v),
            "w": bool(node.export_w),
            "p": bool(node.export_p),
            "t": bool(node.export_t),
            "smoke": bool(node.export_smoke),
            "fuel": bool(node.export_fuel),
            "flame": bool(node.export_flame),
        },
        "output_path": output_path,
    }


def _serialize_viewer_node(node):
    """Serialize one viewer node."""
    return {
        "node_name": node.name,
    }


def _build_simulation_entry(simulation_node):
    """Build a grouped config entry for one simulation node."""
    domain_nodes = _linked_input_nodes(simulation_node, "Domain", "BLENDERCFD_DOMAIN_NODE")
    physics_nodes = _linked_input_nodes(simulation_node, "Physics", "BLENDERCFD_PHYSICS_NODE")
    reference_frame_nodes = _linked_input_nodes(
        simulation_node,
        "Reference Frame",
        "BLENDERCFD_REFERENCE_FRAME_NODE",
    )
    source_nodes = _linked_input_nodes(simulation_node, "Source", "BLENDERCFD_SOURCE_NODE")
    obstacle_nodes = _linked_input_nodes(simulation_node, "Obstacles", "BLENDERCFD_OBSTACLE_NODE")
    force_nodes = _linked_input_nodes(simulation_node, "Forces")

    output_nodes = _linked_output_nodes(simulation_node, "Result", "BLENDERCFD_OUTPUT_NODE")
    viewer_nodes = _linked_output_nodes(simulation_node, "Result", "BLENDERCFD_VIEWER_NODE")

    return {
        "node_name": simulation_node.name,
        "settings": {
            "simulation_length": float(simulation_node.simulation_length),
            "cfl": float(simulation_node.cfl),
            "iterations": int(simulation_node.iterations),
        },
        "domain": _serialize_domain_node(domain_nodes[0]) if domain_nodes else None,
        "physics": _serialize_physics_node(physics_nodes[0]) if physics_nodes else None,
        "reference_frame": (
            _serialize_reference_frame_node(reference_frame_nodes[0])
            if reference_frame_nodes
            else None
        ),
        "sources": [_serialize_source_node(node) for node in source_nodes],
        "obstacles": [_serialize_obstacle_node(node) for node in obstacle_nodes],
        "forces": [_serialize_force_node(node) for node in force_nodes],
        "outputs": [_serialize_output_node(node) for node in output_nodes],
        "viewers": [_serialize_viewer_node(node) for node in viewer_nodes],
    }


def _collect_tree_geometry_nodes(node_tree):
    """Collect all geometry nodes for visibility in the exported JSON."""
    geometry_entries = []
    for node in node_tree.nodes:
        if getattr(node, "bl_idname", "") != "BLENDERCFD_GEOMETRY_NODE":
            continue
        source_object = getattr(node, "source_object", None)
        geometry_entries.append(
            {
                "node_name": node.name,
                "object_name": source_object.name if source_object is not None else None,
            }
        )
    return geometry_entries


def _resolve_node_tree(context=None):
    """Resolve the active BlenderCFD node tree or fall back to the first one."""
    if context is not None:
        space_data = getattr(context, "space_data", None)
        edit_tree = getattr(space_data, "edit_tree", None)
        if edit_tree is not None and getattr(edit_tree, "bl_idname", "") == NODE_TREE_ID:
            return edit_tree

    for node_group in bpy.data.node_groups:
        if getattr(node_group, "bl_idname", "") == NODE_TREE_ID:
            return node_group
    return None


def _resolve_export_directory(simulation_entries):
    """Choose a directory for the exported JSON."""
    for simulation_entry in simulation_entries:
        for output_entry in simulation_entry["outputs"]:
            output_path = output_entry.get("output_path", "")
            if output_path:
                return Path(output_path)

    blend_directory = bpy.path.abspath("//")
    if blend_directory:
        return Path(blend_directory)
    return Path.cwd()


def build_config_dict(context=None):
    """Evaluate the BlenderCFD node tree and return a grouped config dict."""
    node_tree = _resolve_node_tree(context)
    if node_tree is None:
        raise RuntimeError("No BlenderCFD node tree found.")

    simulation_nodes = [
        node
        for node in node_tree.nodes
        if getattr(node, "bl_idname", "") == "BLENDERCFD_SIMULATION_NODE"
    ]

    config_dict = {
        "meta": {
            "node_tree_name": node_tree.name,
            "exported_at_utc": datetime.now(timezone.utc).isoformat(),
            "simulation_count": len(simulation_nodes),
        },
        "simulations": [_build_simulation_entry(node) for node in simulation_nodes],
        "geometry_nodes": _collect_tree_geometry_nodes(node_tree),
    }
    return config_dict


def export_config_dict(context=None):
    """Evaluate the node tree, write a JSON file, and return both path and dict."""
    config_dict = build_config_dict(context)
    export_directory = _resolve_export_directory(config_dict["simulations"])
    export_directory.mkdir(parents=True, exist_ok=True)

    file_name = f"{config_dict['meta']['node_tree_name']}_config_export.json"
    file_path = export_directory / file_name
    file_path.write_text(json.dumps(config_dict, indent=2), encoding="utf-8")
    return file_path, config_dict


class BlenderCFD_OT_export_config_dict(bpy.types.Operator):
    """Export the evaluated BlenderCFD node tree to JSON."""

    bl_idname = "blendercfd.export_config_dict"
    bl_label = "Export Config Dict"
    bl_description = "Evaluate the BlenderCFD node tree and write a JSON config file"

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


classes = (
    BlenderCFD_OT_export_config_dict,
)
