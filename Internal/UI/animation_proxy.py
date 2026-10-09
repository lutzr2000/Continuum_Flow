import uuid
import re
import traceback

import bpy

PROXY_PREFIX = "cf_"
VALID_SUBTYPES = {
    "PIXEL",
    "PIXEL_DIAMETER",
    "UNSIGNED",
    "PERCENTAGE",
    "FACTOR",
    "MASS",
    "ANGLE",
    "TIME",
    "TIME_ABSOLUTE",
    "DISTANCE",
    "DISTANCE_DIAMETER",
    "DISTANCE_CAMERA",
    "POWER",
    "TEMPERATURE",
    "WAVELENGTH",
    "COLOR_TEMPERATURE",
    "FREQUENCY",
    "COLOR",
    "TRANSLATION",
    "DIRECTION",
    "VELOCITY",
    "ACCELERATION",
    "MATRIX",
    "EULER",
    "QUATERNION",
    "AXISANGLE",
    "XYZ",
    "XYZ_LENGTH",
    "COLOR_GAMMA",
    "COORDINATES",
    "LAYER",
    "LAYER_MEMBER",
}

UNIT_SUBTYPES = {
    "LENGTH": "DISTANCE",
    "TEMPERATURE": "TEMPERATURE",
    "VELOCITY": "VELOCITY",
}


def node_proxy_key(node, property_name):
    proxy_id = node.get("continuum_flow_animation_id")

    if not proxy_id:
        ensure_node_proxy_id(node)
        proxy_id = node.get("continuum_flow_animation_id")

    return f"{PROXY_PREFIX}{proxy_id[:12]}_{property_name}"


def ensure_node_proxy_id(node):
    if node.get("continuum_flow_animation_id"):
        return

    node["continuum_flow_animation_id"] = uuid.uuid4().hex


def ensure_scene_proxy(scene, node, property_name):
    key = node_proxy_key(node, property_name)

    if key not in scene:
        value = getattr(node, property_name)
        initial_value = list(value) if hasattr(value, "__len__") else value
        scene[key] = initial_value

    rna_property = node.bl_rna.properties[property_name]
    ui_data = scene.id_properties_ui(key)
    metadata = {"description": rna_property.description}
    unit = str(getattr(rna_property, "unit", "") or "")
    subtype = UNIT_SUBTYPES.get(unit)

    if subtype is None:
        subtype = str(getattr(rna_property, "subtype", "") or "")
    if subtype in VALID_SUBTYPES:
        metadata["subtype"] = subtype

    if hasattr(rna_property, "hard_min"):
        metadata.update(
            min=rna_property.hard_min,
            max=rna_property.hard_max,
            soft_min=rna_property.soft_min,
            soft_max=rna_property.soft_max,
            precision=rna_property.precision,
            step=rna_property.step,
        )

    ui_data.update(**metadata)
    return key


def draw_scene_proxy(layout, scene, node, property_name, text=None):
    key = node_proxy_key(node, property_name)

    if key not in scene:
        layout.prop(node, property_name, text=text)
        return

    layout.prop(
        scene,
        f'["{key}"]',
        text=text if text is not None else node.bl_rna.properties[property_name].name,
    )


def scene_proxy_value(scene, node, property_name):
    """Return the displayed proxy value without modifying the node data-block."""
    if scene is not None:
        key = node_proxy_key(node, property_name)
        if key in scene:
            return scene[key]
    return getattr(node, property_name)


def ensure_node_scene_proxies(scene, node):
    if scene is None:
        return

    for property_name in getattr(node, "animation_proxy_properties", ()):
        ensure_scene_proxy(scene, node, property_name)


def remove_node_scene_proxies(node):
    """Remove every scene value and F-curve owned by a deleted node."""
    proxy_id = node.get("continuum_flow_animation_id")
    if not proxy_id:
        return

    key_prefix = f"{PROXY_PREFIX}{str(proxy_id)[:12]}_"
    for scene in getattr(bpy.data, "scenes", ()):
        owned_keys = {
            str(key) for key in scene.keys() if str(key).startswith(key_prefix)
        }
        if not owned_keys:
            continue

        animation_data = getattr(scene, "animation_data", None)
        action = getattr(animation_data, "action", None)
        if action is not None:
            owned_paths = {f'["{key}"]' for key in owned_keys}
            for curves, _groups in _curve_collections(action):
                for curve in tuple(curves):
                    if curve.data_path in owned_paths:
                        curves.remove(curve)

        for key in owned_keys:
            if key in scene:
                del scene[key]


def sync_scene_animation_proxies(scene):
    if scene is None:
        return

    for node_tree in getattr(bpy.data, "node_groups", ()):
        if getattr(node_tree, "bl_idname", "") != "CONTINUUM_FLOW_NODE_TREE":
            continue

        for node in node_tree.nodes:
            for property_name in getattr(node, "animation_proxy_properties", ()):
                key = node_proxy_key(node, property_name)
                if key in scene:
                    setattr(node, property_name, scene[key])


def scene_proxy_data_path(node, property_name):
    key = node_proxy_key(node, property_name)
    return f'["{key}"]'


def _curve_collections(action):
    legacy_curves = getattr(action, "fcurves", None)

    if legacy_curves is not None:
        yield legacy_curves, getattr(action, "groups", None)
        return

    for layer in getattr(action, "layers", ()):
        for strip in getattr(layer, "strips", ()):
            for channelbag in getattr(strip, "channelbags", ()):
                yield channelbag.fcurves, channelbag.groups


def maintain_animation_proxies():
    try:
        nodes_by_prefix = {}
        for node_tree in getattr(bpy.data, "node_groups", ()):
            if getattr(node_tree, "bl_idname", "") != "CONTINUUM_FLOW_NODE_TREE":
                continue
            for node in node_tree.nodes:
                proxy_id = node.get("continuum_flow_animation_id")
                if proxy_id:
                    nodes_by_prefix[str(proxy_id)[:12]] = node

        path_pattern = re.compile(r'^\["(cf_([0-9a-f]{12})_.+)"\]$')

        for scene in getattr(bpy.data, "scenes", ()):
            for key in tuple(scene.keys()):
                if not str(key).startswith(PROXY_PREFIX):
                    continue
                parts = str(key).split("_", 2)
                if len(parts) < 3 or parts[1] not in nodes_by_prefix:
                    del scene[key]

            animation_data = getattr(scene, "animation_data", None)
            action = getattr(animation_data, "action", None)
            if action is None:
                continue

            for curves, groups in _curve_collections(action):
                for curve in tuple(curves):
                    match = path_pattern.match(curve.data_path)
                    if match is None:
                        continue
                    key, proxy_prefix = match.groups()
                    node = nodes_by_prefix.get(proxy_prefix)
                    if node is None or key not in scene:
                        curves.remove(curve)
                        continue

                    group = groups.get(node.name) or groups.new(node.name)
                    curve.group = group
                    should_hide = not bool(node.select)
                    if curve.hide != should_hide:
                        curve.hide = should_hide
    except Exception:
        traceback.print_exc()
    return 0.25
