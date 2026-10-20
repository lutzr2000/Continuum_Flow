import tempfile
import threading
from pathlib import Path

import bpy

from . import volume_renderer
from .export import export_config
from .solver import solver_status
from .solver.solver_manager import solver_manager
from ..UI import animation_proxy


class PreviewUpdateChannel:
    def __init__(self):
        self._lock = threading.Lock()
        self._values = None

    def publish(self, values):
        with self._lock:
            self._values = values

    def values(self):
        with self._lock:
            return self._values


def node_reference(node):
    return (
        str(getattr(getattr(node, "id_data", None), "name", "")),
        str(getattr(node, "name", "")),
    )


def preview_running_for(viewer_node):
    operator = solver_status.active_preview_operator
    return bool(
        solver_status.preview_running
        and operator is not None
        and operator.viewer_reference == node_reference(viewer_node)
    )


def request_preview_stop():
    operator = solver_status.active_preview_operator
    if operator is not None:
        operator.request_stop()


def get_simulation_node(viewer_node):
    result_socket = viewer_node.inputs.get("Result")
    if result_socket is None:
        return None
    for link in result_socket.links:
        node = getattr(link, "from_node", None)
        if getattr(node, "bl_idname", "") == "CONTINUUM_FLOW_SIMULATION_NODE":
            return node
    return None


class CONTINUUM_FLOW_OT_viewer_preview(bpy.types.Operator):
    bl_idname = "continuum_flow.viewer_preview"
    bl_label = "Live Preview"
    bl_description = "Start or stop the continuous in-memory simulation preview"

    def execute(self, context):
        viewer_node = getattr(context, "node", None)

        if solver_status.preview_running:
            request_preview_stop()
            return {"FINISHED"}

        if solver_status.bake_running or solver_manager.is_busy():
            self.report({"ERROR"}, "The solver is already busy.")
            return {"CANCELLED"}

        simulation_node = get_simulation_node(viewer_node)
        self.viewer_reference = node_reference(viewer_node)
        self.job_id = None
        self.event_timer = None
        self.stop_event = threading.Event()
        self.temporary_directory = None
        self.cleanup_done = False
        self.last_timeline_frame = None
        self.update_channel = PreviewUpdateChannel()

        try:
            config = export_config.build_config_dict(
                context=context,
                simulation_node=simulation_node,
                preview_enabled=True,
            )
            self.temporary_directory = tempfile.TemporaryDirectory(
                prefix="continuum_flow_preview_"
            )

            work_directory = Path(self.temporary_directory.name).resolve()
            export_config.set_subdirectory_paths(config, work_directory)
            export_config.export_geometry_usd(config, work_directory)
            export_config.export_particle_system_npzs(config, work_directory)

            simulation = config["simulation"]
            simulation["settings"]["continuous_preview"] = True
            self.start_frame = int(simulation["settings"].get("start_frame", 1))

            for output in simulation.get("outputs", []):
                for field in (output.get("fields") or {}).values():
                    field["enabled"] = False

            config.setdefault("meta", {})["cancel_event"] = self.stop_event
            config["meta"]["preview_update_channel"] = self.update_channel

            solver_status.preview_running = True
            solver_status.active_preview_operator = self
            self.job_id = solver_manager.start_job(config)
            self.event_timer = context.window_manager.event_timer_add(
                0.1, window=context.window
            )
            context.window_manager.modal_handler_add(self)
            volume_renderer.redraw_viewports()
            return {"RUNNING_MODAL"}
        except Exception as exc:
            self.cleanup(context)
            return {"CANCELLED"}

    def modal(self, context, event):
        if event.type == "ESC":
            self.request_stop()
            return {"RUNNING_MODAL"}

        if event.type == "TIMER" and self.job_id is not None:
            self.update_timeline(context)
            self.publish_live_values(context)
            result = solver_manager.get_job_result(self.job_id)
            if result is not None:
                self.cleanup(context)
                if not self.stop_event.is_set() and not result.get("success", False):
                    self.report({"ERROR"}, result.get("message") or "Preview failed")
                    return {"CANCELLED"}
                return {"FINISHED"}

        return {"PASS_THROUGH"}

    def request_stop(self):
        self.stop_event.set()

    def update_timeline(self, context):
        solver_frame = solver_manager.get_stats().get("frame")
        if solver_frame is None:
            return

        timeline_frame = self.start_frame + int(solver_frame) - 1
        if timeline_frame == self.last_timeline_frame:
            return

        scene = getattr(context, "scene", None)
        if scene is not None:
            scene.frame_set(timeline_frame)
            self.last_timeline_frame = timeline_frame

    def publish_live_values(self, context):
        node_tree = bpy.data.node_groups.get(self.viewer_reference[0])
        if node_tree is None:
            return

        viewer_node = node_tree.nodes.get(self.viewer_reference[1])
        simulation_node = get_simulation_node(viewer_node)
        if simulation_node is None:
            return

        scene = getattr(context, "scene", None)
        source_nodes = export_config.multiple_linked_nodes(
            simulation_node, "Source", direction="input"
        )
        source_values = {}
        for node in source_nodes:

            def current(property_name):
                return animation_proxy.scene_proxy_value(scene, node, property_name)

            source_values[node.name] = {
                "noise": {
                    "scale": float(node.randomness_scale),
                    "seed": int(node.randomness_seed),
                },
                "fuel": {
                    "value": float(current("fuel")),
                    "randomness": float(node.fuel_randomness),
                },
                "smoke": {
                    "value": float(current("smoke")),
                    "randomness": float(node.smoke_randomness),
                },
                "temperature": {
                    "value": float(current("temperature")),
                    "randomness": float(node.temperature_randomness),
                },
                "extra_pressure": {
                    "value": float(current("extra_pressure")),
                    "randomness": float(node.extra_pressure_randomness),
                },
                "velocity": {
                    "value": [float(value) for value in current("velocity")],
                    "space": str(node.velocity_space),
                    "randomness": float(node.velocity_randomness),
                },
            }

        physics_node = export_config.linked_node(
            simulation_node, "Physics", direction="input"
        )
        physics_values = {}
        if physics_node is not None:
            for section_name, property_names in physics_node.property_groups:
                if section_name == "Fluid":
                    continue
                section_key = section_name.lower()
                physics_values[section_key] = {}
                for property_name in property_names:
                    value = float(
                        animation_proxy.scene_proxy_value(
                            scene, physics_node, property_name
                        )
                    )
                    if property_name in export_config.PERCENTAGE_MAPPING:
                        minimum, maximum = export_config.PERCENTAGE_MAPPING[
                            property_name
                        ]
                        value = minimum + (value / 100.0) * (maximum - minimum)
                    config_name = {
                        "temperature_dissipation": "dissipation",
                        "temperature_production_rate": "production_rate",
                        "reference_temperature": "reference_temperature",
                        "buoyancy": "buoyancy",
                        "expansion_rate": "expansion_rate",
                        "smoke_dissipation": "dissipation",
                        "smoke_production_rate": "production_rate",
                        "fuel_dissipation": "dissipation",
                        "fuel_burn_rate": "burn_rate",
                        "fuel_ignition_temperature": "ignition_temperature",
                        "fuel_ignition_temperature_width": "ignition_temperature_width",
                        "vorticity": "vorticity",
                    }[property_name]
                    physics_values[section_key][config_name] = value

        particle_values = {}
        for source_node in source_nodes:
            for particle_node in export_config.multiple_linked_nodes(
                source_node, "Particle Systems", direction="input"
            ):
                particle_values[particle_node.name] = {
                    "radius": float(particle_node.radius),
                    "velocity_transfer": float(particle_node.velocity_transfer) / 100.0,
                }

        self.update_channel.publish(
            {
                "sources": source_values,
                "physics": physics_values,
                "particles": particle_values,
            }
        )

    def cleanup(self, context):
        if self.cleanup_done:
            return
        self.cleanup_done = True
        if self.event_timer is not None:
            context.window_manager.event_timer_remove(self.event_timer)
            self.event_timer = None
        if self.temporary_directory is not None:
            self.temporary_directory.cleanup()
            self.temporary_directory = None
        solver_status.preview_running = False
        solver_status.active_preview_operator = None
        volume_renderer.clear_live_preview()
