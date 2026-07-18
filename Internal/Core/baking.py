import bpy
import shutil
import threading
import time
import sys
from pathlib import Path

from .export import export_config
from . import load_result
from .solver import solver_worker
from .solver import solver_status
from .writer import writer_manager


VDBWatcher = load_result.VDBWatcher()
status_workspace = None

#-------------- get methods ----------------
def get_output_node(context):
    node = getattr(context, "node", None)
    if getattr(node, "bl_idname", "") == "CONTINUUM_FLOW_OUTPUT_NODE":
        return node
    return None


def get_connected_simulation_node(output_node):
    linked_simulation_nodes = get_linked_simulation_nodes(output_node)
    return linked_simulation_nodes[0]


def get_linked_simulation_nodes(output_node):
    result_socket = output_node.inputs.get("Result")
    if result_socket is None or not result_socket.is_linked:
        return []

    simulation_nodes = []
    for link in result_socket.links:
        simulation_node = getattr(link, "from_node", None)
        if getattr(simulation_node, "bl_idname", "") == "CONTINUUM_FLOW_SIMULATION_NODE":
            simulation_nodes.append(simulation_node)
    return simulation_nodes


def get_linked_viewer_node(simulation_node):
    result_socket = simulation_node.outputs.get("Result")
    if result_socket is None or not result_socket.is_linked:
        return []

    viewer_nodes = []
    for link in result_socket.links:
        viewer_node = getattr(link, "to_node", None)
        if getattr(viewer_node, "bl_idname", "") == "CONTINUUM_FLOW_VIEWER_NODE":
            viewer_nodes.append(viewer_node)
    return viewer_nodes


#-------------- progress managment ----------------
def draw_bake_progress(self, context):
    layout = self.layout
    layout.separator_spacer()
    layout.label(text="Bake:")

    progress_row = layout.row(align=True)
    progress_row.ui_units_x = 13

    progress_row.progress(
        factor=float(solver_status.progress),
        type='BAR',
        text=solver_status.progress_text,
    )
    layout.separator_spacer()


def set_status_progress(context):
    global status_workspace
    workspace = getattr(context, "workspace", None)
    if workspace is None:
        return
    status_workspace = workspace
    workspace.status_text_set(draw_bake_progress)
    ui_redraw()


def set_bake_progress(current_frames, total_frames):
    total_frames = max(1, total_frames)
    solver_status.progress_current_frames = current_frames
    solver_status.progress_total_frames = total_frames

    progress = min(1.0, current_frames / total_frames)
    percent = int(round(progress * 100.0))
    solver_status.progress = progress
    solver_status.progress_text = f"{percent}%"

    ui_redraw()


def clear_status_progress(context):
    global status_workspace
    workspace = status_workspace or getattr(context, "workspace", None)
    if workspace is not None:
        workspace.status_text_set(None)
    status_workspace = None
    ui_redraw()


#-------------- UI ----------------
def ui_redraw():
    window_manager = getattr(bpy.context, "window_manager")

    for window in window_manager.windows:
        screen = getattr(window, "screen")

        for area in screen.areas:
            if area.type in {"STATUSBAR", "NODE_EDITOR", "PROPERTIES"}:
                area.tag_redraw()


#-------------- Data and paths ----------------
def normalize_directory_path(path_value):
    try:
        return Path(path_value).resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        return None


def output_directory_has_vdbs(output_directory):
    output_directory = normalize_directory_path(output_directory)
    return bool(
        output_directory
        and output_directory.exists()
        and output_directory.is_dir()
        and any(output_directory.glob("*.vdb"))
    )


#-------------- bake ----------------
def is_live_preview_enabled(simulation_node):
    viewer_nodes = get_linked_viewer_node(simulation_node)
    if not viewer_nodes:
        return False
    return any(bool(getattr(viewer_node, "live_preview", True)) for viewer_node in viewer_nodes)


class CONTINUUM_FLOW_OT_bake(bpy.types.Operator):
    bl_idname = "continuum_flow.bake"
    bl_label = "Bake"

    def execute(self, context):
        self.output_node = get_output_node(context)
        self.simulation_node = get_connected_simulation_node(self.output_node)
        self.job_id = None
        self.job_result = None
        self.writer_server = None
        self.bake_directory = None
        self.output_directory = None
        self.cancel_flag_path = None
        self.cleanup_done = False
        self.cleanup_lock = threading.Lock()
        self.event_timer = None
        self.cancel_requested = False

        solver_status.bake_running = True
        solver_status.active_bake_operator = self
        set_status_progress(context)

        try:
            self.run_bake(context)
            self.event_timer = context.window_manager.event_timer_add(0.1, window=context.window)
            context.window_manager.modal_handler_add(self)
        except Exception as exc:
            print("Failed to start bake:", exc)
            if self.job_id is not None:
                self.cancel_bake()
            else:
                self.cleanup()
            self.report({'ERROR'}, f"Failed to start bake: {exc}")
            return {'CANCELLED'}

        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'ESC':
            print("Bake cancelled by user")
            self.cancel_bake()
            return {'CANCELLED'}

        if event.type == 'TIMER' and self.job_id is not None:
            job_result = solver_worker.get_job_result(self.job_id)
            if job_result is not None:
                self.job_result = job_result
                self.cleanup()

                if self.cancel_requested:
                    return {'CANCELLED'}

                if not bool(job_result.get("success", False)):
                    message = job_result.get("message") or "Bake failed"
                    traceback_text = job_result.get("traceback")
                    if traceback_text:
                        print(traceback_text)
                    self.report({'ERROR'}, message)
                    return {'CANCELLED'}

                return {'FINISHED'}

        return {'PASS_THROUGH'}

    def cleanup(self):
        with self.cleanup_lock:
            if self.cleanup_done:
                return

            self.cleanup_done = True
            solver_status.bake_running = False
            solver_status.active_bake_operator = None
            bpy.context.window_manager.event_timer_remove(self.event_timer)
            self.event_timer = None
            bake_completed_successfully = (
                not self.cancel_requested
                and bool(self.job_result)
                and bool(self.job_result.get("success", False))
            )

            if self.writer_server:
                self.writer_server.stop()

            if self.cancel_flag_path:
                self.cancel_flag_path.unlink(missing_ok=True)

            if bake_completed_successfully:
                VDBWatcher.finish_bake()

            VDBWatcher.stop()

            geometry_directory = Path(self.output_directory).resolve() / "geometry"
            shutil.rmtree(geometry_directory)
            print("Removed geometry directory:", geometry_directory)
        
            self.output_node.last_bake_directory = str(self.output_directory) 
            set_bake_progress(0, 0)
            clear_status_progress(bpy.context)


    def cancel_bake(self):
        self.cancel_requested = True

        solver_worker.shutdown_worker()
        self.job_result = {
            "type": "job_finished",
            "job_id": self.job_id,
            "success": False,
            "message": "Bake cancelled",
        }
        solver_worker.start_worker_in_background()

        self.cleanup()


    def launch_writer_manager(self, config_dict):
        simulation_config = ((config_dict.get("simulations") or [{}])[0])
        output_config = (simulation_config.get("outputs") or [{}])[0]
        performance_config = output_config.get("performance") or {}
        writer_process_count = int(performance_config.get("writer_processes", 4))
        writer_config = {
            "simulations": [{
                "domain": simulation_config.get("domain") or {},
                "outputs": [{
                    "precision": output_config.get("precision", "float32"),
                }],
            }],
        }

        server = writer_manager.HostVDBWriterServer(
            writer_process_count=writer_process_count,
            writer_config=writer_config,
        )
        server.start()
        return server


    def update_bake_progress(self, loaded_frame_count):
        set_bake_progress(loaded_frame_count, solver_status.progress_total_frames)


    def run_bake(self, context):
        config_dict = export_config.build_config_dict(context=context, simulation_node=self.simulation_node)
        bake_directory, config_dict = export_config.export_config_dict(config_dict)

        writer_server = self.launch_writer_manager(config_dict)
        self.writer_server = writer_server
        simulation_config = config_dict["simulation"][0]
        simulation_config["outputs"][0]["host_vdb_writer"] = (writer_server.endpoint())
        self.bake_directory = Path(bake_directory).resolve()
        self.cancel_flag_path = (self.bake_directory / "cancel_requested.flag")
        config_dict["bake_directory"] = str(self.bake_directory)
        
        meta_config = config_dict.setdefault("meta", {})
        meta_config["cancel_flag_path"] = str(self.cancel_flag_path)
        meta_config["parent_sys_path"] = list(sys.path)

        output_config = simulation_config["outputs"][0]
        simulation_settings = simulation_config.get("settings") or {}
        solver_backend = str(simulation_settings.get("solver_backend", "GPU")).strip().upper()
        start_frame = int(simulation_settings.get("start_frame", 1))
        end_frame = int(simulation_settings.get("end_frame", start_frame))
        total_frames = max(1, end_frame - start_frame)
        set_bake_progress(0, total_frames)

        vdb_output_dir = Path(output_config["output_path"]).resolve()
        self.output_directory = vdb_output_dir

        VDBWatcher.start(
            vdb_output_dir,
            start_frame_index=start_frame,
            live_preview_enabled=is_live_preview_enabled(self.simulation_node),
            progress_callback=self.update_bake_progress,
        )

        solver_worker.ensure_worker_running(
            wait=True,
            timeout=120.0,
            preload_backend=solver_backend,
        )
        self.job_id = solver_worker.start_job(config_dict)


#-------------- free bake ----------------
def output_node_has_baked_data(output_node):
    return get_bake_directory(output_node, persist=False) is not None


def bake_directory_is_old(output_node, bake_directory):
    target_directory = normalize_directory_path(bpy.path.abspath(getattr(output_node, "output_path", "")))
    try:
        bake_directory.relative_to(target_directory)
    except ValueError:
        return False
    return True


def get_latest_bake_directory(output_node):
    target_directory = normalize_directory_path(bpy.path.abspath(getattr(output_node, "output_path", "")))
    if target_directory is None or not target_directory.exists() or not target_directory.is_dir():
        return None

    candidates = []
    for child in target_directory.iterdir():
        if not child.is_dir():
            continue
        if not output_directory_has_vdbs(child):
            continue
        candidates.append(child)

    if not candidates:
        return None

    return max(candidates, key=lambda directory: directory.stat().st_mtime)


def get_bake_directory(output_node, persist=False):
    bake_directory = normalize_directory_path(getattr(output_node, "last_bake_directory", ""))
    if bake_directory_is_old(output_node, bake_directory) and output_directory_has_vdbs(bake_directory):
        return bake_directory

    discovered_directory = get_latest_bake_directory(output_node)
    if persist and discovered_directory is not None:
        output_node.last_bake_directory = str(discovered_directory)
    return discovered_directory


def clear_bake_directory(output_directory):
    output_directory = Path(output_directory).resolve()
    deleted_count = sum(1 for _ in output_directory.glob("*.vdb"))
    try:
        shutil.rmtree(output_directory)
    except OSError as exc:
        print("Failed to remove bake directory:", output_directory, exc)
        return 0

    print("Removed bake directory:", output_directory)
    return deleted_count


def free_bake_output(output_node):
    output_directory = get_bake_directory(output_node, persist=False)
    VDBWatcher.clear_loaded_sequence_for_directory(output_directory)
    deleted_count = clear_bake_directory(output_directory)
    output_node.last_bake_directory = str(output_directory)
    return deleted_count


class CONTINUUM_FLOW_OT_free_bake(bpy.types.Operator):
    bl_idname = "continuum_flow.free_bake"
    bl_label = "Free Bake"

    def execute(self, context):
        output_node = get_output_node(context)
        deleted_count = free_bake_output(output_node)
        if deleted_count:
            self.report({'INFO'}, f"Removed {deleted_count} VDB files")
        else:
            self.report({'INFO'}, "No VDB files found to remove")
        return {'FINISHED'}