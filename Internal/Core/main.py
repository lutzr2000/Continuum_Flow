import bpy
import shutil
import threading
import time
import sys
from pathlib import Path

from . import export_config
from . import load_result
from . import solver_process
from . import solver_status
from . import writer_manager


VDBWatcher = load_result.VDBWatcher()
status_workspace = None

#-------------- get methods ----------------
def get_output_node(context):
    node = getattr(context, "node", None)
    if getattr(node, "bl_idname", "") == "CONTINUUM_FLOW_OUTPUT_NODE":
        return node
    return None

def get_connected_simulation_node(output_node):
    linked_simulation_nodes = _linked_simulation_nodes_from_output_node(output_node)
    return linked_simulation_nodes[0]







#-------------- progress managment ----------------
def draw_bake_progress(self, context):
    window_manager = getattr(context, "window_manager")
    layout = self.layout
    layout.separator_spacer()
    layout.label(text="Bake:")

    progress_row = layout.row(align=True)
    progress_row.ui_units_x = 13

    row = progress_row.row(align=True)
    row.enabled = False
    row.prop(
        window_manager,
        "continuum_flow_bake_progress",
        text=solver_status.progress_text,
        slider=True,
    )

    cancel_row = layout.row(align=True)
    cancel_row.operator(
        "continuum_flow.cancel_bake",
        text="",
        icon="PANEL_CLOSE",
        emboss=False,
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

    window_manager = getattr(bpy.context, "window_manager")

    if hasattr(window_manager, "continuum_flow_bake_progress"):
        window_manager.continuum_flow_bake_progress = float(solver_status.progress)
    ui_redraw()


def _clear_status_progress(context):
    global status_workspace
    workspace = status_workspace or getattr(context, "workspace", None)
    if workspace is not None:
        workspace.status_text_set(None)
    status_workspace = None
    ui_redraw()







def ui_redraw():
    window_manager = getattr(bpy.context, "window_manager")

    for window in window_manager.windows:
        screen = getattr(window, "screen")

        for area in screen.areas:
            if area.type in {"STATUSBAR", "NODE_EDITOR", "PROPERTIES"}:
                area.tag_redraw()


def _normalize_directory_path(path_value):
    if not path_value:
        return None
    try:
        return Path(path_value).resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        return None


def output_directory_has_vdbs(output_directory):
    output_directory = _normalize_directory_path(output_directory)
    return bool(
        output_directory
        and output_directory.exists()
        and output_directory.is_dir()
        and any(output_directory.glob("*.vdb"))
    )


def _linked_simulation_nodes_from_output_node(output_node):
    if output_node is None:
        return []

    result_socket = output_node.inputs.get("Result")
    if result_socket is None or not result_socket.is_linked:
        return []

    simulation_nodes = []
    for link in result_socket.links:
        simulation_node = getattr(link, "from_node", None)
        if getattr(simulation_node, "bl_idname", "") == "CONTINUUM_FLOW_SIMULATION_NODE":
            simulation_nodes.append(simulation_node)
    return simulation_nodes


def _linked_viewer_nodes_from_simulation_node(simulation_node):
    if simulation_node is None:
        return []

    result_socket = simulation_node.outputs.get("Result")
    if result_socket is None or not result_socket.is_linked:
        return []

    viewer_nodes = []
    for link in result_socket.links:
        viewer_node = getattr(link, "to_node", None)
        if getattr(viewer_node, "bl_idname", "") == "CONTINUUM_FLOW_VIEWER_NODE":
            viewer_nodes.append(viewer_node)
    return viewer_nodes


def _simulation_live_preview_enabled(simulation_node):
    viewer_nodes = _linked_viewer_nodes_from_simulation_node(simulation_node)
    if not viewer_nodes:
        return False
    return any(bool(getattr(viewer_node, "live_preview", True)) for viewer_node in viewer_nodes)



def _output_node_last_bake_directory(output_node):
    if output_node is None:
        return None
    return _normalize_directory_path(getattr(output_node, "last_bake_directory", ""))


def _output_node_target_directory(output_node):
    if output_node is None:
        return None
    return _normalize_directory_path(bpy.path.abspath(getattr(output_node, "output_path", "")))


def _is_bake_directory_inside_target(output_node, bake_directory):
    target_directory = _output_node_target_directory(output_node)
    bake_directory = _normalize_directory_path(bake_directory)
    if target_directory is None or bake_directory is None:
        return False

    try:
        bake_directory.relative_to(target_directory)
    except ValueError:
        return False
    return True


def _discover_latest_bake_directory(output_node):
    target_directory = _output_node_target_directory(output_node)
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


def _resolved_output_node_bake_directory(output_node, persist=False):
    bake_directory = _output_node_last_bake_directory(output_node)
    if _is_bake_directory_inside_target(output_node, bake_directory) and output_directory_has_vdbs(bake_directory):
        return bake_directory

    discovered_directory = _discover_latest_bake_directory(output_node)
    if persist and discovered_directory is not None:
        _store_output_node_last_bake_directory(output_node, discovered_directory)
    return discovered_directory


def _store_output_node_last_bake_directory(output_node, output_directory):
    if output_node is None:
        return
    output_node.last_bake_directory = str(output_directory) if output_directory else ""


def output_node_has_baked_data(output_node):
    return _resolved_output_node_bake_directory(output_node, persist=False) is not None




def _set_bake_available_state(is_available, output_directory=None):
    solver_status.bake_available = bool(is_available)
    solver_status.last_output_directory = str(output_directory) if output_directory else None
    ui_redraw()


def _update_bake_available_from_output(output_directory):
    has_vdbs = output_directory_has_vdbs(output_directory)
    output_directory = _normalize_directory_path(output_directory)
    _set_bake_available_state(has_vdbs, output_directory if has_vdbs else None)
    return has_vdbs


def _clear_bake_directory(output_directory):
    if output_directory is None:
        return 0

    output_directory = Path(output_directory).resolve()
    if not output_directory.exists() or not output_directory.is_dir():
        return 0

    deleted_count = sum(1 for _ in output_directory.glob("*.vdb"))

    try:
        shutil.rmtree(output_directory)
    except OSError as exc:
        print("Failed to remove bake directory:", output_directory, exc)
        return 0

    print("Removed bake directory:", output_directory)
    return deleted_count


def _clear_geometry_directory(output_directory, retries=5, retry_delay=0.2):
    output_directory = _normalize_directory_path(output_directory)
    if output_directory is None:
        return False

    geometry_directory = output_directory / "geometry"
    if not geometry_directory.exists():
        return False

    for attempt in range(retries):
        try:
            shutil.rmtree(geometry_directory)
            print("Removed geometry directory:", geometry_directory)
            return True
        except FileNotFoundError:
            return False
        except OSError as exc:
            if attempt == retries - 1:
                print("Failed to remove geometry directory:", geometry_directory, exc)
                return False
            time.sleep(retry_delay)

    return False


def _free_bake_output(output_node):
    output_directory = _resolved_output_node_bake_directory(output_node, persist=False)
    VDBWatcher.clear_loaded_sequence_for_directory(output_directory)
    deleted_count = _clear_bake_directory(output_directory)
    _store_output_node_last_bake_directory(output_node, None)
    return deleted_count


class CONTINUUM_FLOW_OT_cancel_bake(bpy.types.Operator):
    bl_idname = "continuum_flow.cancel_bake"
    bl_label = "Cancel Bake"

    def execute(self, context):
        active_operator = solver_status.active_bake_operator
        if not solver_status.bake_running or active_operator is None:
            self.report({'WARNING'}, "No bake is currently running")
            return {'CANCELLED'}

        print("Bake cancelled by user")
        active_operator.cancel_bake()
        return {'FINISHED'}


class CONTINUUM_FLOW_OT_free_bake(bpy.types.Operator):
    bl_idname = "continuum_flow.free_bake"
    bl_label = "Free Bake"

    def execute(self, context):
        output_node = get_output_node(context)
        deleted_count = _free_bake_output(output_node)
        if deleted_count:
            self.report({'INFO'}, f"Removed {deleted_count} VDB files")
        else:
            self.report({'INFO'}, "No VDB files found to remove")
        return {'FINISHED'}


class CONTINUUM_FLOW_OT_BAKE(bpy.types.Operator):
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
        self._cleanup_done = False
        self._cleanup_lock = threading.Lock()
        self._event_timer = None
        self.cancel_requested = False

        solver_status.bake_running = True
        solver_status.active_bake_operator = self
        set_status_progress(context)

        try:
            self.run_bake(context)
            self._event_timer = context.window_manager.event_timer_add(0.1, window=context.window)
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
            job_result = solver_process.get_job_result(self.job_id)
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
        with self._cleanup_lock:
            if self._cleanup_done:
                return

            self._cleanup_done = True
            solver_status.bake_running = False
            if solver_status.active_bake_operator is self:
                solver_status.active_bake_operator = None

            if self._event_timer is not None and bpy.context is not None:
                try:
                    bpy.context.window_manager.event_timer_remove(self._event_timer)
                except Exception:
                    pass
                self._event_timer = None

            bake_completed_successfully = (
                not self.cancel_requested
                and bool(self.job_result)
                and bool(self.job_result.get("success", False))
            )

            if self.writer_server:
                try:
                    self.writer_server.stop()
                except Exception as exc:
                    print("Failed to stop writer server:", exc)

            if self.cancel_flag_path is not None:
                try:
                    self.cancel_flag_path.unlink(missing_ok=True)
                except Exception as exc:
                    print("Failed to remove cancel flag:", exc)

            if bake_completed_successfully:
                try:
                    VDBWatcher.finish_bake()
                except Exception as exc:
                    print("Failed to load final VDB sequence:", exc)

            VDBWatcher.stop()
            _clear_geometry_directory(self.output_directory)

            has_vdbs = _update_bake_available_from_output(self.output_directory)
            _store_output_node_last_bake_directory(
                self.output_node,
                self.output_directory if has_vdbs else None,
            )
            set_bake_progress(0, 0)
            if bpy.context is not None:
                _clear_status_progress(bpy.context)

    def cancel_bake(self):
        self.cancel_requested = True
        VDBWatcher.stop()

        if self.job_id is not None:
            job_result = solver_process.wait_for_job(self.job_id, timeout=10.0)
            if job_result is None:
                print("Solver worker did not stop promptly; restarting worker")
                solver_process.shutdown_worker(restart=True)
                self.job_result = {
                    "type": "job_finished",
                    "job_id": self.job_id,
                    "success": False,
                    "message": "Bake cancelled",
                }
            else:
                self.job_result = job_result

        self.cleanup()

    def _update_progress_from_loaded_frames(self, loaded_frame_count):
        set_bake_progress(loaded_frame_count, solver_status.progress_total_frames)

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

    def run_bake(self, context):
        config_dict = export_config.build_config_dict(context=context, simulation_node=self.simulation_node)
        bake_directory, config_dict = export_config.export_config_dict(config_dict)

        writer_server = self.launch_writer_manager(config_dict)
        self.writer_server = writer_server
        config_dict["simulations"][0]["outputs"][0]["host_vdb_writer"] = writer_server.endpoint()

        self.bake_directory = Path(bake_directory).resolve()
        self.cancel_flag_path = self.bake_directory / "cancel_requested.flag"
        config_dict["bake_directory"] = str(self.bake_directory)
        meta_config = config_dict.setdefault("meta", {})
        meta_config["cancel_flag_path"] = str(self.cancel_flag_path)
        meta_config["parent_sys_path"] = list(sys.path)

        output_config = config_dict["simulations"][0]["outputs"][0]
        simulation_settings = config_dict["simulations"][0].get("settings") or {}
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
            live_preview_enabled=_simulation_live_preview_enabled(self.simulation_node),
            progress_callback=self._update_progress_from_loaded_frames,
        )

        solver_process.ensure_worker_running(
            wait=True,
            timeout=120.0,
            preload_backend=solver_backend,
        )
        self.job_id = solver_process.start_job(config_dict)

