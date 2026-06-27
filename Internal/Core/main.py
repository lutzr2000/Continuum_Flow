import bpy
import json
import shutil
import subprocess
import threading
from pathlib import Path

from . import export_config
from . import writer_manager
from . import load_result
from . import solver_status


def _venv_python_path():
    addon_root = Path(__file__).resolve().parents[2]
    return addon_root / "ContinuumFlow_env" / "Scripts" / "python.exe"


def _read_solver_stdout(process):
    for line in process.stdout:
        print("[Solver]", line, end="")


_vdb_watcher = load_result.VDBWatcher()
_status_workspace = None


def draw_bake_progress(self, context):
    if not solver_status.bake_running:
        return

    window_manager = getattr(context, "window_manager", None)
    if window_manager is None:
        return

    layout = self.layout
    layout.separator_spacer()
    layout.label(text="Bake:")

    progress_row = layout.row(align=True)
    progress_row.ui_units_x = 13

    if hasattr(progress_row, "progress"):
        progress_row.progress(
            factor=float(solver_status.progress),
            type='BAR',
            text=solver_status.progress_text,
        )
    elif hasattr(window_manager, "continuum_flow_bake_progress"):
        fallback_row = progress_row.row(align=True)
        fallback_row.enabled = False
        fallback_row.prop(
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

def _tag_ui_redraw():
    window_manager = getattr(bpy.context, "window_manager", None)
    if window_manager is None:
        return

    for window in window_manager.windows:
        screen = getattr(window, "screen", None)
        if screen is None:
            continue

        for area in screen.areas:
            if area.type == "STATUSBAR":
                area.tag_redraw()


def _set_status_progress(context):
    global _status_workspace
    workspace = getattr(context, "workspace", None)
    if workspace is None:
        return
    _status_workspace = workspace
    workspace.status_text_set(draw_bake_progress)
    _tag_ui_redraw()


def _clear_status_progress(context):
    global _status_workspace
    workspace = _status_workspace or getattr(context, "workspace", None)
    if workspace is not None:
        workspace.status_text_set(None)
    _status_workspace = None
    _tag_ui_redraw()


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


def _resolve_output_node_from_context(context):
    node = getattr(context, "node", None)
    if getattr(node, "bl_idname", "") == "CONTINUUM_FLOW_OUTPUT_NODE":
        return node
    return None


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


def refresh_bake_state_from_output_nodes():
    active_output_directory = None

    for node_tree in getattr(bpy.data, "node_groups", ()):
        for node in getattr(node_tree, "nodes", ()): 
            if getattr(node, "bl_idname", "") != "CONTINUUM_FLOW_OUTPUT_NODE":
                continue

            candidate_directory = _resolved_output_node_bake_directory(node, persist=False)
            if candidate_directory is not None:
                active_output_directory = candidate_directory
                break

        if active_output_directory is not None:
            break

    _set_bake_available_state(active_output_directory is not None, active_output_directory)
    return active_output_directory


def _set_bake_progress(current_frames, total_frames):
    current_frames = max(0, int(current_frames or 0))
    total_frames = max(0, int(total_frames or 0))

    solver_status.progress_current_frames = current_frames
    solver_status.progress_total_frames = total_frames

    if total_frames > 0:
        progress = min(1.0, float(current_frames) / float(total_frames))
        percent = int(round(progress * 100.0))
        solver_status.progress = progress
        solver_status.progress_text = f"{percent}%"
    else:
        solver_status.progress = 0.0
        solver_status.progress_text = "0%"

    window_manager = getattr(bpy.context, "window_manager", None)
    if window_manager is None:
        return

    if hasattr(window_manager, "continuum_flow_bake_progress"):
        window_manager.continuum_flow_bake_progress = float(solver_status.progress)
    _tag_ui_redraw()


def _set_bake_available_state(is_available, output_directory=None):
    solver_status.bake_available = bool(is_available)
    solver_status.last_output_directory = str(output_directory) if output_directory else None
    _tag_ui_redraw()


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


class main(bpy.types.Operator):
    bl_idname = "continuum_flow.bake"
    bl_label = "Bake"

    def execute(self, context):
        self.output_node = _resolve_output_node_from_context(context)
        refresh_bake_state_from_output_nodes()

        if solver_status.bake_running:
            self.report({'WARNING'}, "Bake is already running")
            return {'CANCELLED'}

        if output_node_has_baked_data(self.output_node):
            deleted_count = self.free_bake()
            if deleted_count:
                self.report({'INFO'}, f"Removed {deleted_count} VDB files")
            else:
                self.report({'INFO'}, "No VDB files found to remove")
            return {'FINISHED'}

        self.process = None
        self.writer_server = None
        self.bake_directory = None
        self.output_directory = None
        self._cleanup_done = False
        self._cleanup_lock = threading.Lock()

        solver_status.bake_running = True
        solver_status.active_bake_operator = self
        _set_status_progress(context)

        self.do_bake(context)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'ESC':
            print("Bake cancelled by user")
            self.cancel_bake()
            return {'CANCELLED'}

        if self.process and self.process.poll() is not None:
            self.cleanup()
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

            _vdb_watcher.stop()

            if self.writer_server:
                try:
                    self.writer_server.stop()
                except Exception as exc:
                    print("Failed to stop writer server:", exc)


            has_vdbs = _update_bake_available_from_output(self.output_directory)
            _store_output_node_last_bake_directory(
                self.output_node,
                self.output_directory if has_vdbs else None,
            )
            refresh_bake_state_from_output_nodes()
            _set_bake_progress(0, 0)
            if bpy.context is not None:
                _clear_status_progress(bpy.context)

    def cancel_bake(self):
        _vdb_watcher.stop()

        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
            except Exception as exc:
                print("Failed to terminate solver:", exc)

        self.cleanup()

    def free_bake(self):
        output_directory = _resolved_output_node_bake_directory(self.output_node, persist=False)
        if output_directory is None:
            output_directory = solver_status.last_output_directory

        _vdb_watcher.clear_loaded_sequence()
        deleted_count = _clear_bake_directory(output_directory)
        _store_output_node_last_bake_directory(self.output_node, None)
        refresh_bake_state_from_output_nodes()
        return deleted_count

    def _update_progress_from_loaded_frames(self, loaded_frame_count):
        _set_bake_progress(loaded_frame_count, solver_status.progress_total_frames)

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

    def do_bake(self, context):
        config_dict = export_config.build_config_dict()
        bake_directory, config_dict = export_config.export_config_dict(config_dict)

        writer_server = self.launch_writer_manager(config_dict)
        config_dict["simulations"][0]["outputs"][0]["host_vdb_writer"] = writer_server.endpoint()

        self.bake_directory = Path(bake_directory).resolve()
        self.writer_server = writer_server

        output_config = config_dict["simulations"][0]["outputs"][0]
        simulation_settings = config_dict["simulations"][0].get("settings") or {}
        start_frame = int(simulation_settings.get("start_frame", 1))
        end_frame = int(simulation_settings.get("end_frame", start_frame))
        total_frames = max(0, (end_frame - start_frame) + 1)
        _set_bake_progress(0, total_frames)

        vdb_output_dir = Path(output_config["output_path"]).resolve()
        self.output_directory = vdb_output_dir

        _vdb_watcher.start(
            vdb_output_dir,
            progress_callback=self._update_progress_from_loaded_frames,
        )

        addon_root = Path(__file__).resolve().parents[2]

        self.process = subprocess.Popen(
            [
                str(_venv_python_path()),
                "-u",
                "-m",
                "Solver.General.main",
                str(self.bake_directory),
            ],
            cwd=str(addon_root),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        if self.process.stdin is not None:
            self.process.stdin.write(json.dumps(config_dict))
            self.process.stdin.close()

        threading.Thread(
            target=_read_solver_stdout,
            args=(self.process,),
            daemon=True,
        ).start()


