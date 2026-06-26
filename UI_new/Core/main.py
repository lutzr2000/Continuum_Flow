import bpy
import json
import subprocess
from pathlib import Path
from . import export_config
from . import writer_manager

def _venv_python_path():
    addon_root = Path(__file__).resolve().parents[2]
    return addon_root / "ContinuumFlow_env" / "Scripts" / "python.exe"


class main(bpy.types.Operator):
    bl_idname = "continuum_flow.bake"
    bl_label = "Bake"

    def execute(self, context):
        self.do_bake(context)
        return {'FINISHED'}
    
    def launch_writer_manager(self, config_dict):
        output_config = ((config_dict.get("simulations") or [{}])[0].get("outputs") or [{}])[0]
        performance_config = output_config.get("performance") or {}
        writer_process_count = int(performance_config.get("writer_processes", 4))
        config_path = output_config.get("_writer_config_path")

        server = writer_manager.HostVDBWriterServer(
            writer_process_count=writer_process_count,
            config_path=config_path,
        )
        server.start()
        return server

    def do_bake(self, context):
        config_dict = export_config.build_config_dict()
        config_path, config_dict = export_config.export_config_dict(config_dict)
        config_dict["simulations"][0]["outputs"][0]["_writer_config_path"] = str(config_path)

        writer_server = self.launch_writer_manager(config_dict)
        config_dict["simulations"][0]["outputs"][0]["host_vdb_writer"] = writer_server.endpoint()
        config_dict["simulations"][0]["outputs"][0].pop("_writer_config_path", None)
        config_path.write_text(json.dumps(config_dict, indent=2), encoding="utf-8")

        addon_root = Path(__file__).resolve().parents[2]

        process = subprocess.Popen(
            [
                str(_venv_python_path()),
                "-m",
                "Solver.General.main",
                str(config_path),
            ],
            cwd=str(addon_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        try:
            for line in process.stdout:
                print("[Solver]", line, end="")
        finally:
            print("Solver Exit Code:", process.wait())
            writer_server.stop()
