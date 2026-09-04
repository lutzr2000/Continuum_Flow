from pxr import Usd, UsdGeom
import numpy as np
from pathlib import Path


def load_usd_file(filepath, frame):
    stage = Usd.Stage.Open(str(filepath))
    meshes = []

    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue

        mesh = UsdGeom.Mesh(prim)

        meshes.append({
            "points": np.asarray(
                mesh.GetPointsAttr().Get(frame),
                dtype=np.float32,
            ),
            "face_counts": np.asarray(
                mesh.GetFaceVertexCountsAttr().Get(frame),
                dtype=np.int32,
            ),
            "face_indices": np.asarray(
                mesh.GetFaceVertexIndicesAttr().Get(frame),
                dtype=np.int32,
            ),
        })

    return meshes


def update_masks(t, simulation):
    settings = simulation["settings"]
    timeline = simulation["animation_timeline"]
    output_path = Path(simulation["outputs"][0]["output_path"])

    start_frame = settings["start_frame"]
    end_frame = settings["end_frame"]
    frame_before = min(
        max(start_frame, start_frame + int(t * timeline["fps"])),
        end_frame - 1,
    )

    frame_after = min(frame_before + 1, end_frame - 1)

    obstacle_file = next((
        output_path / geometry["mesh_file"]
        for obstacle in simulation.get("obstacles", [])
        for geometry in obstacle.get("geometry_inputs", [])
        if geometry.get("mesh_file")
    ), None)

    source_files = dict.fromkeys(
        output_path / geometry["mesh_file"]
        for source in simulation.get("sources", [])
        for geometry in source.get("geometry_inputs", [])
        if geometry.get("mesh_file")
    )

    if obstacle_file:
        load_usd_file(obstacle_file, frame_before)
        load_usd_file(obstacle_file, frame_after)

    for source_file in source_files:
        load_usd_file(source_file, frame_before)
        load_usd_file(source_file, frame_after)
