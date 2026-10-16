from pathlib import Path

import bpy
import numpy as np


def export_particle_system_as_npz(
    source_object,
    particle_system_name,
    file_path,
    start_frame,
    end_frame,
    fps,
):
    """
    Sample an evaluated Blender particle system and write an uncompressed NPZ.

    If no particle system is selected, the vertices of the evaluated mesh are
    exported through the same particle timeline format instead.  Evaluating and
    converting the object for every frame is important here: it includes the
    complete modifier stack and Geometry Nodes, whose vertex count may change.
    """
    scene = bpy.context.scene
    current_frame = int(scene.frame_current)
    times = []
    offsets = [0]
    frame_positions = []
    frame_ids = []
    frame_sizes = []
    frame_velocities = []

    try:
        for frame in range(int(start_frame), int(end_frame)):
            scene.frame_set(frame)
            depsgraph = bpy.context.evaluated_depsgraph_get()
            object_eval = source_object.evaluated_get(depsgraph)
            particle_system = None
            if particle_system_name:
                particle_system = object_eval.particle_systems.get(particle_system_name)

            times.append(float(frame - int(start_frame)) / float(max(1, fps)))

            if particle_system is not None:
                positions = []
                particle_ids = []
                sizes = []
                velocities = []
                for particle_id, particle in enumerate(particle_system.particles):
                    if getattr(particle, "alive_state", "ALIVE") != "ALIVE":
                        continue

                    positions.append(tuple(float(value) for value in particle.location))
                    particle_ids.append(particle_id)
                    sizes.append(float(particle.size))
                    velocities.append(
                        tuple(float(value) for value in particle.velocity)
                    )

                frame_positions.append(
                    np.asarray(list(positions), dtype=np.float32).reshape((-1, 3))
                )
                frame_ids.append(np.asarray(particle_ids, dtype=np.uint32))
                frame_sizes.append(np.asarray(sizes, dtype=np.float32))
                frame_velocities.append(
                    np.asarray(list(velocities), dtype=np.float32).reshape((-1, 3))
                )
            elif not particle_system_name:
                positions = vertex_positions(object_eval, depsgraph)
                frame_positions.append(positions)
                frame_ids.append(np.arange(len(positions), dtype=np.uint32))
                frame_sizes.append(np.ones(len(positions), dtype=np.float32))
                frame_velocities.append(None)
            else:
                frame_positions.append(np.empty((0, 3), dtype=np.float32))
                frame_ids.append(np.empty((0,), dtype=np.uint32))
                frame_sizes.append(np.empty((0,), dtype=np.float32))
                frame_velocities.append(np.empty((0, 3), dtype=np.float32))

            offsets.append(offsets[-1] + len(frame_positions[-1]))
    finally:
        scene.frame_set(current_frame)

    vertex_velocities(frame_positions, frame_velocities, fps)

    positions = (
        np.concatenate(frame_positions).astype(np.float32, copy=False)
        if frame_positions and any(len(frame) for frame in frame_positions)
        else np.empty((0, 3), dtype=np.float32)
    )
    particle_ids = (
        np.concatenate(frame_ids).astype(np.uint32, copy=False)
        if frame_ids and any(len(frame) for frame in frame_ids)
        else np.empty((0,), dtype=np.uint32)
    )
    sizes = (
        np.concatenate(frame_sizes).astype(np.float32, copy=False)
        if frame_sizes and any(len(frame) for frame in frame_sizes)
        else np.empty((0,), dtype=np.float32)
    )
    velocities = (
        np.concatenate(frame_velocities).astype(np.float32, copy=False)
        if frame_velocities and any(len(frame) for frame in frame_velocities)
        else np.empty((0, 3), dtype=np.float32)
    )

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        file_path,
        times=np.asarray(times, dtype=np.float32),
        offsets=np.asarray(offsets, dtype=np.uint64),
        positions=positions,
        particle_ids=particle_ids,
        sizes=sizes,
        velocities=velocities,
    )


def vertex_positions(object_eval, depsgraph):
    """Return evaluated mesh vertices in world space and release the temp mesh."""
    if getattr(object_eval, "type", None) != "MESH":
        return np.empty((0, 3), dtype=np.float32)

    mesh = None
    try:
        mesh = object_eval.to_mesh(
            preserve_all_data_layers=False,
            depsgraph=depsgraph,
        )
        matrix_world = object_eval.matrix_world
        return np.asarray(
            list(matrix_world @ vertex.co for vertex in mesh.vertices),
            dtype=np.float32,
        ).reshape((-1, 3))
    finally:
        if mesh is not None:
            object_eval.to_mesh_clear()


def vertex_velocities(frame_positions, frame_velocities, fps):
    """Calculate per-index mesh velocities while allowing changing vertex counts."""
    frame_rate = float(max(1, fps))

    for frame_index, velocities in enumerate(frame_velocities):
        if velocities is not None:
            continue

        current = frame_positions[frame_index]
        result = np.zeros_like(current, dtype=np.float32)

        if frame_index + 1 < len(frame_positions):
            other = frame_positions[frame_index + 1]
            shared_count = min(len(current), len(other))
            result[:shared_count] = (
                other[:shared_count] - current[:shared_count]
            ) * frame_rate
        elif frame_index > 0:
            other = frame_positions[frame_index - 1]
            shared_count = min(len(current), len(other))
            result[:shared_count] = (
                current[:shared_count] - other[:shared_count]
            ) * frame_rate

        frame_velocities[frame_index] = result
