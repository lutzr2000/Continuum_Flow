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
    Sample one evaluated Blender particle system and write an uncompressed NPZ.
    """
    scene = bpy.context.scene
    current_frame = int(scene.frame_current)
    times = []
    offsets = [0]
    positions = []
    sizes = []
    velocities = []

    try:
        for frame in range(int(start_frame), int(end_frame)):
            scene.frame_set(frame)
            depsgraph = bpy.context.evaluated_depsgraph_get()
            object_eval = source_object.evaluated_get(depsgraph)
            particle_system = object_eval.particle_systems.get(particle_system_name)

            times.append(float(frame - int(start_frame)) / float(max(1, fps)))

            if particle_system is not None:
                for particle in particle_system.particles:
                    if getattr(particle, "alive_state", "ALIVE") != "ALIVE":
                        continue

                    positions.append(tuple(float(value) for value in particle.location))
                    sizes.append(float(particle.size))
                    velocities.append(
                        tuple(float(value) for value in particle.velocity)
                    )

            offsets.append(len(positions))
    finally:
        scene.frame_set(current_frame)

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        file_path,
        times=np.asarray(times, dtype=np.float32),
        offsets=np.asarray(offsets, dtype=np.uint64),
        positions=np.asarray(positions, dtype=np.float32).reshape((-1, 3)),
        sizes=np.asarray(sizes, dtype=np.float32),
        velocities=np.asarray(velocities, dtype=np.float32).reshape((-1, 3)),
    )
