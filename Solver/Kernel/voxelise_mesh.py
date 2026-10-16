from typing import Any
from pathlib import Path

import numpy as np
import pyopencl as cl
import trimesh

from . import helper


def voxelise_all_meshes(
    context: cl.Context,
    queue: cl.CommandQueue,
    surface_kernel: cl.Kernel,
    delta: float,
    geometry_inputs: list[dict[str, Any]] | None,
    bake_path: str,
) -> list[dict[str, Any]]:
    """Build base-mask entries only, preserving the mesh object metadata."""

    base_masks = []

    for mesh_object in geometry_inputs or ():
        if not mesh_object:
            continue

        path = Path(bake_path) / mesh_object.get("mesh_file")
        mesh = trimesh.load_mesh(str(path), process=False)

        triangles = np.ascontiguousarray(
            mesh.vertices[mesh.faces],
            dtype=np.float32,
        )

        voxels = voxelize_triangles(
            context,
            queue,
            surface_kernel,
            triangles,
            delta,
        )

        if voxels is not None:
            base_masks.append(
                {
                    "mesh_object": mesh_object,
                    "voxels": voxels,
                }
            )

    return base_masks


def voxelize_triangles(
    context: cl.Context,
    queue: cl.CommandQueue,
    surface_kernel: cl.Kernel,
    triangles: np.ndarray,
    delta: float,
) -> dict[str, Any] | None:
    """
    Rasterize triangle surfaces into a padded object-local Boolean voxel mask.
    """
    if triangles.size == 0:
        return None

    triangles = np.asarray(triangles, dtype=np.float32)
    vertices = triangles.reshape(-1, 3)

    bounds_min = vertices.min(axis=0).astype(np.float32)
    bounds_max = vertices.max(axis=0).astype(np.float32)

    lo = np.floor(bounds_min / delta).astype(np.int32) - 1
    hi = np.ceil(bounds_max / delta).astype(np.int32) + 1

    shape = tuple((hi - lo + 1).tolist())

    origin = np.asarray(
        lo * delta,
        dtype=np.float32,
    )

    triangles_gpu = helper.to_device(context, triangles)

    voxel_count = int(np.prod(shape))
    mask_word_count = (voxel_count + 31) // 32

    mask = helper.device_array(context, (mask_word_count,), dtype=np.uint32)
    cl.enqueue_fill_buffer(queue, mask, np.uint32(0), 0, mask.size)

    surface_kernel(
        queue,
        (int(triangles.shape[0]),),
        None,
        triangles_gpu,
        mask,
        np.int32(triangles.shape[0]),
        np.int32(shape[0]),
        np.int32(shape[1]),
        np.int32(shape[2]),
        np.float32(delta),
        np.float32(origin[0]),
        np.float32(origin[1]),
        np.float32(origin[2]),
    )

    return {
        "mask": mask,
        "shape": shape,
        "origin": origin,
        "bounds_min": origin,
        "bounds_max": np.asarray(
            hi * delta,
            dtype=np.float32,
        ),
    }
