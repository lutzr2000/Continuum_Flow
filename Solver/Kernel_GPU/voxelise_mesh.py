"""CUDA voxelization of object-local base masks only."""

import math
from pathlib import Path

import numpy as np
import trimesh
from numba import cuda

import Solver.Kernel_GPU.kernel_config as kernel_config


def voxelise_all_meshes(delta, mesh_objects, bake_path):
    """Build base-mask entries only, preserving the mesh object metadata."""

    base_masks = []

    for mesh_object in mesh_objects or ():
        if not mesh_object:
            continue

        path = Path(bake_path) / mesh_object.get("mesh_file")
        mesh = trimesh.load_mesh(str(path), process=False)

        triangles = np.ascontiguousarray(
            mesh.vertices[mesh.faces],
            dtype=np.float32,
        )

        voxels = voxelize_triangles(
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


def voxelize_triangles(triangles, delta):
    if triangles.size == 0:
        return None

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

    triangles_gpu = cuda.to_device(triangles)

    mask = cuda.to_device(
        np.zeros(
            shape,
            dtype=np.bool_,
        )
    )

    blocks = tuple(
        (shape[i] + kernel_config.THREADS_PER_BLOCK_3D[i] - 1)
        // kernel_config.THREADS_PER_BLOCK_3D[i]
        for i in range(3)
    )

    surface[
        blocks,
        kernel_config.THREADS_PER_BLOCK_3D,
    ](
        triangles_gpu,
        mask,
        np.float32(delta),
        origin[0],
        origin[1],
        origin[2],
    )

    return {
        "mask": mask,
        "origin": origin,
        "bounds_min": origin,
        "bounds_max": np.asarray(
            hi * delta,
            dtype=np.float32,
        ),
    }


@cuda.jit(cache=True)
def surface(
    triangles,
    mask,
    delta,
    ox,
    oy,
    oz,
):
    i, j, k = cuda.grid(3)

    if i >= mask.shape[0] or j >= mask.shape[1] or k >= mask.shape[2]:
        return

    x = ox + i * delta
    y = oy + j * delta
    z = oz + k * delta

    for t in range(triangles.shape[0]):
        tri = triangles[t]

        xmin = min(
            tri[0, 0],
            tri[1, 0],
            tri[2, 0],
        )
        xmax = max(
            tri[0, 0],
            tri[1, 0],
            tri[2, 0],
        )

        ymin = min(
            tri[0, 1],
            tri[1, 1],
            tri[2, 1],
        )
        ymax = max(
            tri[0, 1],
            tri[1, 1],
            tri[2, 1],
        )

        zmin = min(
            tri[0, 2],
            tri[1, 2],
            tri[2, 2],
        )
        zmax = max(
            tri[0, 2],
            tri[1, 2],
            tri[2, 2],
        )

        if (
            x < xmin - delta
            or x > xmax + delta
            or y < ymin - delta
            or y > ymax + delta
            or z < zmin - delta
            or z > zmax + delta
        ):
            continue

        ax = tri[1, 0] - tri[0, 0]
        ay = tri[1, 1] - tri[0, 1]
        az = tri[1, 2] - tri[0, 2]

        bx = tri[2, 0] - tri[0, 0]
        by = tri[2, 1] - tri[0, 1]
        bz = tri[2, 2] - tri[0, 2]

        nx = ay * bz - az * by
        ny = az * bx - ax * bz
        nz = ax * by - ay * bx

        length = math.sqrt(nx * nx + ny * ny + nz * nz)

        if length == 0.0:
            continue

        distance = abs(
            nx * (x - tri[0, 0]) + ny * (y - tri[0, 1]) + nz * (z - tri[0, 2])
        )

        if distance > delta * 0.87 * length:
            continue

        # Surface voxel.
        mask[i, j, k] = 1

        # Normalize outward triangle normal.
        nx /= length
        ny /= length
        nz /= length

        # Move one voxel inward:
        # inward = negative outward normal.
        ii = i
        jj = j
        kk = k

        anx = abs(nx)
        any_ = abs(ny)
        anz = abs(nz)

        if anx >= any_ and anx >= anz:
            if nx > 0.0:
                ii -= 1
            else:
                ii += 1

        elif any_ >= anz:
            if ny > 0.0:
                jj -= 1
            else:
                jj += 1

        else:
            if nz > 0.0:
                kk -= 1
            else:
                kk += 1

        if (
            0 <= ii < mask.shape[0]
            and 0 <= jj < mask.shape[1]
            and 0 <= kk < mask.shape[2]
        ):
            mask[ii, jj, kk] = 1

        return
