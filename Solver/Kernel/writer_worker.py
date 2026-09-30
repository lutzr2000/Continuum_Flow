import json
import os
import sys
import numpy as np
import numpy.typing as npt
import openvdb

from multiprocessing import shared_memory
from time import perf_counter
from typing import Any, TypeAlias, TypedDict

BRICK_TILES = 32

IntArray: TypeAlias = npt.NDArray[np.intp]
TileMap: TypeAlias = npt.NDArray[np.int32]
FieldArray: TypeAlias = npt.NDArray[np.float32]
Shape3D: TypeAlias = tuple[int, int, int]


class Brick(TypedDict):
    origin: Shape3D
    slots: IntArray
    local_i: IntArray
    local_j: IntArray
    local_k: IntArray
    shape: Shape3D


def open_array(
    info: dict[str, Any],
) -> tuple[np.ndarray, shared_memory.SharedMemory]:
    """
    Open an existing shared-memory block as a NumPy array.

    Args:
        info: Shared-memory name, array shape and dtype.

    Returns:
        NumPy array and its shared-memory handle.
    """
    shm = shared_memory.SharedMemory(name=info["shm_name"])

    array = np.ndarray(
        tuple(info["shape"]),
        dtype=np.dtype(info["dtype"]),
        buffer=shm.buf,
    )

    return array, shm


def prepare_bricks(
    tile_map: TileMap,
    tile_size: int,
) -> list[Brick]:
    """
    Prepare active tiles grouped into cropped bricks.

    Each brick contains:
        origin: Voxel-space origin of the cropped brick.
        slots: Pool indices of its active tiles.
        local_i, local_j, local_k: Tile indices inside the cropped brick.
        shape: Cropped brick dimensions in tiles.

    Args:
        tile_map: Mapping from tile coordinates to field-pool slots.
        tile_size: Number of voxels along each tile axis.

    Returns:
        List of prepared bricks.
    """
    active_positions = np.argwhere(tile_map >= 0)

    active_tile_count = len(active_positions)

    if active_tile_count == 0:
        return []

    active_slots = tile_map[
        active_positions[:, 0],
        active_positions[:, 1],
        active_positions[:, 2],
    ].copy()

    brick_coordinates = active_positions // BRICK_TILES

    order = np.lexsort(
        (
            brick_coordinates[:, 2],
            brick_coordinates[:, 1],
            brick_coordinates[:, 0],
        )
    )

    active_positions = active_positions[order]
    active_slots = active_slots[order]
    brick_coordinates = brick_coordinates[order]

    group_changes = np.any(
        brick_coordinates[1:] != brick_coordinates[:-1],
        axis=1,
    )

    group_starts = np.concatenate(
        (
            np.array([0], dtype=np.intp),
            np.flatnonzero(group_changes) + 1,
            np.array([active_tile_count], dtype=np.intp),
        )
    )

    bricks: list[Brick] = []

    for start, end in zip(group_starts[:-1], group_starts[1:]):
        positions = active_positions[start:end]
        slots = active_slots[start:end]

        minimum = positions.min(axis=0)
        maximum = positions.max(axis=0)

        shape = maximum - minimum + 1
        local_positions = positions - minimum

        origin: Shape3D = (
            int(minimum[0]) * tile_size,
            int(minimum[1]) * tile_size,
            int(minimum[2]) * tile_size,
        )

        bricks.append(
            {
                "origin": origin,
                "slots": slots,
                "local_i": local_positions[:, 0].astype(np.intp),
                "local_j": local_positions[:, 1].astype(np.intp),
                "local_k": local_positions[:, 2].astype(np.intp),
                "shape": (
                    int(shape[0]),
                    int(shape[1]),
                    int(shape[2]),
                ),
            }
        )

    return bricks


def get_brick_buffer(
    cache: dict[tuple[Shape3D, str], np.ndarray],
    shape: Shape3D,
    tile_size: int,
    dtype: np.dtype,
) -> np.ndarray:
    """
    Retrieve or allocate a reusable dense brick buffer.

    Args:
        cache: Temporary buffers indexed by shape and dtype.
        shape: Brick dimensions in tiles.
        tile_size: Number of voxels along each tile axis.
        dtype: NumPy data type of the field.

    Returns:
        Zero-initialized dense brick array.
    """
    key = (shape, np.dtype(dtype).str)

    buffer = cache.get(key)

    if buffer is None:
        buffer = np.empty(
            (
                shape[0] * tile_size,
                shape[1] * tile_size,
                shape[2] * tile_size,
            ),
            dtype=dtype,
        )

        cache[key] = buffer

    buffer.fill(0)

    return buffer


def copy_field_to_grid(
    grid: Any,
    array: FieldArray,
    bricks: list[Brick],
    tile_size: int,
    brick_cache: dict[tuple[Shape3D, str], np.ndarray],
) -> None:
    """
    Pack active tiles into dense bricks and copy them into OpenVDB.

    Single-tile bricks are copied directly. Multi-tile bricks
    are assembled using NumPy advanced indexing.

    Args:
        grid: Destination OpenVDB grid.
        array: Field data indexed by tile-pool slot.
        bricks: Prepared brick layout.
        tile_size: Number of voxels along each tile axis.
        brick_cache: Reusable temporary brick buffers.
    """
    copy_from_array = grid.copyFromArray

    for brick in bricks:
        slots = brick["slots"]
        origin = brick["origin"]

        if len(slots) == 1:
            copy_from_array(
                array[int(slots[0])],
                ijk=origin,
            )
            continue

        shape = brick["shape"]

        brick_values = get_brick_buffer(
            brick_cache,
            shape,
            tile_size,
            array.dtype,
        )

        brick_view = brick_values.reshape(
            shape[0],
            tile_size,
            shape[1],
            tile_size,
            shape[2],
            tile_size,
        )

        brick_view[
            brick["local_i"],
            :,
            brick["local_j"],
            :,
            brick["local_k"],
            :,
        ] = array[slots]

        copy_from_array(
            brick_values,
            ijk=origin,
        )


def copy_velocity_to_grid(
    grid: Any,
    arrays: tuple[FieldArray, FieldArray, FieldArray],
    bricks: list[Brick],
    tile_size: int,
) -> None:
    """Pack three scalar velocity components into one OpenVDB vector grid."""
    copy_from_array = grid.copyFromArray

    for brick in bricks:
        slots = brick["slots"]
        origin = brick["origin"]

        if len(slots) == 1:
            slot = int(slots[0])
            velocity = np.stack(
                (arrays[0][slot], arrays[1][slot], arrays[2][slot]),
                axis=-1,
            )
        else:
            shape = brick["shape"]
            velocity = np.zeros(
                (
                    shape[0] * tile_size,
                    shape[1] * tile_size,
                    shape[2] * tile_size,
                    3,
                ),
                dtype=arrays[0].dtype,
            )
            velocity_view = velocity.reshape(
                shape[0],
                tile_size,
                shape[1],
                tile_size,
                shape[2],
                tile_size,
                3,
            )

            target = (
                brick["local_i"],
                slice(None),
                brick["local_j"],
                slice(None),
                brick["local_k"],
                slice(None),
            )
            for component, array in enumerate(arrays):
                velocity_view[target + (component,)] = array[slots]

        copy_from_array(velocity, ijk=origin)


def write_vdb(payload: dict[str, Any]) -> None:
    """
    Convert shared-memory field data into OpenVDB grids and save them.

    The writer groups active tiles into cropped bricks, builds one
    grid per field and writes all grids into a single VDB file.

    Only the openvdb.write() call is timed.

    Args:
        payload: Frame metadata, shared-memory descriptors and output path.
    """
    tile_size = int(payload["tile_size"])
    delta = float(payload["delta"])
    nx = int(payload["nx"])
    ny = int(payload["ny"])
    precision = str(payload.get("precision", "float32")).lower()

    transform = openvdb.createLinearTransform(
        voxelSize=delta,
    )

    transform.postTranslate(
        (
            -0.5 * nx * delta,
            -0.5 * ny * delta,
            0.0,
        )
    )

    tile_map, tile_map_shm = open_array(
        payload["tile_map"],
    )

    try:
        bricks = prepare_bricks(
            tile_map,
            tile_size,
        )
    finally:
        del tile_map
        tile_map_shm.close()

    grids = []
    brick_cache: dict[tuple[Shape3D, str], np.ndarray] = {}

    velocity_names = ("velocity_x", "velocity_y", "velocity_z")
    velocity_infos = [payload["fields"].get(name) for name in velocity_names]
    if any(info is not None for info in velocity_infos):
        if not all(info is not None for info in velocity_infos):
            raise ValueError("Velocity output requires x, y and z components")

        velocity_arrays = []
        velocity_shms = []
        try:
            for info in velocity_infos:
                array, shm = open_array(info)
                velocity_arrays.append(array)
                velocity_shms.append(shm)

            grid = openvdb.Vec3SGrid(background=(0.0, 0.0, 0.0))
            grid.name = "velocity"
            grid.transform = transform
            grid.saveFloatAsHalf = precision == "float16"

            copy_velocity_to_grid(
                grid=grid,
                arrays=tuple(velocity_arrays),
                bricks=bricks,
                tile_size=tile_size,
            )

            grid.prune((0.0, 0.0, 0.0))
            grids.append(grid)
        finally:
            velocity_arrays.clear()
            for shm in velocity_shms:
                shm.close()

    for name, info in payload["fields"].items():
        if name in velocity_names:
            continue

        array, shm = open_array(info)

        try:
            grid = openvdb.FloatGrid(background=0.0)
            grid.name = name
            grid.transform = transform
            grid.saveFloatAsHalf = precision == "float16"

            copy_field_to_grid(
                grid=grid,
                array=array,
                bricks=bricks,
                tile_size=tile_size,
                brick_cache=brick_cache,
            )

            grid.prune()
            grids.append(grid)

        finally:
            del array
            shm.close()

    brick_cache.clear()

    output_path = payload["output_path"]
    output_dir = os.path.dirname(output_path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    temp_path = output_path + ".tmp"

    try:
        openvdb.write(
            temp_path,
            grids=grids,
        )

        os.replace(
            temp_path,
            output_path,
        )

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def main() -> None:
    """
    Process JSON-line write requests from the parent process.

    Reads frame payloads from stdin and sends JSON status
    responses to stdout. The quit command stops the worker.
    """
    for line in sys.stdin:
        line = line.strip()

        if line == "__QUIT__":
            break

        try:
            payload = json.loads(line)
            write_vdb(payload)

            response = {
                "status": "ok",
            }

        except Exception as exc:
            response = {
                "status": "error",
                "message": str(exc),
            }

        print(
            json.dumps(response),
            flush=True,
        )


if __name__ == "__main__":
    main()
