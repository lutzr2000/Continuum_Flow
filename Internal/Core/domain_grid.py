import math


TILE_SIZE = 4


def cell_count(length, resolution, tile_size=TILE_SIZE):
    """Return enough cells for *length*, rounded up to complete tiles."""
    length = float(length)
    resolution = float(resolution)
    tile_size = int(tile_size)

    if resolution <= 0.0:
        raise ValueError("resolution must be greater than zero")
    if tile_size <= 0:
        raise ValueError("tile_size must be greater than zero")

    required_cells = max(1, math.ceil(length / resolution))
    return math.ceil(required_cells / tile_size) * tile_size


def grid_shape(domain_node):
    """Return the tile-aligned cell counts for a domain node."""
    resolution = float(domain_node.resolution)
    return tuple(
        cell_count(getattr(domain_node, axis), resolution)
        for axis in ("lx", "ly", "lz")
    )


def dimensions(domain_node):
    """Return the actual exported domain dimensions in Blender length units."""
    resolution = float(domain_node.resolution)
    return tuple(count * resolution for count in grid_shape(domain_node))
