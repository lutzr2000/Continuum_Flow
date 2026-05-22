ACTIVE_TILE_SIZE = 4
ACTIVE_TILE_PADDING_CELLS = 4


def active_tile_shape(shape, tile_size=ACTIVE_TILE_SIZE):
    """
    Return the number of active-mask tiles along each axis.
    """
    return tuple((int(shape[axis]) + tile_size - 1) // tile_size for axis in range(3))


def active_tile_padding_tiles(
    tile_size=ACTIVE_TILE_SIZE, padding_cells=ACTIVE_TILE_PADDING_CELLS
):
    """
    Return the tile-radius used to dilate the active scalar region.
    """
    return max(0, (int(padding_cells) + tile_size - 1) // tile_size)
