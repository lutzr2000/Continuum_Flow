bl_info = {
    "name": "Continuum Flow",
    "author": "...",
    "version": (0, 1, 0),
    "blender": (5, 0, 0),
    "category": "Node",
}

from .Internal.UI import register as registry


def register():
    registry.register()


def unregister():
    registry.unregister()