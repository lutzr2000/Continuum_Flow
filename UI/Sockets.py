import bpy
from bpy.props import IntProperty


class BlenderCFDIntSocket(bpy.types.NodeSocket):
    """
    Integer socket used by BlenderCFD nodes to expose bounded scalar values.
    """

    bl_idname = "BLENDERCFD_INT_SOCKET"
    bl_label = "BlenderCFD Integer"

    value: IntProperty(  # type: ignore
        name="Value",
        default=128,
        min=32,
        max=4096,
        soft_min=32,
        soft_max=4096,
    )

    def draw(self, context, layout, node, text):
        """Draw the socket UI in the node editor."""
        if self.is_output or self.is_linked:
            layout.label(text=text)
        else:
            layout.prop(self, "value", text=text)

    def draw_color(self, context, node):
        """Return the display color of the socket."""
        return (0.90, 0.55, 0.20, 1.0)


class BlenderCFDLinkSocket(bpy.types.NodeSocket):
    """
    Generic link socket used to connect logical BlenderCFD node outputs.
    """

    bl_idname = "BLENDERCFD_LINK_SOCKET"
    bl_label = "BlenderCFD Link"

    def draw(self, context, layout, node, text):
        """Draw the socket label in the node editor."""
        layout.label(text=text)

    def draw_color(self, context, node):
        """Return the display color of the socket."""
        return (0.90, 0.55, 0.20, 1.0)


class BlenderCFDForceSocket(bpy.types.NodeSocket):
    """
    Dedicated socket used for force-related links in the BlenderCFD graph.
    """

    bl_idname = "BLENDERCFD_FORCE_SOCKET"
    bl_label = "BlenderCFD Force"

    def draw(self, context, layout, node, text):
        """Draw the socket label in the node editor."""
        layout.label(text=text)

    def draw_color(self, context, node):
        """Return the display color of the socket."""
        return (0.45, 0.65, 0.95, 1.0)


class BlenderCFDResultSocket(bpy.types.NodeSocket):
    """
    Result socket used by the simulation node to expose the final output link.
    """

    bl_idname = "BLENDERCFD_RESULT_SOCKET"
    bl_label = "BlenderCFD Result"

    def draw(self, context, layout, node, text):
        """Draw the socket label in the node editor."""
        layout.label(text=text)

    def draw_color(self, context, node):
        """Return the display color of the socket."""
        return (0.65, 0.35, 0.85, 1.0)


class BlenderCFDReferenceFrameSocket(bpy.types.NodeSocket):
    """
    Socket used for reference-frame links in the BlenderCFD graph.
    """

    bl_idname = "BLENDERCFD_REFERENCE_FRAME_SOCKET"
    bl_label = "BlenderCFD Reference Frame"

    def draw(self, context, layout, node, text):
        """Draw the socket label in the node editor."""
        layout.label(text=text)

    def draw_color(self, context, node):
        """Return the display color of the socket."""
        return (0.95, 0.45, 0.75, 1.0)


classes = (
    BlenderCFDIntSocket,
    BlenderCFDLinkSocket,
    BlenderCFDForceSocket,
    BlenderCFDResultSocket,
    BlenderCFDReferenceFrameSocket,
)
