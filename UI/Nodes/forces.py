"""Continuum Flow force nodes with shared socket and UI behavior."""

import bpy
try:
    import continuum_flow_general_nodes as GeneralNodes
except ImportError:
    import blendercfd_general_nodes as GeneralNodes
from bpy.props import FloatProperty, FloatVectorProperty

BlenderCFDForceSocket = GeneralNodes.BlenderCFDForceSocket
BlenderCFDNodeTree = GeneralNodes.BlenderCFDNodeTree
is_bake_running = GeneralNodes.is_bake_running


class _BlenderCFDBaseForceNode(GeneralNodes.BlenderCFDBaseNode):
    """Shared socket and property drawing helpers for force nodes."""

    draw_property_names = ()

    def _ensure_output_socket(self):
        socket = self.outputs.get("Force")
        if socket is None:
            socket = self.outputs.new(BlenderCFDForceSocket.bl_idname, "Force")
        return socket

    def _sync_node(self):
        self._ensure_output_socket()

    def draw_buttons(self, context, layout):
        self._set_layout_enabled(context, layout)
        col = layout.column(align=True)
        for property_name in self.draw_property_names:
            col.prop(self, property_name)


class BlenderCFDForceConstantNode(_BlenderCFDBaseForceNode):
    """Node used to define a constant force vector for the CFD simulation."""

    bl_idname = "BLENDERCFD_FORCE_CONSTANT_NODE"
    bl_label = "Force Constant"
    bl_icon = "FORCE_FORCE"
    bl_width_default = 220.0
    bl_width_min = 200.0
    bl_width_max = 320.0
    draw_property_names = ("fx", "fy", "fz")

    fx: FloatProperty(name="Fx", default=0.0, description="Force in the x-direction", options={"ANIMATABLE"})  # type: ignore
    fy: FloatProperty(name="Fy", default=0.0, description="Force in the y-direction", options={"ANIMATABLE"})  # type: ignore
    fz: FloatProperty(name="Fz", default=0.0, description="Force in the z-direction", options={"ANIMATABLE"})  # type: ignore


class BlenderCFDForceSwirlNode(_BlenderCFDBaseForceNode):
    """Node used to define a swirl force around an origin and axis."""

    bl_idname = "BLENDERCFD_FORCE_SWIRL_NODE"
    bl_label = "Force Swirl"
    bl_icon = "EMPTY_ARROWS"
    bl_width_default = 220.0
    bl_width_min = 200.0
    bl_width_max = 340.0
    draw_property_names = ("strength", "origin", "axis", "radius")

    strength: FloatProperty(name="Strength", default=0.0, description="Strength of swirl", options={"ANIMATABLE"})  # type: ignore
    origin: FloatVectorProperty(name="Origin", size=3, subtype="XYZ", default=(0.0, 0.0, 0.0), unit="LENGTH", description="Origin of the swirl, flow will rotate about this point", options={"ANIMATABLE"})  # type: ignore
    axis: FloatVectorProperty(name="Axis", size=3, subtype="XYZ", default=(0.0, 0.0, 1.0), description="Axis of swirl, flow will rotate around this axis", options={"ANIMATABLE"})  # type: ignore
    radius: FloatProperty(name="Radius", default=1.0, min=0.0, unit="LENGTH", description="Radius until which the swirl is applied", options={"ANIMATABLE"})  # type: ignore


class BlenderCFDForcePointNode(_BlenderCFDBaseForceNode):
    """Node used to define a smoothed divergence source around one origin."""

    bl_idname = "BLENDERCFD_FORCE_POINT_NODE"
    bl_label = "Force Point"
    bl_icon = "EMPTY_ARROWS"
    bl_width_default = 220.0
    bl_width_min = 200.0
    bl_width_max = 340.0
    draw_property_names = ("strength", "origin", "radius")

    strength: FloatProperty(name="Strength", default=0.0, description="Strength of force", options={"ANIMATABLE"})  # type: ignore
    origin: FloatVectorProperty(name="Origin", size=3, subtype="XYZ", default=(0.0, 0.0, 0.0), unit="LENGTH", description="Origin of the point force", options={"ANIMATABLE"})  # type: ignore
    radius: FloatProperty(name="Radius", default=1.0, min=0.0, unit="LENGTH", description="Radius in which the force is applied", options={"ANIMATABLE"})  # type: ignore


class BlenderCFDForceTurbulenceNode(_BlenderCFDBaseForceNode):
    """Node used to define a procedural turbulence force field."""

    bl_idname = "BLENDERCFD_FORCE_TURBULENCE_NODE"
    bl_label = "Force Turbulence"
    bl_icon = "FORCE_TURBULENCE"
    bl_width_default = 220.0
    bl_width_min = 200.0
    bl_width_max = 340.0
    draw_property_names = ("scale", "frequency", "amplitude", "seed")

    scale: FloatProperty(name="Scale", default=1.0, min=0.0, description="Scale of turbulent force, bigger means more large scale fluctuations", options=set())  # type: ignore
    frequency: FloatProperty(name="Frequency", default=1.0, min=0.0, description="Frequency of turbulence, bigger means faster fluctuations", options=set())  # type: ignore
    amplitude: FloatProperty(name="Amplitude", default=1.0, min=0.0, description="Amplitude of turbulent force", options=set())  # type: ignore
    seed: bpy.props.IntProperty(name="Seed", default=0, description="Random seed for turbulence", options=set())  # type: ignore


classes = (
    BlenderCFDForceConstantNode,
    BlenderCFDForceSwirlNode,
    BlenderCFDForcePointNode,
    BlenderCFDForceTurbulenceNode,
)
