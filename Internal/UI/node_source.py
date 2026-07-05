from . import helper_functions
from . import sockets
from . import node_base
from bpy.props import BoolProperty
from bpy.props import FloatProperty
from bpy.props import FloatVectorProperty
from bpy.props import IntProperty

class ContinuumFlowSourceNode(node_base.ContinuumFlowBaseNode):
    """
    Node used to define a generic CFD source region and its scalar and velocity targets.
    """

    bl_idname = "CONTINUUM_FLOW_SOURCE_NODE"
    bl_label = "Source"
    bl_icon = "LIGHT_SUN"
    bl_width_default = 220.0
    bl_width_min = 200.0
    bl_width_max = 360.0
    scalar_property_names = ("fuel", "smoke", "temperature", "extra_pressure")

    fuel: FloatProperty(name="Fuel Emission", default=0.0, min=0.0, max=100.0, soft_min=0.0, soft_max=100.0, subtype="PERCENTAGE", description="How much fuel is emitted", options={"ANIMATABLE"})  # type: ignore
    smoke: FloatProperty(name="Smoke Emission", default=0.0, min=0.0, max=100.0, soft_min=0.0, soft_max=100.0, subtype="PERCENTAGE", description="How much smoke is emitted", options={"ANIMATABLE"})  # type: ignore
    temperature: FloatProperty(name="Temperature", default=300.0, min=0.0, max=2000.0, soft_min=0.0, soft_max=2000.0, unit="TEMPERATURE", description="Amount of temperature to spawn", options={"ANIMATABLE"})  # type: ignore
    extra_pressure: FloatProperty(name="Extra Pressure", default=0.0, precision=4, description="Additional pressure added in the source", options={"ANIMATABLE"})  # type: ignore
    source_noise: BoolProperty(name="Source Noise", default=False, description="Modulate the source emission with a procedural random field", options=set())  # type: ignore
    noise_scale: FloatProperty(name="Scale", default=6.0, min=1.0, soft_min=1.0, soft_max=64.0, precision=2, description="Approximate noise feature size in source voxels", options=set())  # type: ignore
    noise_seed: IntProperty(name="Seed", default=0, description="Random seed used for the source noise pattern", options=set())  # type: ignore
    noise_amplitude: FloatProperty(name="Amplitude", default=25.0, min=0.0, max=100.0, soft_min=0.0, soft_max=100.0, subtype="PERCENTAGE", description="How strongly the source noise modulates temperature, smoke, fuel and extra pressure", options=set())  # type: ignore
    velocity: FloatVectorProperty(name="Velocity", size=3, default=(0.0, 0.0, 0.0), subtype="VELOCITY", description="Source velocity", options={"ANIMATABLE"})  # type: ignore

    def _sync_node(self):
        helper_functions.ensure_geometry_input(self)
        helper_functions.ensure_named_output(self, sockets.ContinuumFlowIntSocket.bl_idname, "Source")

    def draw_buttons(self, context, layout):
        self._set_layout_enabled(context, layout)
        col = layout.column(align=True)
        for property_name in self.scalar_property_names:
            col.prop(self, property_name)

        noise_col = layout.column(align=True)
        noise_col.prop(self, "source_noise")
        if self.source_noise:
            noise_col.prop(self, "noise_scale")
            noise_col.prop(self, "noise_seed")
            noise_col.prop(self, "noise_amplitude")

        velocity_col = layout.column(align=True)
        velocity_col.label(text="Velocity")
        velocity_col.prop(self, "velocity", text="")


