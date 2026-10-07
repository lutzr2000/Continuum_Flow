from . import sockets
from . import node_base
from bpy.props import EnumProperty
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
    animation_proxy_properties = (
        "fuel",
        "smoke",
        "temperature",
        "extra_pressure",
        "velocity",
    )

    fuel: FloatProperty(name="Fuel Emission", default=0.0, min=0.0, max=100.0, soft_min=0.0, soft_max=100.0, subtype="PERCENTAGE", description="How much fuel is emitted", options={"ANIMATABLE"})  # type: ignore
    smoke: FloatProperty(name="Smoke Emission", default=0.0, min=0.0, max=100.0, soft_min=0.0, soft_max=100.0, subtype="PERCENTAGE", description="How much smoke is emitted", options={"ANIMATABLE"})  # type: ignore
    temperature: FloatProperty(name="Temperature", default=300.0, min=0.0, max=2000.0, soft_min=0.0, soft_max=2000.0, unit="TEMPERATURE", description="Amount of temperature to spawn", options={"ANIMATABLE"})  # type: ignore
    extra_pressure: FloatProperty(name="Extra Pressure", default=0.0, precision=4, description="Additional pressure added in the source", options={"ANIMATABLE"})  # type: ignore
    randomness_scale: FloatProperty(name="Scale", default=1.0, min=0.000001, unit="LENGTH", description="World-space scale of the source randomness field", options=set())  # type: ignore
    randomness_seed: IntProperty(name="Seed", default=0, description="Seed of the source randomness field", options=set())  # type: ignore
    fuel_randomness: FloatProperty(name="Fuel Randomness", default=0.0, min=0.0, max=100.0, subtype="PERCENTAGE", options=set())  # type: ignore
    smoke_randomness: FloatProperty(name="Smoke Randomness", default=0.0, min=0.0, max=100.0, subtype="PERCENTAGE", options=set())  # type: ignore
    temperature_randomness: FloatProperty(name="Temperature Randomness", default=0.0, min=0.0, max=100.0, subtype="PERCENTAGE", options=set())  # type: ignore
    extra_pressure_randomness: FloatProperty(name="Extra Pressure Randomness", default=0.0, min=0.0, max=100.0, subtype="PERCENTAGE", options=set())  # type: ignore
    velocity_randomness: FloatProperty(name="Velocity Randomness", default=0.0, min=0.0, max=100.0, subtype="PERCENTAGE", options=set())  # type: ignore
    velocity_space: EnumProperty(
        name="Space",
        items=(
            ("WORLD", "World Space", "Interpret velocity in world coordinates"),
            (
                "LOCAL",
                "Local Space",
                "Interpret velocity in each source object's local coordinates",
            ),
        ),
        default="WORLD",
        options=set(),
    )  # type: ignore
    velocity: FloatVectorProperty(name="Velocity", size=3, default=(0.0, 0.0, 0.0), subtype="VELOCITY", description="Source velocity", options={"ANIMATABLE"})  # type: ignore

    def _sync_node(self):
        self._ensure_geometry_input()
        self._ensure_particle_system_input()
        self._ensure_named_output(sockets.ContinuumFlowIntSocket.bl_idname, "Source")

    def draw_buttons(self, context, layout):
        self._set_layout_enabled(context, layout)

        randomness_box = layout.box()
        randomness_col = randomness_box.column(align=True)
        randomness_col.label(text="Noise Settings")
        randomness_col.prop(self, "randomness_scale")
        randomness_col.prop(self, "randomness_seed")

        for property_name in self.scalar_property_names:
            box = layout.box()
            col = box.column(align=True)
            col.label(text=property_name.replace("_", " ").title())
            self._draw_property(col, property_name, text="Value")
            col.prop(self, f"{property_name}_randomness", text="Randomness")

        velocity_box = layout.box()
        velocity_col = velocity_box.column(align=True)
        velocity_col.label(text="Velocity")
        velocity_col.prop(self, "velocity_space", text="")
        velocity_col.prop(self, "velocity_randomness", text="Randomness")
        self._draw_property(velocity_col, "velocity", text="")
