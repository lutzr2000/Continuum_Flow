import helper_functions
import sockets
import node_base

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
    velocity_space_items = (
        ("WORLD", "World Space", "Apply the source velocity in world coordinates"),
        (
            "LOCAL",
            "Local Space",
            "Apply the source velocity in each linked object's local coordinates",
        ),
    )

    fuel: FloatProperty(name="Fuel Emission", default=0.0, min=0.0, max=100.0, soft_min=0.0, soft_max=100.0, subtype="PERCENTAGE", description="Fuel emission rate in percent per second", options={"ANIMATABLE"})  # type: ignore
    smoke: FloatProperty(name="Smoke Emission", default=0.0, min=0.0, max=100.0, soft_min=0.0, soft_max=100.0, subtype="PERCENTAGE", description="Smoke emission rate in percent per second", options={"ANIMATABLE"})  # type: ignore
    temperature: FloatProperty(name="Temperature", default=300.0, min=0.0, max=2000.0, soft_min=0.0, soft_max=2000.0, unit="TEMPERATURE", description="Amount of temperature to spawn", options={"ANIMATABLE"})  # type: ignore
    extra_pressure: FloatProperty(name="Extra Pressure", default=0.0, precision=4, description="Additional pressure added in the source", options={"ANIMATABLE"})  # type: ignore
    velocity_space: EnumProperty(name="Space", items=velocity_space_items, default="WORLD", options=set())  # type: ignore
    velocity: FloatVectorProperty(name="Velocity", size=3, default=(0.0, 0.0, 0.0), subtype="VELOCITY", description="Source velocity", options={"ANIMATABLE"})  # type: ignore

    def _sync_node(self):
        helper_functions.ensure_geometry_input(self)
        helper_functions.ensure_named_output(self, sockets.ContinuumFlowIntSocket.bl_idname, "Source")

    def draw_buttons(self, context, layout):
        self._set_layout_enabled(context, layout)
        col = layout.column(align=True)
        for property_name in self.scalar_property_names:
            col.prop(self, property_name)

        velocity_col = layout.column(align=True)
        velocity_col.label(text="Velocity")
        velocity_col.prop(self, "velocity_space", text="Space")
        velocity_col.prop(self, "velocity", text="")
