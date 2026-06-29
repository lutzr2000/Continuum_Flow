from . import helper_functions
from . import sockets
from . import node_base
from bpy.props import FloatProperty

class ContinuumFlowPhysicsNode0(node_base.ContinuumFlowBaseNode):
    """
    Node used to store the physical coefficients of the CFD simulation.
    """

    bl_idname = "CONTINUUM_FLOW_PHYSICS_NODE"
    bl_label = "Physics"
    bl_icon = "MOD_PHYSICS"
    bl_width_default = 220.0
    bl_width_min = 200.0
    bl_width_max = 360.0
    property_groups = (
        ("Fluid", ("fluid_density", "fluid_viscosity")),
        (
            "Temperature",
            (
                "temperature_dissipation",
                "temperature_production_rate",
                "reference_temperature",
                "buoyancy",
                "expansion_rate",
            ),
        ),
        ("Smoke", ("smoke_dissipation", "smoke_production_rate")),
        (
            "Fuel",
            (
                "fuel_dissipation",
                "fuel_burn_rate",
                "fuel_ignition_temperature",
                "minimum_oxygen_concentration",
            ),
        ),
        ("Extras", ("vorticity",)),
    )

    fluid_density: FloatProperty(name="Fluid Density", default=1.225, min=0.0001, precision=4, description="Density of the fluid, default is air", options=set())  # type: ignore
    fluid_viscosity: FloatProperty(name="Fluid Viscosity", default=1.81e-5, min=0.0, precision=6, description="Viscosity of the fluid, default is air", options=set())  # type: ignore
    temperature_dissipation: FloatProperty(name="Temperature Dissipation", default=10, min=0.0, soft_min=0.0, soft_max=100.0, precision=2, subtype="PERCENTAGE", description="how quickly temperature dissipates")  # type: ignore
    temperature_production_rate: FloatProperty(name="Temperature Production Rate", default=10, min=0.0, soft_min=0.0, soft_max=100.0, precision=2, subtype="PERCENTAGE", description="how much temperature is produced due to burning")  # type: ignore
    reference_temperature: FloatProperty(name="Reference Temperature", default=300.0, min=0.0, max=2000, unit="TEMPERATURE", description="Air cooler than this goes down, warmer than this goes up")  # type: ignore
    buoyancy: FloatProperty(name="Buoyancy", default=30, min=0.0, soft_min=0.0, soft_max=100.0, precision=2, subtype="PERCENTAGE", description="Strength of buoyancy")  # type: ignore
    expansion_rate: FloatProperty(name="Expansion Rate", default=10, min=0.0, soft_min=0.0, soft_max=100.0, precision=2, subtype="PERCENTAGE", description="how strongly hot air expands")  # type: ignore
    smoke_dissipation: FloatProperty(name="Smoke Dissipation", default=0, min=0.0, soft_min=0.0, soft_max=100.0, precision=2, subtype="PERCENTAGE", description="how quickly smoke dissipates")  # type: ignore
    smoke_production_rate: FloatProperty(name="Smoke Production Rate", default=50, min=0.0, soft_min=0.0, soft_max=100.0, precision=2, subtype="PERCENTAGE", description="how much smoke is produced due to burning")  # type: ignore
    fuel_dissipation: FloatProperty(name="Fuel Dissipation", default=0, min=0.0, soft_min=0.0, soft_max=100.0, precision=2, subtype="PERCENTAGE", description="how quickly fuel dissipates")  # type: ignore
    fuel_burn_rate: FloatProperty(name="Fuel Burn Rate", default=30, min=0.0, soft_min=0.0, soft_max=100.0, precision=2, subtype="PERCENTAGE", description="How quickly fuel burns away")  # type: ignore
    fuel_ignition_temperature: FloatProperty(name="Fuel Ignition Temperature", default=500.0, min=0.0, max=2000.0, soft_min=0.0, soft_max=2000.0, unit="TEMPERATURE", description="If the air is warmer than this and contains fuel, the fuel will ignite")  # type: ignore
    minimum_oxygen_concentration: FloatProperty(name="Minimum Oxygen Concentration", default=0.0, min=0.0, max=100.0, soft_min=0.0, soft_max=100.0, subtype="PERCENTAGE", description="Minimum oxygen concentration required for fuel to burn")  # type: ignore
    vorticity: FloatProperty(name="Vorticity", default=40, min=0.0, soft_min=0.0, soft_max=100.0, precision=2, subtype="PERCENTAGE", description="How much extra vorticity is added")  # type: ignore

    def _sync_node(self):
        helper_functions.ensure_named_output(self, sockets.ContinuumFlowIntSocket.bl_idname, "Physics")

    def draw_buttons(self, context, layout):
        self._set_layout_enabled(context, layout)
        for title, property_names in self.property_groups:
            self._draw_group(layout, title, property_names)