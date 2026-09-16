import bpy
from bpy.props import FloatProperty, PointerProperty, StringProperty

from . import node_base
from . import sockets


def clear_particle_system(self, context):
    """
    Clear a particle-system selection when its owning object changes.
    """
    self.particle_system = ""


class ContinuumFlowParticleSystemNode(node_base.ContinuumFlowBaseNode):
    """
    Node used to reference a Blender object that provides a particle system.
    """

    bl_idname = "CONTINUUM_FLOW_PARTICLE_SYSTEM_NODE"
    bl_label = "Particle System"
    bl_icon = "PARTICLES"
    bl_width_default = 220.0
    bl_width_min = 200.0
    bl_width_max = 360.0

    source_object: PointerProperty(
        name="Object",
        type=bpy.types.Object,
        update=clear_particle_system,
    )  # type: ignore
    particle_system: StringProperty(
        name="Particle System",
        description="Particle system to use from the selected object",
    )  # type: ignore
    radius: FloatProperty(
        name="Radius",
        default=0.01,
        min=0.0,
        soft_min=0.001,
        soft_max=1.0,
        unit="LENGTH",
        subtype="DISTANCE",
        description="Radius of the particles in meters",
    )  # type: ignore
    velocity_transfer: FloatProperty(
        name="Velocity Transfer",
        default=1.0,
        min=-1.0,
        max=1.0,
        description="Amount of particle velocity transferred to the fluid, -1 means velocity opposite to movement",
    )  # type: ignore

    def _sync_node(self):
        self._ensure_named_output(
            sockets.ContinuumFlowParticleSystemSocket.bl_idname,
            "Particle System",
        )

    def draw_buttons(self, context, layout):
        self._set_layout_enabled(context, layout)
        layout.prop(self, "source_object", text="Object")
        particle_systems = getattr(self.source_object, "particle_systems", ())
        if particle_systems:
            layout.prop_search(
                self,
                "particle_system",
                self.source_object,
                "particle_systems",
                text="Particle System",
            )
        elif self.source_object is not None:
            layout.label(text="No particle systems", icon="INFO")
        layout.prop(self, "radius")
        layout.prop(self, "velocity_transfer", slider=True)
