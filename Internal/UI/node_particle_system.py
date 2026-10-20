import bpy
from bpy.props import FloatProperty, PointerProperty, StringProperty

from . import node_base
from . import sockets
from ..Core.solver import solver_status


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
        options=set(),
    )  # type: ignore
    velocity_transfer: FloatProperty(
        name="Velocity Transfer",
        default=100.0,
        min=0.0,
        soft_max=100.0,
        subtype="PERCENTAGE",
        description="Percentage of particle velocity transferred to the fluid",
        options=set(),
    )  # type: ignore

    def _sync_node(self):
        self._ensure_named_output(
            sockets.ContinuumFlowParticleSystemSocket.bl_idname,
            "Particle System",
        )

    def draw_buttons(self, context, layout):
        locked_settings = layout.column()
        locked_settings.enabled = not (
            solver_status.bake_running or solver_status.preview_running
        )
        locked_settings.prop(self, "source_object", text="Object")
        if self.source_object is not None:
            locked_settings.prop_search(
                self,
                "particle_system",
                self.source_object,
                "particle_systems",
                text="Particle System",
            )
            if (
                not self.particle_system
                and getattr(self.source_object, "type", None) == "MESH"
            ):
                locked_settings.label(text="Using vertices as particles", icon="INFO")
        else:
            row = locked_settings.row()
            row.enabled = False
            row.prop(self, "particle_system", text="Particle System")

        live_settings = layout.column()
        live_settings.enabled = not solver_status.bake_running
        live_settings.prop(self, "radius")
        live_settings.prop(self, "velocity_transfer", slider=True)
