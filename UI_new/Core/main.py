import bpy
from . import export_config

class main(bpy.types.Operator):
    bl_idname = "continuum_flow.bake"
    bl_label = "Bake"

    def execute(self, context):
        self.do_bake(context)
        return {'FINISHED'}

    def do_bake(self, context):
        export_config.export_config_dict()
