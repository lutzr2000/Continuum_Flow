
# is called by the bake button
# 1. evaluates the node tree
# 2. launch writer processes
# 3. build config
# 4. send config

import bpy

class main(bpy.types.Operator):
    bl_idname = "continuum_flow.bake"
    bl_label = "Bake"

    def execute(self, context):
        self.do_bake(context)
        return {'FINISHED'}

    def do_bake(self, context):
        print("Baking...")
