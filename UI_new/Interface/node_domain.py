import helper_functions
import sockets
import node_base


class ContinuumFlowDomainNode(node_base.ContinuumFlowBaseNode):
    """
    Node used to define the CFD domain resolution and boundary conditions.
    """

    bl_idname = "CONTINUUM_FLOW_DOMAIN_NODE"
    bl_label = "Domain"
    bl_icon = "MESH_GRID"
    bl_width_default = 220.0
    bl_width_min = 200.0
    bl_width_max = 360.0

    boundary_axes = (
        ("X Low", "x_low_bc", "x_low_velocity"),
        ("X High", "x_high_bc", "x_high_velocity"),
        ("Y Low", "y_low_bc", "y_low_velocity"),
        ("Y High", "y_high_bc", "y_high_velocity"),
        ("Z Low", "z_low_bc", "z_low_velocity"),
        ("Z High", "z_high_bc", "z_high_velocity"),
    )
    boundary_condition_items = (
        ("WALL", "Wall", "No-slip wall boundary"),
        ("SLIP_WALL", "Slip Wall", "Slip wall boundary"),
        ("OUTFLOW", "Outflow", "Outflow boundary"),
        ("INFLOW", "Inflow", "Inflow boundary with prescribed velocity"),
    )
    resolution: FloatProperty(name="Resolution", default=0.1, min=0.000001, soft_min=0.000001, unit="LENGTH", description="Grid resolution", options=set())  # type: ignore
    nx: IntProperty(name="NX", default=128, min=32, max=8192, soft_min=32, description="Grid cells in x", options=set())  # type: ignore
    ny: IntProperty(name="NY", default=128, min=32, max=8192, soft_min=32, description="Grid cells in y", options=set())  # type: ignore
    nz: IntProperty(name="NZ", default=128, min=32, max=8192, soft_min=32, description="Grid cells in z", options=set())  # type: ignore
    x_low_bc: bpy.props.EnumProperty(name="X Low", items=boundary_condition_items, default="OUTFLOW", options=set())  # type: ignore
    x_high_bc: bpy.props.EnumProperty(name="X High", items=boundary_condition_items, default="OUTFLOW", options=set())  # type: ignore
    y_low_bc: bpy.props.EnumProperty(name="Y Low", items=boundary_condition_items, default="OUTFLOW", options=set())  # type: ignore
    y_high_bc: bpy.props.EnumProperty(name="Y High", items=boundary_condition_items, default="OUTFLOW", options=set())  # type: ignore
    z_low_bc: bpy.props.EnumProperty(name="Z Low", items=boundary_condition_items, default="OUTFLOW", options=set())  # type: ignore
    z_high_bc: bpy.props.EnumProperty(name="Z High", items=boundary_condition_items, default="OUTFLOW", options=set())  # type: ignore
    x_low_velocity: FloatVectorProperty(name="Velocity", size=3, subtype="XYZ", default=(0.0, 0.0, 0.0), unit="VELOCITY", options=set())  # type: ignore
    x_high_velocity: FloatVectorProperty(name="Velocity", size=3, subtype="XYZ", default=(0.0, 0.0, 0.0), unit="VELOCITY", options=set())  # type: ignore
    y_low_velocity: FloatVectorProperty(name="Velocity", size=3, subtype="XYZ", default=(0.0, 0.0, 0.0), unit="VELOCITY", options=set())  # type: ignore
    y_high_velocity: FloatVectorProperty(name="Velocity", size=3, subtype="XYZ", default=(0.0, 0.0, 0.0), unit="VELOCITY", options=set())  # type: ignore
    z_low_velocity: FloatVectorProperty(name="Velocity", size=3, subtype="XYZ", default=(0.0, 0.0, 0.0), unit="VELOCITY", options=set())  # type: ignore
    z_high_velocity: FloatVectorProperty(name="Velocity", size=3, subtype="XYZ", default=(0.0, 0.0, 0.0), unit="VELOCITY", options=set())  # type: ignore

    def _sync_node(self):
        helper_functions.ensure_named_output(self, sockets.ContinuumFlowIntSocket.bl_idname, "Domain")

    def _draw_boundary_controls(self, layout, label, condition_attr, velocity_attr):
        box = layout.box()
        row = box.row(align=True)
        row.label(text=label)
        row.prop(self, condition_attr, text="")
        if getattr(self, condition_attr) == "INFLOW":
            box.prop(self, velocity_attr, text="Velocity")

    def draw_buttons(self, context, layout):
        self._set_layout_enabled(context, layout)
        col = layout.column(align=True)
        for property_name in ("resolution", "nx", "ny", "nz"):
            col.prop(self, property_name)
        layout.separator()
        layout.label(text="Boundary Conditions")
        for boundary_args in self.boundary_axes:
            self._draw_boundary_controls(layout, *boundary_args)