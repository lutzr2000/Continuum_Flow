Solver Theory
=============

This section describes the numerical model used by the Continuum flow solver. The simulation is based on the incompressible Navier-Stokes equations and uses a regular Cartesian grid with a finite-difference discretization. The goal is not to reproduce every detail of real fluid physics, but to provide a stable and fast solver for visually convincing smoke and fire effects. The solver simulates the following fields:

- velocity vector field in x,y,z-diretions, denoted by: :math:`\vec{u}`
- pressure scalar field, denoted by: :math:`p`
- temperature scalar field, denoted by: :math:`\theta`
- scalar fields like smoke, flame and fuel denoted by :math:`\phi`


Governing Equations
-------------------

Conservation of mass:

.. math::
   :label: eq-continuity

   \nabla \cdot \vec{u} = 0 

Conservation of momentum:

.. math::
   :label: eq-momentum

   \frac{\partial \vec{u}}{\partial t}
   + \left( \vec{u} \cdot \nabla \right) \vec{u}
   = -\frac{1}{\rho} \nabla p
   + \nu \nabla^2 \vec{u}
   + \vec{f}

In equation :eq:`eq-continuity` the scalar multiplication of the nabla operator with the velocity vector field :math:`\vec{u}` denotes the divergence of that vector field. In equation :eq:`eq-momentum` :math:`t` denotes time, :math:`\rho` denotes density, :math:`\nu` denotes viscocity and :math:`\vec{f}` denotes an additional force vector field.

In every flow simulation at least pressure and velocity need to be computed. They are coupled, but there is no single equation to compute pressure. To construct the equation for pressure we take the divergence of equation :eq:`eq-momentum`. This becomes then a poisson equation of form:

.. math::
   :label: eq-poisson_pressure

   \nabla^2 p = b 

with :math:`b` being the right hand side. This is an additional equation we can use for computing pressure. In the next section the solution procedure is described in further detail.


General Solution Procedure
--------------------------
 
When running the simulation the solver initalises a flow field and then starts the time itteration. In the time itteration the following procedure is done:

1. Update all parameters, force fields, sources and osbtacles
2. Compute vorticiry forces
3. WHen using the adaptive Domain option, building the active mask
4. Compute the semi-lagrangian advection
5. Apply MacCormacks correction 
6. Build the right-hand-side (rhs) :math:`b` of the pressure poisson equation :eq:`eq-poisson_pressure`
7. Solve the pressure poisson equation
8. Use pressure for the velocity projection 
9. Apply boundary conditions
10. Advect the scalar fields with semi-lagrangian advection
11. Apply MacCormacks correction for scalar variables
12. Apply boundary conditions again
13. Optional: Output data
14. Compute new stable time step

Repeat


Discretization
--------------

The equations above are describing continuus fluids (hence the nem Continuum Flow) as differential equations. In practical application we can not solve analytically and have to rely on numerics. In this solver finite difference approximation is used for computing the derivatives. This means the whole simulation volume (domain) is subdivided into a grid with uniform resolution :math:`\Delta` resulting in a grid with :math:`n_x \cdot n_y \cdot n_z` grid cells. For every cell a value for all fields needs to be computed. The derivatives for pressure, diffusion processes and vorticity are computing using the following approximations:

A one dimnensional first derivative in the x-direction is computed using central differencing:

.. math::
   :label: eq-central_first_derivative

   \frac{\partial f}{\partial x}\Big|_i
   \approx
   \frac{f_{i+1} - f_{i-1}}{2\Delta x}

A one dimnensional second derivative in the x-direction with central differencing:

.. math::
   :label: eq-central_second_derivative

   \frac{\partial^2 f}{\partial x^2}\Big|_i
   \approx
   \frac{f_{i+1} - 2f_i + f_{i-1}}{\Delta x^2}


Advection Procedure
-------------------

The treatment of flow advection is critical for flow simulation. It can be a strict constrain on the time step. To avoid this here semi-lagrangian advection is used which allows for very larger time steps and thus fast simualtions.

We first ask: Where has this fluid been a time interval of :math:`\Delta t` ago? This is called backtracing. In this solver the backtracing is actually subdivided into three substeps to allow for curvature of the path.

.. math::
   :label: eq-backtracing

   \mathbf{x}_{\mathrm{old}}
   =
   \mathbf{x}
   -
   \Delta t \, \mathbf{u}(\mathbf{x})

Next we take the quantity :math:`q` (velocity, smoke, temperature, ...) we found at this backtraced position and move it to the departure cell.

.. math::
   :label: eq-semi_lagrangian_update

   q^{*}(\mathbf{x})
   =
   q^n(\mathbf{x}_{\mathrm{old}})

Since our backtracing position :math:`x_{old}` will never end up exactly in a cell center, we need to interpolate the quantity :math:`q` at the position we end up through backtracing. This introduces numerical diffusion, meaning the flow becomes less swirly and in VFX terms: Boring to look at. 

To migitate this we apply MacCormacks correction:

.. math::
   :label: eq-maccormack-correction

   q'
   =
   q^{*}
   +
   \alpha
   \left(
      q^n - q^{**}
   \right)

Here :math:`q^{n+1}` is the quantity at the next time step and :math:`q^{n}` is the quantity from the current time step. :math:`q^{*}` is the quantity we computed with semi lagrangian advection and :math:`q^{**}` is the reversly advected velocity field. If there is no numerical diffusion :math:`q^{n}` and :math:`q^{**}` would be equal and the whole term would cancel out. In practice it never is and thus we use it for correction. :math:`\alpha` determines the strength of this correction. This correction can produce visual artefacts.


Pressure Solve
--------------

To compute pressure we solve the discreized version of equation :eq:`eq-poisson_pressure`. First we need to compute the rhs :math:`b`. In accordance with "Stable fluids" (https://doi.org/10.1145/311535.311548) we neglect non linear terms of the :eq:`eq-momentum` equation and only compute :math:`b` as:

.. math::
   :label: eq-pressure-rhs

   b
   =
   \nabla \cdot \vec{u'}

Here :math:`u'` is the previously advected velocity field. Additional divergence terms can be added optionally like thermal divergence, divgerence due to the point force or an extra pressure term added by sources. 

Equation :eq:`eq-poisson_pressure` is then solved by a red-black-Gauss-Seidel solver for as many itterations as specified by the users. In every itteration the Neumann boundary conditions for pressure are applied. Since the absolute level of pressure is not defined with only Neumman boundary conditions we substract :math:`b-\bar{b}`. We remove the mean of :math:`b` to improve stability of the pressure field.


Velocity Projection
-------------------

In accordance with "Stable fluids" (https://doi.org/10.1145/311535.311548) we then project the velocity to be divergence free using the following projection:

.. math::
   :label: eq-pressure_projection

   \mathbf{u}^{\,n+1}
   =
   \mathbf{u}'
   -
   \frac{\Delta t}{\rho}
   \nabla p

With this we have computed a new pressure and velocity field for the time step.


Scalar Fields
-------------

The scalar fields are primarly advected through the field as passive scalars. We neglect diffusion for the quantitys fuel, smoke and temperature.



Time Step
---------

Vorticity
---------


Bouancy
---------






