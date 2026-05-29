Solver Theory
=============

This section describes the numerical model used by the Continuum Flow solver. The simulation is based on the incompressible Navier-Stokes equations and uses a regular Cartesian grid with a finite-difference discretization. The goal is not to reproduce every detail of real fluid physics, but to provide a stable and fast solver for visually convincing smoke and fire effects. The solver simulates the following fields:

- velocity vector field in x-, y-, and z-directions, denoted by: :math:`\vec{u}`
- pressure scalar field, denoted by: :math:`p`
- temperature scalar field, denoted by: :math:`\phi_{\theta}`
- scalar fields like smoke, flame, and fuel denoted by :math:`\phi`


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

In equation :eq:`eq-continuity` the scalar multiplication of the nabla operator with the velocity vector field :math:`\vec{u}` denotes the divergence of that vector field. In equation :eq:`eq-momentum`, :math:`t` denotes time, :math:`\rho` denotes density, :math:`\nu` denotes viscosity, and :math:`\vec{f}` denotes an additional force vector field.

In every flow simulation at least pressure and velocity need to be computed. They are coupled, but there is no single equation to compute pressure. To construct the equation for pressure we take the divergence of equation :eq:`eq-momentum`. This then becomes a Poisson equation of the form:

.. math::
   :label: eq-poisson_pressure

   \nabla^2 p = b 

with :math:`b` being the right hand side. This is an additional equation we can use for computing pressure. In the next section the solution procedure is described in further detail.


General Solution Procedure
--------------------------
 
When running the simulation, the solver initializes a flow field and then starts the time iteration. During the time iteration, the following procedure is performed:

1. Update all parameters, force fields, sources, and obstacles
2. Compute vorticity forces
3. When using the adaptive Domain option, build the active mask
4. Compute the semi-lagrangian advection
5. Apply MacCormack's correction
6. Build the right-hand side (rhs) :math:`b` of the pressure Poisson equation :eq:`eq-poisson_pressure`
7. Solve the pressure Poisson equation
8. Use pressure for the velocity projection
9. Apply boundary conditions
10. Advect the scalar fields with semi-lagrangian advection
11. Apply MacCormack's correction for scalar variables
12. Apply boundary conditions again
13. Optional: Output data
14. Compute new stable time step

Repeat


Discretization
--------------

The equations above describe continuous fluids, hence the name Continuum Flow, as differential equations. In practice, we cannot solve them analytically and therefore have to rely on numerical methods. In this solver, finite-difference approximation is used to compute the derivatives. This means the whole simulation volume, the domain, is subdivided into a grid with uniform resolution :math:`\Delta`, resulting in a grid with :math:`n_x \cdot n_y \cdot n_z` cells. For every cell, a value for all fields needs to be computed. The derivatives for pressure, diffusion processes, and vorticity are computed using the following approximations:

A one-dimensional first derivative in the x-direction is computed using central differencing:

.. math::
   :label: eq-central_first_derivative

   \frac{\partial f}{\partial x}\Big|_i
   \approx
   \frac{f_{i+1} - f_{i-1}}{2\Delta x}

A one-dimensional second derivative in the x-direction is computed with central differencing:

.. math::
   :label: eq-central_second_derivative

   \frac{\partial^2 f}{\partial x^2}\Big|_i
   \approx
   \frac{f_{i+1} - 2f_i + f_{i-1}}{\Delta x^2}


Advection Procedure
-------------------

The treatment of flow advection is critical for flow simulation. It can be a strict constraint on the time step. To avoid this, semi-Lagrangian advection is used here, which allows for much larger time steps and thus fast simulations.

We first ask: where was this fluid a time interval of :math:`\Delta t` ago? This is called backtracing. In this solver, the backtracing is subdivided into three substeps to allow for curvature of the path.

.. math::
   :label: eq-backtracing

   \mathbf{x}_{\mathrm{old}}
   =
   \mathbf{x}
   -
   \Delta t \, \vec{u}(\mathbf{x})

Next, we take the quantity :math:`q` (velocity, smoke, temperature, ...) found at this backtraced position and move it to the departure cell.

.. math::
   :label: eq-semi_lagrangian_update

   q^{*}(\mathbf{x})
   =
   q^n(\mathbf{x}_{\mathrm{old}})

Since our backtracing position :math:`\mathbf{x}_{\mathrm{old}}` will never end up exactly at a cell center, we need to interpolate the quantity :math:`q` at the position reached through backtracing. This introduces numerical diffusion, meaning the flow becomes less swirly and, in VFX terms, less interesting to look at.

To mitigate this we apply MacCormack's correction:

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

Here :math:`q^{n+1}` is the quantity at the next time step and :math:`q^{n}` is the quantity from the current time step. :math:`q^{*}` is the quantity we computed with semi-Lagrangian advection and :math:`q^{**}` is the reversely advected velocity field. If there is no numerical diffusion, :math:`q^{n}` and :math:`q^{**}` would be equal and the whole term would cancel out. In practice, this is never the case, and thus we use it for correction. :math:`\alpha` determines the strength of this correction. This correction can produce visual artifacts.


Pressure Solve
--------------

To compute pressure we solve the discretized version of equation :eq:`eq-poisson_pressure`. First we need to compute the rhs :math:`b`. In accordance with "Stable fluids" (https://doi.org/10.1145/311535.311548) we neglect non-linear terms of the :eq:`eq-momentum` equation and only compute :math:`b` as:

.. math::
   :label: eq-pressure-rhs

   b
   =
   \nabla \cdot \vec{u}'

Here :math:`\vec{u}'` is the previously advected velocity field. Additional divergence terms can optionally be added, like thermal divergence, divergence due to the point force, or an extra pressure term added by sources.

Equation :eq:`eq-poisson_pressure` is then solved by a red-black Gauss-Seidel solver for as many iterations as specified by the user. In every iteration the Neumann boundary conditions for pressure are applied. Since the absolute level of pressure is not defined with only Neumann boundary conditions, we subtract :math:`b-\bar{b}`. We remove the mean of :math:`b` to improve stability of the pressure field.


Velocity Projection
-------------------

In accordance with "Stable fluids" (https://doi.org/10.1145/311535.311548) we then project the velocity to be divergence free using the following projection:

.. math::
   :label: eq-pressure_projection

   \vec{u}^{\,n+1}
   =
   \vec{u}'
   -
   \frac{\Delta t}{\rho}
   \nabla p

With this we have computed a new pressure and velocity field for the time step.


Scalar Fields
-------------

The scalar fields are primarily advected through the field as passive scalars. We neglect diffusion for the quantities fuel, smoke, and temperature. For the scalar update we first compute the advected and MacCormack-corrected fields :math:`\phi_{\mathrm{fuel}}^{c}`, :math:`\phi_{\mathrm{smoke}}^{c}`, and :math:`\phi_{\mathrm{temperature}}^{c}`. The source terms are then evaluated from this corrected state.

The local oxygen concentration is approximated from the smoke field as:

.. math::
   :label: eq-oxygen-from-smoke

   \phi_{\mathrm{oxygen}}
   =
   100
   -
   \phi_{\mathrm{smoke}}'

Fuel is dissipated with a linear decay term. If the temperature exceeds the ignition threshold, fuel is present, and enough oxygen is available, an additional combustion term is activated. Altogether this yields the fuel source term:

.. math::
   :label: eq-fuel-source

   S_{\mathrm{fuel}}
   =
   \begin{cases}
   -k_{\mathrm{fuel,diss}} \, \phi_{\mathrm{fuel}}^{n+1}
   -
   k_{\mathrm{burn}} \, \phi_{\mathrm{fuel}}^{n+1},
   &
   \text{burning condition}
   \\
   -k_{\mathrm{fuel,diss}} \, \phi_{\mathrm{fuel}}^{n+1},
   & \text{otherwise}
   \end{cases}

Here, the burning condition means:

.. math::

   \phi_{\mathrm{temperature}}^{n+1} > \phi_{\mathrm{ignition}},
   \qquad
   \phi_{\mathrm{fuel}}^{n+1} > 0,
   \qquad
   \phi_{\mathrm{oxygen}} \geq \phi_{\mathrm{oxygen,min}}

The temperature source term consists of a relaxation towards the reference temperature and a production term caused by burning fuel:

.. math::
   :label: eq-temperature-source

   S_{\mathrm{\theta}}
   =
   -k_{\mathrm{\theta,diss}}
   \left(
      \phi_{\mathrm{\theta}}^{n+1}
      -
      \phi_{\mathrm{\theta,ref}}
   \right)
   +
   k_{\mathrm{\theta,prod}}
   \left(
      -S_{\mathrm{fuel,burn}}
   \right)

The temperature is not completely passive in the flow since it affects thermal divergence and buoyancy. The smoke source term is modeled as smoke production from combustion and linear smoke dissipation:

.. math::
   :label: eq-smoke-source

   S_{\mathrm{smoke}}
   =
   k_{\mathrm{smoke,prod}}
   \left(
      -S_{\mathrm{fuel,burn}}
   \right)
   -
   k_{\mathrm{smoke,diss}}
   \phi_{\mathrm{smoke}}^{n+1}


The flame field is not evolved by its own transport equation. Instead it is defined diagnostically from the instantaneous fuel consumption rate as

.. math::
   :label: eq-flame-definition

   \phi_{\mathrm{flame}}
   =
   \max\left(
      -S_{\mathrm{fuel,burn}},
      0
   \right)

Hence :math:`\phi_{\mathrm{flame}}` marks where fuel is currently burning and vanishes in cells without active combustion. Pure fuel dissipation does not contribute to the flame field.


Time Step
---------

To keep the explicit parts of the solver stable, the time step is adapted dynamically. In the implementation three restrictions are evaluated: a convective CFL limit, a diffusive limit and a forcing-based limit. The time step is chosen as the minimum of these values.

The convective restriction is computed componentwise from the maximum absolute velocity components:

.. math::
   :label: eq-dt-convective

   \Delta t_{\mathrm{conv}}
   =
   \min\left(
      \frac{\mathrm{CFL} \cdot \Delta}{\max(|u|)},
      \frac{\mathrm{CFL} \cdot \Delta}{\max(|v|)},
      \frac{\mathrm{CFL} \cdot \Delta}{\max(|w|)}
   \right)

The diffusive stability limit is:

.. math::
   :label: eq-dt-diffusive

   \Delta t_{\mathrm{diff}}
   =
   \frac{\Delta^2}{6 \nu}

For body forces we use an additional restriction based on the maximum force components:

.. math::
   :label: eq-dt-forcing

   \Delta t_{\mathrm{force}}
   =
   \min\left(
      \frac{\mathrm{CFL} \cdot \Delta \, \rho}{\max(|f_x|)},
      \frac{\mathrm{CFL} \cdot \Delta \, \rho}{\max(|f_y|)},
      \frac{\mathrm{CFL} \cdot \Delta \, \rho}{\max(|f_z|)}
   \right)

The solver then advances with:

.. math::
   :label: eq-dt-final

   \Delta t
   =
   \min\left(
      \Delta t_{\mathrm{conv}},
      \Delta t_{\mathrm{diff}},
      \Delta t_{\mathrm{force}},
      \Delta t_{\max}
   \right)

where :math:`\Delta t_{\max}` is an upper bound from the output frame rate.

Vorticity
---------

To preserve small-scale swirling motion, the solver adds a vorticity confinement force. First the vorticity vector is computed as the curl of the velocity field:

.. math::
   :label: eq-vorticity-definition

   \vec{\omega}
   =
   \nabla \times \vec{u}

with the components:

.. math::
   :label: eq-vorticity-components

   \omega_x = \frac{\partial w}{\partial y} - \frac{\partial v}{\partial z},
   \qquad
   \omega_y = \frac{\partial u}{\partial z} - \frac{\partial w}{\partial x},
   \qquad
   \omega_z = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}

From this we compute the vorticity magnitude :math:`|\vec{\omega}|` and its normalized gradient:

.. math::
   :label: eq-vorticity-direction

   \vec{N}
   =
   \frac{\nabla |\vec{\omega}|}{|\nabla |\vec{\omega}||}

The confinement force is then given by:

.. math::
   :label: eq-vorticity-confinement

   \vec{f}_{\mathrm{vort}}
   =
   \epsilon
   \left(
      \vec{N} \times \vec{\omega}
   \right)

where :math:`\epsilon` is the user-controlled vorticity strength. This force is added to the existing body-force field before the momentum update.

Buoyancy
---------

Buoyancy is modeled with a Boussinesq-type approximation based on the local temperature difference to a reference temperature. In this solver the buoyancy force acts only in the vertical :math:`z`-direction:

.. math::
   :label: eq-buoyancy-force

   f_z^{\mathrm{buoy}}
   =
   g \, \beta
   \left(
      \phi_{\mathrm{\theta}}
      -
      \phi_{\mathrm{\theta,ref}}
   \right)

Here :math:`g` is the gravitational acceleration constant set to 9.81 and :math:`\beta` is the configured buoyancy factor. Positive temperature deviations therefore produce upward acceleration, while colder fluid produces a downward contribution.





