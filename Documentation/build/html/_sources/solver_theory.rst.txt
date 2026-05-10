Solver Theory
=============

This section describes the numerical model used by the Continuum flow solver. The
simulation is based on the incompressible Navier-Stokes equations and uses a
regular Cartesian grid with a finite-difference discretization. The goal is not
to reproduce every detail of real fluid physics, but to provide a stable and
fast solver for visually convincing smoke and fire effects.


Governing Equations
-------------------

The solver starts from the incompressible Navier-Stokes equations in vector
form.

Conservation of mass:

.. math::

   \nabla \cdot \mathbf{u} = 0

Conservation of momentum:

.. math::

   \frac{\partial \mathbf{u}}{\partial t}
   + \left( \mathbf{u} \cdot \nabla \right) \mathbf{u}
   = -\frac{1}{\rho} \nabla p
   + \nu \nabla^2 \mathbf{u}
   + \mathbf{f}

Here, :math:`\mathbf{u}` is the velocity field, :math:`p` is the pressure,
:math:`\rho` is the fluid density, :math:`\nu` is the kinematic viscosity and
:math:`\mathbf{f}` contains external forces such as buoyancy, turbulence and
user-authored force fields.


Finite-Difference Discretization
--------------------------------

The simulation domain is discretized into a uniform grid with cell spacing
:math:`\Delta`. All derivatives are approximated directly on this grid by finite
differences.

For a scalar quantity :math:`\phi`, the solver uses central differences for
diffusion and for most derivatives appearing in the pressure equation:

.. math::

   \frac{\partial \phi}{\partial x}
   \approx \frac{\phi_{i+1,j,k} - \phi_{i-1,j,k}}{2\Delta}

.. math::

   \frac{\partial^2 \phi}{\partial x^2}
   \approx \frac{\phi_{i+1,j,k} - 2\phi_{i,j,k} + \phi_{i-1,j,k}}{\Delta^2}

The three-dimensional Laplacian is built from the sum of the second
derivatives in :math:`x`, :math:`y` and :math:`z`. The viscous term in the
momentum equation therefore becomes a standard 7-point stencil on the grid.

Time integration is explicit for advection, diffusion and force application. A
stable timestep is chosen automatically from the most restrictive of three
limits:

.. math::

   \Delta t_{\mathrm{conv}}
   = \min \left(
   \frac{\mathrm{CFL}\,\Delta}{|u|_{\max}},
   \frac{\mathrm{CFL}\,\Delta}{|v|_{\max}},
   \frac{\mathrm{CFL}\,\Delta}{|w|_{\max}}
   \right)

.. math::

   \Delta t_{\mathrm{diff}} = \frac{\Delta^2}{6\nu}

.. math::

   \Delta t_{\mathrm{force}}
   = \min \left(
   \frac{\mathrm{CFL}\,\Delta\,\rho}{|F_x|_{\max}},
   \frac{\mathrm{CFL}\,\Delta\,\rho}{|F_y|_{\max}},
   \frac{\mathrm{CFL}\,\Delta\,\rho}{|F_z|_{\max}}
   \right)

The final timestep is

.. math::

   \Delta t = \min\left(
   \Delta t_{\mathrm{conv}},
   \Delta t_{\mathrm{diff}},
   \Delta t_{\mathrm{force}}
   \right)

and is additionally clamped to the current output interval.


Advection and Upwind Schemes
----------------------------

The nonlinear convection term :math:`(\mathbf{u}\cdot\nabla)\mathbf{u}` is the
most delicate part of the transport step. To keep the method stable, the solver
uses upwind differencing. The basic idea is simple: derivatives are evaluated
using values from the upstream side of the flow. This follows the physical
transport direction and suppresses the oscillations that often appear with pure
central differencing.

First-Order Upwind
~~~~~~~~~~~~~~~~~~

In the first-order upwind scheme, each derivative uses the immediately adjacent
cell in the upwind direction. For example, for a scalar quantity :math:`\phi`
in :math:`x` direction:

.. math::

   \frac{\partial \phi}{\partial x}
   \approx
   \begin{cases}
   \dfrac{\phi_{i,j,k} - \phi_{i-1,j,k}}{\Delta}, & u_{i,j,k} \ge 0 \\
   \dfrac{\phi_{i+1,j,k} - \phi_{i,j,k}}{\Delta}, & u_{i,j,k} < 0
   \end{cases}

This scheme is robust and dissipative. In practice that means it is hard to
destabilize, but it also smooths the flow and reduces small vortical detail.
For fast previews and difficult setups, this is usually the safest choice.

Second-Order Upwind
~~~~~~~~~~~~~~~~~~~

The second-order upwind scheme uses a larger stencil and therefore preserves
sharper gradients. When the local flow is positive, the derivative in
:math:`x` direction is approximated by

.. math::

   \frac{\partial \phi}{\partial x}
   \approx
   \frac{3\phi_{i,j,k} - 4\phi_{i-1,j,k} + \phi_{i-2,j,k}}{2\Delta}

and for negative flow by

.. math::

   \frac{\partial \phi}{\partial x}
   \approx
   \frac{-3\phi_{i,j,k} + 4\phi_{i+1,j,k} - \phi_{i+2,j,k}}{2\Delta}

The same idea is applied in all three coordinate directions and for all three
velocity components. Near domain boundaries, where the wider stencil is not
available, the solver automatically falls back to first-order upwind.

This second-order method is less diffusive and keeps more swirl and structure
in the motion, but it is also less forgiving. In particular it usually requires
a smaller CFL number than first-order upwinding.

The transported scalar fields temperature, smoke and fuel are currently updated
with first-order upwinding.


Pressure Solve and Enforcing Incompressibility
----------------------------------------------

After force assembly, the solver computes a pressure field that counteracts
velocity divergence. The pressure equation is built from the divergence of the
momentum equation and solved as a Poisson problem:

.. math::

   \nabla^2 p = b

The right-hand side :math:`b` is assembled from three parts:

.. math::

   b =
   \frac{\rho}{\Delta t}
   \left(
   \nabla \cdot \mathbf{u}
   + d_{\mathrm{authored}}
   - d_{\mathrm{thermal}}
   \right)
   - \rho \, N

where :math:`N` is the nonlinear term coming from
:math:`\nabla \cdot ((\mathbf{u}\cdot\nabla)\mathbf{u})`,
:math:`d_{\mathrm{authored}}` is an additional user-authored divergence source,
and :math:`d_{\mathrm{thermal}}` is the thermal expansion term discussed below.

The pressure field is solved iteratively with a red-black Gauss-Seidel method.
Each iteration updates alternating checkerboard cells and applies homogeneous
Neumann boundary conditions, meaning the normal pressure gradient at the domain
boundary is zero. In discrete form, boundary pressure values are copied from the
adjacent interior cells.

Because a Poisson equation with pure Neumann boundary conditions only defines
pressure up to an arbitrary constant, the discrete right-hand side must be
compatible. The solver therefore subtracts the mean value of :math:`b` before
starting the iterations, which removes the null-space inconsistency.

The number of pressure iterations is user-controlled. Higher values improve the
enforcement of incompressibility, while lower values are faster but can make the
motion softer or less stable.


Buoyancy
--------

Buoyancy is modeled with a Boussinesq-style approximation. Instead of solving a
fully variable-density flow, the solver assumes that density differences are
small and only matter in the body-force term. The buoyancy force is applied in
the vertical direction:

.. math::

   F_z \leftarrow F_z + g \, \beta \, (T - T_{\mathrm{ref}})

Here, :math:`g = 9.81`, :math:`\beta` is the user-controlled buoyancy factor,
:math:`T` is the local temperature and :math:`T_{\mathrm{ref}}` is the
reference temperature. If a cell is hotter than the reference temperature, the
vertical force becomes positive and the fluid rises. If it is colder, the force
becomes negative and the fluid sinks.

This is a deliberate simplification. The solver does not compute a full
equation of state, but this approximation is efficient and gives intuitive
control over rising hot smoke and sinking cold fluid.


Thermal Expansion
-----------------

Thermal expansion is modeled separately from buoyancy. Instead of only pushing
hot fluid upward, the solver also allows hot regions to act as volumetric
sources in the pressure projection step. Here it is important to note that we deviate from real physics here. The implementation is done via an artificial
divergence term:

.. math::

   d_{\mathrm{thermal}} =
   \alpha \, (T - T_{\mathrm{ref}})

where :math:`\alpha` is the user-controlled expansion rate. This term appears
in the pressure right-hand side with a negative sign:

.. math::

   b \sim \frac{\rho}{\Delta t}
   \left(
   \nabla \cdot \mathbf{u}
   + d_{\mathrm{authored}}
   - d_{\mathrm{thermal}}
   \right)

As a result, temperatures above the reference temperature reduce the pressure
right-hand side locally and create outward motion after the projection. In
visual terms, hot regions do not only rise because of buoyancy, they also
expand and push neighboring fluid away.

Vorticity
---------

Vorticity is used to recover small-scale swirling motion that is often damped by
grid discretization and by the numerically diffusive transport schemes. The
solver first computes the vorticity vector

.. math::

   \boldsymbol{\omega} = \nabla \times \mathbf{u}

with components

.. math::

   \omega_x = \frac{\partial w}{\partial y} - \frac{\partial v}{\partial z},
   \qquad
   \omega_y = \frac{\partial u}{\partial z} - \frac{\partial w}{\partial x},
   \qquad
   \omega_z = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}

using central differences on the grid. From this, the vorticity magnitude

.. math::

   |\boldsymbol{\omega}| =
   \sqrt{\omega_x^2 + \omega_y^2 + \omega_z^2}

is evaluated in each fluid cell.

The user-facing vorticity setting does not change the physical viscosity.
Instead, it adds a vorticity-confinement force that re-injects rotational
energy into regions where vortices already exist. The method computes the
normalized gradient of vorticity magnitude

.. math::

   \mathbf{N} =
   \frac{\nabla |\boldsymbol{\omega}|}{|\nabla |\boldsymbol{\omega}||}

and then applies the force

.. math::

   \mathbf{f}_{\mathrm{vc}} = \epsilon \, (\mathbf{N} \times \boldsymbol{\omega})

where :math:`\epsilon` is the vorticity strength chosen by the user.

Intuitively, :math:`\mathbf{N}` points from regions of weaker rotation toward
regions of stronger rotation. The cross product with :math:`\boldsymbol{\omega}`
creates a force that wraps around existing vortices and sharpens them. This
helps the flow keep visually pleasing curls and rolling structures, especially
in smoke simulations where first-order advection would otherwise smooth them
out.

The confinement force is only applied in fluid cells and is skipped near solid
obstacles and very close to the domain boundary, where the required stencil
would be incomplete.


Scalar Fields: Temperature, Smoke and Fuel
------------------------------------------

The scalar fields temperature, smoke and fuel are all transported by the
velocity field with first-order upwinding. In addition, each field has simple
source and dissipation terms.

Fuel burns only when the temperature in a cell exceeds the ignition threshold:

.. math::

   S_{\mathrm{fuel}} =
   \begin{cases}
   -k_{\mathrm{burn}} \, f, & T > T_{\mathrm{ignition}} \\
   0, & \text{otherwise}
   \end{cases}

The temperature source combines dissipation toward the reference temperature and
heat release from burning fuel:

.. math::

   S_T =
   -k_T (T - T_{\mathrm{ref}})
   + q_T (-S_{\mathrm{fuel}})

Smoke is produced from burning fuel and dissipates over time:

.. math::

   S_{\mathrm{smoke}} =
   q_{\mathrm{smoke}} (-S_{\mathrm{fuel}})
   - k_{\mathrm{smoke}}

These source terms are intentionally simple. They are designed for direct
artistic control instead of chemically accurate combustion.


Solution Procedure
------------------

For each simulation substep, the solver advances the system in the following
order:

1. Update dynamic masks, animated constants and time-dependent force inputs.
2. Assemble the body-force field from authored forces and turbulence.
3. Add buoyancy to the vertical force component.
4. Build and solve the pressure Poisson equation.
5. Optionally add vorticity forces.
6. Update the velocity field with either first-order or second-order upwind
   advection, plus pressure, viscosity and body forces.
7. Advect and update temperature, smoke, fuel and flame.
8. Apply domain, obstacle and source boundary conditions.
9. Write output if the simulation time reached the next output point.
10. Recompute a stable timestep from the updated state.

This ordering is important because pressure, buoyancy, thermal expansion and
scalar transport all interact. In particular, temperature affects buoyancy
through the body-force term and affects thermal expansion through the pressure
equation right-hand side.


Boundary Conditions and Practical Notes
---------------------------------------

The solver works on interior cells and handles boundaries explicitly. Pressure
uses zero-gradient boundary conditions during the Poisson solve. Velocity and
scalar fields are then overwritten by the configured domain, obstacle and source
conditions.

Near boundaries, the second-order upwind stencil is not always available, so
the scheme falls back to first-order upwind locally. This is one reason why the
solver remains robust even with more accurate advection enabled.

To improve stability further, the velocity update also limits the maximum
per-step velocity change to improve stability.
