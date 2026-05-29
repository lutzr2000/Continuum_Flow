Best Practice
=============

In this final section, a few tips for working with Continuum Flow are collected.

- The most important rule: when something looks weird or the solver diverges, try reducing the CFL number. This sadly comes at the cost of longer compute times, but it can often fix issues.
- While a large CFL value is desirable for quick simulations, some cases really benefit from lower CFL values, like the 04 explosion example.
- When setting up your simulation, start with low resolutions and only go higher towards the end of your iteration procedure.
- Only output the data you really need. Saving volume grids can produce very large amounts of data very quickly, which can fill up your disk and slow the solver down.
- When doing your final high-resolution simulation, turn off the "live preview". This frees up RAM for the solver.

With these tips, I wish you fun and success while gathering experience with Continuum Flow!

