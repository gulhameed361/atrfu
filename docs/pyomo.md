# Relation to Pyomo's trust-region solver

Pyomo ships a trust-region filter method for grey-box problems,
`pyomo.contrib.trustregion` (`SolverFactory('trustregion')`). It also takes
black boxes as `ExternalFunction`s. A-TRFu belongs to the same family of
methods and adds:

- a **funnel** in place of the filter, to accept steps;
- **separate errors for each black box**, with the sampling region driven by
  their hypervolume;
- a **feasibility-restoration phase** with two radius policies (R1 and R2).

The method and the comparison are described in the thesis listed under
[citing](citing.md). To move a problem from Pyomo's solver, keep the model,
list the external functions by black box in `efmap`, and call
`TrustRegionSolver().solve(model, efmap)`.
