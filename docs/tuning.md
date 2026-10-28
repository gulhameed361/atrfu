# Tuning guide

The defaults are a sound starting point. Change options only when a run
shows a reason to. This page explains how to set options, which ones matter,
and what to try for common symptoms.

## Setting options

```python
from atrfu import TrustRegionSolver

solver = TrustRegionSolver(max_it=300, sample_radius=0.05)
solver = TrustRegionSolver(**{"max it": 300})                 # spaces work too
solver = TrustRegionSolver(preset="a-trfu-r2", n_restoration_max=20)
solver = TrustRegionSolver(solver_options={"linear_solver": "ma57"})   # passed to IPOPT
TrustRegionSolver.options()            # every option with its default and meaning
TrustRegionSolver.options("Funnel")    # one group
```

The solver checks the options when it is created. An unknown name gets a
"did you mean" hint. Values that break a required ordering are rejected with a
message that names the ordering:

- `gamma_c < 1 < gamma_e`
- `0 < eta1 < eta2 < 1`
- `sigma_dec < 1 < sigma_inc`
- `sigma_eta3 < 1`
- `sample_radius_adjust <= 1`
- `delta_min <= ep_delta` and `delta_min <= sample_radius`
- `radius_max >= trust_radius`
- `beta`, `kappa_f`, `gamma_f` and `gamma_theta` all `< 1`

## The options that matter most

**Bounds and `scaling`.** With the default `scaling=0`, distances are
measured in units of each variable's bound range. The first trust region is
the whole box, and `trust_radius` is ignored. Bounds that are much wider than
the region of interest therefore make the first steps coarse. Give realistic
bounds. If only the black-box inputs have meaningful bounds, use `scaling=1`
and set `trust_radius` in the variables' units.

**`sample_radius`.** This is the finite-difference step used to build the
surrogates (with `scaling=0`, a fraction of each bound range). The solver
adapts it as the run proceeds. The initial value matters most when the black
boxes are noisy, because a step that is too small then measures the noise
rather than the slope. Use a larger value in that case.

**Termination: `ep_i`, `ep_chi`, `ep_delta`, `ep_s`, `max_it`.**
- `ep_i` is the feasibility tolerance on the black-box errors. Set it to the
  accuracy you need from the black boxes.
- `ep_chi` and `ep_delta` define "optimal".
- `ep_s` is the short-step exit.
- `max_it` caps the iterations. Every iteration costs at least one black-box
  evaluation per input.

**`efmap` grouping and `multiple_black_boxes`.** Give each independent
black box its own `efmap` entry (see the [user guide](guide.md)).

**Leave at their defaults unless you have a reason:** the radius factors
(`gamma_c`, `gamma_e`, `eta1`, `eta2`), the funnel and filter parameters, and
the hypervolume factors (`sigma_*`). They govern how the method converges,
not the problem.

## Symptoms and what to try

| symptom | what to try |
|---|---|
| `POSSIBLY AN OPTIMAL SOLUTION IS FOUND` but `final_chi` is not small | The run stalled rather than converged. Check that independent black boxes have separate `efmap` entries, and try another starting point. A smaller `ep_s` alone only delays the exit. |
| `Maximum iterations` while `theta` and `obj` in `result["trajectory"]` are still improving | Raise `max_it`. |
| `Maximum iterations` with `theta` stuck just above `ep_i` | Check that `ep_i` is not tighter than the black boxes' accuracy. Try the default configuration if you changed presets. |
| Many rejected steps, slow progress | Tighten the variable bounds (`scaling=0`), or use `scaling=1` with a `trust_radius` that matches the problem's scale. |
| `RESTORATION STATIONARY -- NOT FEASIBLE` (R2) | If `theta` was still falling, raise `n_restoration_max`; otherwise use R1 (`restoration_strategy=1`). |
| `... STALLED -- NOT FEASIBLE` | The subproblems keep failing. Check that the glass-box model is feasible within the bounds, start from a better point, or raise `n_radius_resets`. |
| `WARNING: TRSP solve failed` lines in the log | IPOPT failed on a subproblem. Pass IPOPT options through `solver_options` (for example a different `linear_solver`), and check the model's scaling. |
| Noisy black boxes | Use a larger `sample_radius`. Set `bb_cache=0` only if repeated calls at the same point should return new values. |
| The black boxes are slow | Watch `bb_evals` in the summary. A looser `ep_i` or `ep_chi`, or a smaller `max_it`, stops earlier. |
