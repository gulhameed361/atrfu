# User guide

## Defining a grey-box problem

A problem is a Pyomo `ConcreteModel` plus an `efmap`.

**Black boxes.** Each black-box output is a Python function wrapped in
`pyomo.environ.ExternalFunction`. It is called with floats and returns one
float. A black box with several outputs has one function per output. Call
each function in a constraint that links it to an output variable:

```python
m.bb_conv = ExternalFunction(conversion)          # conversion(T, tau) -> float
m.bb_sel = ExternalFunction(selectivity)
m.link_conv = Constraint(expr=m.conv == m.bb_conv(m.T, m.tau))
m.link_sel = Constraint(expr=m.sel == m.bb_sel(m.T, m.tau))
```

**Inputs.** The arguments of a black-box call must be distinct, unfixed
variables with finite bounds. For a constant, a parameter or an expression,
introduce a bounded variable and constrain it to that value.

**efmap.** `efmap` groups the functions by black box: `{name: [functions]}`.
For example:

```python
efmap = {"reactor": [m.bb_conv, m.bb_sel], "flash": [m.bb_split]}
```

Every function that the model calls must appear exactly once in `efmap`. The
error of a black box is the sum of the errors of its outputs. With the default
`multiple_black_boxes=1`, the solver tracks the error of each black box
separately. So give each independent black box its own entry, and group only
the outputs that really come from one model. On the quick start, the two
functions as separate black boxes reach the optimum. Grouped as one black
box, the run stops early at f = 2.20 instead of 1.53.

**Bounds.** By default (`scaling=0`), the trust region is measured in units of
each variable's bound range, so every unfixed variable needs finite bounds.
With `scaling=1`, only the black-box inputs need bounds, and `trust_radius` is
in the variables' own units.

**Objective.** The model needs exactly one active objective. It may minimise
or maximise. Results are reported in the objective's own sense.

**Expensive outputs that share one simulation.** The solver calls each output
function separately. If several outputs come from one expensive simulation,
cache the simulation in your own code (for example with
`functools.lru_cache` on the inputs), so it runs once per point.

## Choosing a configuration

The default is A-TRFu: a funnel, the hypervolume sampling region and
restoration R1. Four switches select the components, and presets name the
common combinations:

| preset | `globalization_strategy` | `multiple_black_boxes` | `restoration_enabled` | `restoration_strategy` |
|---|---|---|---|---|
| `"a-trfu"` (default) | 1 funnel | 1 per-black-box errors, hypervolume | 1 | 1 (R1) |
| `"a-trfu-r2"` | 1 | 1 | 1 | 2 (R2) |
| `"no-hv"` | 1 | 0 one aggregate error | 1 | 1 |
| `"no-restoration"` | 1 | 1 | 0 | 1 |
| `"filter"` | 0 filter | 1 | 1 | 1 |

```python
TrustRegionSolver(preset="filter", max_it=300)   # explicit options override the preset
```

Start with the default. The other presets are mainly for comparison. On the
quick start, they stall or stop before the optimum.

## Reading the result

`solve` returns a dict and leaves the final point in the model's variables.

| key | meaning |
|---|---|
| `status` | exit status (next section) |
| `feasible` | `final_theta < ep_i` |
| `final_obj` | objective at the final point, in the model's sense |
| `final_theta` | infeasibility: 1-norm of true black-box outputs minus output variables |
| `final_chi` | criticality measure at the final point (small near a stationary point) |
| `iterations` | index of the final iterate |
| `bb_evals`, `bb_evals_detail` | true black-box evaluations, total and per black box |
| `bb_cache_hits` | calls answered from the evaluation cache (`bb_cache`) |
| `f_steps`, `theta_steps`, `rejected` | accepted objective steps, accepted feasibility steps, rejected steps |
| `restoration_phases`, `frp_solves`, `restorations` | restoration phases entered, restoration subproblems solved, rejected restoration iterations |
| `wall_s`, `cpu_s` | run time |
| `n_vars`, `n_vars_xyz`, `n_constraints`, ... | problem size (x = black-box inputs, y = outputs, z = others) |
| `duals` | constraint multipliers, with `report_duals=1` |
| `trajectory` | per iterate, from the starting point (iteration 0): `theta`, `obj`, `chi`, `trust_radius`, `sample_radius`, `step_norm` |

## Exit statuses

| status | meaning |
|---|---|
| `OPTIMAL SOLUTION FOUND` | feasible, `chi < ep_chi` and sampling radius `< ep_delta` |
| `POSSIBLY AN OPTIMAL SOLUTION IS FOUND` | feasible and the last step was shorter than `ep_s`. **Check `final_chi`**: a short step can also mean the run has stalled. |
| `FEASIBLE SOLUTION FOUND` (…) | feasible, but the trust region shrank to `delta_min` or restoration stopped, so optimality is not established |
| `Maximum iterations` | `max_it` reached |
| `RESTORATION STATIONARY -- NOT FEASIBLE` | R2 restoration reached `n_restoration_max` without reaching feasibility |
| `TRSP STALLED -- NOT FEASIBLE`, `RESTORATION STALLED -- NOT FEASIBLE` | subproblems kept failing at the smallest radius and the radius resets were used up |

With `verbose=2` (the default), the log shows each iteration and each
restoration step. `verbose=1` prints only the banner and the summary, and
`verbose=0` prints nothing.
