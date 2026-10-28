# How the solver works

A-TRFu is Algorithm 7.1 of the thesis listed under [citing](citing.md). In
outline, one iteration $k$ does the following:

1. **Surrogates.** Each black box $d_b$ is replaced by a linear model $r_b$,
   built from one-sided finite differences of size $\sigma_k$ (the *sampling
   radius*) around the current point.
2. **Criticality.** The measure $\chi_k$ (small near a stationary point) is
   computed. If $\sigma_k > \xi\,\chi_k$ (`criticality_check`), the
   sampling radius is halved, so the surrogates become more accurate as the
   run converges.
3. **Termination.** The run stops when the point is feasible
   ($\theta_k = \sum_b \|d_b(x_k) - y_b\|_1 < \epsilon_\theta$, `ep_i`) and either it is critical
   ($\chi_k < \epsilon_\chi$ and $\sigma_k < \epsilon_\Delta$), or the last
   step was shorter than `ep_s`.
4. **Step.** The trust-region subproblem minimises the objective subject to
   the glass-box constraints, the surrogates, and the box
   $\|x - x_k\|_\infty \le \Delta_k$. It is solved with IPOPT. After two
   consecutive rejected steps, a *restoration* subproblem is solved instead.
   It minimises the surrogate mismatch $\sum_b \|r_b(x) - y_b\|_2^2$
   (thesis (7.11)), within the same trust region and glass-box constraints.
5. **Acceptance.** The black boxes are evaluated at the trial point. The
   **funnel** (or the **filter**) classifies the step:
   - an *f-step* reduces the objective while keeping $\theta$ within the funnel;
   - a *θ-step* reduces $\theta$ and narrows the funnel;
   - otherwise the step is rejected.
6. **Radii.**
   - $\Delta_k$ grows after f-steps and after successful θ-steps, and shrinks
     after rejections. Under R2 restoration, the ratio test decides instead.
   - With several black boxes, $\sigma_k$ follows the hypervolume of the
     per-black-box errors: it grows when a step improves that hypervolume as
     predicted, and it shrinks when the step does not improve it.

The code follows the same structure. The loop is `atrfu._algorithm.TRF`. The
acceptance tests are in `atrfu._funnel` and `atrfu._filter`, and the
hypervolume is in `atrfu._hypervolume`. The Pyomo model transformation and the
subproblems are in `atrfu._interface`.
