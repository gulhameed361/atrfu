# atrfu

**atrfu** solves *grey-box* optimisation problems with A-TRFu, an advanced
trust-region funnel method. Part of the model is written in Pyomo (the
*glass box*). The rest comes from black boxes, such as simulations or legacy
code, that can only be evaluated, with no derivatives.

```python
from atrfu import TrustRegionSolver

result = TrustRegionSolver().solve(model, efmap)
```

The solver replaces each black box with a local linear surrogate built from
finite differences. It then solves a sequence of trust-region subproblems
with IPOPT and accepts or rejects each step with a funnel. With several black
boxes, it measures the error of each one separately and adapts the sampling
region from the hypervolume of their errors. A feasibility-restoration phase
recovers from runs of rejected steps.

```{toctree}
:maxdepth: 2
:caption: Using atrfu

install
quickstart
guide
tuning
options
```

```{toctree}
:maxdepth: 2
:caption: Reference

algorithm
api
pyomo
changelog
citing
contributing
```
