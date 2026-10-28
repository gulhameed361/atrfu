# atrfu

A-TRFu, an advanced trust-region funnel method for **grey-box optimisation**
with [Pyomo](https://www.pyomo.org). A grey-box problem couples an
equation-based Pyomo model with black boxes, such as simulations or legacy
code, that can only be evaluated.

**Documentation:** <https://atrfu.readthedocs.io>

## Install

```bash
pip install atrfu
conda install -c conda-forge ipopt     # the subproblem solver (not pip-installable)
atrfu check                            # confirms IPOPT and gjh are found
```

AMPL's gjh is downloaded automatically on first use, or in advance with
`atrfu install-gjh`.

## Example

```python
from pyomo.environ import ConcreteModel, Constraint, ExternalFunction, Objective, Var
from atrfu import TrustRegionSolver

def distance(a, b):                    # a black box: only its values are used
    return a**2 + b**2

m = ConcreteModel()
m.x1 = Var(bounds=(-3, 3), initialize=0.5)
m.x2 = Var(bounds=(-3, 3), initialize=0.5)
m.y = Var(bounds=(0, 18))
m.bb = ExternalFunction(distance)
m.link = Constraint(expr=m.y == m.bb(m.x1, m.x2))
m.disc = Constraint(expr=m.y <= 1)
m.obj = Objective(expr=(m.x1 - 2)**2 + (m.x2 - 1)**2)

result = TrustRegionSolver(verbose=1).solve(m, {"distance": [m.bb]})
print(result["status"], result["final_obj"], m.x1.value, m.x2.value)
```

See the [user guide](https://atrfu.readthedocs.io/en/stable/guide.html) and
the [tuning guide](https://atrfu.readthedocs.io/en/stable/tuning.html).

## Citing

Please cite G. Hameed, *Rigorous Trust-Region Algorithms for Surrogate-based
Grey-box Optimisation*, PhD thesis, University of Surrey, 2026, and the
software release (`CITATION.cff`, Zenodo DOI per release).

## License

MIT. Files derived from Pyomo keep their BSD-3-Clause notice.
