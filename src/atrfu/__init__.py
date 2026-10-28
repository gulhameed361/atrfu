"""A-TRFu: an advanced trust-region funnel method for grey-box optimisation.

A grey-box problem couples an equation-oriented ("glass-box") Pyomo model with
black-box truth models that can only be evaluated. A-TRFu replaces each black
box by a local surrogate and solves a sequence of trust-region subproblems.

    from atrfu import TrustRegionSolver
    result = TrustRegionSolver().solve(model, efmap)
"""
from ._gjh import ensure_gjh
from .solver import TrustRegionSolver

__version__ = "1.0.0"

__all__ = ["TrustRegionSolver", "ensure_gjh", "__version__"]
