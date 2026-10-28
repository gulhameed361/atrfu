"""A-TRFu: an advanced trust-region funnel method for grey-box optimisation.

A grey-box problem couples an equation-oriented ("glass-box") Pyomo model with
black-box truth models that can only be evaluated. A-TRFu replaces each black
box by a local surrogate and solves a sequence of trust-region subproblems.

    from atrfu import TrustRegionSolver
    result = TrustRegionSolver().solve(model, efmap)
"""
__version__ = "1.0.0"

from ._gjh import ensure_gjh  # noqa: E402
from .presets import PRESETS  # noqa: E402
from .solver import TrustRegionSolver  # noqa: E402

__all__ = ["PRESETS", "TrustRegionSolver", "ensure_gjh", "__version__"]
