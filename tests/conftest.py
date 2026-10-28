import importlib.util
import math
from pathlib import Path

import pytest
from pyomo.environ import SolverFactory

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
#: Optimum of examples/quickstart.py: the point of the unit disc nearest (2, 1)
F_STAR = (math.sqrt(5) - 1) ** 2
X_STAR = (2 / math.sqrt(5), 1 / math.sqrt(5))


def load_example(name):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def quickstart():
    """The quickstart problem: (model, efmap)."""
    return load_example('quickstart').build_model()


requires_ipopt = pytest.mark.skipif(
    not SolverFactory('ipopt').available(exception_flag=False),
    reason='IPOPT not installed')
