"""Problem checks that run before the first solve, with messages naming the fix."""
import pytest
from pyomo.environ import Constraint, ExternalFunction, Objective, Param

from atrfu import TrustRegionSolver
from conftest import requires_ipopt


def solve(model, efmap):
    return TrustRegionSolver(verbose=0).solve(model, efmap)


@pytest.mark.parametrize('efmap, match', [
    ({}, 'non-empty dict'),
    ({'distance': []}, 'non-empty list'),
    ({'distance': ['bb1']}, 'not an ExternalFunction'),
])
def test_malformed_efmap(quickstart, efmap, match):
    model, _ = quickstart
    with pytest.raises(ValueError, match=match):
        solve(model, efmap)


def test_function_in_two_blocks(quickstart):
    m, _ = quickstart
    with pytest.raises(ValueError, match='appears in both'):
        solve(m, {'a': [m.bb1, m.bb2], 'b': [m.bb2]})


def test_unmapped_function(quickstart):
    m, _ = quickstart
    with pytest.raises(ValueError, match='missing from efmap: bb2'):
        solve(m, {'distance': [m.bb1]})


def test_unused_block(quickstart):
    m, efmap = quickstart
    m.bb3 = ExternalFunction(lambda a: a)
    with pytest.raises(ValueError, match='no active constraint calls: spare'):
        solve(m, {**efmap, 'spare': [m.bb3]})


@pytest.mark.parametrize('n_objectives', [0, 2])
def test_needs_one_objective(quickstart, n_objectives):
    m, efmap = quickstart
    if n_objectives == 0:
        m.obj.deactivate()
    else:
        m.obj2 = Objective(expr=m.x1)
    with pytest.raises(ValueError, match=f'exactly one active objective .found {n_objectives}'):
        solve(m, efmap)


def test_black_box_input_must_be_a_free_variable(quickstart):
    m, efmap = quickstart
    m.p = Param(initialize=1.0, mutable=True)
    m.link2.deactivate()
    m.link3 = Constraint(expr=m.y2 == m.bb2(m.x1, m.p))
    with pytest.raises(ValueError, match='distinct, unfixed variables'):
        solve(m, efmap)


@requires_ipopt
def test_unbounded_input(quickstart):
    m, efmap = quickstart
    m.x1.setlb(None)
    with pytest.raises(ValueError, match='variables without finite bounds: x1'):
        solve(m, efmap)
