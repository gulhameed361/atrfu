"""End-to-end solves of the quickstart problem (need IPOPT; gjh is fetched on demand)."""
import runpy

import numpy as np
import pytest
from pyomo.environ import Objective, maximize

from atrfu import PRESETS, TrustRegionSolver
from conftest import EXAMPLES, F_STAR, X_STAR, load_example, requires_ipopt

pytestmark = requires_ipopt
TOL = 1e-4
#: Iteration budget for tests of the mechanics rather than of convergence
SHORT = 15


def assert_optimal(result, model, f_star=F_STAR):
    assert result['feasible']
    assert result['final_obj'] == pytest.approx(f_star, abs=TOL)
    assert (model.x1.value, model.x2.value) == pytest.approx(X_STAR, abs=1e-3)


def test_quickstart_reaches_the_optimum(quickstart):
    model, efmap = quickstart
    result = TrustRegionSolver(verbose=0).solve(model, efmap)
    assert_optimal(result, model)
    assert result['bb_evals'] == sum(result['bb_evals_detail'].values())
    assert set(result['bb_evals_detail']) == set(efmap)
    # one entry per iterate, from the starting point (iteration 0) to the last
    assert result['trajectory']['iter'] == list(range(result['iterations'] + 1))


def test_verbose_0_is_silent(quickstart, capsys):
    TrustRegionSolver(verbose=0, max_it=SHORT).solve(*quickstart)
    assert capsys.readouterr().out == ''


# Only the default configuration is tested for convergence: on the quickstart
# the others stall or stop early (as does the A-TRFu reference code).
@pytest.mark.slow
@pytest.mark.parametrize('preset', sorted(set(PRESETS) - {'a-trfu'}))
def test_presets_run(quickstart, preset):
    result = TrustRegionSolver(preset=preset, verbose=0, max_it=SHORT).solve(*quickstart)
    assert result['restoration_strategy'] == PRESETS[preset]['restoration_strategy']
    assert 0 < result['iterations'] <= SHORT + 1


@pytest.mark.slow
def test_maximisation_reports_the_users_sense(quickstart):
    m, efmap = quickstart
    m.obj.deactivate()
    m.neg = Objective(expr=-m.obj.expr, sense=maximize)
    result = TrustRegionSolver(verbose=0).solve(m, efmap)
    assert_optimal(result, m, -F_STAR)
    assert result['trajectory']['obj'][-1] == pytest.approx(-F_STAR, abs=TOL)


@pytest.mark.slow
def test_one_black_box_with_two_outputs(quickstart):
    model, _ = quickstart
    result = TrustRegionSolver(verbose=0, max_it=SHORT).solve(
        model, {'both': [model.bb1, model.bb2]})
    assert list(result['bb_evals_detail']) == ['both']
    assert result['bb_evals_detail']['both'] == result['bb_evals'] > 0


@pytest.mark.slow
def test_solve_is_repeatable():
    solver = TrustRegionSolver(verbose=0, max_it=SHORT)
    before = solver.config.value()
    runs = [solver.solve(*load_example('quickstart').build_model()) for _ in range(2)]
    assert solver.config.value() == before
    np.testing.assert_equal(runs[0]['trajectory'], runs[1]['trajectory'])   # NaN-aware
    assert runs[0]['bb_evals'] == runs[1]['bb_evals']


@pytest.mark.slow
@pytest.mark.parametrize('script', sorted(p.name for p in EXAMPLES.glob('*.py')))
def test_examples_run(script):
    runpy.run_path(str(EXAMPLES / script), run_name='__main__')


@pytest.mark.slow
def test_readme_example():
    readme = (EXAMPLES.parent / 'README.md').read_text(encoding='utf-8')
    code = readme.split('```python\n', 1)[1].split('```', 1)[0]
    exec(compile(code, 'README.md', 'exec'), {})
