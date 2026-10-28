"""Pareto front, hypervolume and the hypervolume sampling-radius rule."""
from types import SimpleNamespace

import numpy as np
import pytest

from atrfu._algorithm import hv_reduction_ratio, update_sampling_region
from atrfu._hypervolume import calculate_hypervolume, is_dominated, update_pareto_front


def test_dominance():
    front = [np.array([1.0, 2.0])]
    assert is_dominated(np.array([2.0, 3.0]), front)
    assert not is_dominated(np.array([0.5, 3.0]), front)
    assert not is_dominated(np.array([1.0, 2.0]), front)   # equal is not dominated
    assert not is_dominated(np.array([1.0, 1.0]), [])


def test_front_update():
    front = [np.array([1.0, 2.0]), np.array([2.0, 1.0])]
    assert len(update_pareto_front(front, np.array([3.0, 3.0]))) == 2      # dominated
    assert len(update_pareto_front(front, np.array([1.0, 2.0]))) == 2      # duplicate
    assert len(update_pareto_front(front, np.array([0.5, 3.0]))) == 3      # new trade-off
    assert len(update_pareto_front(front, np.array([0.5, 0.5]))) == 1      # dominates all


def test_hypervolume_by_inclusion_exclusion():
    ref = np.array([3.0, 3.0])
    assert calculate_hypervolume([], ref) == 0.0
    assert calculate_hypervolume([np.array([1.0, 2.0])], ref) == pytest.approx(2.0)
    # union of two boxes: 2 + 2 - 1
    front = [np.array([1.0, 2.0]), np.array([2.0, 1.0])]
    assert calculate_hypervolume(front, ref) == pytest.approx(3.0)
    assert calculate_hypervolume([np.array([1.0, 1.0, 1.0])],
                                 np.array([2.0, 3.0, 4.0])) == pytest.approx(6.0)


def test_reduction_ratio():
    front, ref = [np.array([2.0, 2.0])], np.array([4.0, 4.0])
    hv = calculate_hypervolume(front, ref)
    # reaching zero infeasibility realises all of the predicted improvement
    rho, _ = hv_reduction_ratio(front, ref, [0.0, 0.0], hv)
    assert rho == pytest.approx(1.0)
    rho, _ = hv_reduction_ratio(front, ref, [3.0, 3.0], hv)
    assert rho == 0.0


@pytest.mark.parametrize('rho, expected', [(0.9, 0.15), (0.3, 0.1), (-0.1, 0.05)])
def test_sampling_radius_rule(rho, expected):
    config = SimpleNamespace(sample_radius=0.1, trust_radius=1.0, delta_min=1e-6,
                             sigma_eta3=0.5, sigma_inc=1.5, sigma_dec=0.5)
    front = [np.array([1.0, 1.0])]
    new_front, _ = update_sampling_region(config, rho, [0.5, 0.5], 1.0, 0.5, front)
    assert config.sample_radius == pytest.approx(expected)
    assert len(new_front) == 1 and (new_front[0] == (0.5 if rho > 0 else 1.0)).all()
