"""Filter and funnel acceptance tests, and the trust-radius rules."""
from types import SimpleNamespace

import pytest

from atrfu._algorithm import contract, expand, radius_by_ratio, radius_on_reject
from atrfu._filter import Filter
from atrfu._funnel import Funnel


def make_filter(**kw):
    args = dict(kappa_theta=0.1, gamma_s=2.0, theta_min=1e-4, theta_max=50.0,
                gamma_f=0.01, gamma_theta=0.01)
    return Filter(**{**args, **kw})


def make_funnel(phi=1.0, **kw):
    args = dict(phi_min=1e-8, kappa_f=0.5, beta=0.8, switch_coeff=0.01,
                switch_exponent=2.0, eta=0.01, kappa_r=1.1)
    return Funnel(phi_init=phi, **{**args, **kw})


class TestFilter:
    def test_f_step_needs_switching_and_small_theta(self):
        flt = make_filter()
        assert flt.classify(1e-5, 1e-5, f_k=1.0, f=0.5, delta=1) == 'f'
        # theta_k >= theta_min: theta-step even with a large decrease
        assert flt.classify(1e-3, 1e-3, f_k=1.0, f=0.5, delta=1) == 'theta'

    def test_theta_step_augments_filter_and_dominated_points_are_rejected(self):
        flt = make_filter()
        flt.accept('theta', theta=1.0, f=2.0)
        assert flt.entries == [(2.0 - 0.01, 0.99)]
        assert flt.classify(0.5, 1.0, f_k=2.0, f=2.5, delta=1) == 'reject'
        assert flt.classify(0.5, 0.5, f_k=2.0, f=2.5, delta=1) == 'theta'

    def test_new_entry_removes_the_entries_it_dominates(self):
        flt = make_filter()
        flt.abandon(theta=1.0, f=2.0)
        flt.abandon(theta=0.5, f=1.0)
        assert len(flt.entries) == 1 and flt.entries[0][1] == pytest.approx(0.495)

    def test_theta_max_except_on_first_iteration(self):
        flt = make_filter(theta_max=10.0)
        assert flt.classify(1.0, 20.0, f_k=1, f=1, delta=1) == 'reject'
        assert flt.classify(1.0, 20.0, f_k=1, f=1, delta=1,
                            first_iteration=True) == 'theta'

    def test_f_step_leaves_filter_unchanged(self):
        flt = make_filter()
        flt.accept('f', theta=1.0, f=2.0)
        assert flt.entries == []


class TestFunnel:
    def test_inside_funnel(self):
        fun = make_funnel(phi=1.0)
        assert fun.classify(0.1, 0.5, f_k=1.0, f=0.5, delta=1.0) == 'f'
        # switching holds but the decrease is below eta * delta: rejected
        assert fun.classify(0.1, 0.5, f_k=1.0, f=0.995, delta=1.0) == 'reject'
        # no switching: theta-step needs theta+ <= beta * phi
        assert fun.classify(0.5, 0.7, f_k=1.0, f=1.0, delta=1.0) == 'theta'
        assert fun.classify(0.5, 0.9, f_k=1.0, f=1.0, delta=1.0) == 'reject'

    def test_outside_funnel(self):
        fun = make_funnel(phi=1.0)
        assert fun.classify(0.1, 1.05, f_k=1.0, f=0.5, delta=1.0) == 'theta-relax'
        assert fun.classify(0.1, 1.2, f_k=1.0, f=0.5, delta=1.0) == 'reject'
        assert fun.classify(0.5, 1.05, f_k=1.0, f=1.0, delta=1.0) == 'reject'

    def test_theta_steps_contract_and_f_steps_and_abandon_do_not(self):
        fun = make_funnel(phi=1.0)
        fun.accept('f', theta=0.2, f=0.0)
        fun.abandon(theta=0.2, f=0.0)
        assert fun.phi == 1.0
        fun.accept('theta', theta=0.2, f=0.0)
        assert fun.phi == pytest.approx(0.6)
        fun.accept('theta-relax', theta=0.2, f=0.0)
        assert fun.phi == pytest.approx(0.4)

    def test_width_floor(self):
        fun = make_funnel(phi=0.0, phi_min=1e-3)
        assert fun.phi == 1e-3
        fun.accept('theta', theta=0.0, f=0.0)
        assert fun.phi == 1e-3


def radius_config(**kw):
    args = dict(trust_radius=1.0, gamma_c=0.5, gamma_e=2.5, delta_min=1e-6,
                radius_max=4.0, eta1=0.05, eta2=0.2, restoration_strategy=1)
    return SimpleNamespace(**{**args, **kw})


class TestRadius:
    def test_contract_and_expand_respect_limits(self):
        c = radius_config()
        contract(c)
        assert c.trust_radius == 0.5
        for _ in range(3):
            expand(c)
        assert c.trust_radius == 4.0
        c.trust_radius = 1e-6
        contract(c)
        assert c.trust_radius == 1e-6

    @pytest.mark.parametrize('rho, expected', [(0.0, 0.5), (0.1, 1.0), (0.5, 2.5)])
    def test_ratio_rule(self, rho, expected):
        c = radius_config()
        radius_by_ratio(c, rho)
        assert c.trust_radius == expected

    def test_rejection_r1_always_contracts(self):
        c = radius_config(restoration_strategy=1)
        radius_on_reject(c, restoration_mode=True, rhok=0.9)
        assert c.trust_radius == 0.5

    def test_rejection_r2_restoration_uses_ratio(self):
        c = radius_config(restoration_strategy=2)
        radius_on_reject(c, restoration_mode=True, rhok=0.9)
        assert c.trust_radius == 2.5
        radius_on_reject(c, restoration_mode=False, rhok=0.9)
        assert c.trust_radius == 1.25
