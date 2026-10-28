"""Options: defaults, spellings, presets, cross-option checks and their docs."""
import pytest

from atrfu import PRESETS, TrustRegionSolver
from atrfu.solver import CONFIG, DOCS, GROUPS


def test_every_option_is_grouped_and_documented_once():
    grouped = [name for members in GROUPS.values() for name in members]
    assert sorted(grouped) == sorted(CONFIG) == sorted(DOCS)
    assert all(DOCS[name].strip() for name in DOCS)


def test_default_is_the_a_trfu_preset():
    config = TrustRegionSolver().config
    assert all(config[k] == v for k, v in PRESETS['a-trfu'].items())


def test_spaced_and_underscored_names_are_one_option():
    assert TrustRegionSolver(max_it=7).config.max_it == 7
    assert TrustRegionSolver(**{'max it': 7}).config['max it'] == 7


def test_preset_and_override():
    config = TrustRegionSolver(preset='filter', restoration_strategy=2).config
    assert config.globalization_strategy == 0
    assert config.restoration_strategy == 2


def test_radius_max_defaults_to_1000_trust_radius():
    assert TrustRegionSolver(trust_radius=2.0).config.radius_max == 2000.0


def test_solvers_do_not_share_state():
    a, b = TrustRegionSolver(), TrustRegionSolver(solver_options={'tol': 1e-8})
    assert 'tol' not in a.config.solver_options and b.config.solver_options.tol == 1e-8


@pytest.mark.parametrize('options, match', [
    (dict(max_itt=5), r"'max_itt' \(did you mean max_it"),
    (dict(preset='funnel'), 'unknown preset'),
    (dict(globalization_strategy=2), 'globalization strategy'),
    (dict(gamma_c=0.5, gamma_e=0.9), 'gamma_c < 1 < gamma_e'),
    (dict(eta1=0.5, eta2=0.2), 'eta1 < eta2'),
    (dict(sigma_inc=0.9), 'sigma_dec < 1 < sigma_inc'),
    (dict(delta_min=1e-3, ep_delta=1e-5), 'delta_min <= ep_delta'),
    (dict(trust_radius=2.0, radius_max=1.0), 'radius_max >= trust_radius'),
    (dict(beta=1.0), 'beta < 1'),
])
def test_invalid_options_are_rejected(options, match):
    with pytest.raises(ValueError, match=match):
        TrustRegionSolver(**options)


def test_options_listing(capsys):
    TrustRegionSolver.options('Funnel')
    out = capsys.readouterr().out
    assert out.lstrip().startswith('Funnel') and 'kappa_f' in out and 'max_it' not in out
    TrustRegionSolver.options()
    out = capsys.readouterr().out
    assert all(group in out for group in GROUPS)
