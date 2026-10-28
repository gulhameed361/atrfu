"""Checks run before a solve, with messages that name what to fix."""
from pyomo.core.base.external import PythonCallbackFunction

# (condition on the config, message) for each relation between options
_RELATIONS = [
    (lambda c: c.gamma_c < 1 < c.gamma_e, 'gamma_c < 1 < gamma_e'),
    (lambda c: c.eta1 < c.eta2 < 1, '0 < eta1 < eta2 < 1'),
    (lambda c: c.sigma_dec < 1 < c.sigma_inc, 'sigma_dec < 1 < sigma_inc'),
    (lambda c: c.sigma_eta3 < 1, '0 < sigma_eta3 < 1'),
    (lambda c: c.sample_radius_adjust <= 1, '0 < sample_radius_adjust <= 1'),
    (lambda c: c.delta_min <= c.ep_delta, 'delta_min <= ep_delta'),
    (lambda c: c.delta_min <= c.sample_radius, 'delta_min <= sample_radius'),
    (lambda c: c.radius_max >= c.trust_radius, 'radius_max >= trust_radius'),
    (lambda c: c.beta < 1 and c.kappa_f < 1, '0 < beta < 1 and 0 < kappa_f < 1'),
    (lambda c: c.gamma_f < 1 and c.gamma_theta < 1,
     '0 < gamma_f < 1 and 0 < gamma_theta < 1'),
]


def check_options(config):
    """Raise ValueError if the options violate a required relation."""
    broken = [msg for ok, msg in _RELATIONS if not ok(config)]
    if broken:
        raise ValueError("invalid options: need " + '; '.join(broken))


def check_efmap(efmap):
    """Raise ValueError unless efmap maps block names to lists of Python
    ExternalFunctions, each in one block only."""
    if not isinstance(efmap, dict) or not efmap:
        raise ValueError("efmap must be a non-empty dict "
                         "{block name: [ExternalFunction, ...]}")
    seen = {}
    for block, efs in efmap.items():
        if not isinstance(efs, (list, tuple)) or not efs:
            raise ValueError(f"efmap[{block!r}] must be a non-empty list of "
                             "ExternalFunctions")
        for ef in efs:
            if not isinstance(ef, PythonCallbackFunction):
                raise ValueError(
                    f"efmap[{block!r}] contains {getattr(ef, 'name', ef)} "
                    f"({type(ef).__name__}), which is not an "
                    "ExternalFunction wrapping a Python function "
                    "(pyomo.environ.ExternalFunction(my_function))")
            if id(ef) in seen:
                raise ValueError(f"{ef.name} appears in both efmap[{seen[id(ef)]!r}] "
                                 f"and efmap[{block!r}]")
            seen[id(ef)] = block
