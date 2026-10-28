"""TrustRegionSolver: the A-TRFu solver and its options."""
import difflib
import textwrap

from pyomo.common.config import (
    ConfigBlock,
    ConfigValue,
    In,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
)
from pyomo.environ import Objective, Var, value

from ._algorithm import TRF
from ._validate import check_options
from .presets import PRESETS

CONFIG = ConfigBlock('A-TRFu')
#: Option names by purpose, in declaration order, and their descriptions
#: (the source of options() and of the options page of the documentation).
GROUPS = {}
DOCS = {}


def _option(group, name, default, domain, doc):
    GROUPS.setdefault(group, []).append(name)
    DOCS[name] = doc
    CONFIG.declare(name, ConfigValue(default=default, domain=domain, description=doc))


_Binary = In([0, 1])

# ---- Strategy ---------------------------------------------------------------
_option('Strategy', 'globalization strategy', 1, _Binary,
        'How trial points are accepted: 1 = funnel (A-TRFu), 0 = filter.')
_option('Strategy', 'multiple black boxes', 1, _Binary,
        '1 = measure the infeasibility of each black box separately and set the '
        'sampling radius from the hypervolume of their Pareto front (thesis '
        'Section 7.2); 0 = one aggregate infeasibility, no hypervolume rule.')
_option('Strategy', 'restoration enabled', 1, _Binary,
        '1 = after two consecutive rejected (or failed) steps, solve the '
        'feasibility restoration problem (7.11) instead of the trust-region '
        'subproblem until a step is accepted; 0 = never.')
_option('Strategy', 'restoration strategy', 1, In([1, 2]),
        'Trust-radius policy during restoration (thesis Section 7.3). 1 = R1: '
        'keep the current radius and contract it by gamma_c on every rejected '
        'restoration step. 2 = R2: on entering restoration reset the radius to '
        'min(gamma_e * last accepted radius, radius_max) (with the filter, the '
        'abandoned iterate is also added to the filter); a rejected restoration '
        'step updates the radius by the ratio test (eta1, eta2); after '
        'n_restoration_max consecutive restoration iterations the run stops.')

# ---- Budget and termination -------------------------------------------------
_option('Budget and termination', 'max it', 100, PositiveInt,
        'Maximum number of iterations.')
_option('Budget and termination', 'ep i', 1e-5, PositiveFloat,
        'Feasibility tolerance. The infeasibility theta is the 1-norm of the true '
        'black-box outputs minus the output variables y; an iterate is feasible '
        'when theta < ep_i (result["feasible"]). The OPTIMAL, POSSIBLY OPTIMAL '
        'and FEASIBLE SOLUTION exits require it.')
_option('Budget and termination', 'ep chi', 1e-3, PositiveFloat,
        'Criticality tolerance: the run stops ("OPTIMAL SOLUTION FOUND") at a '
        'feasible iterate with chi < ep_chi and sampling radius < ep_delta.')
_option('Budget and termination', 'ep delta', 1e-3, PositiveFloat,
        'Sampling-radius tolerance of the "OPTIMAL SOLUTION FOUND" test.')
_option('Budget and termination', 'ep s', 1e-4, PositiveFloat,
        'Step tolerance: the run stops ("POSSIBLY AN OPTIMAL SOLUTION IS FOUND") '
        'at a feasible iterate when the last trial step, accepted or not, had '
        '||s||_inf < ep_s.')

# ---- Trust region and sampling region ---------------------------------------
_option('Trust region and sampling region', 'scaling', 0, _Binary,
        "0 = measure the trust region and the finite-difference steps in units "
        "of each variable's bound range; every unfixed variable then needs finite "
        "bounds, and trust_radius and radius_max are set to 1 (the whole box). "
        "1 = absolute units; only the black-box inputs need finite bounds.")
_option('Trust region and sampling region', 'trust radius', 1.0, PositiveFloat,
        'Initial trust-region radius Delta_0 (infinity norm). Ignored when '
        'scaling=0.')
_option('Trust region and sampling region', 'radius max', None, PositiveFloat,
        'Largest trust radius. Default 1000 * trust_radius; 1 when scaling=0.')
_option('Trust region and sampling region', 'delta min', 1e-6, PositiveFloat,
        'Smallest trust radius, and the lower limit of the sampling radius when '
        'it is reduced by sigma_dec or sample_radius_adjust. A feasible iterate '
        'at this radius on two consecutive iterations stops the run ("FEASIBLE '
        'SOLUTION FOUND"). A subproblem that fails at this radius stops the run '
        'if the iterate is feasible, and otherwise triggers a radius reset '
        '(n_radius_resets). Must be <= ep_delta.')
_option('Trust region and sampling region', 'sample radius', 0.1, PositiveFloat,
        'Initial sampling radius sigma_0: the size of the one-sided finite-'
        'difference steps that build the linear surrogate of each black box.')
_option('Trust region and sampling region', 'sample radius adjust', 0.5, PositiveFloat,
        'When the trust radius drops below the sampling radius, the sampling '
        'radius becomes max(sample_radius_adjust * Delta, delta_min); in (0, 1].')
_option('Trust region and sampling region', 'criticality check', 0.5, PositiveFloat,
        'Criticality factor xi: at each iteration the sampling radius is halved '
        'if it exceeds xi * chi, so the surrogate is refined as the iterate '
        'approaches stationarity.')
_option('Trust region and sampling region', 'gamma c', 0.5, PositiveFloat,
        'Trust-radius contraction factor (< 1): applied after a failed '
        'subproblem, a rejected step (except rejected R2 restoration steps, '
        'which use the ratio test), and a theta-step with ratio rho < eta1.')
_option('Trust region and sampling region', 'gamma e', 2.5, PositiveFloat,
        'Trust-radius expansion factor (> 1): applied after an f-step, after a '
        'theta-step with rho >= eta2, and on entering R2 restoration; the radius '
        'never exceeds radius_max.')
_option('Trust region and sampling region', 'eta1', 0.05, PositiveFloat,
        'Ratio test of theta-steps (and of rejected R2 restoration steps), with '
        'rho = 1 - (theta+ - ep_i) / max(theta_k, ep_i): contract when '
        'rho < eta1. Need 0 < eta1 < eta2 < 1.')
_option('Trust region and sampling region', 'eta2', 0.2, PositiveFloat,
        'Ratio test: expand when rho >= eta2; between eta1 and eta2 the radius '
        'is kept.')
_option('Trust region and sampling region', 'n radius resets', 3, NonNegativeInt,
        'Number of radius resets allowed when a subproblem fails at the smallest '
        'radius while the iterate is infeasible. The k-th reset sets the radius '
        'to min(radius_max, max(1e-2, 100 * delta_min) * 10^(k-1)); when they '
        'are used up the run stops ("... STALLED -- NOT FEASIBLE").')

# ---- Hypervolume sampling region (multiple_black_boxes=1) -------------------
_option('Hypervolume sampling region', 'sigma eta3', 0.5, PositiveFloat,
        'Hypervolume rule (thesis Eqs 11-14): after each trial step, rho_HV = '
        'actual / predicted improvement of the hypervolume of the per-black-box '
        'infeasibilities. rho_HV > sigma_eta3 expands the sampling radius, '
        '0 < rho_HV <= sigma_eta3 keeps it, rho_HV <= 0 shrinks it. In (0, 1).')
_option('Hypervolume sampling region', 'sigma inc', 1.5, PositiveFloat,
        'Sampling-radius expansion factor (> 1); the radius never exceeds the '
        'trust radius.')
_option('Hypervolume sampling region', 'sigma dec', 0.5, PositiveFloat,
        'Sampling-radius contraction factor (< 1), applied when rho_HV <= 0; '
        'floored at delta_min.')

# ---- Funnel (globalization_strategy=1) --------------------------------------
_option('Funnel', 'mu s', 0.01, PositiveFloat,
        'Switching condition: the step counts as objective-driven when '
        'f_k - f+ >= mu_s * theta_k^funnel_exponent.')
_option('Funnel', 'funnel exponent', 2.0, PositiveFloat,
        'Exponent of theta_k in the switching condition.')
_option('Funnel', 'eta', 0.01, PositiveFloat,
        'Sufficient decrease: a trial point inside the funnel that satisfies the '
        'switching condition is accepted (f-step) only if f_k - f+ >= eta * '
        'Delta_k, otherwise rejected.')
_option('Funnel', 'beta', 0.8, PositiveFloat,
        'A trial point inside the funnel that fails the switching condition is '
        'accepted (theta-step) only if theta+ <= beta * phi; in (0, 1).')
_option('Funnel', 'kappa f', 0.5, PositiveFloat,
        'Funnel update after a theta-step: phi = max(phi_min, '
        '(1 - kappa_f) theta+ + kappa_f phi); in (0, 1).')
_option('Funnel', 'kappa r', 1.1, PositiveFloat,
        'A trial point outside the funnel (theta+ > phi) that satisfies the '
        'switching condition is accepted as a theta-step if theta+ <= kappa_r * '
        'phi; otherwise rejected.')
_option('Funnel', 'phi min', 1e-8, PositiveFloat,
        'Smallest funnel width phi (the initial width is the initial '
        'infeasibility).')

# ---- Filter (globalization_strategy=0) --------------------------------------
_option('Filter', 'kappa theta', 0.1, PositiveFloat,
        'Switching condition: an acceptable trial point is an f-step when '
        'f_k - f+ >= kappa_theta * theta_k^gamma_s and theta_k < theta_min, '
        'otherwise a theta-step.')
_option('Filter', 'gamma s', 2.0, PositiveFloat,
        'Exponent of theta_k in the switching condition.')
_option('Filter', 'theta min', 1e-4, PositiveFloat,
        'f-steps are possible only from iterates with theta_k < theta_min.')
_option('Filter', 'theta max', 50.0, PositiveFloat,
        'Trial points with theta+ > theta_max are rejected (from iteration 1 on).')
_option('Filter', 'gamma f', 0.01, PositiveFloat,
        'Filter margin: a theta-step adds the pair (f+ - gamma_f theta+, '
        '(1 - gamma_theta) theta+), and trial points dominated by a pair are '
        'rejected; in (0, 1).')
_option('Filter', 'gamma theta', 0.01, PositiveFloat,
        'Filter margin on the infeasibility (see gamma_f); in (0, 1).')

# ---- Restoration ------------------------------------------------------------
_option('Restoration', 'n restoration max', 10, NonNegativeInt,
        'R2 only: after this many consecutive restoration iterations the run '
        'stops ("restoration stationary"). The cap keeps R2 convergent; raise it '
        'if theta is still decreasing when it is reached.')
_option('Restoration', 'frp proximity', 0.0, NonNegativeFloat,
        'Weight w of a term w * sum(((v - v_k)/(ub - lb))^2) over the bounded x '
        'and z variables, added to the restoration objective so that variables '
        'which do not affect the infeasibility stay at the current iterate. '
        '0 = the restoration problem (7.11) exactly.')

# ---- Subproblem solver ------------------------------------------------------
_option('Subproblem solver', 'solver', 'ipopt', str,
        'NLP solver for the subproblems (a Pyomo SolverFactory name). With '
        'ipopt, the trust-region subproblems use tuned settings (tol 1e-6, '
        'constr_viol_tol 1e-6, max_iter 20000, ...), and solver_options are '
        'applied on top.')
DOCS['solver_options'] = ('Options passed to the subproblem solver, on top of the '
                          'built-in IPOPT settings (e.g. {"linear_solver": "ma57"}).')
CONFIG.declare('solver_options', ConfigBlock(implicit=True,
                                             description=DOCS['solver_options']))
GROUPS['Subproblem solver'].append('solver_options')

# ---- Output and diagnostics -------------------------------------------------
_option('Output and diagnostics', 'verbose', 2, In([0, 1, 2]),
        '0 = silent; 1 = banner, exit status and run summary; 2 = also every '
        'iteration, restoration step and subproblem failure, and the final '
        'iteration table.')
_option('Output and diagnostics', 'print variables', False, bool,
        'With verbose=2, also print the variable vector at every iteration.')
_option('Output and diagnostics', 'bb cache', 1, _Binary,
        '1 = reuse the stored output when a black box is called again at '
        'exactly the same inputs. For deterministic black boxes this changes '
        'only the evaluation count; use 0 if repeated calls should give new '
        '(e.g. noisy) values.')
_option('Output and diagnostics', 'report duals', 0, _Binary,
        "1 = report the multipliers of the model's own constraints from the last "
        'successful subproblem solve (result["duals"]).')


class TrustRegionSolver:
    """The A-TRFu trust-region solver for grey-box optimisation.

    Parameters
    ----------
    preset : str, optional
        Start from a named configuration in ``atrfu.PRESETS`` (e.g. ``"filter"``);
        explicit options override it. Default: the A-TRFu defaults.
    **options
        Any option listed by :meth:`options`. Names may use spaces or
        underscores (``max_it`` and ``"max it"`` are the same option).

    Examples
    --------
    >>> solver = TrustRegionSolver(max_it=200)                       # doctest: +SKIP
    >>> solver = TrustRegionSolver(preset="filter", restoration_strategy=2)  # doctest: +SKIP
    >>> result = solver.solve(model, {"reactor": [m.bb1], "flash": [m.bb2]})  # doctest: +SKIP
    """

    CONFIG = CONFIG

    def __init__(self, preset=None, **options):
        if preset is not None:
            if preset not in PRESETS:
                raise ValueError(f"unknown preset {preset!r}; available: {', '.join(PRESETS)}")
            options = {**PRESETS[preset], **options}
        try:
            self.config = self.CONFIG(options, preserve_implicit=True)
        except ValueError as e:
            raise ValueError(_unknown_option_message(options) or str(e)) from None
        if self.config.radius_max is None:
            self.config.radius_max = 1000.0 * self.config.trust_radius
        check_options(self.config)

    @classmethod
    def options(cls, group=None):
        """Print every option (or one group) with its default and meaning."""
        for name, members in GROUPS.items():
            if group is not None and name != group:
                continue
            print(f"\n{name}\n{'-' * len(name)}")
            for opt in members:
                default = cls.CONFIG.get(opt).value()
                default = '{}' if opt == 'solver_options' else repr(default)
                print(f"  {opt.replace(' ', '_')} = {default}")
                print(textwrap.fill(DOCS[opt], 79, initial_indent=' ' * 6,
                                    subsequent_indent=' ' * 6))

    def solve(self, model, efmap):
        """Solve a grey-box problem.

        Parameters
        ----------
        model : pyomo.environ.ConcreteModel
            The glass-box model, with one active objective. Each black box is a
            Pyomo ``ExternalFunction`` wrapping a Python callable, used in the
            constraints (e.g. ``m.y == m.bb(m.x1, m.x2)``). The variables
            receive the final point.
        efmap : dict
            ``{block name: [ExternalFunction, ...]}``: the black boxes, each
            with the ExternalFunctions (outputs) it provides.

        Returns
        -------
        dict
            ``status`` (exit message), ``feasible`` (theta < ep_i),
            ``final_obj`` (in the model's sense), ``final_theta``,
            ``final_chi``, ``iterations``, ``bb_evals`` and ``bb_evals_detail``
            (true black-box evaluations, total and per block),
            ``bb_cache_hits``, ``f_steps``, ``theta_steps``, ``rejected``,
            ``restorations`` (rejected restoration iterations), ``frp_solves``,
            ``restoration_phases``, ``wall_s``, ``cpu_s``, problem sizes
            (``n_vars``, ``n_vars_xyz``, ``n_constraints``, ...), ``duals``
            (with report_duals=1) and ``trajectory`` (theta, obj, chi,
            trust_radius, sample_radius, step_norm at each iterate, from the
            starting point, iteration 0).
        """
        model_vars = list(model.component_data_objects(Var))
        # The clone deep-copies config too, so the run (which mutates config
        # as state) leaves self.config untouched and solve() is repeatable.
        model._atrfu_data = (model_vars, efmap, self.config)
        try:
            inst = model.clone()
        finally:
            del model._atrfu_data
        inst_vars, inst_efmap, inst_config = inst._atrfu_data
        result = TRF(inst, inst_efmap, inst_config)

        for inst_var, var in zip(inst_vars, model_vars, strict=True):
            var.set_value(value(inst_var))
        objective = next(model.component_data_objects(Objective, active=True))
        result['final_obj'] = value(objective)
        return result


def _unknown_option_message(options):
    known = {name.replace(' ', '_') for name in CONFIG}
    unknown = [k for k in options if k.replace(' ', '_') not in known]
    if not unknown:
        return None
    hints = []
    for k in unknown:
        close = difflib.get_close_matches(k.replace(' ', '_'), known, n=3)
        hints.append(f"{k!r}" + (f" (did you mean {', '.join(close)}?)" if close else ''))
    return ("unknown option(s): " + '; '.join(hints)
            + ". TrustRegionSolver.options() lists every option.")
