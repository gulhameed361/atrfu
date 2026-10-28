from pyomo.common.config import (
    ConfigBlock,
    ConfigValue,
    In,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
)
from pyomo.environ import Objective, Var, value

from ._algorithm import TRF

# =============================================================================
# TrustRegionSolver -- the user-facing solver wrapper + full configuration.
# Consolidated here (was duplicated across RunFile.py and the HDA_main_*.py
# case studies, which had drifted out of sync). Case-study / run scripts now
# do `from TRF import TrustRegionSolver` and contain only the model + run.
# =============================================================================

class TrustRegionSolver:
    CONFIG = ConfigBlock('Trust Region')

    CONFIG.declare('solver', ConfigValue(
        default='ipopt',
        description='Subproblem solver (e.g. ipopt). solver_options are merged '
                    'on top of the curated IPOPT settings.',
    ))

    CONFIG.declare('solver_options', ConfigBlock(
        implicit=True,
        description='Options to pass to the subproblem solver',
    ))

    CONFIG.declare('max it', ConfigValue(
        default=100,
        domain=PositiveInt,
        description='Maximum number of trust-region iterations.',
    ))

    # Initialize trust radius
    CONFIG.declare('trust radius', ConfigValue(
        default=1.0,
        domain=PositiveFloat,
        description='Initial trust-region radius Delta_0 (on the scaled '
                    'unit box when scaling=0).',
    ))

    # Initialize sample region and radius
    CONFIG.declare('sample region', ConfigValue(
        default=True,
        domain=bool,
        description='Tie the ROM sampling region to the criticality measure '
                    'chi during the criticality phase.',
    ))

    CONFIG.declare('sample radius', ConfigValue(
        default=0.1,
        domain=PositiveFloat,
        description='Initial ROM sampling radius sigma_0.',
    ))

    # Placeholder for 'radius max', value to be set in __init__
    CONFIG.declare('radius max', ConfigValue(
        default=None,
        domain=PositiveFloat,
        description='Upper bound on the trust-region radius; '
                    'defaults to 1000 * trust radius.',
    ))

    # Termination tolerances
    CONFIG.declare('ep i', ConfigValue(
        default=1e-5,
        domain=PositiveFloat,
        description='Feasibility tolerance: an iterate is reported FEASIBLE '
                    'only when thetak < ep i (strict; never upgraded).',
    ))

    CONFIG.declare('ep s', ConfigValue(
        default=1e-4,
        domain=PositiveFloat,
        description='Step-norm tolerance for the "possibly optimal" '
                    'termination test (||s_k|| < ep s).',
    ))

    CONFIG.declare('ep delta', ConfigValue(
        default=1e-3,
        domain=PositiveFloat,
        description='Trust-region radius tolerance for the optimal-'
                    'termination test.',
    ))

    CONFIG.declare('ep chi', ConfigValue(
        default=1e-3,
        domain=PositiveFloat,
        description='Criticality (stationarity) tolerance for the optimal-'
                    'termination test (chik < ep chi).',
    ))

    CONFIG.declare('delta min', ConfigValue(
        default=1e-6,
        domain=PositiveFloat,
        description='delta min <= ep delta',
    ))

    # Criticality Check Parameters
    CONFIG.declare('criticality check', ConfigValue(
        default=0.5,
        domain=PositiveFloat,
    ))

    # Trust region update parameters
    CONFIG.declare('gamma c', ConfigValue(
        default=0.5,
        domain=PositiveFloat,
        description='Trust-region CONTRACTION factor (<1), applied on '
                    'rejected or poor (rho<eta1) steps.',
    ))

    CONFIG.declare('gamma e', ConfigValue(
        default=2.5,
        domain=PositiveFloat,
        description='Trust-region EXPANSION factor (>1), applied on '
                    'successful f-type (or good rho>=eta2) steps.',
    ))

    # Switching Condition
    CONFIG.declare('gamma s', ConfigValue(
        default=2.0,
        domain=PositiveFloat,
        description='Filter switching exponent (strategy 0): an f-type step '
                    'needs f_k - f+ >= kappa_theta * theta_k^gamma_s.',
    ))

    CONFIG.declare('kappa theta', ConfigValue(
        default=0.1,
        domain=PositiveFloat,
        description='Filter switching coefficient (strategy 0).',
    ))

    CONFIG.declare('kappa f', ConfigValue(
        default=0.5,
        domain=PositiveFloat,
        description='funnel‑shrink factor after f‑type',
    ))

    CONFIG.declare('kappa r', ConfigValue(
        default=1.1,
        domain=PositiveFloat,
        description='funnel width factor of a theta-type step taken outside '
                    'the funnel (theta+ <= kappa_r * phi)',
    ))

    CONFIG.declare('theta min', ConfigValue(
        default=1e-4,
        domain=PositiveFloat,
    ))

    CONFIG.declare('phi min', ConfigValue(
        default=1e-8,
        domain=PositiveFloat,
        description='hard floor on funnel width',
    ))

    # Filter
    CONFIG.declare('gamma f', ConfigValue(
        default=0.01,
        domain=PositiveFloat,
        description='gamma_f and gamma_theta in (0,1) are fixed parameters',
    ))

    CONFIG.declare('gamma theta', ConfigValue(
        default=0.01,
        domain=PositiveFloat,
        description='gamma_f and gamma_theta in (0,1) are fixed parameters',
    ))

    CONFIG.declare('theta max', ConfigValue(
        default=50,
        domain=PositiveInt,
    ))


    CONFIG.declare('beta', ConfigValue(
        default=0.8,
        domain=PositiveFloat,
        description='extra shrink required for theta‑type',
    ))

    CONFIG.declare('mu s', ConfigValue(
        default=0.01,
        domain=PositiveFloat,
        description='funnel switching coefficient: an f-type step needs '
                    'f_k - f+ >= mu_s * theta_k^(funnel exponent)',
    ))

    CONFIG.declare('eta', ConfigValue(
        default=0.01,
        domain=PositiveFloat,
        description='funnel Armijo coefficient: an f-type step needs '
                    'f_k - f+ >= eta * Delta_k',
    ))

    CONFIG.declare('funnel exponent', ConfigValue(
        default=2.0,
        domain=PositiveFloat,
        description='funnel switching exponent',
    ))

    # Ratio test parameters (for theta steps)
    CONFIG.declare('eta1', ConfigValue(
        default=0.05,
        domain=PositiveFloat,
    ))

    CONFIG.declare('eta2', ConfigValue(
        default=0.2,
        domain=PositiveFloat,
    ))

    # Hypervolume-based sampling-region update (Eq 14)
    CONFIG.declare('sigma eta3', ConfigValue(
        default=0.5,
        domain=PositiveFloat,
        description='HV reduction-ratio expand threshold eta3 (Eq 14); in (0,1)',
    ))

    CONFIG.declare('sigma inc', ConfigValue(
        default=1.5,
        domain=PositiveFloat,
        description='sampling-region expansion factor gamma_sigma_inc (>1)',
    ))

    CONFIG.declare('sigma dec', ConfigValue(
        default=0.5,
        domain=PositiveFloat,
        description='sampling-region contraction factor gamma_sigma_dec (in (0,1))',
    ))

    CONFIG.declare('n radius resets', ConfigValue(
        default=3,
        domain=NonNegativeFloat,
        description='max radius resets allowed when restoration wedges at the '
                    'minimum radius while infeasible (escape hatch; tolerances '
                    'are never relaxed)',
    ))

    # --- Restoration phase and its Delta policy ---
    CONFIG.declare('restoration enabled', ConfigValue(
        default=1,
        domain=In([0, 1]),
        description='1 = enter the feasibility restoration phase (7.11) after '
                    '>=2 consecutive step rejections (the default). 0 = never '
                    'enter restoration and keep solving the plain TRSP '
                    '(Table 7.2, no-restoration column).',
    ))

    #
    # Two strategies, selected by 'restoration strategy':
    #
    #   R1 (default) -- classic. The FRP (compatibilityCheck) is solved at the
    #       current Delta, and a rejected restoration iteration contracts Delta
    #       unconditionally by gamma_c like any other rejection.
    #   R2 -- relaxed. On ENTERING a phase Delta is reset to
    #       min(gamma_e * accepted_radius, radius_max) before the first FRP
    #       solve (and, under the filter, the abandoned iterate is pushed into
    #       the filter); a rejected restoration iteration then updates Delta by
    #       the theta-reduction ratio rhok, with the n_restoration_max cap
    #       active to keep the phase finite.
    #
    # Both solve the SAME subproblem -- compatibilityCheck, the TR-boxed FRP.
    # They differ only in the Delta policy. R1 is the default (Table 7.3).
    CONFIG.declare('restoration strategy', ConfigValue(
        default=1,
        domain=In([1, 2]),
        description='Feasibility-restoration policy. '
                    '1 = R1 classic (FRP at the current Delta; a rejected '
                    'restoration iteration contracts unconditionally), '
                    '2 = R2 relaxed (entry reset to gamma_e*accepted_radius '
                    'plus filter insertion; a rejected restoration iteration '
                    'updates Delta by rhok; n_restoration_max cap active).',
    ))

    # Opt-in regularisation of the feasibility restoration problem (7.11).
    # (7.11) minimises only the surrogate residual, so any variable that does
    # not affect it (e.g. an over-bounded cost-side duty) is left free and the
    # NLP solver may place it anywhere within the trust region. A positive
    # weight adds  w * sum(((v - v_k)/(ub - lb))^2)  over the x and z variables
    # with finite bounds, which pins such variables at the centre and barely
    # touches the residual-reducing ones. Default 0 = the thesis's (7.11)
    # exactly (Chapter 7 results). Used only for the DFM-PSA case study.
    CONFIG.declare('frp proximity', ConfigValue(
        default=0.0,
        domain=float,
        description='Weight of the range-normalised proximity term added to '
                    'the restoration objective (0 = off, the default)',
    ))

    CONFIG.declare('n restoration max', ConfigValue(
        default=10,
        domain=NonNegativeFloat,
        description='max CONSECUTIVE feasibility-restoration (FRP) iterations '
                    'before the run exits as RESTORATION STATIONARY. Only takes '
                    'effect under R2 (restoration strategy 2), because it '
                    'is LOAD-BEARING for that mode rather than a diagnostic cap: '
                    'the relaxed rule breaks the "infinitely many rejections => '
                    'Delta -> 0" leg of the proof, which survives only because '
                    'hitting this cap TERMINATES the run, making an infinite tail '
                    'of rejections without an intervening acceptance impossible. '
                    'Raise it if theta is still improving when the cap is hit',
    ))

    # Output level (replace with real print levels)
    CONFIG.declare('print variables', ConfigValue(
        default=False,
        domain=bool,
    ))

    # Built-in black-box evaluation cache (memoization of true BB callback
    # calls, exact-key -> transparent to results; reports hits/misses).
    CONFIG.declare('bb cache', ConfigValue(
        default=1,
        domain=In([0, 1]),
        description='0 = off, 1 = memoise true black-box evaluations in the '
                    'solver (exact input key; never changes results, only the '
                    'evaluation count). Reported as cache hits/misses in the '
                    'run summary. Problems with their own block-level cache '
                    '(e.g. HDA) keep it -- the two layers are complementary.',
    ))

    # Report constraint dual values (multipliers) from the final successful
    # subproblem solve in the run summary. Off by default (attaches an IMPORT
    # dual Suffix). The (equality | inequality) constraint COUNTS are always
    # reported in the banner / run summary regardless of this flag.
    CONFIG.declare('report duals', ConfigValue(
        default=0,
        domain=In([0, 1]),
        description='0 = off, 1 = capture and report constraint dual values '
                    '(Lagrange multipliers) from the final subproblem solve in '
                    'the run summary (and the solve() result dict under "duals").',
    ))


    # Sample Radius reset parameter
    CONFIG.declare('sample radius adjust', ConfigValue(
        default=0.5,
        domain=PositiveFloat,
    ))

    # Default globalization strategy
    CONFIG.declare('globalization strategy', ConfigValue(
        default=0,
        domain=In([0, 1]),
        description='0 = Filter, 1 = Funnel',
    ))

    # Default problem scaling
    CONFIG.declare('scaling', ConfigValue(
        default=0,
        domain=In([0, 1]),
        description='0 = Yes, 1 = No',
    ))

    # Default number of black-boxes setting
    CONFIG.declare('multiple black boxes', ConfigValue(
        default=0,
        domain=In([0, 1]),
        description='0 = one aggregate infeasibility, 1 = per-black-box '
                    'infeasibility vector with hypervolume sampling-region '
                    'management (Eq 14)',
    ))

    def __init__(self, **kwargs):
        self.config = self.CONFIG(kwargs, preserve_implicit=True)

        # Set 'radius max' if not already provided
        if self.config['radius max'] is None:
            self.config['radius max'] = 1000.0 * self.config['trust radius']

    def solve(self, model, eflist, **kwds):
        # Store all data needed to change in the original model
        model._tmp_trf_data = (list(model.component_data_objects(Var)), eflist, self.config)

        # Clone the model to work on it. NOTE: clone() deep-copies the attached
        # _tmp_trf_data tuple too, so inst gets its OWN copy of self.config. The
        # main loop mutates config (trust_radius, sample_radius, ...) as run
        # state, but only on that copy -- self.config is left untouched, so
        # calling solve() again on this solver starts fresh (verified). Do not
        # "optimise" this by passing self.config directly without a copy.
        inst = model.clone()

        # Call the TRF function on the cloned model
        result = TRF(inst, inst._tmp_trf_data[1], inst._tmp_trf_data[2])

        # Copy potentially changed variable values back to the original model
        for inst_var, orig_var in zip(inst._tmp_trf_data[0], model._tmp_trf_data[0], strict=True):
            orig_var.set_value(value(inst_var))

        # Add the objective in the user's own sense, read off the original
        # model's active objective after copy-back.
        objective = next(model.component_data_objects(Objective, active=True))
        result['final_obj'] = value(objective)
        return result
