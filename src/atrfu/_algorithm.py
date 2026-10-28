"""The A-TRFu main loop (thesis Algorithm 7.1)."""
import time

import numpy as np

from ._filter import Filter
from ._funnel import Funnel
from ._gjh import ensure_gjh
from ._helper import cloneXYZ, packXYZ, system_report
from ._hypervolume import calculate_hypervolume, update_pareto_front
from ._interface import PyomoInterface
from ._logger import Logger

STEP_FLAG = {'f': 'fStep', 'theta': 'thetaStep', 'theta-relax': 'relaxthetaStep'}


# =============================================================================
# Sampling-region update (Eq 14): hypervolume-driven 3-case rule
# =============================================================================

def update_sampling_region(config, reduction_ratio_HV, theta_values, HV, HVk,
                           k_population):
    """Update sigma_{k+1} from the HV reduction ratio (Eq 14).

    rho_HV > eta3   -> expand  (sigma *= sigma_inc, capped at the trust radius)
    0 < rho <= eta3 -> maintain
    rho_HV <= 0     -> shrink  (sigma *= sigma_dec, floored at delta_min)

    The Pareto front and HVk are updated only when rho_HV > 0 (Eq 14 text).
    """
    if reduction_ratio_HV > 0:
        if reduction_ratio_HV > config.sigma_eta3:
            config.sample_radius = min(config.sigma_inc * config.sample_radius,
                                       config.trust_radius)
        else:
            config.sample_radius = min(config.sample_radius,
                                       config.trust_radius)
        k_population = update_pareto_front(k_population, np.array(theta_values))
        HVk = HV
    else:
        config.sample_radius = min(
            max(config.sigma_dec * config.sample_radius, config.delta_min),
            config.trust_radius)
    return k_population, HVk


def hv_reduction_ratio(k_population, ref_point, error_vec_plus, HVk):
    """Return (rho_HV, HV_actual) (Eqs 11-13).

    actual    = HV(update(P_k, phi_{k+1})) - HV_k
    predicted = HV(update(P_k, 0)) - HV_k
    rho_HV    = actual / predicted, 0 if predicted <= 0
    """
    HV_act = calculate_hypervolume(
        update_pareto_front(k_population, np.array(error_vec_plus)), ref_point)
    HV_pred = calculate_hypervolume(
        update_pareto_front(k_population, np.zeros_like(
            np.asarray(error_vec_plus, dtype=float))), ref_point)
    predicted = HV_pred - HVk
    rho = 0.0 if predicted <= 0 else (HV_act - HVk) / predicted
    return rho, HV_act


# =============================================================================
# Trust-region radius updates
# =============================================================================

def contract(config):
    config.trust_radius = max(config.gamma_c * config.trust_radius,
                              config.delta_min)


def expand(config):
    config.trust_radius = min(config.gamma_e * config.trust_radius,
                              config.radius_max)


def radius_by_ratio(config, rhok):
    """Theta-reduction ratio rule: contract if rho < eta1, expand if
    rho >= eta2, otherwise keep Delta."""
    if rhok < config.eta1:
        contract(config)
    elif rhok >= config.eta2:
        expand(config)


def radius_on_reject(config, restoration_mode, rhok):
    """Delta update for a rejected trial step.

    R1: every rejected step contracts unconditionally, restoration or not.
    That is the convergence anchor ("infinitely many rejections => Delta -> 0").

    R2: a rejected restoration iteration follows the theta-reduction ratio
    instead, so persistent rejections cannot strangle Delta at an infeasible
    point; n_restoration_max keeps the phase finite (see TRF()).
    """
    if restoration_mode and config.restoration_strategy == 2:
        radius_by_ratio(config, rhok)
    else:
        contract(config)


# =============================================================================
# Main loop
# =============================================================================

def make_globalisation(config, theta0):
    if config.globalization_strategy == 0:
        return Filter(kappa_theta=config.kappa_theta, gamma_s=config.gamma_s,
                      theta_min=config.theta_min, theta_max=config.theta_max,
                      gamma_f=config.gamma_f, gamma_theta=config.gamma_theta)
    return Funnel(phi_init=theta0, phi_min=config.phi_min,
                  kappa_f=config.kappa_f, beta=config.beta,
                  switch_coeff=config.mu_s,
                  switch_exponent=config.funnel_exponent, eta=config.eta,
                  kappa_r=config.kappa_r)


def TRF(m, eflist, config):
    """Run A-TRFu on the (cloned) model ``m``; ``config`` is mutated as run
    state. Returns the result dict documented in TrustRegionSolver.solve."""
    _t_wall0 = time.perf_counter()
    _t_cpu0 = time.process_time()

    ensure_gjh(verbose=True)

    logger = Logger()
    problem = PyomoInterface(m, eflist, config)
    x, y, z = problem.getInitialValue()
    lx, ly, lz = problem.lx, problem.ly, problem.lz
    neq, nineq = problem.count_constraints()

    # ---- startup banner -----------------------------------------------------
    # x = black-box inputs, y = black-box outputs, z = remaining variables.
    strat_name = {0: 'Filter', 1: 'Funnel'}[config.globalization_strategy]
    frp_name = {1: 'R1 classic (contract on reject)',
                2: 'R2 relaxed (entry reset + rho radius)'}[
                    config.restoration_strategy]
    print("========== A-TRFu SOLVER ==========")
    print(f"Globalization strategy: {config.globalization_strategy} "
          f"({strat_name})")
    print(f"Hypervolume sampling  : multiple_black_boxes="
          f"{config.multiple_black_boxes}")
    print(f"Restoration strategy  : {config.restoration_strategy} "
          f"({frp_name})")
    print(f"Problem size          : {lx + ly + lz} variables  "
          f"(x|y|z = {lx}|{ly}|{lz});  "
          f"{len(problem.TRF.y)} black-box block(s)")
    print("    x = black-box inputs | y = black-box outputs | z = remaining")
    print(f"Constraints           : {neq + nineq}  "
          f"(equality {neq} | inequality {nineq})")
    print(f"Tolerances            : ep_i={config.ep_i:.1e}, "
          f"ep_chi={config.ep_chi:.1e}, ep_delta={config.ep_delta:.1e}, "
          f"max_it={config.max_it}")
    print("===================================\n")

    if config.scaling == 0:
        config.trust_radius = 1
        config.radius_max = 1

    romParam, yr = problem.buildROM(x, config.sample_radius, config.scaling)
    g, J, varlist, conlist = problem.grad_hess_calc(x, y, z, romParam,
                                                    config.scaling)
    rebuildROM = False
    xk, yk, zk = cloneXYZ(x, y, z)
    chik = 1e8
    thetak = np.linalg.norm(yr - yk, 1)
    objk = problem.evaluateObj(x, y, z)

    if config.multiple_black_boxes == 1:
        # yr holds the black-box outputs at xk (buildROM evaluated them).
        thetak_values = problem.block_thetas(yr, yk)
        k_population = [thetak_values]
        ref_point = thetak_values * 2 + 10
        HVk = calculate_hypervolume(k_population, ref_point)

    globalisation = make_globalisation(config, thetak)

    iteration = -1
    stepNorm = 1e10
    consecutive_rejections = 0
    radius_resets = 0
    # Consecutive restoration (FRP) iterations and the theta they started from;
    # capped by n_restoration_max under R2, which terminates the solve.
    restoration_iters = 0
    restoration_theta0 = None
    # Total FRP solves and restoration phases entered. 'restorations' in the
    # result counts only REJECTED restoration iterations.
    frp_solves = 0
    restoration_phases = 0
    subopt_flag = False
    exit_msg = "UNKNOWN"
    prev_iter_start = time.time()
    # Radius of the most recent accepted step (or wedge reset): R2 restoration
    # resets Delta from it rather than from the contracted value.
    accepted_radius = config.trust_radius

    while True:
        if iteration >= 0:
            logger.printIteration(iteration)
        now = time.time()
        IT = now - prev_iter_start
        prev_iter_start = now

        iteration += 1
        if iteration > config.max_it:
            exit_msg = "Maximum iterations"
            print("EXIT: Maximum iterations\n")
            break

        if iteration == 1:
            config.sample_region = False

        # Keep the sampling region within the trust region
        if config.trust_radius < config.sample_radius:
            config.sample_radius = max(
                config.sample_radius_adjust * config.trust_radius,
                config.delta_min)
            rebuildROM = True

        # Surrogate r_k, kappa-fully linear on the sampling region sigma_k
        if rebuildROM:
            romParam, yr = problem.buildROM(x, config.sample_radius,
                                            config.scaling)
            g, J, varlist, conlist = problem.grad_hess_calc(
                x, y, z, romParam, config.scaling)

        # Criticality measure chi, always on the UNIT box (thesis (3.21)): a
        # Delta-sized box would make chi proportional to Delta and the
        # termination test pass trivially as Delta -> 0.
        if iteration > 0:
            flag, chik = problem.criticalityCheck(
                x, y, z, romParam, g, J, varlist, conlist,
                criticality_radius=1)
            if not flag:
                raise RuntimeError("Criticality check failed")

        # An iterate is FEASIBLE only when thetak < ep_i (strict).
        feas_ok = thetak < config.ep_i

        logger.newIter(iteration, xk, yk, zk, thetak, objk, chik, IT,
                       config.print_variables)

        # Termination tests (accuracy radius: the sampling radius sigma)
        if (feas_ok and chik < config.ep_chi
                and config.sample_radius < config.ep_delta):
            exit_msg = "OPTIMAL SOLUTION FOUND"
            print(f"EXIT: OPTIMAL SOLUTION FOUND "
                  f"(thetak={thetak:.3e}, chik={chik:.3e})")
            break

        if feas_ok and stepNorm < config.ep_s:
            exit_msg = "POSSIBLY AN OPTIMAL SOLUTION IS FOUND"
            print(f"EXIT: POSSIBLY AN OPTIMAL SOLUTION IS FOUND "
                  f"(thetak={thetak:.3e}, chik={chik:.3e})")
            break

        if config.trust_radius <= config.delta_min and feas_ok:
            if subopt_flag:
                exit_msg = "FEASIBLE SOLUTION FOUND"
                print(f"EXIT: FEASIBLE SOLUTION FOUND (thetak={thetak:.3e})")
                break
            subopt_flag = True
        else:
            subopt_flag = False

        # Criticality phase (sigma-based)
        if not config.sample_region:
            if config.sample_radius > chik * config.criticality_check:
                config.sample_radius = config.sample_radius / 2
        else:
            config.sample_radius = max(
                min(config.sample_radius, chik * config.criticality_check),
                config.delta_min)

        logger.setCurIter(trustRadius=config.trust_radius,
                          sampleRadius=config.sample_radius)

        # Restoration (the FRP, (7.11)) is entered after two consecutive
        # rejections, unless 'restoration enabled' = 0.
        restoration_mode = (consecutive_rejections >= 2
                            and config.restoration_enabled == 1)

        if not restoration_mode:
            # Any acceptance (or wedge reset) zeroes consecutive_rejections,
            # so this covers every exit from a restoration run.
            restoration_iters = 0
            restoration_theta0 = None
        else:
            if restoration_theta0 is None:
                restoration_theta0 = thetak
            # Under R2 the cap is load-bearing for convergence: terminating
            # here makes an infinite tail of rejections impossible.
            if (config.restoration_strategy == 2
                    and restoration_iters >= config.n_restoration_max):
                problem.setVarValue(x=x, y=y, z=z)
                run = (f"{restoration_iters} consecutive restoration "
                       f"iterations")
                trend = (f"theta over the restoration run: "
                         f"{restoration_theta0:.3e} -> {thetak:.3e}")
                if feas_ok:   # stalled, but already feasible
                    exit_msg = "FEASIBLE SOLUTION FOUND (restoration stationary)"
                    print(f"EXIT: FEASIBLE SOLUTION FOUND (restoration "
                          f"stationary after {run}, thetak={thetak:.3e} < "
                          f"ep_i={config.ep_i:.1e}; {trend})")
                    break
                exit_msg = "RESTORATION STATIONARY -- NOT FEASIBLE"
                print(f"EXIT: RESTORATION STATIONARY -- NOT FEASIBLE ({run}, "
                      f"thetak={thetak:.3e} >= ep_i={config.ep_i:.1e}; "
                      f"{trend})")
                break

            if restoration_iters == 0:
                restoration_phases += 1
                if config.restoration_strategy == 2:
                    # R2 entry, before the first FRP solve: reset Delta from
                    # the last accepted radius and expand once, and tell the
                    # globalisation the current iterate is abandoned (the
                    # filter records it; the funnel is left unchanged).
                    delta_before = config.trust_radius
                    config.trust_radius = min(
                        config.gamma_e * accepted_radius, config.radius_max)
                    globalisation.abandon(thetak, objk)
                    clamped = (', CLAMPED at radius_max'
                               if config.trust_radius >= config.radius_max
                               else '')
                    print(f"[FRP] entering restoration phase "
                          f"{restoration_phases}: Delta {delta_before:.3e} -> "
                          f"{config.trust_radius:.3e} (gamma_e * "
                          f"accepted_radius={accepted_radius:.3e}{clamped}), "
                          f"thetak={thetak:.3e}")

        # Solve TRSP_k, or the restoration problem (7.11)
        if not restoration_mode:
            flag, obj, _ = problem.TRSPk(x, y, z, xk, yk, zk, romParam,
                                         config.trust_radius, config.scaling)
        else:
            flag, obj, _ = problem.compatibilityCheck(
                x, y, z, xk, yk, zk, romParam, config.trust_radius,
                config.scaling)
            obj = problem.evaluateObj(x, y, z)
            restoration_iters += 1
            frp_solves += 1
            at_max = (' (at radius_max)'
                      if config.trust_radius >= config.radius_max else '')
            print(f"[FRP] solve {restoration_iters} of phase "
                  f"{restoration_phases}, thetak={thetak:.3e}, "
                  f"Delta={config.trust_radius:.3e}{at_max}")

        # A failed subproblem solve is a rejected step. At the radius floor,
        # either exit or escape an infeasible wedge with an escalating reset.
        if not flag:
            which = "restoration" if restoration_mode else "TRSP"
            print(f"WARNING: {which} solve failed; treating as a rejected step")
            x, y, z = cloneXYZ(xk, yk, zk)
            if config.trust_radius <= config.delta_min:
                if feas_ok:
                    # The model may hold the failed solve's point; the final
                    # copy-back reads the model.
                    problem.setVarValue(x=x, y=y, z=z)
                    exit_msg = (f"FEASIBLE SOLUTION FOUND "
                                f"({which} stalled at minimum radius)")
                    print(f"EXIT: FEASIBLE SOLUTION FOUND ({which} stalled "
                          f"at minimum radius, thetak={thetak:.3e})")
                    break
                if radius_resets < config.n_radius_resets:
                    # Escalating reset (1e-2, 1e-1, 1, ... capped at
                    # radius_max); tolerances are not relaxed.
                    radius_resets += 1
                    config.trust_radius = min(
                        config.radius_max,
                        max(1e-2, 100.0 * config.delta_min)
                        * (10.0 ** (radius_resets - 1)))
                    consecutive_rejections = 0
                    rebuildROM = True
                    accepted_radius = config.trust_radius
                    print(f"WARNING: infeasible wedge at minimum radius "
                          f"(thetak={thetak:.3e}); radius reset "
                          f"{radius_resets}/{config.n_radius_resets} "
                          f"to {config.trust_radius:.1e}")
                    continue
                problem.setVarValue(x=x, y=y, z=z)
                exit_msg = f"{which.upper()} STALLED -- NOT FEASIBLE"
                print(f"EXIT: {exit_msg} (thetak={thetak:.3e} >= "
                      f"ep_i={config.ep_i:.1e}, radius resets exhausted)")
                break
            contract(config)
            rebuildROM = True
            logger.iterlog.rejected = True
            consecutive_rejections += 1
            continue

        # True black boxes at the trial point
        yr = problem.evaluateDx(x)
        stepNorm = np.linalg.norm(packXYZ(x - xk, y - yk, z - zk), np.inf)
        logger.setCurIter(stepNorm=stepNorm)
        theta = np.linalg.norm(yr - y, 1)

        # Hypervolume-driven sampling-region (sigma) management
        if config.multiple_black_boxes == 1:
            theta_values = problem.block_thetas(yr, y)
            reduction_ratio_HV, HV = hv_reduction_ratio(
                k_population, ref_point, theta_values, HVk)
            k_population, HVk = update_sampling_region(
                config, reduction_ratio_HV, theta_values, HV, HVk,
                k_population)

        # Theta-reduction ratio for the theta-step radius update
        rhok = 1 - ((theta - config.ep_i) / max(thetak, config.ep_i))

        if restoration_mode and config.restoration_strategy == 2:
            action = ('contract' if rhok < config.eta1 else
                      'expand' if rhok >= config.eta2 else 'maintain')
            print(f"[FRP] solve {restoration_iters} outcome: "
                  f"theta+={theta:.3e} (thetak={thetak:.3e}), "
                  f"rhok={rhok:.3e} -> {action}")

        status = globalisation.classify(thetak, theta, objk, obj,
                                        config.trust_radius,
                                        first_iteration=iteration == 0)
        if status == 'reject':
            radius_on_reject(config, restoration_mode, rhok)
            rebuildROM = True
            x, y, z = cloneXYZ(xk, yk, zk)
            if restoration_mode:
                logger.iterlog.restoration = True
            else:
                logger.iterlog.rejected = True
            consecutive_rejections += 1
            continue

        globalisation.accept(status, theta, obj)
        setattr(logger.iterlog, STEP_FLAG[status], True)
        consecutive_rejections = 0
        if status == 'f':
            expand(config)
        else:
            radius_by_ratio(config, rhok)
        accepted_radius = config.trust_radius

        rebuildROM = True
        xk, yk, zk = cloneXYZ(x, y, z)
        # The true outputs at the new centre were just evaluated.
        yk = yr.copy()
        thetak = theta
        objk = obj

    logger.printVectors()

    # ---- run summary --------------------------------------------------------
    wall_s = time.perf_counter() - _t_wall0
    cpu_s = time.process_time() - _t_cpu0
    y_per_block = ", ".join(f"{blk}={len(problem.TRF.y[blk])}"
                            for blk in sorted(problem.TRF.y))
    print("\n========== RUN SUMMARY ==========")
    print(f"Iterations            : {iteration}")
    print(f"Problem size          : {lx + ly + lz} variables  "
          f"(x|y|z = {lx}|{ly}|{lz})")
    print("    x = black-box inputs | y = black-box outputs | z = remaining")
    print(f"Black-box blocks      : {len(problem.TRF.y)}  "
          f"(outputs per block: {y_per_block})")
    print(f"Constraints           : {neq + nineq}  "
          f"(equality {neq} | inequality {nineq})")
    print(f"Wall time (s)         : {wall_s:.3f}")
    print(f"CPU time (s)          : {cpu_s:.3f}   (process_time)")
    print("External BB evaluations (true black-box callback calls):")
    for blk in sorted(problem.bb_eval_counts):
        print(f"    {blk:<12}: {problem.bb_eval_counts[blk]}")
    print(f"    {'TOTAL':<12}: {problem.bb_eval_total}")
    if problem.bb_cache_enabled:
        hits, misses = problem.bb_cache_hits, problem.bb_eval_total
        requests = hits + misses
        rate = (100.0 * hits / requests) if requests else 0.0
        print(f"BB cache              : {hits} hits / {requests} requests "
              f"({rate:.1f}% hit rate); {misses} true evaluations")
    else:
        print("BB cache              : disabled")

    duals = dict(problem.last_duals)
    if problem.report_duals:
        print(f"Dual values           : {len(duals)} captured "
              f"(last subproblem solve); largest |dual|:")
        for name, val in sorted(duals.items(), key=lambda t: -abs(t[1]))[:15]:
            print(f"    {name:<32}: {val:.6g}")

    print("System:")
    print(system_report())
    print("=================================\n")

    def series(attr):
        return [float('nan') if getattr(it, attr) is None
                else float(getattr(it, attr)) for it in logger.iters]

    def count(flag):
        return sum(1 for it in logger.iters if getattr(it, flag))

    return {
        'status':          exit_msg,
        'iterations':      iteration,
        'feasible':        bool(thetak < config.ep_i),
        'final_theta':     float(thetak),
        'final_chi':       float(chik),
        'bb_evals':        int(problem.bb_eval_total),
        'bb_evals_detail': {blk: int(problem.bb_eval_counts[blk])
                            for blk in sorted(problem.bb_eval_counts)},
        'bb_cache_hits':   int(problem.bb_cache_hits),
        'bb_cache_enabled': bool(problem.bb_cache_enabled),
        'f_steps':         count('fStep'),
        'theta_steps':     count('thetaStep'),
        'rejected':        count('rejected'),
        'restorations':    count('restoration'),
        'frp_solves':      int(frp_solves),
        'restoration_phases': int(restoration_phases),
        'restoration_strategy': int(config.restoration_strategy),
        'wall_s':          float(wall_s),
        'cpu_s':           float(cpu_s),
        'n_vars':          int(lx + ly + lz),
        'n_vars_xyz':      (int(lx), int(ly), int(lz)),
        'n_eq_constraints':   int(neq),
        'n_ineq_constraints': int(nineq),
        'n_constraints':      int(neq + nineq),
        'duals':           duals,
        # Per-iteration trajectory; obj in the user's sense.
        'trajectory': {
            'iter':          [int(it.iteration) for it in logger.iters],
            'theta':         series('thetak'),
            'obj':           [problem.sense * float(it.objk)
                              for it in logger.iters],
            'chi':           series('chik'),
            'trust_radius':  series('trustRadius'),
            'sample_radius': series('sampleRadius'),
            'step_norm':     series('stepNorm'),
        },
    }
