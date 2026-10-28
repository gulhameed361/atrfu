"""Developer sanity check (not packaged): solve the A-TRFu benchmark problems
with this package and compare against the A-TRFu thesis reference results.

The benchmark problems and reference results live in the A-TRFu repository
(https://github.com/gulhameed361/A-TRFu), expected next to this one; pass
--atrfu-repo otherwise. Each run executes in its own process and leaves
dev/out/<tag>/<config>/<problem>.json, so two tags (e.g. before and after a
refactor) can be compared exactly with --diff. --orig runs the original
A-TRFu solver code instead of this package (an exact baseline on the
current platform).

    python dev/crosscheck.py --tag orig --orig
    python dev/crosscheck.py --tag new --diff orig
    python dev/crosscheck.py --tag new --configs filter full_R2 --problems powell
"""
import argparse
import contextlib
import csv
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')
DEFAULT_REPO = os.path.join(HERE, '..', '..', 'A-TRFu')
# The Chapter 7 configurations (A-TRFu experiments/run_chapter7.py); 200 iterations.
CONFIGS = {
    'full':           dict(globalization_strategy=1, multiple_black_boxes=1,
                           restoration_strategy=1),
    'full_R2':        dict(globalization_strategy=1, multiple_black_boxes=1,
                           restoration_strategy=2),
    'no_hv':          dict(globalization_strategy=1, multiple_black_boxes=0,
                           restoration_strategy=1),
    'no_restoration': dict(globalization_strategy=1, multiple_black_boxes=1,
                           restoration_strategy=1, restoration_enabled=0),
    'filter':         dict(globalization_strategy=0, multiple_black_boxes=1,
                           restoration_strategy=1),
}
KEYS = ('status', 'feasible', 'iterations', 'bb_evals', 'f_steps',
        'theta_steps', 'rejected', 'restorations', 'frp_solves',
        'final_obj', 'final_theta')


def run_one(repo, tag, config, problem, orig):
    sys.path.insert(0, os.path.join(repo, 'problems'))
    if orig:
        sys.path.insert(0, os.path.join(repo, 'atrfu'))
        from TRF import TrustRegionSolver
    else:
        from atrfu import TrustRegionSolver
    from benchmark_problems import PROBLEMS

    out_dir = os.path.join(OUT, tag, config)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, problem)
    with open(stem + '.log', 'w', encoding='utf-8') as log, \
            contextlib.redirect_stdout(log):
        model, efmap, kwargs = PROBLEMS[problem][0]()
        options = {**CONFIGS[config], 'max_it': 200, **kwargs}
        result = TrustRegionSolver(**options).solve(model, efmap)
    with open(stem + '.json', 'w', encoding='utf-8') as f:
        json.dump({k: result[k] for k in KEYS}, f, indent=1)


def load(tag, config, problem):
    path = os.path.join(OUT, tag, config, problem + '.json')
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    return None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--diff', metavar='TAG', help='compare exactly with an earlier tag')
    ap.add_argument('--orig', action='store_true',
                    help="run A-TRFu's own solver code instead of this package")
    ap.add_argument('--atrfu-repo', default=DEFAULT_REPO)
    ap.add_argument('--configs', nargs='+', default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument('--problems', nargs='+')
    ap.add_argument('--one', nargs=2, help=argparse.SUPPRESS)
    args = ap.parse_args()
    repo = os.path.abspath(args.atrfu_repo)

    if args.one:
        run_one(repo, args.tag, *args.one, args.orig)
        return

    with open(os.path.join(repo, 'results', 'reference', 'runs.csv'),
              encoding='utf-8') as f:
        ref = {(r['config'], r['problem']): r for r in csv.DictReader(f)
               if r['problem'] != 'HDA'}

    n_bad = 0
    print(f"{'config':15s} {'problem':20s} {'it':>4s} {'ref':>4s} "
          f"{'final_obj':>14s} {'rel diff':>9s}  vs reference"
          + (f'  vs {args.diff}' if args.diff else ''))
    for config in args.configs:
        problems = args.problems or [p for c, p in ref if c == config]
        for p in problems:
            if load(args.tag, config, p) is None:
                proc = subprocess.run(
                    [sys.executable, __file__, '--tag', args.tag,
                     '--atrfu-repo', repo, '--one', config, p]
                    + (['--orig'] if args.orig else []),
                    capture_output=True, text=True)
                if proc.returncode:
                    n_bad += 1
                    print(f'{config:15s} {p:20s} FAILED: '
                          f'{proc.stderr.strip().splitlines()[-1:]}')
                    continue
            r, rr = load(args.tag, config, p), ref[config, p]
            f_ref = float(rr['final_obj'])
            rel = abs(r['final_obj'] - f_ref) / (1 + abs(f_ref))
            same_path = all(str(r[k] if k != 'feasible' else int(r[k])) == rr[k]
                            for k in ('status', 'feasible', 'iterations', 'bb_evals'))
            ok = same_path and rel < 1e-6
            verdict = 'same path' if ok else 'CHECK'
            if args.diff:
                same = r == load(args.diff, config, p)
                verdict += '  identical' if same else '  DIFFERENT'
                ok = ok and same
            n_bad += not ok
            print(f"{config:15s} {p:20s} {r['iterations']:4d} "
                  f"{int(rr['iterations']):4d} {r['final_obj']:14.7g} "
                  f"{rel:9.1e}  {verdict}")
    print(f'\n{n_bad} run(s) need attention')


if __name__ == '__main__':
    main()
