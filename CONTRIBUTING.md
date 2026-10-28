# Contributing

Bug reports and pull requests are welcome at
<https://github.com/gulhameed361/atrfu>.

## Development setup

```bash
conda create -n atrfu-dev -c conda-forge python=3.12 ipopt
conda activate atrfu-dev
pip install -e ".[dev,docs]"
atrfu install-gjh
```

## Checks

```bash
ruff check .
pytest -m "not slow"     # quick, about a minute
pytest                   # everything, as in CI
sphinx-build -W docs docs/_build
```

CI runs these on Linux, Windows and macOS with Python 3.10 to 3.13.

## Code map

| module | contents |
|---|---|
| `solver.py` | `TrustRegionSolver` and the option table (`CONFIG`, `GROUPS`, `DOCS`), the single source of the options page |
| `presets.py` | named configurations |
| `_validate.py` | option and `efmap` checks |
| `_algorithm.py` | the main loop `TRF` and the radius rules |
| `_funnel.py`, `_filter.py` | step acceptance |
| `_hypervolume.py` | Pareto front and hypervolume |
| `_interface.py` | Pyomo model transformation, surrogates and subproblems |
| `_gjh.py`, `_readgjh.py` | finding gjh and reading its derivatives |
| `_logger.py`, `_helper.py` | iteration log and small utilities |
| `cli.py` | the `atrfu` command |

## Rules

- A change to the numerical behaviour needs a reason, and a check that the
  results move only as intended.
- A new option is declared once, with `_option(...)` in `solver.py`, together
  with its description. Tests and docs pick it up from there.
- Keep the code `ruff`-clean and add a test for each fix.

## Releasing

1. Update `__version__` in `src/atrfu/__init__.py`, `CHANGELOG.md` and `CITATION.cff`.
2. Commit and push. Wait for CI to pass.
3. Optional dry run: Actions > Release > Run workflow. This uploads to
   TestPyPI and solves the quick start with the installed package.
4. Create a GitHub release with tag `vX.Y.Z` (the tag must match
   `__version__`). The release workflow repeats the TestPyPI check and then
   publishes to PyPI. Zenodo archives the release, and Read the Docs builds
   the docs for the tag.
