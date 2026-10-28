# Changelog

## 1.0.0

First release of the A-TRFu solver as a package. It is ported from the
A-TRFu thesis archive ([10.5281/zenodo.23173541](https://doi.org/10.5281/zenodo.23173541)),
with the same numerical behaviour.

Changes from the archive:
- The default is A-TRFu (funnel, hypervolume sampling region, restoration R1).
  Other configurations are available as presets.
- Every option is documented. Options are checked when the solver is created,
  and unknown names get "did you mean" hints.
- The problem is checked before solving, with messages that name what to fix.
- Output is controlled by `verbose` (0, 1 or 2).
- gjh is found or downloaded automatically. The `atrfu check` and
  `atrfu install-gjh` commands are added.
