# API reference

```{eval-rst}
.. autoclass:: atrfu.TrustRegionSolver
   :members: solve, options

.. autofunction:: atrfu.ensure_gjh
```

`atrfu.PRESETS` maps each preset name to its options; the presets are listed
in [Choosing a configuration](guide.md#choosing-a-configuration).

## Command line

| command | effect |
|---|---|
| `atrfu check` | Print the versions and the IPOPT and gjh in use. Exit with status 1 if IPOPT is missing. |
| `atrfu install-gjh [--dir DIR]` | Download gjh for this platform (by default into Pyomo's `bin` directory). |
| `atrfu --version` | Print the version. |
