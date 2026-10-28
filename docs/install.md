# Installation

atrfu needs Python 3.10 or newer. It also needs two external programs that pip
cannot install: **IPOPT**, which solves the subproblems, and AMPL's **gjh**,
which supplies the derivatives of the glass-box model.

## 1. The package

```bash
pip install atrfu
```

## 2. IPOPT

The simplest route on Linux, macOS and Windows is conda-forge:

```bash
conda install -c conda-forge ipopt
```

Other options are `idaes get-extensions` (from the `idaes-pse` package) or a
COIN-OR binary on your `PATH`. Any IPOPT that Pyomo can find works.

## 3. gjh

gjh is downloaded automatically from AMPL the first time you solve, into
Pyomo's configuration directory (not into your Python environment). To fetch
it in advance, for example on a machine that will run offline:

```bash
atrfu install-gjh
```

The solver uses the first gjh it finds, in this order:
1. the `ATRFU_GJH` environment variable (the full path to an executable);
2. AMPL's `ampl_module_gjh` Python package;
3. your `PATH`;
4. Pyomo's `bin` directory, which is where `atrfu install-gjh` puts it.

## 4. Check the setup

```bash
atrfu check
```

This prints the versions and the IPOPT and gjh that will be used. It exits
with status 1 if IPOPT is missing.
