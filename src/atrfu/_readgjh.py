#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
"""Read the objective gradient and constraint Jacobian written by AMPL's gjh,
with the variable and constraint names from the .col and .row files
(requires symbolic_solver_labels=True)."""


def _read_sparse(f, header, rows):
    """Entries of the sparse 'param <name> :=' block after ``header``,
    0-based: [i, value] for a vector, [row, col, value] for a matrix."""
    line = f.readline()
    while line != header:
        line = f.readline()
    entries, row = [], None
    line = f.readline()
    while line[0] != ';':
        if rows and line[0] == '[':
            row = int(''.join(filter(str.isdigit, line))) - 1
            line = f.readline()
        idx, val = line.split()[:2]
        entries.append(([row] if rows else []) + [int(idx) - 1, float(val)])
        line = f.readline()
    return entries


def readgjh(fname):
    """Return (g, J, varlist, conlist) from the gjh output ``fname``."""
    with open(fname) as f:
        g = _read_sparse(f, "param g :=\n", rows=False)
        J = _read_sparse(f, "param J :=\n", rows=True)
    stem = fname[:-3]
    with open(stem + 'col') as f:
        varlist = f.read().split()
    with open(stem + 'row') as f:
        conlist = f.read().split()
    return g, J, varlist, conlist
