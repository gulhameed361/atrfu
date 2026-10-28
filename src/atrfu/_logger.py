#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
"""Per-iteration record of a run and its printed log."""
import numpy as np

from ._helper import packXYZ

# Step-type flag -> label printed in the iteration log
STEP_LABELS = {'fStep': 'f-type step', 'thetaStep': 'theta-type step',
               'relaxthetaStep': 'relax-theta-type step',
               'rejected': 'step rejected', 'restoration': 'RESTORATION'}


class IterLog:
    """State of one iteration: the centre (xk, yk, zk), its measures, the
    radii and the step taken."""

    def __init__(self, iteration, xk, yk, zk, print_vars):
        self.iteration = iteration
        self.xk, self.yk, self.zk = xk, yk, zk
        self.print_vars = print_vars
        self.thetak = self.objk = self.chik = self.IT = None
        self.trustRadius = self.sampleRadius = self.stepNorm = None
        for flag in STEP_LABELS:
            setattr(self, flag, False)

    def fprint(self):
        print("\n**************************************")
        print(f"Iteration {self.iteration:d}:")
        if self.print_vars:
            print(packXYZ(self.xk, self.yk, self.zk))
        print(f"thetak = {self.thetak}")
        print(f"objk = {self.objk}")
        print(f"trustRadius = {self.trustRadius}")
        print(f"sampleRadius = {self.sampleRadius}")
        print(f"stepNorm = {self.stepNorm}")
        print(f"chi = {self.chik}")
        print(f"Iteration time (s) = {self.IT}")
        for flag, label in STEP_LABELS.items():
            if getattr(self, flag):
                print(label)
        print("**************************************\n")


class Logger:
    def __init__(self):
        # Instance attribute: a class-level list would be shared by every
        # solve in the process.
        self.iters = []

    def newIter(self, iteration, xk, yk, zk, thetak, objk, chik, IT, print_vars):
        self.iterlog = IterLog(iteration, xk, yk, zk, print_vars)
        self.iterlog.thetak, self.iterlog.objk = thetak, objk
        self.iterlog.chik, self.iterlog.IT = chik, IT
        self.iters.append(self.iterlog)

    def setCurIter(self, **values):
        """Set trustRadius, sampleRadius and/or stepNorm of this iteration."""
        for name, val in values.items():
            setattr(self.iterlog, name, val)

    def printIteration(self, iteration):
        if iteration < len(self.iters):
            self.iters[iteration].fprint()

    def printVectors(self):
        last = self.iterlog
        for it in self.iters:
            dis = np.linalg.norm(packXYZ(it.xk - last.xk, it.yk - last.yk,
                                         it.zk - last.zk), np.inf)
            print("\t".join(str(v) for v in (
                it.iteration, it.thetak, it.objk, it.chik, "", it.IT,
                it.trustRadius, it.sampleRadius, it.stepNorm, dis)))
        print("\n========== Optimization Summary ==========")
        print(f"Total Time (s): {sum(it.IT for it in self.iters if it.IT is not None):.6f}")
        print(f"Total Iterations: {max(it.iteration for it in self.iters)}")
        print("==========================================\n")
