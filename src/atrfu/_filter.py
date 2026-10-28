"""Filter globalisation (``globalization_strategy=0``).

The filter stores (objective, infeasibility) pairs. A trial point is
acceptable when no stored pair dominates it and theta+ <= theta_max. An
acceptable point is an f-type step when the switching condition
f_k - f+ >= kappa_theta * theta_k^gamma_s holds and theta_k < theta_min;
otherwise it is a theta-type step and the filter is augmented with the
envelope (f+ - gamma_f theta+, (1 - gamma_theta) theta+).
"""
from math import pow


class Filter:
    """Filter of non-dominated (objective, infeasibility) pairs."""

    def __init__(self, kappa_theta, gamma_s, theta_min, theta_max,
                 gamma_f, gamma_theta):
        self.kappa_theta = kappa_theta
        self.gamma_s = gamma_s
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.gamma_f = gamma_f
        self.gamma_theta = gamma_theta
        # Instance attribute: a class-level list would be shared by every
        # solve in the process.
        self.entries = []

    def _dominated(self, f, theta):
        return any(f >= fe and theta >= te for fe, te in self.entries)

    def classify(self, theta_k, theta, f_k, f, delta, first_iteration=False):
        """Return 'f', 'theta' or 'reject'. The filter test is skipped on the
        first iteration; ``delta`` is unused (shared interface with Funnel)."""
        if not first_iteration and (theta > self.theta_max
                                    or self._dominated(f, theta)):
            return 'reject'
        if (f_k - f >= self.kappa_theta * pow(theta_k, self.gamma_s)
                and theta_k < self.theta_min):
            return 'f'
        return 'theta'

    def accept(self, status, theta, f):
        """Record an accepted step: a theta-type step augments the filter."""
        if status != 'f':
            self.abandon(theta, f)

    def abandon(self, theta, f):
        """Add the envelope of (f, theta), forbidding a return near it."""
        f, theta = f - self.gamma_f * theta, (1 - self.gamma_theta) * theta
        if self._dominated(f, theta):
            return
        self.entries = [(fe, te) for fe, te in self.entries
                        if not (f <= fe and theta <= te)]
        self.entries.append((f, theta))
