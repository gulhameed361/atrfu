"""Funnel globalisation (``globalization_strategy=1``).

The funnel keeps one scalar width phi that bounds the infeasibility theta of
the iterates and shrinks as feasibility improves. A trial point with
infeasibility theta+ and objective f+, from an iterate with theta_k and f_k,
is classified as

  inside the funnel (theta+ <= phi):
    f-type      if  f_k - f+ >= mu_s * theta_k^p          (switching)
                and f_k - f+ >= eta * Delta_k             (Armijo)
                -- rejected if switching holds but Armijo fails;
    theta-type  if switching fails and theta+ <= beta * phi;
  outside the funnel:
    theta-type  (relaxed) if switching holds and theta+ <= kappa_r * phi;
  otherwise rejected.

After a theta-type step the funnel contracts,
phi <- max(phi_min, (1 - kappa_f) theta+ + kappa_f phi).
"""


class Funnel:
    """Scalar funnel of width phi.

    Parameters
    ----------
    phi_init        : initial width (the initial infeasibility)
    phi_min         : floor on the width
    kappa_f         : contraction weight after a theta-type step
    beta            : required shrink for a theta-type step inside the funnel
    switch_coeff    : switching coefficient mu_s
    switch_exponent : switching exponent p
    eta             : Armijo coefficient
    kappa_r         : width factor for a theta-type step outside the funnel
    """

    def __init__(self, phi_init, phi_min, kappa_f, beta, switch_coeff,
                 switch_exponent, eta, kappa_r):
        self.phi = max(phi_min, phi_init)
        self.phi_min = phi_min
        self.kappa_f = kappa_f
        self.beta = beta
        self.switch_coeff = switch_coeff
        self.switch_exponent = switch_exponent
        self.eta = eta
        self.kappa_r = kappa_r

    def classify(self, theta_k, theta, f_k, f, delta, first_iteration=False):
        """Return 'f', 'theta', 'theta-relax' or 'reject' for the trial point.
        ``first_iteration`` is unused (shared interface with Filter)."""
        switching = (f_k - f) >= self.switch_coeff * (theta_k ** self.switch_exponent)
        if theta <= self.phi:
            if switching:
                return 'f' if (f_k - f) >= self.eta * delta else 'reject'
            return 'theta' if theta <= self.beta * self.phi else 'reject'
        if switching and theta <= self.kappa_r * self.phi:
            return 'theta-relax'
        return 'reject'

    def accept(self, status, theta, f):
        """Record an accepted step: a theta-type step contracts the funnel."""
        if status != 'f':
            self.phi = max(self.phi_min,
                           (1 - self.kappa_f) * theta + self.kappa_f * self.phi)

    def abandon(self, theta, f):
        """Restoration entry: the funnel is deliberately left unchanged, as the
        convergence proof needs phi to be non-increasing."""
