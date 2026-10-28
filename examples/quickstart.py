"""Quick start: a grey-box problem with two black boxes.

    minimise    (x1 - 2)^2 + (x2 - 1)^2
    subject to  y1 = bb1(x1, x2),  y1 <= 1      (bb1 is the squared distance from the origin)
                y2 = bb2(x1, x2),  y2 >= 0.5    (bb2 = sin(x1) + x2, inactive at the solution)

The solver sees bb1 and bb2 only through evaluations. The solution is the
point of the unit disc nearest to (2, 1): x* = (2, 1)/sqrt(5), with
f* = (sqrt(5) - 1)^2 ~= 1.5279.
"""
import math

from pyomo.environ import ConcreteModel, Constraint, ExternalFunction, Objective, Var

from atrfu import TrustRegionSolver


def bb1(a, b):          # stands in for an expensive simulation
    return a ** 2 + b ** 2


def bb2(a, b):
    return math.sin(a) + b


def build_model():
    m = ConcreteModel()
    # Every variable has finite bounds: by default (scaling=0) the trust region
    # is measured in units of each variable's bound range.
    m.x1 = Var(bounds=(-3, 3), initialize=0.5)
    m.x2 = Var(bounds=(-3, 3), initialize=0.5)
    m.y1 = Var(bounds=(0, 18), initialize=0.0)
    m.y2 = Var(bounds=(-4, 4), initialize=0.0)

    m.bb1 = ExternalFunction(bb1)
    m.bb2 = ExternalFunction(bb2)
    m.link1 = Constraint(expr=m.y1 == m.bb1(m.x1, m.x2))
    m.link2 = Constraint(expr=m.y2 == m.bb2(m.x1, m.x2))

    m.disc = Constraint(expr=m.y1 <= 1)
    m.floor = Constraint(expr=m.y2 >= 0.5)
    m.obj = Objective(expr=(m.x1 - 2) ** 2 + (m.x2 - 1) ** 2)
    # One entry per black box: its name and the ExternalFunctions it provides.
    efmap = {"distance": [m.bb1], "trig": [m.bb2]}
    return m, efmap


if __name__ == "__main__":
    model, efmap = build_model()
    result = TrustRegionSolver(verbose=1).solve(model, efmap)
    print(result["status"], "| feasible:", result["feasible"])
    print(f"f = {result['final_obj']:.6f}   x = ({model.x1.value:.6f}, {model.x2.value:.6f})")
    print("black-box evaluations:", result["bb_evals"])
