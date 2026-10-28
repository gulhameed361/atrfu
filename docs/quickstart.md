# Quick start

The problem: find the point of the unit disc nearest to (2, 1). The distance
from the origin is computed by a black box:

$$
\min_{x}\ (x_1 - 2)^2 + (x_2 - 1)^2 \quad \text{s.t.}\quad
\underbrace{x_1^2 + x_2^2}_{\text{black box } bb_1} \le 1,\qquad
\underbrace{\sin x_1 + x_2}_{\text{black box } bb_2} \ge 0.5 .
$$

The solution is $x^* = (2, 1)/\sqrt{5}$, with $f^* = (\sqrt{5} - 1)^2 \approx 1.5279$.

```{literalinclude} ../examples/quickstart.py
:language: python
```

Run it with `python examples/quickstart.py`:

```text
POSSIBLY AN OPTIMAL SOLUTION IS FOUND | feasible: True
f = 1.527863   x = (0.894093, 0.447884)
black-box evaluations: 342
```

Three steps turn a model into a grey-box problem:

1. **Wrap each black-box function** in `ExternalFunction(my_function)`. The
   function takes floats and returns a float.
2. **Link it to an output variable** with a constraint, such as
   `m.y1 == m.bb1(m.x1, m.x2)`. Then use the output variable (`m.y1`) in the
   rest of the model.
3. **Group the functions by black box** in `efmap`. Each entry maps a name of
   your choice to the list of `ExternalFunction`s that black box provides.

After `solve`, the model's variables hold the final point, and the returned
dict describes the run (see [Reading the result](guide.md#reading-the-result)).
