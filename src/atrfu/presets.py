"""Named solver configurations: ``TrustRegionSolver(preset=name, **overrides)``.

``"a-trfu"`` is the default configuration (thesis Chapter 7); the others switch
off or replace one of its components, as in the thesis's ablation study
(Table 7.2) and the comparison of restoration strategies (Table 7.3).
"""

PRESETS = {
    # Funnel, hypervolume sampling-region management, restoration R1
    'a-trfu': dict(globalization_strategy=1, multiple_black_boxes=1,
                   restoration_enabled=1, restoration_strategy=1),
    # ... with restoration strategy R2
    'a-trfu-r2': dict(globalization_strategy=1, multiple_black_boxes=1,
                      restoration_enabled=1, restoration_strategy=2),
    # ... without hypervolume: one aggregate infeasibility
    'no-hv': dict(globalization_strategy=1, multiple_black_boxes=0,
                  restoration_enabled=1, restoration_strategy=1),
    # ... without the restoration phase
    'no-restoration': dict(globalization_strategy=1, multiple_black_boxes=1,
                           restoration_enabled=0, restoration_strategy=1),
    # ... with a filter instead of the funnel
    'filter': dict(globalization_strategy=0, multiple_black_boxes=1,
                   restoration_enabled=1, restoration_strategy=1),
}
