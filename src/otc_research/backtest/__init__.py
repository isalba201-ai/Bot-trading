"""Phase 4: backtesting engine.

Runs a Strategy (see strategy.py) over stored, point-in-time Feature rows
(never recomputing indicators ad hoc — see FEATURES.md/ARCHITECTURE.md),
under a strict temporal train/validation/test split (splits.py) and three
execution-realism scenarios (execution.py), and reports statistically
grounded results (metrics.py) — see BACKTESTING.md for the full
methodology this package implements.

What this package does NOT do yet: walk-forward analysis (Phase 7), Monte
Carlo (Phase 8), parameter/robustness sensitivity sweeps (Phase 6), or the
H1-H10 strategies themselves (Phase 5) — see the module docstrings for
what's built vs. still planned.
"""
