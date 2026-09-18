"""Phase 8: statistical research layer.

Approved plan: /root/.claude/plans/sequential-sparking-candle.md (the
project's pivot away from hand-picked indicator combinations, after
H1-H20 showed no robust edge, toward systematic bottom-up statistical
discovery with strict multiple-testing control).

Nothing here executes trades or talks to any broker. This package only
ever reads already-validated ``Candle``/``Feature`` rows (never a live
source, never Pocket Option) and produces statistics, discovered
conditions, and — only if warranted — simple models. See dataset.py's
docstring for the point-in-time contract every function here relies on.
"""
