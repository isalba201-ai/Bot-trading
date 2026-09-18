"""Phase 9 (approved plan sub-phase B, /root/.claude/plans/sequential-
sparking-candle.md): the manual-review signal lifecycle. A ``Signal`` row
is a proposal for a person to consider, never an order -- this package
never places a trade, never touches Pocket Option or any broker, and has
no live-execution path. See ``otc_research.notifications`` for how a
signal is actually delivered, and ``signals.decisions``/``signals.
performance`` for how a person's real, manual choice is recorded and
compared against the theoretical read.
"""
