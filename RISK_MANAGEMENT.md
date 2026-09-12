# Risk management

## The system never executes anything

This is not a configurable option, and there is no plan to make it one. The
system's only output is information: a signal (or "NO HAY SEÑAL") together
with its supporting data, shown to you. It never places an order, never
connects to a broker account, never touches funds, and never stores broker
credentials anywhere in this repository. Every trade decision — whether to
enter, at what size, in which broker — is made by you, manually, outside
this system. The rules below are things the system **tracks and warns
about** based on the signal journal (SIGNAL_ENGINE.md), not things it
enforces by blocking its own orders — it has no orders to block.

Risk management is intentionally independent of any strategy's logic —
these rules apply no matter which hypothesis produced a signal, and are
configured in `config/config.yaml` under `risk:` / `signals:`.

**Status: these are the rules the system is designed to track and surface.
The components that consume them (journal reconciliation, dashboard
warnings) are built in Phases 10+ and do not exist yet — this document
fixes the rules in advance.**

## Core rules

- `risk_per_trade` (default `0.04`, i.e. 4% of balance) — configurable, but
  **must never change automatically as a result of a loss**. No martingale,
  no "double after a loss," no loss-triggered position sizing of any kind.
- `daily_loss_limit` (default `0.10`) — stop trading for the rest of the day
  once cumulative daily loss reaches this fraction of balance.
- `max_consecutive_losses` (default `3`) — trigger a cooldown once reached.
- `max_trades_per_day` (default `10`) — hard cap, regardless of how many
  A+ signals appear.
- `max_drawdown` (default `0.20`) — stop trading entirely (not just for the
  day) if account drawdown exceeds this.
- `cooldown_after_loss_minutes` (default `30`) — mandatory pause after a
  loss before another signal can be acted on.

Any of these limits being reached means the system will surface a **STOP
TRADING** warning — not "reduce size and continue," not "wait for a better
setup." Since the system does not place trades itself, honoring the stop is
up to you; the warning exists so the decision is based on a clear signal
rather than being talked out of it in the moment.

## Signal quality tiers

Signals (once the signal engine — Phase 9 — exists) are classified:

- **A+** — every condition required by a ROBUST EDGE strategy is met.
  **Default: only A+ signals are surfaced.**
- **A** — sufficient conditions met, but not the full A+ bar.
- **B** — weak signal.
- **NO_TRADE** — any uncertainty at all.

`signals.default_min_quality_tier` in config.yaml defaults to `A+`
specifically so the system does not trade constantly — see the discipline
rule below.

## Discipline rule (central, non-negotiable)

> If the necessary conditions aren't present, don't trade.
> If there's doubt, don't trade.
> If the perfect entry appears, take it once and accept the result.

Technically: **NO SIGNAL = NO TRADE.** There is no fallback signal, no
"close enough," no discretionary override built into the system.

## Live execution

`signals.allow_live_execution` in config.yaml defaults to `false`, and
`otc_research.config.load_config()` **raises an error** if it's set to
`true` — automated live execution is not implemented in this codebase at
all yet (see project brief, section 22: data → backtest → out-of-sample →
walk-forward → paper trading → live signals → *only then* consider
automation). No credentials are stored in this repository, and none should
ever be committed to it.
