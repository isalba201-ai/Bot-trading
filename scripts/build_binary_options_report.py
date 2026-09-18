#!/usr/bin/env python
"""Step 9 addendum: renders the master + summary binary-options backtest
comparison tables (the user's exact requested schemas) from:

1. ``data/binary_options_backtest_results.jsonl`` — this addendum's own
   new H1-H20 base/NOTRADE/PERFORMANCE and H9-restricted-discovery
   evaluations (written by ``run_binary_options_backtest.py``).
2. The already-computed, corrected-candidacy re-run for discovered
   conditions (``run_candidacy_corrected_rerun.py``'s log,
   ``/tmp/step_corrected_rerun_full.log``) — PARSED, never re-evaluated:
   re-running would touch the already-frozen EUR/USD 15m TEST result a
   second time, which the TEST-once discipline forbids regardless of
   whether the number would come out identical.
3. Triple-barrier FDR counts (``condition_trials``, ``step8d-triplebarrier``
   run family) — reported ONLY as an auxiliary footnote, per
   ``EXPIRY_UNIVERSE_AUDIT.md`` Section 1's discovery/candidacy target
   mismatch finding — never in the primary master table.

Payout sensitivity ({70,75,80,85,90}%) is computed POST-HOC here from the
stored win rates — a binary option's WIN/LOSS outcome doesn't depend on
payout, only its break-even/EV/margin do (see
``backtest.metrics.break_even_win_rate``/``payout_adjusted_expectancy``).

Writes the full row set to ``--csv-out`` (every evaluation, for audit) and
prints the markdown master/summary tables (filtered to gate-1+ survivors,
per the "not every strategy" instruction) plus the four-category
classification breakdown to stdout / ``--markdown-out``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from collections import Counter, defaultdict

from otc_research.backtest.metrics import break_even_win_rate, payout_adjusted_expectancy

DB_PATH = "data/otc_research.db"
CORRECTED_LOG_PATH = "/tmp/step_corrected_rerun_full.log"
DEFAULT_JSONL = "data/binary_options_backtest_results.jsonl"
PAYOUTS = (0.70, 0.75, 0.80, 0.85, 0.90)

LOG_LINE_RE = re.compile(
    r"CORRECTED (?P<asset>\w+)/(?P<timeframe>\w+) (?P<direction>CALL|PUT) h=(?P<h>\d+) "
    r"(?P<label>.*?) -- train_n=(?P<n>\S+) train_wr=(?P<wr>\S+) accepted=(?P<accepted>True|False) "
    r"rejected_at=(?P<rejected_at>\S+)"
)


def _load_jsonl(path: str) -> list[dict]:
    try:
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        return []


def _parse_discovered_condition_log(path: str) -> list[dict]:
    """One row per already-evaluated discovered condition, REUSED verbatim
    from the corrected rerun's own log -- train_n/train_wr/accepted/
    rejected_at only (the log never recorded robustness/walk-forward/TEST
    detail beyond accepted/rejected_at, which is enough for the
    classification and gate-distribution reporting this table needs).
    """
    by_key: dict[tuple, dict] = {}
    try:
        with open(path) as f:
            for line in f:
                m = LOG_LINE_RE.search(line)
                if not m:
                    continue
                accepted = m.group("accepted") == "True"
                rejected_at = None if m.group("rejected_at") == "None" else m.group("rejected_at")
                train_wr = None if m.group("wr") == "n/a" else float(m.group("wr"))
                key = (m.group("asset"), m.group("timeframe"), m.group("direction"), m.group("h"), m.group("label"))
                # The log has exact-duplicate lines from an earlier foreground
                # run that timed out partway before being relaunched in the
                # background (see conversation history) -- last occurrence
                # wins, later lines are never less complete than earlier ones
                # for a byte-identical, deterministic (seeded) re-evaluation.
                by_key[key] = {
                    "family": "discovered-condition",
                    "code": None,
                    "label": m.group("label"),
                    "direction": m.group("direction"),
                    "asset": m.group("asset"),
                    "timeframe": m.group("timeframe"),
                    "h": int(m.group("h")),
                    "expiry_seconds": None,
                    "payout_used": 0.85,
                    "accepted": accepted,
                    "rejected_at_gate": rejected_at,
                    "train_sample_size": None if m.group("n") == "None" else int(m.group("n")),
                    "train_win_rate": train_wr,
                    "test_sample_size": None,
                    "test_win_rate": None,
                    "test_win_rate_ci_low": None,
                }
    except FileNotFoundError:
        pass
    return list(by_key.values())


def _classify_row(row: dict) -> str:
    if row.get("classification"):
        return row["classification"]
    if row["accepted"]:
        return "ACCEPTED"
    if row["rejected_at_gate"] == "test":
        return "PROMISING_BUT_UNPROVEN"
    if row["rejected_at_gate"] == "sample_size_and_margin" and (row.get("train_sample_size") or 0) < 100:
        return "EXPLORATORY"
    if row["rejected_at_gate"] == "walk_forward" and row.get("wf_n_folds_sufficiently_sampled") is not None and row["wf_n_folds_sufficiently_sampled"] < 3:
        return "EXPLORATORY"
    return "REJECTED"


def _known_test_result_overrides() -> dict[tuple, dict]:
    """The one already-published, frozen TEST result (EUR/USD 15m h=5,
    session-conditioned) -- transcribed verbatim from
    BINARY_OPTIONS_REFRAME_AUDIT.md / EXPIRY_UNIVERSE_AUDIT.md, never
    re-derived. Keyed by (asset, timeframe, h) since the log-parsed rows
    don't carry a stable condition id to join on more precisely.
    """
    return {
        ("EUR_USD", "15m", 5): {
            "test_sample_size": 120,
            "test_win_rate": 0.600,
            "test_win_rate_ci_low": 0.511,
        }
    }


def _payout_sensitivity(win_rate: float | None) -> dict[float, dict]:
    if win_rate is None:
        return {}
    out = {}
    for payout in PAYOUTS:
        be = break_even_win_rate(payout)
        out[payout] = {
            "break_even": be,
            "margin": win_rate - be,
            "ev": payout_adjusted_expectancy(win_rate, payout),
        }
    return out


def _triple_barrier_footnote(cur) -> list[dict]:
    cur.execute(
        """
        SELECT asset, timeframe, target_col, COUNT(*),
               SUM(CASE WHEN fdr_significant=1 AND win_rate>0.5 THEN 1 ELSE 0 END)
        FROM condition_trials WHERE run_id LIKE ?
        GROUP BY asset, timeframe, target_col
        """,
        ("step8d-triplebarrier-20260918T170230%",),
    )
    rows = []
    for asset, timeframe, target_col, n, sig in cur.fetchall():
        rows.append({
            "asset": asset, "timeframe": timeframe, "target_col": target_col,
            "trials": n, "fdr_significant": sig or 0,
        })
    return rows


def _fmt_pct(x):
    return f"{x*100:.1f}%" if x is not None else "n/a"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jsonl", default=DEFAULT_JSONL)
    parser.add_argument("--corrected-log", default=CORRECTED_LOG_PATH)
    parser.add_argument("--csv-out", default="data/binary_options_backtest_master_table.csv")
    parser.add_argument("--markdown-out", default=None)
    args = parser.parse_args()

    new_rows = [r for r in _load_jsonl(args.jsonl) if r["family"] not in ("H1-H20-PERFORMANCE",)]
    perf_rows = {(r["code"], r["asset"], r["timeframe"], r["h"]): r for r in _load_jsonl(args.jsonl) if r["family"] == "H1-H20-PERFORMANCE"}
    discovered_rows = _parse_discovered_condition_log(args.corrected_log)
    overrides = _known_test_result_overrides()
    for row in discovered_rows:
        key = (row["asset"], row["timeframe"], row["h"])
        if key in overrides and row["train_sample_size"] and row["train_sample_size"] > 0 and row["rejected_at_gate"] == "test":
            row.update(overrides[key])

    all_rows = new_rows + discovered_rows
    for row in all_rows:
        row["classification"] = _classify_row(row)

    # -------------------------------------------------------------- CSV --
    fieldnames = sorted({k for r in all_rows for k in r.keys()})
    with open(args.csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    # ---------------------------------------------------- classification --
    lines = []
    lines.append("## Classification breakdown (all families, all evaluations)\n")
    by_family_class: dict[str, Counter] = defaultdict(Counter)
    for row in all_rows:
        by_family_class[row["family"]][row["classification"]] += 1
    lines.append("| Family | Total | ACCEPTED | PROMISING_BUT_UNPROVEN | EXPLORATORY | REJECTED |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for family, counts in sorted(by_family_class.items()):
        total = sum(counts.values())
        lines.append(
            f"| {family} | {total} | {counts['ACCEPTED']} | {counts['PROMISING_BUT_UNPROVEN']} "
            f"| {counts['EXPLORATORY']} | {counts['REJECTED']} |"
        )
    overall = Counter(row["classification"] for row in all_rows)
    lines.append(
        f"| **TOTAL** | **{len(all_rows)}** | **{overall['ACCEPTED']}** | "
        f"**{overall['PROMISING_BUT_UNPROVEN']}** | **{overall['EXPLORATORY']}** | **{overall['REJECTED']}** |"
    )

    # ----------------------------------------------- master table (gate1+) --
    gate1_plus = [
        r for r in all_rows
        if r["classification"] != "REJECTED" or (r.get("rejected_at_gate") not in (None,) and r.get("train_sample_size") and r["train_sample_size"] >= 100)
    ]
    gate1_plus.sort(key=lambda r: (r["classification"] != "ACCEPTED", r["classification"] != "PROMISING_BUT_UNPROVEN", -(r.get("train_win_rate") or 0)))

    lines.append("\n## Master comparison table (every evaluation that reached gate 1 -- full set in the CSV)\n")
    lines.append(
        "| Estrategia | Activo | TF | Expiry | CALL/PUT | Trades | WIN% (TRAIN) | Break-even@85% "
        "| EV/trade@85% | P&L (R) | Max DD (R) | WF | TEST | Estado |"
    )
    lines.append("|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|---|")
    for r in gate1_plus:
        sens = _payout_sensitivity(r.get("train_win_rate"))
        be85 = sens.get(0.85)
        be_str = _fmt_pct(be85["break_even"]) if be85 else "n/a"
        ev_str = f"{be85['ev']:.4f}" if be85 else "n/a"

        wf = r.get("wf_fraction_folds_with_edge")
        wf_str = f"{wf*100:.0f}% folds w/ edge" if wf is not None else "not reached"

        if r.get("test_win_rate") is not None:
            test_str = (
                f"{_fmt_pct(r['test_win_rate'])} (CI low {_fmt_pct(r.get('test_win_rate_ci_low'))}, "
                f"n={r.get('test_sample_size')})"
            )
        else:
            test_str = "not reached"

        perf = perf_rows.get((r.get("code"), r.get("asset"), r.get("timeframe"), r.get("h")))
        pnl_str = f"{perf['perf_cumulative_return_r']:.2f}" if perf else "n/a"
        dd_str = f"{perf['perf_max_drawdown_r']:.2f}" if perf else "n/a"

        expiry = f"{r.get('expiry_seconds')}s" if r.get("expiry_seconds") else "n/a"
        lines.append(
            f"| {r.get('label', r.get('code'))} | {r['asset']} | {r['timeframe']} | {expiry} "
            f"| {r.get('direction', '?')} | {r.get('train_sample_size', '?')} "
            f"| {_fmt_pct(r.get('train_win_rate'))} | {be_str} | {ev_str} | {pnl_str} | {dd_str} "
            f"| {wf_str} | {test_str} | {r['classification']} |"
        )

    # ----------------------------------------------------- summary table --
    lines.append("\n## Summary table (gate-1+ survivors)\n")
    lines.append("| Estrategia | Nº señales (TRAIN) | WIN% TEST | EV@85% | Consistencia WF | Robustez | Estado |")
    lines.append("|---|---:|---:|---:|---|---|---|")
    for r in gate1_plus:
        sens = _payout_sensitivity(r.get("train_win_rate"))
        ev_str = f"{sens[0.85]['ev']:.4f}" if 0.85 in sens else "n/a"
        wf = r.get("wf_fraction_folds_with_edge")
        wf_str = f"{wf*100:.0f}%" if wf is not None else "not reached"
        robustness = r.get("robustness_classification", "not reached")
        test_wr_str = _fmt_pct(r.get("test_win_rate")) if r.get("test_win_rate") is not None else "not reached"
        lines.append(
            f"| {r.get('label', r.get('code'))} ({r['asset']}/{r['timeframe']}/h={r.get('h')}) "
            f"| {r.get('train_sample_size', '?')} | {test_wr_str} | {ev_str} | {wf_str} | {robustness} "
            f"| {r['classification']} |"
        )

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    tb_rows = _triple_barrier_footnote(cur)
    lines.append("\n## Triple-barrier auxiliary footnote (NOT in primary comparison -- see EXPIRY_UNIVERSE_AUDIT.md Section 1)\n")
    lines.append("| Activo | TF | Target | Trials | FDR-sig |")
    lines.append("|---|---|---|---:|---:|")
    for r in tb_rows:
        lines.append(f"| {r['asset']} | {r['timeframe']} | {r['target_col']} | {r['trials']} | {r['fdr_significant']} |")

    output = "\n".join(lines)
    if args.markdown_out:
        with open(args.markdown_out, "w") as f:
            f.write(output)
    print(output)


if __name__ == "__main__":
    main()
