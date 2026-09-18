#!/usr/bin/env python
"""Builds the (dataset x timeframe x horizon x direction) investigation
matrix requested after the corrected candidacy re-run -- distinguishes
signal cadence (timeframe) from expiry duration (h candles x timeframe
seconds) explicitly, per cell, and reports the FURTHEST gate any
FDR-significant winning-direction condition from that cell reached
(never inventing a result for a cell nothing was evaluated in).

Reads condition_trials (FDR counts, per the three run families already
executed: step7 naive-10-feature, step8d-session naive-13-feature,
step8d-triplebarrier) and parses the corrected candidacy re-run's own
log output (scripts/run_candidacy_corrected_rerun.py) for the furthest
gate reached per selected candidate -- never re-simulates or estimates a
result. Prints markdown tables directly.
"""

from __future__ import annotations

import re
import sqlite3
from collections import defaultdict

from otc_research.utils.timeframes import timeframe_to_seconds

DB_PATH = "data/otc_research.db"
CORRECTED_LOG_PATH = "/tmp/step_corrected_rerun_full.log"

GATE_ORDER = ["sample_size_and_margin", "robustness", "walk_forward", "test", "ACCEPTED"]
GATE_RANK = {g: i for i, g in enumerate(GATE_ORDER)}


def fdr_counts(cur, run_like: str) -> dict:
    cur.execute(
        """
        SELECT asset, timeframe, target_col, COUNT(*),
               SUM(CASE WHEN fdr_significant=1 AND win_rate>0.5 THEN 1 ELSE 0 END),
               MAX(CASE WHEN win_rate>0.5 THEN win_rate END)
        FROM condition_trials WHERE run_id LIKE ?
        GROUP BY asset, timeframe, target_col
        """,
        (run_like,),
    )
    return {(a, tf, tc): (n, sig or 0, best) for a, tf, tc, n, sig, best in cur.fetchall()}


LOG_LINE_RE = re.compile(
    r"CORRECTED (?P<asset>\w+)/(?P<timeframe>\w+) (?P<direction>CALL|PUT) h=(?P<h>\d+) "
    r".*? -- train_n=(?P<n>\S+) train_wr=(?P<wr>\S+) accepted=(?P<accepted>True|False) "
    r"rejected_at=(?P<rejected_at>\S+)"
)


def parse_corrected_log(path: str) -> dict:
    """Returns {(asset, timeframe, h, direction): furthest_gate_str} --
    "furthest" = max GATE_RANK among every selected candidate in that cell.
    """
    best: dict[tuple, str] = {}
    with open(path) as f:
        for line in f:
            m = LOG_LINE_RE.search(line)
            if not m:
                continue
            key = (m.group("asset"), m.group("timeframe"), int(m.group("h")), m.group("direction"))
            gate = "ACCEPTED" if m.group("accepted") == "True" else m.group("rejected_at")
            if key not in best or GATE_RANK.get(gate, -1) > GATE_RANK.get(best[key], -1):
                best[key] = gate
    return best


def expiry_label(timeframe: str, h: int) -> str:
    seconds = h * timeframe_to_seconds(timeframe)
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds / 60
    return f"{minutes:g} min"


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    step7 = fdr_counts(cur, "step7-20260918T161113%")
    step8d_session = fdr_counts(cur, "step8d-session-20260918T170230%")
    step8d_tb = fdr_counts(cur, "step8d-triplebarrier-20260918T170230%")
    corrected = parse_corrected_log(CORRECTED_LOG_PATH)

    print("### Naive target (call_wins_h/put_wins_h), core-10 features (Step 7) and core+session-13 features (Step 8d)\n")
    print("| Dataset | Timeframe | h | Vencimiento | Dirección | Trials (S7) | FDR-sig (S7) | Trials (S8d-session) | FDR-sig (S8d-session) | Furthest gate (corrected candidacy) |")
    print("|---|---|---:|---:|---|---:|---:|---:|---:|---|")
    rows = sorted(set(step7) | set(step8d_session))
    for asset, timeframe, target_col in rows:
        h = int(target_col.rsplit("_", 1)[-1])
        direction = "CALL" if target_col.startswith("call_wins") else "PUT"
        s7 = step7.get((asset, timeframe, target_col), (0, 0, None))
        s8 = step8d_session.get((asset, timeframe, target_col), (0, 0, None))
        gate = corrected.get((asset, timeframe, h, direction), "NO EVALUADO en candidacy corregida")
        print(
            f"| {asset} | {timeframe} | {h} | {expiry_label(timeframe, h)} | {direction} "
            f"| {s7[0]} | {s7[1]} | {s8[0]} | {s8[1]} | {gate} |"
        )

    print()
    print("### Triple-barrier target (auxiliary, upper=1.0x ATR lower=1.0x ATR max_horizon=10 candles)\n")
    print("| Dataset | Timeframe | h (max_horizon) | Vencimiento equivalente | Dirección | Trials | FDR-sig | Furthest gate |")
    print("|---|---|---:|---:|---|---:|---:|---|")
    for asset, timeframe, target_col in sorted(step8d_tb):
        direction = "CALL" if "call" in target_col else "PUT"
        h = 10
        tb = step8d_tb[(asset, timeframe, target_col)]
        gate = corrected.get((asset, timeframe, h, direction), "NO EVALUADO en candidacy corregida")
        print(
            f"| {asset} | {timeframe} | {h} | {expiry_label(timeframe, h)} (variable, barrier-stop) | {direction} "
            f"| {tb[0]} | {tb[1]} | {gate} |"
        )


if __name__ == "__main__":
    main()
