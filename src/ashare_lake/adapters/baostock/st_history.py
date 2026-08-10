"""Baostock historical ST labels — backfill source for trading_status (C4).

The daily ``trading_status`` step gets ST flags from EastMoney, which
only expose *today's* ST list — so ST labels in the lake start at the first live
run (2026-07), leaving every earlier backtest window with survivorship /
look-ahead bias (``universe="all_a"`` does not drop names that were ST then).

Baostock's k-data carries a per-day ``isST`` flag back to 2016, so a per-symbol
sweep reconstructs the historical ST label. ``isST`` is binary — it does not
split "ST" from "*ST" — so every ST day maps to ``status="st"``; that is enough
for the universe filter (``EXCLUDED_STATUSES`` covers both). Every genuinely
traded day is emitted: ``status="st"`` when ``isST == 1`` and
``status="normal"`` when ``isST == 0`` — the negative evidence makes a swept
non-ST day query-visible instead of indistinguishable from "never checked".
An explicit Baostock ``tradestatus == "0"`` day is emitted as
``status="suspended" / is_trading=false`` — provider-declared no-trade evidence
(the ``isST`` flag is NOT interpreted on such days; ST stays unknown because
the day was not tradeable). Suspension continues to be reconstructed from bar
gaps as well; both paths may cover the same day and compact resolves the PK.
Malformed ``tradestatus`` / ``isST`` values fail the symbol closed rather than
degrading to a guessed status. Missing rows are NEVER interpreted as suspended.
"""

from __future__ import annotations

import time
from datetime import date

import polars as pl

from ashare_lake.adapters.baostock._session import (
    fetch_per_symbol,
    to_baostock_symbol,
)

__all__ = ["fetch_st_history", "to_baostock_symbol"]

# baostock k-data fields: trading status (1=trading) and the ST flag (1=ST).
_ST_FIELDS = "date,code,tradestatus,isST"

# trading_status columns minus provenance (added by write_fetched).
_OUTPUT_SCHEMA = {
    "symbol": pl.Utf8,
    "trade_date": pl.Date,
    "is_trading": pl.Boolean,
    "status": pl.Utf8,
}


def _fetch_one_st(bs, symbol: str, start: date, end: date) -> list[dict] | None:
    """Trading-day (st/normal) + explicit non-trading (suspended) rows.

    ``None`` means a retryable provider/symbol failure (query error OR a
    malformed ``tradestatus`` / ``isST`` value — never silently treated as a
    guessed status). A symbol with no returned rows returns ``[]``; absence of
    rows is never interpreted as suspension.
    """
    rs = bs.query_history_k_data_plus(
        to_baostock_symbol(symbol),
        _ST_FIELDS,
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        frequency="d",
        adjustflag="3",  # ST flag is adjust-independent
    )
    if getattr(rs, "error_code", "0") != "0":
        return None
    out: list[dict] = []
    while rs.next():
        trade_raw, _code, tradestatus, is_st = rs.get_row_data()
        if tradestatus not in ("0", "1") or is_st not in ("0", "1"):
            # Malformed/unexpected vocabulary: fail the symbol closed instead
            # of silently recording a guessed status (st/normal/suspended).
            return None
        if tradestatus == "1":
            out.append(
                {
                    "symbol": symbol,
                    "trade_date": date.fromisoformat(trade_raw),
                    "is_trading": True,
                    "status": "st" if is_st == "1" else "normal",
                }
            )
        else:
            # Provider-declared no-trade day: isST is NOT interpreted here;
            # the day was not tradeable, so ST stays unknown (is_st=None later).
            out.append(
                {
                    "symbol": symbol,
                    "trade_date": date.fromisoformat(trade_raw),
                    "is_trading": False,
                    "status": "suspended",
                }
            )
    return out


def fetch_st_history(
    symbols: list[str],
    start: date,
    end: date,
    *,
    bs=None,
    sleep=time.sleep,
    config=None,
) -> tuple[pl.DataFrame, list[str]]:
    """Per-symbol historical trading_status rows over ``[start, end]``.

    Returns ``(dataframe, failed_symbols)``. Fail-loud on login failure; each
    symbol is retried with a fresh session + backoff and the still-failing ones
    are returned so the caller can surface them and resume. A symbol with no
    trading days contributes zero rows; a traded never-ST symbol contributes
    ``normal`` rows (negative evidence) — neither is a failure.

    ``bs`` / ``sleep`` / ``config`` are injectable for offline tests. Pass
    ``config`` in production for ``[sources.baostock]`` pacing.
    """
    rows, failed = fetch_per_symbol(
        symbols,
        start,
        end,
        _fetch_one_st,
        bs=bs,
        sleep=sleep,
        label="baostock ST",
        config=config,
    )
    df = pl.DataFrame(rows, schema=_OUTPUT_SCHEMA) if rows else pl.DataFrame(schema=_OUTPUT_SCHEMA)
    return df, failed
