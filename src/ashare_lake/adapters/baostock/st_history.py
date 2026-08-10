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
Suspension is reconstructed separately from bar gaps, so this path emits NO
rows for non-trading days (``tradestatus != 1``) and stays purely about the ST
label. Malformed ``isST`` values fail the symbol closed rather than degrading
to ``normal``.
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
    """Trading-day rows (st/normal) for one symbol, or ``None`` on failure.

    ``None`` means a retryable provider/symbol failure (query error OR a
    malformed ``isST`` value — never silently treated as normal). A symbol
    with no trading days returns ``[]``.
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
        if tradestatus != "1":
            # Non-trading / suspended days are owned by the derived bar-gap
            # suspension path; Baostock must not emit rows for them.
            continue
        if is_st not in ("0", "1"):
            # Malformed/unexpected vocabulary: fail the symbol closed instead
            # of silently recording a normal day.
            return None
        out.append(
            {
                "symbol": symbol,
                "trade_date": date.fromisoformat(trade_raw),
                "is_trading": True,
                "status": "st" if is_st == "1" else "normal",
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
