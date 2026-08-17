"""L0 reference steps: instruments, trading_calendar, trading_status."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from datetime import date, timedelta

import polars as pl

from ashare_lake.adapters.eastmoney.instruments import enrich_instrument_list_dates
from ashare_lake.adapters.tdx_protocol.client import (
    fetch_instruments,
    fetch_trading_calendar,
    fetch_trading_status,
    normalize_with_source,
)
from ashare_lake.config import Config
from ashare_lake.domain.schemas import with_provenance
from ashare_lake.domain.symbols import is_all_a_symbol, is_tdx_servable, parse_symbol
from ashare_lake.orchestrator.registry import register_step
from ashare_lake.steps.common import (
    BACKFILL_START,
    fetch_incremental_daily,
    load_bar_universe,
    load_symbols,
    write_simple,
)
from ashare_lake.steps.http_common import write_fetched

logger = logging.getLogger(__name__)

# Resume marker for the ST-history backfill: which symbols baostock has already
# been swept for. Needed because ~85% of names are never ST and so contribute no
# rows — data-presence alone (as valuation uses) cannot tell "done" from "todo".
_ST_BACKFILL_STATE = "trading_status_st_backfill"
# Flush + checkpoint every chunk so a mid-sweep baostock login death does not
# discard hours of already-fetched ST rows (observed ~2950/5204 lost on one run).
_ST_BACKFILL_CHUNK = 200


@register_step("instruments", group="core", requires_workers=False)
def step_instruments(config: Config, trade_date: date, run_id: str, context: dict) -> dict:
    rl = config.tdx_rate_limit_spec()
    df = fetch_instruments(rate_limit=rl, allow_mock=config.tdx_allow_mock, config=config)
    df = normalize_with_source(df)
    df = enrich_instrument_list_dates(config, df)
    df = _merge_untdxable_instruments(config, df)
    if getattr(config, "_backfill", False):
        df = _merge_delisted_instruments(config, df)
    return write_simple(config, run_id, "instruments", df)


def _merge_untdxable_instruments(config: Config, df: pl.DataFrame) -> pl.DataFrame:
    """Add listed symbols the TDX security list structurally cannot contain.

    TDX serves Shanghai and Shenzhen only, so the Beijing exchange never
    appeared in the snapshot and the lake carried zero BJ instruments — meaning
    ``universe="all_a"`` quietly resolved to two exchanges out of three. The
    code-space sweep is what discovers them (``asl delisted discover``); this
    reads its live-but-missing bucket so the daily bar step has symbols to
    route to the fallback vendor.

    Runs every day, not only under --backfill: without it the next instruments
    compact would see every BJ name as absent from the snapshot and start
    inferring delistings for stocks that are trading normally.
    """
    from ashare_lake.steps.delisted import load_live_missing

    try:
        live_missing = load_live_missing(config)
    except Exception as exc:  # noqa: BLE001 — a missing catalogue is not fatal
        logger.debug("no delisted catalogue to read untdxable instruments from: %s", exc)
        return df
    known = set(df["symbol"].to_list()) if not df.is_empty() else set()
    recovered = sorted(s for s in live_missing if s not in known and not is_tdx_servable(s))
    if not recovered:
        return df

    logger.info(
        "instruments: +%d listed symbol(s) with no TDX route (e.g. %s)",
        len(recovered),
        ", ".join(recovered[:3]),
    )
    rows = pl.DataFrame(
        {
            "symbol": recovered,
            "name": [None] * len(recovered),
            "exchange": [s.split(".")[1] for s in recovered],
            "asset_type": ["stock"] * len(recovered),
            "list_date": pl.Series([None] * len(recovered), dtype=pl.Date),
            "delist_date": pl.Series([None] * len(recovered), dtype=pl.Date),
            "prev_symbol": [None] * len(recovered),
        }
    )
    return pl.concat(
        [df, with_provenance(rows, source="sina", data_version="v1")], how="diagonal_relaxed"
    )


def _merge_delisted_instruments(config: Config, df: pl.DataFrame) -> pl.DataFrame:
    """Add baostock's delisted names to a live-snapshot instrument list.

    TDX and EastMoney both answer "what is listed today", so on their own they
    build a survivors-only lake (audit: ``universe_survivorship_absent``).
    baostock's ``query_stock_basic`` is the one free source that also returns
    codes that *stopped* existing, which is what makes a point-in-time universe
    possible at all.

    Only rows baostock marks delisted are appended. Names it calls listed but the
    live snapshot omits are ambiguous — a delisting the snapshot has not caught up
    with, or a baostock staleness artefact — and appending them would inject
    untradable symbols into ``all_a``; they are counted and logged instead.

    Fail-loud: this runs only under an explicit ``--backfill``, whose entire
    purpose is the delisted set, so a broken baostock session must not quietly
    degrade into "no delisted names exist".
    """
    if not config.sources.get("baostock", False):
        logger.warning(
            "instruments backfill: [sources.baostock] disabled — delisted symbols "
            "cannot be recovered from TDX/EastMoney alone; universe stays survivors-only"
        )
        return df

    from ashare_lake.adapters.baostock.instruments import fetch_instrument_basics

    config.rate_limit("baostock")
    basics = fetch_instrument_basics()
    if basics.is_empty():
        raise RuntimeError(
            "baostock query_stock_basic returned no rows; refusing to write a "
            "survivors-only instrument list under --backfill"
        )

    live = set(df["symbol"].to_list())
    delisted = basics.filter(pl.col("delist_date").is_not_null() & ~pl.col("symbol").is_in(live))
    unlisted_unknown = basics.filter(
        pl.col("delist_date").is_null() & ~pl.col("symbol").is_in(live)
    ).height

    # baostock's ipoDate reaches further back than EastMoney's clist, so it also
    # fills list_date holes on names that are still trading.
    known_list_dates = basics.filter(pl.col("list_date").is_not_null()).select(
        ["symbol", pl.col("list_date").alias("_bs_list_date")]
    )
    df = (
        df.join(known_list_dates, on="symbol", how="left")
        .with_columns(pl.coalesce(pl.col("list_date"), pl.col("_bs_list_date")).alias("list_date"))
        .drop("_bs_list_date")
    )

    logger.info(
        "instruments backfill: +%d delisted symbol(s) from baostock "
        "(%d listed-but-absent skipped as ambiguous)",
        delisted.height,
        unlisted_unknown,
    )
    if delisted.is_empty():
        return df
    return pl.concat(
        [df, with_provenance(delisted, source="baostock", data_version="v1")],
        how="diagonal_relaxed",
    )


def _earliest_bar_date(config: Config) -> date | None:
    """First date any bar dataset carries — the calendar has to reach at least
    that far back or every window over the deep history resolves to zero
    sessions.

    Both datasets, matching ``_trading_days_from_bars`` in the calendar adapter.
    Reading only daily_bars pinned the calendar's start to 2001 while index_bars
    reached 1990-12-19, so 2,538 index_bars dates sat before the calendar
    existed and the audit reported them as bars on non-trading days — a warning
    about the calendar being short, dressed as a warning about the data.
    """
    import polars as pl

    earliest: date | None = None
    for dataset in ("index_bars", "daily_bars"):
        root = config.curated_root / dataset
        if not root.exists():
            continue
        files = list(root.glob("**/*.parquet"))
        if not files:
            continue
        frames = [pl.read_parquet(f, columns=["trade_date"]) for f in files]
        first = pl.concat(frames, how="diagonal_relaxed")["trade_date"].min()
        if first is not None and (earliest is None or first < earliest):
            earliest = first
    return earliest


@register_step("trading_calendar", group="core")
def step_trading_calendar(config: Config, trade_date: date, run_id: str, context: dict) -> dict:
    if getattr(config, "_backfill", False):
        start = getattr(config, "_backfill_start", None) or BACKFILL_START
        # Deep history reaches back further than BACKFILL_START; follow the data.
        earliest = _earliest_bar_date(config)
        if earliest is not None and earliest < start:
            start = earliest
    else:
        start = trade_date - timedelta(days=30)
    end = trade_date + timedelta(days=365)
    rl = config.tdx_rate_limit_spec()
    seed_path = config.meta_root / "seeds" / "trading_calendar.csv"
    df = fetch_trading_calendar(
        start,
        end,
        rate_limit=rl,
        allow_mock=config.tdx_allow_mock,
        curated_root=config.curated_root,
        seed_path=seed_path if seed_path.exists() else None,
    )
    df = normalize_with_source(df, source="exchange_calendar")
    return write_simple(config, run_id, "trading_calendar", df)


@register_step("trading_status", group="core")
def step_trading_status(config: Config, trade_date: date, run_id: str, context: dict) -> dict:
    if getattr(config, "_backfill", False):
        return _backfill_trading_status_st(config, trade_date, run_id)

    symbols = context.get("symbols") or load_symbols(config)
    rl = config.tdx_rate_limit_spec()

    # EastMoney is the only daily ST feed. An AkShare union used to sit here as a
    # "second source", but `ak.stock_zh_a_st_em` requests the same push2 clist
    # endpoint with the same `fs=m:0+f:4,m:1+f:4` filter that
    # adapters/eastmoney/trading_status.py already queries — same vendor, same
    # board, same filter — so it could only ever repeat this answer or fail. The
    # push2 → push2delay failover in the EastMoney client is the real robustness
    # here. The one genuinely independent ST reading, baostock's per-day `isST`,
    # is a per-symbol sweep and stays where it is affordable: the `--backfill`
    # path below. See issue #3.
    def _fetch(day: date):
        return fetch_trading_status(
            symbols,
            day,
            rate_limit=rl,
            allow_mock=config.tdx_allow_mock,
        )

    df, _findings = fetch_incremental_daily(
        config,
        "trading_status",
        trade_date,
        _fetch,
        allow_empty=True,
    )
    if df.is_empty():
        return {"rows_read": 0, "rows_written": 0}
    # EastMoney is the only daily status provider; rows fetched here are
    # current-state snapshots and must not be stamped as TDX (the dataset's
    # nominal primary source). Correct provenance matters for the
    # derived-suspension merge precedence later.
    df = with_provenance(df.drop("source", strict=False), source="eastmoney", data_version="v1")
    return write_simple(config, run_id, "trading_status", df)


def _is_all_a(symbol: str) -> bool:
    try:
        info = parse_symbol(symbol)
    except ValueError:
        return False
    return is_all_a_symbol(info.code, info.exchange)


def _st_backfill_state_path(config: Config):
    return config.meta_root / "state" / f"{_ST_BACKFILL_STATE}.json"


def _st_backfilled_symbols(config: Config) -> set[str]:
    path = _st_backfill_state_path(config)
    if not path.exists():
        return set()
    return set(json.loads(path.read_text(encoding="utf-8")).get("completed", []))


def _mark_st_backfilled(config: Config, symbols: list[str]) -> None:
    """Atomically add ``symbols`` to the swept-symbol set (resume marker)."""
    path = _st_backfill_state_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    completed = sorted(_st_backfilled_symbols(config) | set(symbols))
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"completed": completed}, handle, indent=2)
        os.replace(tmp, path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _backfill_trading_status_st(config: Config, trade_date: date, run_id: str) -> dict:
    """Historical ST labels from baostock over the all_a universe (2016 → today).

    Fills the ``trading_status`` ST gap so ``universe="all_a"`` excludes names that
    were ST in earlier backtest windows (removes survivorship / look-ahead bias).
    Resumable via a swept-symbol marker: a re-run only retries names a throttled
    session dropped. Progress is checkpointed every ``_ST_BACKFILL_CHUNK`` symbols
    (write staging + mark swept) so a mid-sweep login failure still keeps prior
    chunks. Symbols still failing are surfaced as an audit finding (fail-loud).
    """
    from ashare_lake.adapters.baostock.st_history import fetch_st_history

    universe = [s for s in load_symbols(config) if _is_all_a(s)]
    # Only sweep names that have price bars: a delisted symbol still sitting in the
    # instruments list would otherwise cost a baostock round-trip with no bar to
    # join its ST label against. Skip the constraint on a bars-less lake so a
    # first-time backfill still runs.
    bar_universe = load_bar_universe(config)
    if bar_universe:
        universe = [s for s in universe if s in bar_universe]
    done = _st_backfilled_symbols(config)
    todo = [s for s in universe if s not in done]
    if not todo:
        return {"rows_read": 0, "rows_written": 0, "note": "all symbols already ST-backfilled"}

    rows_read = 0
    rows_written = 0
    all_failed: list[str] = []
    for offset in range(0, len(todo), _ST_BACKFILL_CHUNK):
        batch = todo[offset : offset + _ST_BACKFILL_CHUNK]
        df, failed = fetch_st_history(batch, BACKFILL_START, trade_date, config=config)
        if not df.is_empty():
            chunk = write_fetched(
                config,
                run_id,
                "trading_status",
                df,
                source="baostock",
                batch_id=f"batch-{offset:05d}",
            )
            rows_read += int(chunk.get("rows_read", 0))
            rows_written += int(chunk.get("rows_written", 0))
        failed_set = set(failed)
        swept = [s for s in batch if s not in failed_set]
        if swept:
            _mark_st_backfilled(config, swept)
        all_failed.extend(failed)

    result: dict = {"rows_read": rows_read, "rows_written": rows_written}
    if all_failed:
        result["failed_symbols"] = len(all_failed)
        finding = {
            "dataset": "trading_status",
            "severity": "warning",
            "code": "baostock_st_backfill_incomplete",
            "message": (
                f"{len(all_failed)}/{len(todo)} symbols failed baostock ST backfill "
                "(throttled/dropped); re-run `asl backfill trading_status` to resume."
            ),
        }
        result.setdefault("context_updates", {})["audit_findings"] = [finding]
    return result
