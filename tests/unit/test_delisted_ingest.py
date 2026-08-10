"""Ingesting catalogued delistings into daily_bars + instruments."""

import json
from datetime import date

import polars as pl

from ashare_lake.config import Config
from ashare_lake.domain.schemas import DAILY_BARS_SCHEMA
from ashare_lake.steps.delisted import (
    _ingested_symbols,
    backfill_delisted_bars,
    catalog_path,
    delisted_backfill_targets,
    delisted_symbols_in_window,
)
from ashare_lake.storage.parquet import StagingWriter

_START = date(2016, 1, 1)
_BAR_COLS = [c for c in DAILY_BARS_SCHEMA if c not in ("source", "data_version", "fetched_at")]


def _bars(symbol: str, first: date, last: date) -> pl.DataFrame:
    days = [first, last]
    return pl.DataFrame(
        {
            "symbol": [symbol] * 2,
            "trade_date": days,
            "open": [1.0, 2.0],
            "high": [1.0, 2.0],
            "low": [1.0, 2.0],
            "close": [1.0, 2.0],
            "volume": [10, 20],
            "amount": [None, None],
        },
        schema={c: DAILY_BARS_SCHEMA[c] for c in _BAR_COLS},
    )


def _cfg(tmp_path, catalog: dict[str, str], live=("600519.SH",)):
    cfg = Config(data_root=tmp_path / "data", sources={"sina": True})
    path = catalog_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"delisted": catalog, "never_issued": []}))
    inst = cfg.curated_root / "instruments"
    inst.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "symbol": list(live),
            "name": ["live"] * len(live),
            "exchange": [s.split(".")[1] for s in live],
            "asset_type": ["stock"] * len(live),
            "list_date": pl.Series([date(2000, 1, 1)] * len(live), dtype=pl.Date),
            "delist_date": pl.Series([None] * len(live), dtype=pl.Date),
            "prev_symbol": [None] * len(live),
        }
    ).write_parquet(inst / "part-merged.parquet")
    return cfg


def _staged(cfg, dataset, run_id) -> pl.DataFrame:
    files = StagingWriter(cfg.staging_root).list_run_files(dataset, run_id)
    if not files:
        return pl.DataFrame()
    return pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")


def test_only_delistings_overlapping_the_window_are_fetched(tmp_path):
    """A name gone before the lake starts contributes nothing to a backtest over it."""
    cfg = _cfg(tmp_path, {"600001.SH": "2009-12-15", "600070.SH": "2025-04-10"})

    assert delisted_symbols_in_window(cfg, _START) == ["600070.SH"]


def test_bars_are_staged_with_sina_provenance(tmp_path):
    cfg = _cfg(tmp_path, {"600070.SH": "2025-04-10"})

    result = backfill_delisted_bars(
        cfg, "run-1", _START, fetch=lambda s, c: _bars(s, date(2016, 3, 1), date(2025, 4, 10))
    )

    staged = _staged(cfg, "daily_bars", "run-1")
    assert result["rows_written"] == 2
    assert staged["symbol"].unique().to_list() == ["600070.SH"]
    assert staged["source"].unique().to_list() == ["sina"]


def test_instruments_row_dates_the_delisting_from_the_last_bar(tmp_path):
    cfg = _cfg(tmp_path, {"600070.SH": "2025-04-10"})

    backfill_delisted_bars(
        cfg, "run-1", _START, fetch=lambda s, c: _bars(s, date(2016, 3, 1), date(2025, 4, 10))
    )

    inst = _staged(cfg, "instruments", "run-1")
    row = inst.filter(pl.col("symbol") == "600070.SH")
    assert row["list_date"].item() == date(2016, 3, 1)
    assert row["delist_date"].item() == date(2025, 4, 10)
    assert row["asset_type"].item() == "stock"


def test_instruments_staging_keeps_the_live_snapshot(tmp_path):
    """Staging only recovered names would look like a mass delisting to compact."""
    cfg = _cfg(tmp_path, {"600070.SH": "2025-04-10"}, live=("600519.SH", "000001.SZ"))

    backfill_delisted_bars(
        cfg, "run-1", _START, fetch=lambda s, c: _bars(s, date(2016, 3, 1), date(2025, 4, 10))
    )

    inst = _staged(cfg, "instruments", "run-1")
    assert set(inst["symbol"]) == {"600519.SH", "000001.SZ", "600070.SH"}
    # A live symbol must keep its null delist_date.
    assert inst.filter(pl.col("symbol") == "600519.SH")["delist_date"].item() is None


def test_spans_accumulate_across_staging_chunks(tmp_path):
    """Chunked staging must not drop earlier symbols from the instruments rows."""
    catalog = {f"6001{i:02d}.SH": "2025-04-10" for i in range(120)}
    cfg = _cfg(tmp_path, catalog)

    backfill_delisted_bars(
        cfg, "run-1", _START, fetch=lambda s, c: _bars(s, date(2016, 3, 1), date(2025, 4, 10))
    )

    inst = _staged(cfg, "instruments", "run-1")
    recovered = set(inst["symbol"]) - {"600519.SH"}
    assert len(recovered) == 120, "every chunk's symbols must reach instruments"


def test_a_failed_symbol_is_not_marked_done_so_a_rerun_retries_it(tmp_path):
    cfg = _cfg(tmp_path, {"600070.SH": "2025-04-10", "600083.SH": "2025-01-16"})

    def flaky(symbol, client):
        if symbol == "600070.SH":
            raise ConnectionError("reset")
        return _bars(symbol, date(2016, 3, 1), date(2025, 1, 16))

    result = backfill_delisted_bars(cfg, "run-1", _START, fetch=flaky)

    assert result["failed_symbols"] == 1
    assert "600070.SH" not in _ingested_symbols(cfg)
    assert "600083.SH" in _ingested_symbols(cfg)
    assert delisted_symbols_in_window(cfg, _START) == ["600070.SH"]


def test_rerun_is_a_noop_once_everything_is_ingested(tmp_path):
    cfg = _cfg(tmp_path, {"600070.SH": "2025-04-10"})
    backfill_delisted_bars(
        cfg, "run-1", _START, fetch=lambda s, c: _bars(s, date(2016, 3, 1), date(2025, 4, 10))
    )

    def must_not_be_called(symbol, client):
        raise AssertionError(f"refetched {symbol}")

    result = backfill_delisted_bars(cfg, "run-2", _START, fetch=must_not_be_called)
    assert result["rows_written"] == 0
    assert "no delisted recovery targets" in result["note"]


def test_a_symbol_with_empty_bars_is_kept_retryable_and_not_ingested(tmp_path):
    """Empty history for a window-overlapping target is unresolved, not done."""
    cfg = _cfg(tmp_path, {"600070.SH": "2025-04-10"})

    result = backfill_delisted_bars(
        cfg,
        "run-1",
        _START,
        fetch=lambda s, c: pl.DataFrame(schema={c: DAILY_BARS_SCHEMA[c] for c in _BAR_COLS}),
        probe_last=lambda s, c: None,
    )

    assert result["recovered"] == 0
    assert result["empty_symbols"] == 1
    assert result["failed_symbols"] == 0
    assert "600070.SH" not in _ingested_symbols(cfg)
    assert _staged(cfg, "instruments", "run-1").is_empty()
    # The empty target remains a recovery target for the next run.
    assert "600070.SH" in delisted_backfill_targets(cfg, _START, date.today())


def test_fetch_exception_keeps_target_retryable(tmp_path):
    cfg = _cfg(tmp_path, {"600070.SH": "2025-04-10"})

    def boom(symbol, client):
        raise ConnectionError("provider down")

    result = backfill_delisted_bars(cfg, "run-1", _START, fetch=boom)

    assert result["failed_symbols"] == 1
    assert "600070.SH" not in _ingested_symbols(cfg)
    assert "600070.SH" in delisted_backfill_targets(cfg, _START, date.today())


def test_empty_formal_target_kept_retryable(tmp_path):
    """Formal authority target with an empty vendor response stays retryable."""
    cfg = _cfg(tmp_path, {})  # no catalogue evidence
    inst_path = cfg.curated_root / "instruments" / "part-merged.parquet"
    live = pl.read_parquet(inst_path)
    formal = pl.DataFrame(
        {
            "symbol": ["600071.SH"],
            "name": ["delisted"],
            "exchange": ["SH"],
            "asset_type": ["stock"],
            "list_date": pl.Series([date(1998, 1, 22)], dtype=pl.Date),
            "delist_date": pl.Series([date(2025, 4, 10)], dtype=pl.Date),
            "prev_symbol": [None],
        }
    )
    pl.concat([live, formal], how="diagonal_relaxed").write_parquet(inst_path)

    result = backfill_delisted_bars(
        cfg,
        "run-1",
        _START,
        fetch=lambda s, c: pl.DataFrame(schema={c: DAILY_BARS_SCHEMA[c] for c in _BAR_COLS}),
        probe_last=lambda s, c: None,
    )

    assert result["empty_symbols"] == 1
    assert "600071.SH" not in _ingested_symbols(cfg)
    assert "600071.SH" in delisted_backfill_targets(cfg, _START, date.today())


def test_mixed_chunk_preserves_success_and_keeps_empty_exception_retryable(tmp_path):
    """A successful chunk symbol stays ingested; empty/exception stay targets."""
    cfg = _cfg(
        tmp_path,
        {
            "600070.SH": "2025-04-10",
            "600071.SH": "2025-04-10",
            "600072.SH": "2025-04-10",
        },
    )

    def mixed(symbol, client):
        if symbol == "600070.SH":
            return _bars(symbol, date(2016, 3, 1), date(2025, 4, 10))
        if symbol == "600071.SH":
            return pl.DataFrame(schema={c: DAILY_BARS_SCHEMA[c] for c in _BAR_COLS})
        raise ConnectionError("provider down")

    result = backfill_delisted_bars(
        cfg,
        "run-1",
        _START,
        fetch=mixed,
        probe_last=lambda s, c: None,
    )

    assert result["recovered"] == 1
    assert result["empty_symbols"] == 1
    assert result["failed_symbols"] == 1
    assert set(_ingested_symbols(cfg)) == {"600070.SH"}
    staged = _staged(cfg, "daily_bars", "run-1")
    assert staged["symbol"].unique().to_list() == ["600070.SH"]
    remaining = delisted_backfill_targets(cfg, _START, date.today())
    assert remaining == ["600071.SH", "600072.SH"]
