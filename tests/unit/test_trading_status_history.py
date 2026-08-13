from datetime import date

import polars as pl

from ashare_lake.config import Config
from ashare_lake.derive.trading_status_history import (
    derive_suspension_history,
    status_row_precedence_class,
)


def _write(root, dataset, partition_col, val, df):
    d = root / "curated" / dataset / f"{partition_col}={val}"
    d.mkdir(parents=True, exist_ok=True)
    df.write_parquet(d / "part-merged.parquet")


def test_derive_suspension_from_bar_gaps(tmp_path):
    root = tmp_path / "data"
    cfg = Config(data_root=root)
    # calendar: 3 trading days
    days = [date(2024, 6, 26), date(2024, 6, 27), date(2024, 6, 28)]
    for d in days:
        _write(
            root,
            "trading_calendar",
            "trade_date",
            d.isoformat(),
            pl.DataFrame(
                {
                    "trade_date": [d],
                    "is_trading": [True],
                    "source": ["seed"],
                    "data_version": ["v1"],
                    "fetched_at": ["2024-06-28T00:00:00+00:00"],
                }
            ),
        )

    # bars: 600519 trades all 3 days; 000001 missing the middle day (suspended)
    def bar(sym, d):
        return {
            "symbol": sym,
            "trade_date": d,
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1,
            "amount": 1.0,
            "source": "tdx_protocol",
            "data_version": "v1",
            "fetched_at": "2024-06-28T00:00:00+00:00",
        }

    for d in days:
        rows = [bar("600519.SH", d)]
        if d != date(2024, 6, 27):
            rows.append(bar("000001.SZ", d))
        _write(root, "daily_bars", "trade_date", d.isoformat(), pl.DataFrame(rows))
    # instruments: both listed before window, not delisted
    (root / "curated" / "instruments").mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "symbol": ["600519.SH", "000001.SZ"],
            "name": ["A", "B"],
            "exchange": ["SH", "SZ"],
            "asset_type": ["stock", "stock"],
            "list_date": [date(2010, 1, 1), date(2010, 1, 1)],
            "delist_date": [None, None],
            "prev_symbol": [None, None],
            "source": ["tdx", "tdx"],
            "data_version": ["v1", "v1"],
            "fetched_at": ["2024-06-28T00:00:00+00:00"] * 2,
        }
    ).write_parquet(root / "curated" / "instruments" / "part-merged.parquet")

    n = derive_suspension_history(cfg)
    assert n == 1  # only 000001 on 2024-06-27

    # trading_status is month-partitioned (DatasetSpec); never write day dirs.
    month_dir = root / "curated" / "trading_status" / "trade_date=2024-06"
    assert month_dir.is_dir()
    assert not (root / "curated" / "trading_status" / "trade_date=2024-06-27").exists()

    ts = pl.read_parquet(month_dir / "part-merged.parquet")
    susp = ts.filter(pl.col("status") == "suspended")
    assert susp.height == 1
    assert susp["symbol"][0] == "000001.SZ"
    assert susp["trade_date"][0] == date(2024, 6, 27)
    assert susp["is_trading"][0] is False
    assert susp["source"][0] == "derived_bar_gap"


def test_derive_suspension_empty_lake(tmp_path):
    cfg = Config(data_root=tmp_path / "data")
    assert derive_suspension_history(cfg) == 0


def test_derive_suspension_respects_end_window(tmp_path):
    """Gaps outside [--start,--end] must not be written."""
    root = tmp_path / "data"
    cfg = Config(data_root=root)
    days = [
        date(2015, 12, 30),
        date(2015, 12, 31),
        date(2016, 1, 4),
        date(2016, 1, 5),
    ]

    def bar(sym, d):
        return {
            "symbol": sym,
            "trade_date": d,
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1,
            "amount": 1.0,
            "source": "tdx_protocol",
            "data_version": "v1",
            "fetched_at": "2016-01-05T00:00:00+00:00",
        }

    for d in days:
        _write(
            root,
            "trading_calendar",
            "trade_date",
            d.isoformat(),
            pl.DataFrame(
                {
                    "trade_date": [d],
                    "is_trading": [True],
                    "source": ["seed"],
                    "data_version": ["v1"],
                    "fetched_at": ["2016-01-05T00:00:00+00:00"],
                }
            ),
        )
        # 000001 missing 2015-12-31 and 2016-01-04
        rows = [bar("600519.SH", d)]
        if d not in (date(2015, 12, 31), date(2016, 1, 4)):
            rows.append(bar("000001.SZ", d))
        _write(root, "daily_bars", "trade_date", d.isoformat(), pl.DataFrame(rows))

    (root / "curated" / "instruments").mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "symbol": ["600519.SH", "000001.SZ"],
            "name": ["A", "B"],
            "exchange": ["SH", "SZ"],
            "asset_type": ["stock", "stock"],
            "list_date": [date(2010, 1, 1), date(2010, 1, 1)],
            "delist_date": [None, None],
            "prev_symbol": [None, None],
            "source": ["tdx", "tdx"],
            "data_version": ["v1", "v1"],
            "fetched_at": ["2016-01-05T00:00:00+00:00"] * 2,
        }
    ).write_parquet(root / "curated" / "instruments" / "part-merged.parquet")

    n = derive_suspension_history(cfg, start=date(2015, 1, 1), end=date(2015, 12, 31))
    assert n == 1
    ts = pl.read_parquet(
        root / "curated" / "trading_status" / "trade_date=2015-12" / "part-merged.parquet"
    )
    assert ts.filter(pl.col("status") == "suspended")["trade_date"].to_list() == [
        date(2015, 12, 31)
    ]
    assert not (root / "curated" / "trading_status" / "trade_date=2016-01").exists()


def _ts_row(symbol: str, trade_date: date, *, source: str, fetched_at: str) -> dict:
    return {
        "symbol": symbol,
        "trade_date": trade_date,
        "is_trading": True,
        "status": "normal",
        "source": source,
        "data_version": "v1",
        "fetched_at": fetched_at,
    }


def _write_existing_status(root, trade_date: date, rows: list[dict]) -> None:
    _write(
        root,
        "trading_status",
        "trade_date",
        trade_date.strftime("%Y-%m"),
        pl.DataFrame(rows),
    )


def _read_status(root) -> pl.DataFrame:
    files = sorted((root / "curated" / "trading_status").rglob("*.parquet"))
    return pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")


def _mini_lake(root) -> None:
    cfg = Config(data_root=root)
    days = [date(2024, 6, 26), date(2024, 6, 27), date(2024, 6, 28)]
    for d in days:
        _write(
            root,
            "trading_calendar",
            "trade_date",
            d.isoformat(),
            pl.DataFrame(
                {
                    "trade_date": [d],
                    "is_trading": [True],
                    "source": ["seed"],
                    "data_version": ["v1"],
                    "fetched_at": ["2024-06-28T00:00:00+00:00"],
                }
            ),
        )
    for d in days:
        rows = [
            {
                "symbol": "600519.SH",
                "trade_date": d,
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
                "volume": 1,
                "amount": 1.0,
                "source": "tdx_protocol",
                "data_version": "v1",
                "fetched_at": "2024-06-28T00:00:00+00:00",
            }
        ]
        if d != date(2024, 6, 27):
            rows.append(
                {
                    "symbol": "000001.SZ",
                    "trade_date": d,
                    "open": 1.0,
                    "high": 1.0,
                    "low": 1.0,
                    "close": 1.0,
                    "volume": 1,
                    "amount": 1.0,
                    "source": "tdx_protocol",
                    "data_version": "v1",
                    "fetched_at": "2024-06-28T00:00:00+00:00",
                }
            )
        _write(root, "daily_bars", "trade_date", d.isoformat(), pl.DataFrame(rows))
    (root / "curated" / "instruments").mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "symbol": ["600519.SH", "000001.SZ"],
            "list_date": [date(2000, 1, 1), date(2000, 1, 1)],
            "delist_date": [None, None],
        }
    ).write_parquet(root / "curated" / "instruments" / "part-merged.parquet")
    return cfg


def _derive_with_existing(tmp_path, existing: list[dict]) -> pl.DataFrame:
    root = tmp_path / "data"
    cfg = _mini_lake(root)
    _write_existing_status(root, date(2024, 6, 27), existing)
    derive_suspension_history(cfg)
    return _read_status(root)


def _suspended_row(status: pl.DataFrame) -> pl.DataFrame:
    return status.filter(
        (pl.col("symbol") == "000001.SZ") & (pl.col("trade_date") == date(2024, 6, 27))
    )


def test_derived_wins_over_legacy_tdx_non_pit_row(tmp_path):
    """T1: legacy tdx_protocol normal row fetched on a later session loses."""
    status = _derive_with_existing(
        tmp_path,
        [
            _ts_row(
                "000001.SZ",
                date(2024, 6, 27),
                source="tdx_protocol",
                fetched_at="2024-06-28T00:00:00+00:00",
            )
        ],
    )
    row = _suspended_row(status)
    assert row.height == 1
    assert row["source"][0] == "derived_bar_gap"
    assert row["status"][0] == "suspended"
    assert row["is_trading"][0] is False


def test_derived_wins_over_eastmoney_non_pit_row(tmp_path):
    """T2: eastmoney normal row fetched on a later session loses."""
    status = _derive_with_existing(
        tmp_path,
        [
            _ts_row(
                "000001.SZ",
                date(2024, 6, 27),
                source="eastmoney",
                fetched_at="2024-06-28T00:00:00+00:00",
            )
        ],
    )
    row = _suspended_row(status)
    assert row.height == 1
    assert row["source"][0] == "derived_bar_gap"


def test_same_session_eastmoney_row_keeps_precedence(tmp_path):
    """T3: same-session eastmoney normal row must not be reinterpreted."""
    status = _derive_with_existing(
        tmp_path,
        [
            _ts_row(
                "000001.SZ",
                date(2024, 6, 27),
                source="eastmoney",
                fetched_at="2024-06-27T02:00:00+00:00",
            )
        ],  # Shanghai 06-27
    )
    row = _suspended_row(status)
    assert row.height == 1
    assert row["source"][0] == "eastmoney"
    assert row["status"][0] == "normal"


def test_same_session_eastmoney_suspended_row_kept(tmp_path):
    """T4: same-session suspended row remains authoritative."""
    status = _derive_with_existing(
        tmp_path,
        [
            {
                "symbol": "000001.SZ",
                "trade_date": date(2024, 6, 27),
                "is_trading": False,
                "status": "suspended",
                "source": "eastmoney",
                "data_version": "v1",
                "fetched_at": "2024-06-27T02:00:00+00:00",
            }
        ],
    )
    row = _suspended_row(status)
    assert row.height == 1
    assert row["source"][0] == "eastmoney"
    assert row["status"][0] == "suspended"


def test_baostock_row_keeps_precedence(tmp_path):
    """T5: baostock ST rows are never reclassified."""
    status = _derive_with_existing(
        tmp_path,
        [
            {
                "symbol": "000001.SZ",
                "trade_date": date(2024, 6, 27),
                "is_trading": True,
                "status": "ST",
                "source": "baostock",
                "data_version": "v1",
                "fetched_at": "2024-06-28T00:00:00+00:00",
            }
        ],
    )
    row = _suspended_row(status)
    assert row.height == 1
    assert row["source"][0] == "baostock"


def test_unknown_source_keeps_precedence(tmp_path):
    """T6: unknown sources are preserved, never discarded as NON-PIT."""
    status = _derive_with_existing(
        tmp_path,
        [
            _ts_row(
                "000001.SZ",
                date(2024, 6, 27),
                source="mystery_provider",
                fetched_at="2024-06-28T00:00:00+00:00",
            )
        ],
    )
    row = _suspended_row(status)
    assert row.height == 1
    assert row["source"][0] == "mystery_provider"


def test_no_collision_derive_unchanged(tmp_path):
    """T7: without a collision the derive behaves as before."""
    root = tmp_path / "data"
    cfg = _mini_lake(root)
    result = derive_suspension_history(cfg)
    status = _read_status(root)
    assert result == 1
    assert status.height == 1
    assert status["source"][0] == "derived_bar_gap"


def test_precedence_class_pure_contract():
    """Direct unit tests for the classifier (T1/T3/T5/T6/T8)."""
    tdx_later = {
        "source": "tdx_protocol",
        "trade_date": date(2026, 8, 4),
        "fetched_at": "2026-08-10T01:32:34+08:00",
    }
    assert status_row_precedence_class(tdx_later) == "NON_PIT_DAILY_SNAPSHOT"

    em_same = {
        "source": "eastmoney",
        "trade_date": date(2026, 8, 4),
        "fetched_at": "2026-08-04T02:00:00+00:00",  # Shanghai 08-04
    }
    assert status_row_precedence_class(em_same) == "TRUSTED_SAME_SESSION_DAILY"

    assert (
        status_row_precedence_class(
            {"source": "baostock", "trade_date": date(2026, 8, 4), "fetched_at": None}
        )
        == "BAOSTOCK"
    )
    assert (
        status_row_precedence_class(
            {"source": "weird", "trade_date": date(2026, 8, 4), "fetched_at": None}
        )
        == "OTHER"
    )
    assert (
        status_row_precedence_class(
            {"source": "derived_bar_gap", "trade_date": date(2026, 8, 4), "fetched_at": None}
        )
        == "DERIVED_BAR_GAP"
    )

    # T8: UTC timestamp whose Shanghai date equals trade_date.
    utc_evening = {
        "source": "eastmoney",
        "trade_date": date(2026, 8, 4),
        "fetched_at": "2026-08-03T20:00:00+00:00",  # Shanghai 08-04 04:00
    }
    assert status_row_precedence_class(utc_evening) == "TRUSTED_SAME_SESSION_DAILY"

    # Missing fetched_at cannot prove same-session.
    assert (
        status_row_precedence_class(
            {"source": "eastmoney", "trade_date": date(2026, 8, 4), "fetched_at": None}
        )
        == "NON_PIT_DAILY_SNAPSHOT"
    )


def test_daily_step_stamps_eastmoney_provenance(tmp_path, monkeypatch):
    """The official daily trading_status step must stamp EastMoney provenance."""
    from ashare_lake.steps.reference import step_trading_status

    root = tmp_path / "data"
    cfg = Config(data_root=root)
    fetched = pl.DataFrame(
        [
            {
                "symbol": "600000.SH",
                "trade_date": date(2026, 8, 7),
                "is_trading": True,
                "status": "normal",
            }
        ]
    )

    def fake_fetch(config, dataset, trade_date, fetch_fn, *, allow_empty=False):
        return fetched, []

    monkeypatch.setattr("ashare_lake.steps.reference.fetch_incremental_daily", fake_fetch)
    monkeypatch.setattr("ashare_lake.steps.reference.load_symbols", lambda config: ["600000.SH"])

    step_trading_status(cfg, date(2026, 8, 7), "run-1", {})

    staged = sorted((root / "staging" / "trading_status").rglob("*.parquet"))
    assert staged
    written = pl.read_parquet(staged[0])
    assert written["source"].to_list() == ["eastmoney"]
    assert written["status"].to_list() == ["normal"]
