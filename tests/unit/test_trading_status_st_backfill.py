"""Resume + orchestration for the trading_status ST backfill step (C4).

The baostock fetch and the curated write are stubbed so the test isolates the
step's own logic: the todo set, the swept-symbol resume marker, and the
fail-loud finding on dropped symbols.
"""

from __future__ import annotations

import json
from datetime import date

import polars as pl

from ashare_lake.config import Config
from ashare_lake.steps import reference
from ashare_lake.steps.reference import (
    _backfill_trading_status_st,
    _st_backfilled_symbols,
    _st_completed_symbols,
    _st_scope_checkpoint_path,
)


def _write_instruments(config: Config, symbols: list[str]) -> None:
    root = config.curated_root / "instruments"
    root.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"symbol": symbols}).write_parquet(root / "part-merged.parquet")


def _st_row(symbol: str, d: date) -> dict:
    return {"symbol": symbol, "trade_date": d, "is_trading": True, "status": "st"}


def _patch(monkeypatch, *, returns):
    """Stub the network fetch and the curated write; return a captured-writes list."""
    written: list[pl.DataFrame] = []
    calls: list[tuple] = []

    def fake_fetch(symbols, start, end, **kwargs):
        calls.append((list(symbols), start, end))
        df, failed = returns
        return df, failed

    def fake_write(config, run_id, dataset, df, *, source, batch_id="batch-0"):
        written.append(df)
        return {"rows_read": df.height, "rows_written": df.height}

    monkeypatch.setattr("ashare_lake.adapters.baostock.st_history.fetch_st_history", fake_fetch)
    monkeypatch.setattr(reference, "write_fetched", fake_write)
    return written, calls


def test_writes_st_rows_and_marks_all_swept_symbols(tmp_path, monkeypatch):
    cfg = Config(data_root=tmp_path / "data")
    _write_instruments(cfg, ["600000.SH", "600001.SH"])  # both all_a, no ST for 600001
    df = pl.DataFrame([_st_row("600000.SH", date(2020, 5, 6))])
    written, _calls = _patch(monkeypatch, returns=(df, []))

    result = _backfill_trading_status_st(cfg, date(2026, 7, 1), "run1")

    assert result["rows_written"] == 1
    assert written[0]["symbol"].to_list() == ["600000.SH"]
    # every swept symbol is marked done — including the one that was never ST
    assert _st_backfilled_symbols(cfg) == {"600000.SH", "600001.SH"}


def test_resume_skips_already_swept_symbols(tmp_path, monkeypatch):
    cfg = Config(data_root=tmp_path / "data")
    _write_instruments(cfg, ["600000.SH", "600001.SH"])
    _patch(monkeypatch, returns=(pl.DataFrame([_st_row("600000.SH", date(2020, 5, 6))]), []))
    _backfill_trading_status_st(cfg, date(2026, 7, 1), "run1")

    # Second run: nothing left to do.
    captured: dict = {}

    def fake_fetch(symbols, start, end, **kwargs):
        captured["symbols"] = symbols
        return pl.DataFrame(schema={"symbol": pl.Utf8}), []

    monkeypatch.setattr("ashare_lake.adapters.baostock.st_history.fetch_st_history", fake_fetch)
    result = _backfill_trading_status_st(cfg, date(2026, 7, 1), "run2")
    assert "already ST-backfilled" in result["note"]
    assert "symbols" not in captured  # fetch not even called


def test_failed_symbols_are_not_marked_and_surface_a_finding(tmp_path, monkeypatch):
    cfg = Config(data_root=tmp_path / "data")
    _write_instruments(cfg, ["600000.SH", "600001.SH"])
    df = pl.DataFrame([_st_row("600000.SH", date(2020, 5, 6))])
    _patch(monkeypatch, returns=(df, ["600001.SH"]))  # 600001 dropped by throttling

    result = _backfill_trading_status_st(cfg, date(2026, 7, 1), "run1")

    # only the succeeded symbol is marked; the dropped one stays todo for resume
    assert _st_backfilled_symbols(cfg) == {"600000.SH"}
    assert result["failed_symbols"] == 1
    finding = result["context_updates"]["audit_findings"][0]
    assert finding["code"] == "baostock_st_backfill_incomplete"
    assert finding["severity"] == "warning"


def _scope_cfg(tmp_path, symbols, **attrs):
    cfg = Config(data_root=tmp_path / "data")
    _write_instruments(cfg, symbols)
    for key, value in attrs.items():
        setattr(cfg, key, value)
    return cfg


def test_start_reaches_fetch_st_history(tmp_path, monkeypatch):
    cfg = _scope_cfg(tmp_path, ["600000.SH"], _backfill_start=date(2026, 3, 30))
    df = pl.DataFrame([_st_row("600000.SH", date(2026, 5, 6))])
    _written, calls = _patch(monkeypatch, returns=(df, []))

    _backfill_trading_status_st(cfg, date(2026, 8, 7), "run1")

    assert calls[0][1] == date(2026, 3, 30)


def test_end_reaches_fetch_st_history(tmp_path, monkeypatch):
    cfg = _scope_cfg(
        tmp_path,
        ["600000.SH"],
        _backfill_start=date(2026, 3, 30),
        _backfill_end=date(2026, 8, 7),
    )
    df = pl.DataFrame([_st_row("600000.SH", date(2026, 5, 6))])
    _written, calls = _patch(monkeypatch, returns=(df, []))

    _backfill_trading_status_st(cfg, date(2026, 8, 10), "run1")

    assert calls[0][2] == date(2026, 8, 7)  # NOT date.today() / trade_date


def test_inverted_window_fails_closed(tmp_path, monkeypatch):
    cfg = _scope_cfg(
        tmp_path,
        ["600000.SH"],
        _backfill_start=date(2026, 8, 7),
        _backfill_end=date(2026, 3, 30),
    )
    _patch(monkeypatch, returns=(pl.DataFrame(), []))

    try:
        _backfill_trading_status_st(cfg, date(2026, 8, 10), "run1")
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "inverted" in str(exc)


def test_explicit_symbol_scope_queries_only_those_symbols(tmp_path, monkeypatch):
    cfg = _scope_cfg(
        tmp_path,
        ["600000.SH", "600001.SH", "000001.SZ"],
        _backfill_symbols=["600000.SH", "000001"],
    )
    df = pl.DataFrame([_st_row("600000.SH", date(2026, 5, 6))])
    _written, calls = _patch(monkeypatch, returns=(df, []))

    _backfill_trading_status_st(cfg, date(2026, 8, 7), "run1")

    queried = {s for call in calls for s in call[0]}
    assert queried == {"600000.SH", "000001.SZ"}  # 600001.SH never queried


def test_malformed_explicit_symbol_fails_closed(tmp_path, monkeypatch):
    cfg = _scope_cfg(tmp_path, ["600000.SH"], _backfill_symbols=["NOT_A_SYMBOL"])
    _patch(monkeypatch, returns=(pl.DataFrame(), []))

    try:
        _backfill_trading_status_st(cfg, date(2026, 8, 7), "run1")
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "malformed symbol" in str(exc)


def test_unknown_explicit_symbol_fails_closed(tmp_path, monkeypatch):
    cfg = _scope_cfg(tmp_path, ["600000.SH"], _backfill_symbols=["999999.SH"])
    _patch(monkeypatch, returns=(pl.DataFrame(), []))

    try:
        _backfill_trading_status_st(cfg, date(2026, 8, 7), "run1")
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "unknown instrument" in str(exc)


def test_default_path_remains_all_a_2016(tmp_path, monkeypatch):
    cfg = _scope_cfg(tmp_path, ["600000.SH", "600001.SH"])
    df = pl.DataFrame([_st_row("600000.SH", date(2020, 5, 6))])
    _written, calls = _patch(monkeypatch, returns=(df, []))

    result = _backfill_trading_status_st(cfg, date(2026, 7, 1), "run1")

    assert result["scope_id"] is None
    assert calls[0][1] == date(2016, 1, 1)  # BACKFILL_START
    assert calls[0][2] == date(2026, 7, 1)  # trade_date


def test_same_scope_resume_skips_completed(tmp_path, monkeypatch):
    cfg = _scope_cfg(
        tmp_path,
        ["600000.SH", "000001.SZ"],
        _backfill_start=date(2026, 3, 30),
        _backfill_end=date(2026, 8, 7),
        _backfill_symbols=["600000.SH", "000001.SZ"],
    )
    df = pl.DataFrame([_st_row("600000.SH", date(2026, 5, 6))])
    _written, calls = _patch(monkeypatch, returns=(df, []))
    first = _backfill_trading_status_st(cfg, date(2026, 8, 7), "run1")
    assert first["scope_id"] is not None

    second = _backfill_trading_status_st(cfg, date(2026, 8, 7), "run2")

    assert len(calls) == 1  # first run only
    assert "already ST-backfilled" in second["note"]


def test_zero_st_successful_sweep_counts_completed(tmp_path, monkeypatch):
    cfg = _scope_cfg(
        tmp_path,
        ["600000.SH"],
        _backfill_start=date(2026, 3, 30),
        _backfill_end=date(2026, 8, 7),
        _backfill_symbols=["600000.SH"],
    )
    _written, calls = _patch(monkeypatch, returns=(pl.DataFrame(), []))  # zero ST rows

    result = _backfill_trading_status_st(cfg, date(2026, 8, 7), "run1")

    assert result["rows_written"] == 0
    assert _st_completed_symbols(cfg, result["scope_id"]) == {"600000.SH"}


def test_failed_symbol_remains_todo(tmp_path, monkeypatch):
    cfg = _scope_cfg(
        tmp_path,
        ["600000.SH", "000001.SZ"],
        _backfill_start=date(2026, 3, 30),
        _backfill_end=date(2026, 8, 7),
        _backfill_symbols=["600000.SH", "000001.SZ"],
    )
    _written, calls = _patch(monkeypatch, returns=(pl.DataFrame(), ["000001.SZ"]))
    first = _backfill_trading_status_st(cfg, date(2026, 8, 7), "run1")
    assert first["failed_symbols"] == 1

    second = _backfill_trading_status_st(cfg, date(2026, 8, 7), "run2")
    assert "already ST-backfilled" not in second.get("note", "")
    # Only the failed symbol remains todo (600000.SH was completed).
    assert set(calls[1][0]) == {"000001.SZ"}
    assert second["failed_symbols"] == 1  # mocked fetch fails 000001 again


def test_shallow_checkpoint_does_not_suppress_deeper_run(tmp_path, monkeypatch):
    cfg = _scope_cfg(
        tmp_path,
        ["600000.SH", "000001.SZ"],
        _backfill_start=date(2026, 3, 30),
        _backfill_end=date(2026, 8, 7),
        _backfill_symbols=["600000.SH", "000001.SZ"],
    )
    _written, calls = _patch(monkeypatch, returns=(pl.DataFrame(), []))
    _backfill_trading_status_st(cfg, date(2026, 8, 7), "run1")
    assert len(calls) == 1

    # Deeper/default scope: the shallow completion must NOT suppress A/B.
    cfg._backfill_start = None
    cfg._backfill_end = None
    cfg._backfill_symbols = None
    deep = _backfill_trading_status_st(cfg, date(2026, 8, 7), "run2")
    assert len(calls) == 2  # second fetch happened
    assert deep["scope_id"] is None  # legacy default scope
    assert "already ST-backfilled" not in deep.get("note", "")


def test_different_scope_does_not_poison_another_scope(tmp_path, monkeypatch):
    cfg = _scope_cfg(
        tmp_path,
        ["600000.SH", "000001.SZ"],
        _backfill_start=date(2026, 3, 30),
        _backfill_end=date(2026, 8, 7),
        _backfill_symbols=["600000.SH"],
    )
    _written, calls = _patch(monkeypatch, returns=(pl.DataFrame(), []))
    first = _backfill_trading_status_st(cfg, date(2026, 8, 7), "run1")

    cfg._backfill_symbols = ["000001.SZ"]  # same dates, different symbols
    second = _backfill_trading_status_st(cfg, date(2026, 8, 7), "run2")

    assert first["scope_id"] != second["scope_id"]
    assert len(calls) == 2  # 000001.SZ was NOT suppressed by scope 1


def test_legacy_checkpoint_only_counts_for_legacy_scope(tmp_path, monkeypatch):
    cfg = _scope_cfg(tmp_path, ["600000.SH", "000001.SZ"])
    # Legacy file claims 600000.SH done for the default scope.
    path = reference._st_backfill_state_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"completed": ["600000.SH"]}))

    df = pl.DataFrame([_st_row("000001.SZ", date(2020, 5, 6))])
    _written, calls = _patch(monkeypatch, returns=(df, []))
    legacy = _backfill_trading_status_st(cfg, date(2026, 7, 1), "run1")
    assert len(calls) == 1  # only 000001.SZ todo (600000.SH skipped by legacy)
    assert legacy["scope_id"] is None

    # Custom scope: legacy completion must NOT be reused.
    cfg._backfill_start = date(2026, 3, 30)
    cfg._backfill_end = date(2026, 8, 7)
    cfg._backfill_symbols = ["600000.SH"]
    _backfill_trading_status_st(cfg, date(2026, 8, 7), "run2")
    assert len(calls) == 2  # 600000.SH queried again for the custom scope


def test_st_output_semantics_unchanged(tmp_path, monkeypatch):
    cfg = _scope_cfg(
        tmp_path,
        ["600000.SH"],
        _backfill_start=date(2026, 3, 30),
        _backfill_end=date(2026, 8, 7),
        _backfill_symbols=["600000.SH"],
    )
    df = pl.DataFrame([_st_row("600000.SH", date(2026, 5, 6))])
    written, _calls = _patch(monkeypatch, returns=(df, []))

    _backfill_trading_status_st(cfg, date(2026, 8, 7), "run1")

    row = written[0]
    # Fetch-level ST semantics are passed through unchanged (write_fetched
    # stamps source=baostock in production).
    assert row["status"].to_list() == ["st"]
    assert row["is_trading"].to_list() == [True]
    assert row["symbol"].to_list() == ["600000.SH"]


def test_scope_checkpoint_file_is_per_scope(tmp_path, monkeypatch):
    cfg = _scope_cfg(
        tmp_path,
        ["600000.SH"],
        _backfill_start=date(2026, 3, 30),
        _backfill_end=date(2026, 8, 7),
        _backfill_symbols=["600000.SH"],
    )
    _patch(monkeypatch, returns=(pl.DataFrame(), []))
    result = _backfill_trading_status_st(cfg, date(2026, 8, 7), "run1")

    path = _st_scope_checkpoint_path(cfg, result["scope_id"])
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["scope_version"] == 1
    assert payload["start"] == "2026-03-30"
    assert payload["end"] == "2026-08-07"
    assert payload["symbol_hash"]
    assert payload["completed"] == ["600000.SH"]
