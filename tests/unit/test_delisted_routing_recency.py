"""Delist-aware routing + recency contract on the current upstream.

Ownership contract under test: for a historical window ``[start, end]`` the
generic TDX/EastMoney path only receives symbols active through ``end``;
formally delisted symbols (instrument authority) are known-delisted with an
explicit state; probe-only recent terminals stay quarantined (30-day rule) and
keep coverage fail-closed.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from ashare_lake.config import Config, load_config
from ashare_lake.config.bootstrap import path_for_toml
from ashare_lake.orchestrator.manifest import Manifest
from ashare_lake.steps import bars, delisted
from ashare_lake.steps.common import classify_daily_routing
from ashare_lake.steps.delisted import (
    catalog_path,
    delisted_coverage_report,
    delisted_symbols_in_window,
    known_delisted_instruments,
    load_live_missing,
    pending_codes,
)
from ashare_lake.storage import StagingWriter
from ashare_lake.storage.layout import init_data_layout

WINDOW_START = date(2023, 8, 7)
WINDOW_END = date(2026, 8, 7)

LIVE = "600519.SH"
ACTIVE_AT_END = "000001.SZ"
A_SYMBOL = "600002.SH"  # delisted before window start
B_SYMBOL = "600070.SH"  # delisted inside the window
FUTURE_LISTING = "688001.SH"  # listed after window end


def _spans() -> dict[str, tuple[date | None, date | None]]:
    return {
        LIVE: (date(2001, 8, 27), None),
        ACTIVE_AT_END: (date(1991, 4, 3), date(2027, 6, 1)),
        A_SYMBOL: (date(1998, 1, 22), date(2006, 4, 6)),
        B_SYMBOL: (date(1997, 3, 1), date(2024, 5, 10)),
        FUTURE_LISTING: (date(2027, 1, 1), None),
    }


def _instrument_rows() -> list[dict]:
    spans = _spans()
    return [
        {
            "symbol": sym,
            "name": f"Mock-{sym}",
            "exchange": sym.split(".")[1],
            "asset_type": "stock",
            "list_date": spans[sym][0],
            "delist_date": spans[sym][1],
            "prev_symbol": None,
        }
        for sym in spans
    ]


@pytest.fixture
def cfg(tmp_path):
    cfg_path = tmp_path / "test.toml"
    cfg_path.write_text(
        f"""
[data]
root = "{path_for_toml(tmp_path / "data")}"

[orchestrator]
workers = 1
batch_size = 100

[tdx_protocol]
allow_mock = true
"""
    )
    return load_config(cfg_path)


def _write_instruments(cfg, rows: list[dict]) -> None:
    df = pl.DataFrame(
        rows,
        schema={
            "symbol": pl.Utf8,
            "name": pl.Utf8,
            "exchange": pl.Utf8,
            "asset_type": pl.Utf8,
            "list_date": pl.Date,
            "delist_date": pl.Date,
            "prev_symbol": pl.Utf8,
        },
    )
    out_dir = cfg.staging_root / "instruments" / "run_id=test"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_dir / "part-0.parquet", compression="zstd")


def _staged_daily_symbols(cfg, run_id: str) -> set[str]:
    files = StagingWriter(cfg.staging_root).list_run_files("daily_bars", run_id)
    if not files:
        return set()
    return set(
        pl.scan_parquet([str(f) for f in files])
        .select("symbol")
        .unique()
        .collect()["symbol"]
        .to_list()
    )


def _start_failed_batch(
    manifest: Manifest,
    run_id: str,
    batch_id: str,
    symbols: list[str],
) -> None:
    manifest.start_batch(
        run_id,
        batch_id,
        task_id="daily_bars",
        dataset="daily_bars",
        symbols=symbols,
        window_start=WINDOW_START.isoformat(),
        window_end=WINDOW_END.isoformat(),
    )
    manifest.finish_batch(
        run_id,
        batch_id,
        "failed",
        error_message="daily_bars: TDX returned no bars (set [tdx_protocol].allow_mock for tests)",
    )


def _write_curated_instruments(cfg, rows: list[tuple[str, date | None]]) -> None:
    root = cfg.curated_root / "instruments"
    root.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "symbol": [row[0] for row in rows],
            "delist_date": pl.Series([row[1] for row in rows], dtype=pl.Date),
        }
    ).write_parquet(root / "part-merged.parquet")


def _write_anchor_bar(cfg) -> None:
    """Anchor the catalogue reference date to the frozen window end."""
    part = cfg.curated_root / "daily_bars" / f"trade_date={WINDOW_END.isoformat()}"
    part.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {"symbol": ["600519.SH"], "trade_date": [WINDOW_END], "volume": [100]}
    ).write_parquet(part / "part-merged.parquet")


def _write_bar(cfg, symbol: str, *days: date, volume: int = 100) -> None:
    for day in days:
        part = cfg.curated_root / "daily_bars" / f"trade_date={day.isoformat()}"
        part.mkdir(parents=True, exist_ok=True)
        path = part / "part-merged.parquet"
        incoming = pl.DataFrame({"symbol": [symbol], "trade_date": [day], "volume": [volume]})
        if path.exists():
            incoming = pl.concat([pl.read_parquet(path), incoming])
        incoming.write_parquet(path)


def _write_catalog(cfg, catalog: dict[str, str]) -> None:
    path = catalog_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"delisted": catalog, "never_issued": [], "version": 1}))


# --- routing -----------------------------------------------------------------


def test_classifier_routes_a_b_active_and_future():
    spans = _spans()
    routing = classify_daily_routing(list(spans), spans, WINDOW_START, WINDOW_END)
    assert routing.included == [LIVE, ACTIVE_AT_END]
    assert routing.excluded_expected_no_data == [A_SYMBOL, FUTURE_LISTING]
    assert routing.excluded_delegated_delisted == [B_SYMBOL]
    assert routing.excluded_future_listing == [FUTURE_LISTING]


def test_classifier_unknown_span_treated_active_never_silently_dropped():
    routing = classify_daily_routing(["999999.SH"], {}, WINDOW_START, WINDOW_END)
    assert routing.included == ["999999.SH"]
    assert routing.excluded == []


def test_classifier_delist_exactly_at_end_is_delegated():
    spans = {"600070.SH": (date(1997, 3, 1), WINDOW_END)}
    routing = classify_daily_routing(["600070.SH"], spans, WINDOW_START, WINDOW_END)
    assert routing.excluded_delegated_delisted == ["600070.SH"]


def test_fresh_backfill_fetches_only_generic_symbols(cfg):
    init_data_layout(cfg)
    run_id = Manifest(cfg.manifest_path).start_run("test")
    _write_instruments(cfg, _instrument_rows())
    cfg._backfill = True
    cfg._backfill_start = WINDOW_START
    cfg._backfill_end = WINDOW_END

    out = bars.step_daily_bars(cfg, WINDOW_END, run_id, {})

    assert _staged_daily_symbols(cfg, run_id) == {LIVE, ACTIVE_AT_END}
    assert out["context_updates"]["daily_bars_routing"] == {
        "generic_included": 2,
        "excluded_expected_no_data": 2,
        "excluded_delegated_delisted": 1,
        "excluded_future_listing": 1,
    }
    findings = out["context_updates"]["audit_findings"]
    assert any(f["check"] == "daily_bars_delist_aware_routing" for f in findings)


# --- retry -------------------------------------------------------------------


def test_retry_all_delisted_batch_resolves_without_vendor(cfg):
    init_data_layout(cfg)
    manifest = Manifest(cfg.manifest_path)
    run_id = manifest.start_run("test")
    _write_instruments(cfg, [r for r in _instrument_rows() if r["symbol"] in (A_SYMBOL, B_SYMBOL)])
    batch_id = "2023-08-07_2026-08-07-batch-0"
    symbols = [A_SYMBOL, B_SYMBOL]
    _start_failed_batch(manifest, run_id, batch_id, symbols)

    out = bars.step_daily_bars(
        cfg,
        WINDOW_END,
        run_id,
        {"_retry_batch_specs": [(batch_id, symbols, WINDOW_START, WINDOW_END)]},
    )

    row = manifest.get_batch(run_id, batch_id)
    assert row["status"] == "success"
    assert row["rows_written"] == 0
    assert "ROUTED_OUT_OF_GENERIC" in row["error_message"]
    assert "expected_no_data=1" in row["error_message"]
    assert "delegated_delisted=1" in row["error_message"]
    # Original batch identity/symbol provenance retained; no vendor rows.
    assert json.loads(row["symbols_json"]) == symbols
    assert _staged_daily_symbols(cfg, run_id) == set()
    assert manifest.get_batch(run_id, f"{batch_id}-delegated") is None
    assert out["context_updates"]["daily_bars_routing"]["generic_included"] == 0


def test_retry_mixed_batch_fetches_generic_only_and_records_delegated(cfg):
    init_data_layout(cfg)
    manifest = Manifest(cfg.manifest_path)
    run_id = manifest.start_run("test")
    _write_instruments(
        cfg, [r for r in _instrument_rows() if r["symbol"] in (LIVE, A_SYMBOL, B_SYMBOL)]
    )
    batch_id = "2023-08-07_2026-08-07-batch-0"
    symbols = [LIVE, A_SYMBOL, B_SYMBOL]
    _start_failed_batch(manifest, run_id, batch_id, symbols)

    out = bars.step_daily_bars(
        cfg,
        WINDOW_END,
        run_id,
        {"_retry_batch_specs": [(batch_id, symbols, WINDOW_START, WINDOW_END)]},
    )

    assert _staged_daily_symbols(cfg, run_id) == {LIVE}
    row = manifest.get_batch(run_id, batch_id)
    assert row["status"] == "success"
    assert json.loads(row["symbols_json"]) == [LIVE]
    sibling = manifest.get_batch(run_id, f"{batch_id}-delegated")
    assert sibling is not None
    assert sibling["status"] == "success"
    assert sibling["rows_written"] == 0
    assert "ROUTED_OUT_OF_GENERIC" in sibling["error_message"]
    assert set(json.loads(sibling["symbols_json"])) == {A_SYMBOL, B_SYMBOL}
    counts = out["context_updates"]["daily_bars_routing"]
    assert counts["generic_included"] == 1
    assert counts["excluded_expected_no_data"] == 1
    assert counts["excluded_delegated_delisted"] == 1


# --- discovery live-set ------------------------------------------------------


def test_pending_codes_does_not_mask_delisted_as_live(cfg):
    init_data_layout(cfg)
    _write_instruments(cfg, [r for r in _instrument_rows() if r["symbol"] in (LIVE, A_SYMBOL)])

    pending = pending_codes(cfg)

    assert LIVE not in pending, "a genuinely active instrument stays excluded from discovery"
    assert A_SYMBOL in pending, "a known delisted symbol must become a discovery candidate"


# --- recency contract (real runtime semantics, 30-day rule kept) -------------


def test_probe_only_recent_terminal_is_quarantined(tmp_path, monkeypatch):
    cfg = Config(data_root=tmp_path / "data", sources={"sina": True})
    _write_catalog(cfg, {"600099.SH": "2026-07-20"})  # probe-only, inside 30-day window
    _write_anchor_bar(cfg)
    monkeypatch.setattr("ashare_lake.steps.delisted.pending_codes", lambda cfg: [])

    live_missing = load_live_missing(cfg)
    assert "600099.SH" in live_missing, "probe-only recent terminal stays quarantined"
    assert "600099.SH" not in known_delisted_instruments(cfg, WINDOW_END)

    report = delisted_coverage_report(cfg, WINDOW_START, WINDOW_END)
    assert report["counts"]["recent_quarantined"] == 1
    assert report["samples"]["recent_quarantined"][0]["basis"] == "probe_only"
    assert report["verified"] is False


def test_known_delisted_quarantined_still_blocks_coverage(tmp_path, monkeypatch):
    cfg = Config(data_root=tmp_path / "data", sources={"sina": True})
    _write_curated_instruments(cfg, [("600001.SH", date(2026, 7, 14))])
    _write_catalog(cfg, {"600001.SH": "2026-07-17"})  # probe last within 30 days
    _write_anchor_bar(cfg)
    monkeypatch.setattr("ashare_lake.steps.delisted.pending_codes", lambda cfg: [])

    assert known_delisted_instruments(cfg, WINDOW_END) == {"600001.SH": date(2026, 7, 14)}
    report = delisted_coverage_report(cfg, WINDOW_START, WINDOW_END)

    assert report["counts"]["known_delisted_instruments"] == 1
    assert report["counts"]["known_delisted_in_window"] == 1
    assert report["counts"]["recent_quarantined"] == 1
    entry = report["samples"]["recent_quarantined"][0]
    assert entry["symbol"] == "600001.SH"
    assert entry["basis"] == "known_delisted_quarantined"
    assert report["verified"] is False


def test_known_delisted_covered_becomes_verified(tmp_path, monkeypatch):
    cfg = Config(data_root=tmp_path / "data", sources={"sina": True})
    _write_curated_instruments(cfg, [("600002.SH", date(2024, 5, 10))])
    _write_catalog(cfg, {"600002.SH": "2024-05-10"})
    _write_bar(cfg, "600002.SH", date(2023, 9, 1), date(2024, 5, 10))
    _write_anchor_bar(cfg)
    monkeypatch.setattr("ashare_lake.steps.delisted.pending_codes", lambda cfg: [])

    report = delisted_coverage_report(cfg, WINDOW_START, WINDOW_END)

    assert report["counts"]["known_delisted_in_window"] == 1
    assert report["counts"]["proven_overlap"] == 1
    assert report["counts"]["missing_bars"] == 0
    assert report["counts"]["recent_quarantined"] == 0
    assert report["verified"] is True


def test_known_delisted_with_missing_bars_is_not_verified(tmp_path, monkeypatch):
    cfg = Config(data_root=tmp_path / "data", sources={"sina": True})
    _write_curated_instruments(cfg, [("600003.SH", date(2024, 6, 3))])
    _write_catalog(cfg, {"600003.SH": "2024-06-03"})
    _write_anchor_bar(cfg)
    monkeypatch.setattr("ashare_lake.steps.delisted.pending_codes", lambda cfg: [])

    report = delisted_coverage_report(cfg, WINDOW_START, WINDOW_END)

    assert report["counts"]["missing_bars"] == 1
    assert report["counts"]["recent_quarantined"] == 0
    assert report["counts"]["known_delisted_unreconciled"] == 0
    assert report["verified"] is False


def test_known_delisted_unreconciled_when_catalog_lacks_it(tmp_path, monkeypatch):
    cfg = Config(data_root=tmp_path / "data", sources={"sina": True})
    _write_curated_instruments(cfg, [("600004.SH", date(2024, 6, 3))])
    _write_catalog(cfg, {})  # discovery never filed it
    _write_anchor_bar(cfg)
    monkeypatch.setattr("ashare_lake.steps.delisted.pending_codes", lambda cfg: [])

    report = delisted_coverage_report(cfg, WINDOW_START, WINDOW_END)

    assert report["counts"]["known_delisted_unreconciled"] == 1
    assert report["samples"]["known_delisted_unreconciled"][0]["reason"] == "not_discovered"
    assert report["verified"] is False


def test_classify_catalog_keeps_30_day_quarantine():
    """Regression: LIVE_RECENCY_DAYS stays 30 in runtime semantics."""
    assert delisted.LIVE_RECENCY_DAYS == 30


# --- window semantics --------------------------------------------------------


def test_delisted_symbols_in_window_requires_b_and_skips_a(tmp_path):
    cfg = Config(data_root=tmp_path / "data", sources={"sina": True})
    _write_catalog(
        cfg,
        {
            A_SYMBOL: "2006-04-06",
            B_SYMBOL: "2024-05-10",
        },
    )
    _write_anchor_bar(cfg)

    assert delisted_symbols_in_window(cfg, WINDOW_START) == [B_SYMBOL]


# --- frozen classification replay (read-only, real runtime semantics) --------

REAL_TSV = Path(
    "/Users/luke808/AI/shared-asl-init-failed-daily-bars-ab-classification-0280a169-v01.tsv"
)


def test_frozen_replay_real_runtime_semantics(tmp_path, monkeypatch):
    """Frozen 336 population under the exact runtime contract (no rule disabled)."""
    if not REAL_TSV.exists():
        pytest.skip("real classification TSV not present on this machine")
    spans: dict[str, tuple[date | None, date | None]] = {}
    for line in REAL_TSV.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or line.startswith("symbol"):
            continue
        symbol, _cls, _basis, list_date, delist_date = line.split("\t")
        spans[symbol] = (date.fromisoformat(list_date), date.fromisoformat(delist_date))
    assert len(spans) == 336

    # Routing view: instrument authority is decisive for every known delisting.
    routing = classify_daily_routing(list(spans), spans, WINDOW_START, WINDOW_END)
    assert routing.included == []
    assert len(routing.excluded_expected_no_data) == 229
    assert len(routing.excluded_delegated_delisted) == 107
    assert routing.excluded_future_listing == []

    # Recency-edge regression expectation derived from the frozen TSV:
    # probe-only semantics would quarantine delist dates within 30 days of the
    # window end (2026-07-08..2026-08-07). Runtime must NOT hardcode this list.
    cutoff = WINDOW_END - timedelta(days=delisted.LIVE_RECENCY_DAYS)
    recent = sorted(
        sym
        for sym, (_list, delist_date) in spans.items()
        if WINDOW_START <= delist_date <= WINDOW_END and delist_date > cutoff
    )
    assert recent == ["000004.SZ", "002808.SZ", "002898.SZ", "300029.SZ"]

    # Coverage gate view with the real 30-day rule: 4 recency-edge names are
    # quarantined at the catalogue level, keeping the gate fail-closed.
    cfg = Config(data_root=tmp_path / "data", sources={"sina": True})
    _write_curated_instruments(cfg, [(sym, spans[sym][1]) for sym in spans])
    _write_catalog(cfg, {sym: spans[sym][1].isoformat() for sym in spans})
    _write_anchor_bar(cfg)
    monkeypatch.setattr("ashare_lake.steps.delisted.pending_codes", lambda cfg: [])

    report = delisted_coverage_report(cfg, WINDOW_START, WINDOW_END)
    counts = report["counts"]
    assert counts["known_delisted_instruments"] == 336
    assert counts["known_delisted_in_window"] == 107
    assert counts["expected_no_data"] == 229
    assert counts["recent_quarantined"] == 4
    assert counts["known_delisted_unreconciled"] == 0
    assert counts["missing_bars"] == 103
    assert report["verified"] is False
    quarantined_symbols = {s["symbol"] for s in report["samples"]["recent_quarantined"]}
    assert quarantined_symbols == set(recent)
    assert all(
        s["basis"] == "known_delisted_quarantined" for s in report["samples"]["recent_quarantined"]
    )
