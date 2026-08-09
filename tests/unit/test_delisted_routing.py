"""Delist-aware routing between generic daily backfill and delisted recovery.

Ownership contract under test: for a historical window ``[start, end]`` the
generic TDX/EastMoney path must only receive symbols still active through
``end``. Symbols delisted before ``start`` are expected-no-data; symbols
delisted inside the window are delegated to the dedicated delisted path.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from ashare_lake.config import Config, load_config
from ashare_lake.config.bootstrap import path_for_toml
from ashare_lake.orchestrator.manifest import Manifest
from ashare_lake.steps import bars, delisted
from ashare_lake.steps.common import classify_daily_routing
from ashare_lake.storage import StagingWriter
from ashare_lake.storage.layout import init_data_layout

WINDOW_START = date(2023, 8, 7)
WINDOW_END = date(2026, 8, 7)

LIVE = "600519.SH"  # active, no delist_date
ACTIVE_AT_END = "000001.SZ"  # delist_date after window end
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


# --- pure classifier ---------------------------------------------------------


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


# --- fresh generic path ------------------------------------------------------


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


# --- retry path --------------------------------------------------------------


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
    # Original batch identity/symbol provenance retained.
    assert json.loads(row["symbols_json"]) == symbols
    # No vendor rows staged, no sibling invented for the pure case.
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

    # Vendor receives only the generic-owned symbol.
    assert _staged_daily_symbols(cfg, run_id) == {LIVE}
    row = manifest.get_batch(run_id, batch_id)
    assert row["status"] == "success"
    assert json.loads(row["symbols_json"]) == [LIVE]
    # Excluded A/B recorded in an explicit sibling batch.
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
    _write_instruments(
        cfg, [r for r in _instrument_rows() if r["symbol"] in (LIVE, A_SYMBOL)]
    )

    pending = delisted.pending_codes(cfg)

    assert LIVE not in pending, "a genuinely active instrument stays excluded from discovery"
    assert A_SYMBOL in pending, "a known delisted symbol must become a discovery candidate"


# --- window semantics --------------------------------------------------------


def test_delisted_symbols_in_window_requires_b_and_skips_a(tmp_path):
    cfg = Config(data_root=tmp_path / "data", sources={"sina": True})
    state = cfg.meta_root / "state"
    state.mkdir(parents=True, exist_ok=True)
    (state / "delisted_catalog.json").write_text(
        json.dumps(
            {
                "delisted": {
                    A_SYMBOL: "2006-04-06",
                    B_SYMBOL: "2024-05-10",
                },
                "never_issued": [],
                "version": 1,
            }
        )
    )

    assert delisted.delisted_symbols_in_window(cfg, WINDOW_START) == [B_SYMBOL]


# --- classification replay against the real 0280a169 population -------------

REAL_TSV = Path(
    "/Users/luke808/AI/shared-asl-init-failed-daily-bars-ab-classification-0280a169-v01.tsv"
)


def test_replay_real_0280_classification(tmp_path, monkeypatch):
    """Pure/synthetic replay of the frozen A=229 / B=107 population."""
    if not REAL_TSV.exists():
        pytest.skip("real classification TSV not present on this machine")
    spans: dict[str, tuple[date | None, date | None]] = {}
    for line in REAL_TSV.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or line.startswith("symbol"):
            continue
        symbol, _cls, _basis, list_date, delist_date = line.split("\t")
        spans[symbol] = (date.fromisoformat(list_date), date.fromisoformat(delist_date))

    assert len(spans) == 336
    routing = classify_daily_routing(list(spans), spans, WINDOW_START, WINDOW_END)
    assert routing.included == []
    assert len(routing.excluded_expected_no_data) == 229
    assert len(routing.excluded_delegated_delisted) == 107
    assert routing.excluded_future_listing == []

    # Catalog semantics: A must NOT be eligible for dedicated historical fetch;
    # B must be (frozen window semantics, no production constants).
    cfg = Config(data_root=tmp_path / "data", sources={"sina": True})
    state = cfg.meta_root / "state"
    state.mkdir(parents=True, exist_ok=True)
    (state / "delisted_catalog.json").write_text(
        json.dumps(
            {
                "delisted": {
                    sym: delist_date.isoformat() for sym, (_list_date, delist_date) in spans.items()
                },
                "never_issued": [],
                "version": 1,
            }
        )
    )
    monkeypatch.setattr(delisted, "LIVE_RECENCY_DAYS", 0)
    monkeypatch.setattr(delisted, "_reference_date", lambda config: WINDOW_END)
    in_window = delisted.delisted_symbols_in_window(cfg, WINDOW_START)
    assert len(in_window) == 107
    assert set(in_window) == set(routing.excluded_delegated_delisted)
