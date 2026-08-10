"""Formal delisting identity vs last-traded window-overlap contract.

``instruments.delist_date`` is identity/legal authority; it does NOT by itself
prove a security traded inside ``[start, as_of]``. The last-traded authority
(observed bars, then catalogue/Sina terminal) decides overlap: a formal
candidate that stopped trading before ``start`` is EXPECTED_NO_DATA, not
MISSING_BARS and not a recovery target.
"""

from __future__ import annotations

import json
from datetime import date

import polars as pl
import pytest

from ashare_lake.config import Config
from ashare_lake.steps import delisted
from ashare_lake.steps.delisted import (
    _ingested_symbols,
    backfill_delisted_bars,
    catalog_path,
    delisted_backfill_targets,
    delisted_coverage_report,
    known_delisted_instruments,
)

START = date(2023, 8, 7)
AS_OF = date(2026, 8, 7)


def _cfg(tmp_path, *, instruments: dict[str, date], catalog: dict[str, str]) -> Config:
    cfg = Config(data_root=tmp_path / "data", sources={"sina": True})
    root = cfg.curated_root / "instruments"
    root.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "symbol": list(instruments),
            "list_date": pl.Series([date(1998, 1, 1)] * len(instruments), dtype=pl.Date),
            "delist_date": pl.Series(list(instruments.values()), dtype=pl.Date),
        }
    ).write_parquet(root / "part-merged.parquet")
    _write_catalog(cfg, catalog)
    _anchor(cfg)
    return cfg


def _write_catalog(cfg, catalog: dict[str, str]) -> None:
    path = catalog_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"delisted": catalog, "never_issued": []}))


def _anchor(cfg) -> None:
    part = cfg.curated_root / "daily_bars" / f"trade_date={AS_OF.isoformat()}"
    part.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"symbol": ["600519.SH"], "trade_date": [AS_OF], "volume": [100]}).write_parquet(
        part / "part-merged.parquet"
    )


def _write_bar(cfg, symbol: str, day: date) -> None:
    part = cfg.curated_root / "daily_bars" / f"trade_date={day.isoformat()}"
    part.mkdir(parents=True, exist_ok=True)
    path = part / "part-merged.parquet"
    incoming = pl.DataFrame({"symbol": [symbol], "trade_date": [day], "volume": [100]})
    if path.exists():
        incoming = pl.concat([pl.read_parquet(path), incoming])
    incoming.write_parquet(path)


def _empty_bars() -> pl.DataFrame:
    from ashare_lake.domain.schemas import DAILY_BARS_SCHEMA

    cols = [c for c in DAILY_BARS_SCHEMA if c not in ("source", "data_version", "fetched_at")]
    return pl.DataFrame(schema={c: DAILY_BARS_SCHEMA[c] for c in cols})


# --- targeting --------------------------------------------------------------


def test_formal_delist_after_start_last_trade_before_start_is_not_target(tmp_path):
    cfg = _cfg(
        tmp_path,
        instruments={"600001.SH": date(2023, 8, 23)},
        catalog={"600001.SH": "2023-06-13"},  # authoritative terminal before start
    )

    assert delisted_backfill_targets(cfg, START, AS_OF) == []


def test_formal_delist_after_start_with_overlap_is_target(tmp_path):
    cfg = _cfg(
        tmp_path,
        instruments={"600001.SH": date(2023, 8, 23)},
        catalog={"600001.SH": "2023-10-02"},  # terminal inside window
    )

    assert delisted_backfill_targets(cfg, START, AS_OF) == ["600001.SH"]


def test_formal_no_terminal_evidence_is_still_target(tmp_path):
    cfg = _cfg(tmp_path, instruments={"600001.SH": date(2023, 8, 23)}, catalog={})

    assert delisted_backfill_targets(cfg, START, AS_OF) == ["600001.SH"]


# --- empty-fetch probe resolution -------------------------------------------


def test_empty_fetch_probe_prewindow_resolves_no_overlap(tmp_path):
    cfg = _cfg(
        tmp_path,
        instruments={"600001.SH": date(2023, 8, 23)},
        catalog={},
    )

    result = backfill_delisted_bars(
        cfg,
        "run-1",
        START,
        fetch=lambda s, c: _empty_bars(),
        probe_last=lambda s, c: date(2023, 6, 13),
    )

    assert result["no_overlap_symbols"] == 1
    assert result["recovered"] == 0
    assert result.get("empty_symbols", 0) == 0
    assert result.get("failed_symbols", 0) == 0
    assert "600001.SH" not in _ingested_symbols(cfg)
    # Terminal persisted to the atomic catalogue; next run excludes the symbol.
    assert delisted_backfill_targets(cfg, START, AS_OF) == []


def test_empty_fetch_probe_inwindow_is_inconsistent_unresolved(tmp_path):
    cfg = _cfg(
        tmp_path,
        instruments={"600001.SH": date(2023, 8, 23)},
        catalog={},
    )

    result = backfill_delisted_bars(
        cfg,
        "run-1",
        START,
        fetch=lambda s, c: _empty_bars(),
        probe_last=lambda s, c: date(2023, 10, 2),
    )

    assert result["empty_symbols"] == 1
    assert "600001.SH" not in _ingested_symbols(cfg)
    assert "600001.SH" in delisted_backfill_targets(cfg, START, AS_OF)


def test_empty_fetch_probe_none_is_unresolved_not_never_issued(tmp_path):
    cfg = _cfg(
        tmp_path,
        instruments={"600001.SH": date(2023, 8, 23)},
        catalog={},
    )

    result = backfill_delisted_bars(
        cfg,
        "run-1",
        START,
        fetch=lambda s, c: _empty_bars(),
        probe_last=lambda s, c: None,
    )

    assert result["empty_symbols"] == 1
    assert "600001.SH" not in _ingested_symbols(cfg)
    assert "600001.SH" in delisted_backfill_targets(cfg, START, AS_OF)


def test_empty_fetch_probe_exception_is_unresolved(tmp_path):
    cfg = _cfg(
        tmp_path,
        instruments={"600001.SH": date(2023, 8, 23)},
        catalog={},
    )

    def boom(symbol, client):
        raise ConnectionError("probe down")

    result = backfill_delisted_bars(
        cfg,
        "run-1",
        START,
        fetch=lambda s, c: _empty_bars(),
        probe_last=boom,
    )

    assert result["empty_symbols"] == 1
    assert "600001.SH" not in _ingested_symbols(cfg)


def test_no_overlap_symbol_never_enters_ingested(tmp_path):
    cfg = _cfg(
        tmp_path,
        instruments={"600001.SH": date(2023, 8, 23)},
        catalog={},
    )

    backfill_delisted_bars(
        cfg,
        "run-1",
        START,
        fetch=lambda s, c: _empty_bars(),
        probe_last=lambda s, c: date(2023, 6, 13),
    )

    assert "600001.SH" not in _ingested_symbols(cfg)
    payload = json.loads((cfg.meta_root / "state" / "delisted_catalog.json").read_text())
    assert payload["delisted"]["600001.SH"] == "2023-06-13"


# --- coverage contract ------------------------------------------------------


def test_formal_with_bars_and_no_catalog_is_proven_overlap(tmp_path, monkeypatch):
    """Observed bars prove overlap; absent catalogue must not make it unreconciled."""
    cfg = _cfg(tmp_path, instruments={"600001.SH": date(2024, 5, 10)}, catalog={})
    _write_bar(cfg, "600001.SH", date(2024, 5, 10))
    monkeypatch.setattr("ashare_lake.steps.delisted.pending_codes", lambda cfg: [])

    report = delisted_coverage_report(cfg, START, AS_OF)

    assert report["counts"]["known_formal_candidates"] == 1
    assert report["counts"]["proven_overlap"] == 1
    assert report["counts"]["known_delisted_unreconciled"] == 0
    assert report["counts"]["missing_bars"] == 0
    assert report["known_coverage_complete"] is True


def test_formal_prewindow_terminal_is_expected_no_data_not_missing_bars(tmp_path, monkeypatch):
    cfg = _cfg(
        tmp_path,
        instruments={"600001.SH": date(2023, 8, 23)},
        catalog={"600001.SH": "2023-06-13"},
    )
    monkeypatch.setattr("ashare_lake.steps.delisted.pending_codes", lambda cfg: [])

    report = delisted_coverage_report(cfg, START, AS_OF)

    assert report["counts"]["formal_no_overlap"] == 1
    assert report["counts"]["missing_bars"] == 0
    assert report["counts"]["known_overlap_required"] == 0
    assert report["counts"]["known_delisted_unreconciled"] == 0
    assert report["known_coverage_complete"] is True


def test_formal_required_bars_missing_is_missing_bars(tmp_path, monkeypatch):
    cfg = _cfg(
        tmp_path,
        instruments={"600001.SH": date(2023, 8, 23)},
        catalog={"600001.SH": "2023-10-02"},
    )
    monkeypatch.setattr("ashare_lake.steps.delisted.pending_codes", lambda cfg: [])

    report = delisted_coverage_report(cfg, START, AS_OF)

    assert report["counts"]["known_overlap_required"] == 1
    assert report["counts"]["missing_bars"] == 1
    assert report["known_coverage_complete"] is False


def test_full_discovery_stays_separate_from_known_coverage(tmp_path):
    """KNOWN_COVERAGE_COMPLETE can be true while DISCOVERY_COMPLETE is false."""
    cfg = _cfg(
        tmp_path,
        instruments={"600001.SH": date(2024, 5, 10)},
        catalog={"600001.SH": "2024-05-10"},
    )
    _write_bar(cfg, "600001.SH", date(2024, 5, 10))

    report = delisted_coverage_report(cfg, START, AS_OF)

    assert report["known_coverage_complete"] is True
    assert report["discovery_complete"] is False  # issued-code sweep not done
    assert report["verified"] is False


# --- real state read-only replay (injected frozen evidence) -----------------

REAL_CONFIG = "/Users/luke808/AI/asl-shared-config.toml"
REAL_MANIFEST = "/Users/luke808/AI/asl-shared/meta/manifest.db"
# Frozen regression evidence: five formal candidates whose vendor history ends
# before the window (external exchange evidence; used only in the replay/tests).
FROZEN_NO_OVERLAP = {
    "000616.SZ": "2023-06-13",
    "000671.SZ": "2023-06-13",
    "002113.SZ": "2023-06-13",
    "002503.SZ": "2023-06-13",
    "002504.SZ": "2023-06-13",
}


def test_real_run_readonly_overlap_replay(monkeypatch):
    """Read-only replay on the real shared root with injected last-traded evidence."""
    import os

    from ashare_lake.config import load_config

    if not os.path.exists(REAL_MANIFEST):
        pytest.skip("real shared root not present")
    cfg = load_config(REAL_CONFIG)

    ingested = _ingested_symbols(cfg)
    remaining = delisted_backfill_targets(cfg, START, AS_OF)
    known = known_delisted_instruments(cfg, AS_OF)
    known_in_window = [s for s, d in known.items() if d >= START]
    assert len(known_in_window) == 107
    assert len(ingested) == 102
    assert set(remaining) == set(FROZEN_NO_OVERLAP)

    # Inject frozen pre-window terminals via the catalogue store (read-only on
    # real data: the catalogue itself is simulated for the replay).
    monkeypatch.setattr(
        delisted,
        "_read_catalog",
        lambda config: {"delisted": dict(FROZEN_NO_OVERLAP), "never_issued": []},
    )

    report = delisted_coverage_report(cfg, START, AS_OF)
    counts = report["counts"]
    assert counts["known_formal_candidates"] == 107
    assert counts["proven_overlap"] == 102
    assert counts["formal_no_overlap"] == 5
    assert counts["missing_bars"] == 0
    assert counts["known_delisted_unreconciled"] == 0
    assert report["known_coverage_complete"] is True
    assert report["discovery_complete"] is False  # pending issued-code probes
    assert report["verified"] is False

    patched_targets = delisted_backfill_targets(cfg, START, AS_OF)
    assert patched_targets == []  # 102 recovered ingested; 5 proven no-overlap
