"""Ledger correctness for init resume: supersession + current-phase authority.

Two defects under test:

* Bug A — historical ``phase_results`` snapshots permanently poisoned the final
  init status even after their batches were recovered; final status must derive
  from CURRENT manifest batch state.
* Bug B — a successfully retried non-worker step left its old failed/stale
  batch permanently incomplete; the retry must supersede the old attempt while
  keeping it auditable.
"""

from __future__ import annotations

import sqlite3
from datetime import date

import pytest

import ashare_lake.orchestrator.engine as eng_mod
from ashare_lake.config import Config
from ashare_lake.orchestrator.compact_gate import compact_allowed
from ashare_lake.orchestrator.engine import JobEngine
from ashare_lake.orchestrator.init_phases import (
    current_phase_statuses,
    init_run_complete,
    needs_finalize,
    step_incomplete,
    step_succeeded,
)
from ashare_lake.orchestrator.manifest import Manifest
from ashare_lake.orchestrator.registry import StepEntry
from ashare_lake.storage.layout import init_data_layout

_PHASES = [
    "phase1_reference",
    "phase2c_daily_bars_backfill",
    "phase3_index_and_status",
    "phase4_finalize",
]


@pytest.fixture
def cfg(tmp_path):
    return Config(
        data_root=tmp_path / "data",
        init_phases=_PHASES,
        tdx_allow_mock=True,
    )


def _stub_get_step(monkeypatch, *, fail: set[str] | None = None, running: set[str] | None = None):
    """Replace all steps with offline stubs; worker flags off so retry is a step."""
    fail = fail or set()
    running = running or set()

    def _fn(name: str):
        def _call(config, trade_date, run_id, context):
            if name in fail:
                raise RuntimeError(f"simulated {name} failure")
            return {"rows_read": 1, "rows_written": 1}

        return StepEntry(fn=_call, group="test", requires_workers=False)

    monkeypatch.setattr(eng_mod, "get_step", _fn)


def _mk_run(manifest: Manifest, *, historical_failure: bool = False) -> str:
    meta = {"phases": _PHASES, "trade_date": "2024-06-28"}
    if historical_failure:
        meta["phase_results"] = [
            {"phase": "phase1_reference", "status": "success"},
            {"phase": "phase2c_daily_bars_backfill", "status": "failed"},
        ]
    return manifest.start_run("init", meta)


def _add_batch(
    manifest: Manifest,
    run_id: str,
    batch_id: str,
    dataset: str,
    status: str,
    *,
    error: str | None = None,
) -> None:
    manifest.start_batch(run_id, batch_id, task_id=dataset, dataset=dataset)
    manifest.finish_batch(
        run_id,
        batch_id,
        status,
        rows_read=1,
        rows_written=1,
        error_message=error,
    )


def _all_phase_steps_success(manifest: Manifest, run_id: str) -> None:
    for dataset in (
        "instruments",
        "trading_calendar",
        "daily_bars",
        "index_bars",
        "trading_status",
        "compact",
        "derive_adj_factors",
        "derive_industry_index",
        "audit",
    ):
        _add_batch(manifest, run_id, f"{dataset}-b1", dataset, "success")


# --- Bug B: retry supersession ----------------------------------------------


def test_failed_step_retry_supersedes_old_attempt(cfg, monkeypatch):
    init_data_layout(cfg)
    _stub_get_step(monkeypatch)
    manifest = Manifest(cfg.manifest_path)
    engine = JobEngine(cfg)
    run_id = _mk_run(manifest)
    _add_batch(manifest, run_id, "old-ts", "trading_status", "failed", error="provider down")

    result = engine._run_step("trading_status", date(2024, 6, 28), run_id, {}, retry_of="old-ts")

    assert result["status"] == "success"
    old = manifest.get_batch(run_id, "old-ts")
    assert old["status"] == "superseded"
    assert "superseded by retry batch" in old["error_message"]
    assert "provider down" in old["error_message"]  # prior error preserved for audit
    assert manifest.incomplete_batch_count(run_id) == 0
    batches = manifest.get_batches_for_run(run_id)
    assert step_succeeded(batches, "trading_status")
    assert not step_incomplete(batches, "trading_status")


def test_stale_step_retry_supersedes_old_attempt(cfg, monkeypatch):
    init_data_layout(cfg)
    _stub_get_step(monkeypatch)
    manifest = Manifest(cfg.manifest_path)
    engine = JobEngine(cfg)
    run_id = _mk_run(manifest)
    _add_batch(manifest, run_id, "old-ts", "trading_status", "stale", error="heartbeat timeout")

    result = engine._run_step("trading_status", date(2024, 6, 28), run_id, {}, retry_of="old-ts")

    assert result["status"] == "success"
    assert manifest.get_batch(run_id, "old-ts")["status"] == "superseded"
    assert manifest.incomplete_batch_count(run_id) == 0


def test_failed_retry_leaves_attempts_unresolved(cfg, monkeypatch):
    init_data_layout(cfg)
    _stub_get_step(monkeypatch, fail={"trading_status"})
    manifest = Manifest(cfg.manifest_path)
    engine = JobEngine(cfg)
    run_id = _mk_run(manifest)
    _add_batch(manifest, run_id, "old-ts", "trading_status", "failed", error="provider down")

    result = engine._run_step("trading_status", date(2024, 6, 28), run_id, {}, retry_of="old-ts")

    assert result["status"] == "failed"
    assert manifest.get_batch(run_id, "old-ts")["status"] == "failed"  # not superseded
    assert manifest.incomplete_batch_count(run_id) == 2
    batches = manifest.get_batches_for_run(run_id)
    assert not step_succeeded(batches, "trading_status")


def test_running_retry_is_not_resolved(cfg):
    init_data_layout(cfg)
    manifest = Manifest(cfg.manifest_path)
    run_id = _mk_run(manifest)
    _add_batch(manifest, run_id, "old-ts", "trading_status", "failed")
    manifest.start_batch(run_id, "new-ts", task_id="trading_status", dataset="trading_status")

    assert manifest.incomplete_batch_count(run_id) == 2
    batches = manifest.get_batches_for_run(run_id)
    assert not step_succeeded(batches, "trading_status")
    assert current_phase_statuses(_PHASES, batches)["phase3_index_and_status"] == "failed"


def test_unrelated_failed_batch_is_not_accidentally_resolved(cfg, monkeypatch):
    init_data_layout(cfg)
    _stub_get_step(monkeypatch)
    manifest = Manifest(cfg.manifest_path)
    engine = JobEngine(cfg)
    run_id = _mk_run(manifest)
    # trading_status retried successfully (superseded), audit failed and untouched.
    _add_batch(manifest, run_id, "old-ts", "trading_status", "failed")
    _add_batch(manifest, run_id, "audit-fail", "audit", "failed", error="real failure")

    engine._run_step("trading_status", date(2024, 6, 28), run_id, {}, retry_of="old-ts")

    assert manifest.get_batch(run_id, "old-ts")["status"] == "superseded"
    assert manifest.get_batch(run_id, "audit-fail")["status"] == "failed"
    assert manifest.incomplete_batch_count(run_id) == 1
    batches = manifest.get_batches_for_run(run_id)
    assert step_succeeded(batches, "trading_status")
    assert not step_succeeded(batches, "audit")


# --- Bug A: current phase authority -----------------------------------------


def test_historical_failure_does_not_poison_current_success(cfg):
    init_data_layout(cfg)
    manifest = Manifest(cfg.manifest_path)
    engine = JobEngine(cfg)
    run_id = _mk_run(manifest, historical_failure=True)
    _all_phase_steps_success(manifest, run_id)

    status = engine._finalize_init_run(
        run_id,
        [
            {"phase": "phase1_reference", "status": "success"},
            {"phase": "phase2c_daily_bars_backfill", "status": "failed"},
        ],
        rows_written=1,
    )

    assert status == "success"
    assert engine.manifest.get_run(run_id)["status"] == "success"
    meta = engine.manifest.get_run_metadata(run_id)
    # Historical snapshot remains auditable; current view is the authority.
    assert meta["phase_results"][1]["status"] == "failed"
    assert meta["current_phase_status"]["phase2c_daily_bars_backfill"] == "success"
    assert all(v == "success" for v in meta["current_phase_status"].values())


def test_current_batch_failure_still_fails_run(cfg):
    init_data_layout(cfg)
    manifest = Manifest(cfg.manifest_path)
    engine = JobEngine(cfg)
    run_id = _mk_run(manifest)
    _all_phase_steps_success(manifest, run_id)
    _add_batch(manifest, run_id, "ts-fail", "trading_status", "failed", error="still broken")

    status = engine._finalize_init_run(run_id, [], rows_written=1)

    assert status == "failed"
    assert engine.manifest.get_run(run_id)["status"] == "failed"
    meta = engine.manifest.get_run_metadata(run_id)
    assert meta["current_phase_status"]["phase3_index_and_status"] == "failed"


def test_running_retry_blocks_final_success(cfg):
    init_data_layout(cfg)
    manifest = Manifest(cfg.manifest_path)
    engine = JobEngine(cfg)
    run_id = _mk_run(manifest)
    _all_phase_steps_success(manifest, run_id)
    manifest.start_batch(run_id, "ts-running", task_id="trading_status", dataset="trading_status")

    status = engine._finalize_init_run(run_id, [], rows_written=1)

    assert status == "failed"


# --- resume flow (Bug A + B together) ---------------------------------------


def test_resume_reaches_success_after_retry_supersedes_failed_phase(cfg, monkeypatch):
    """Original phase2c failed; retry succeeds; FINAL_RUN_STATUS=success."""
    init_data_layout(cfg)
    _stub_get_step(monkeypatch)
    manifest = Manifest(cfg.manifest_path)
    engine = JobEngine(cfg)
    run_id = _mk_run(manifest, historical_failure=True)
    # phase1/phase3 batches already success; phase2c batch failed (to be retried).
    for ds in ("instruments", "trading_calendar", "index_bars", "trading_status"):
        _add_batch(manifest, run_id, f"{ds}-b1", ds, "success")
    _add_batch(manifest, run_id, "daily-old", "daily_bars", "failed", error="TDX no bars")

    result = engine.resume_init(date(2024, 6, 28), run_id=run_id)

    assert result["status"] == "success"
    assert manifest.get_run(run_id)["status"] == "success"
    assert manifest.get_batch(run_id, "daily-old")["status"] == "superseded"
    assert manifest.incomplete_batch_count(run_id) == 0
    meta = manifest.get_run_metadata(run_id)
    assert meta["current_phase_status"]["phase2c_daily_bars_backfill"] == "success"
    historical = [p for p in meta["phase_results"] if p["phase"] == "phase2c_daily_bars_backfill"]
    assert historical and historical[0]["status"] == "failed"  # auditable


def test_resume_stays_failed_when_retry_also_fails(cfg, monkeypatch):
    init_data_layout(cfg)
    _stub_get_step(monkeypatch, fail={"daily_bars"})
    manifest = Manifest(cfg.manifest_path)
    engine = JobEngine(cfg)
    run_id = _mk_run(manifest)
    for ds in ("instruments", "trading_calendar", "index_bars", "trading_status"):
        _add_batch(manifest, run_id, f"{ds}-b1", ds, "success")
    _add_batch(manifest, run_id, "daily-old", "daily_bars", "failed", error="TDX no bars")

    result = engine.resume_init(date(2024, 6, 28), run_id=run_id)

    assert result["status"] == "failed"
    assert manifest.get_run(run_id)["status"] == "failed"


# --- finalize gate / compact gate -------------------------------------------


def test_finalize_allowed_after_legitimate_supersession(cfg):
    init_data_layout(cfg)
    manifest = Manifest(cfg.manifest_path)
    run_id = _mk_run(manifest)
    for ds in ("instruments", "trading_calendar", "daily_bars", "index_bars", "trading_status"):
        _add_batch(manifest, run_id, f"{ds}-b1", ds, "success")
    # trading_status old attempt superseded by a successful retry.
    _add_batch(manifest, run_id, "old-ts", "trading_status", "failed")
    _add_batch(manifest, run_id, "new-ts", "trading_status", "success")
    manifest.supersede_batch(run_id, "old-ts", superseded_by="new-ts", prior_error="x")

    batches = manifest.get_batches_for_run(run_id)
    assert init_run_complete([p for p in _PHASES if p != "phase4_finalize"], batches)
    assert needs_finalize(_PHASES, batches)
    assert manifest.incomplete_batch_count(run_id) == 0


def test_compact_allowed_after_supersession_but_blocked_by_unresolved(cfg):
    init_data_layout(cfg)
    manifest = Manifest(cfg.manifest_path)
    run_id = _mk_run(manifest)
    _add_batch(manifest, run_id, "old-ts", "trading_status", "failed")
    _add_batch(manifest, run_id, "new-ts", "trading_status", "success")
    _add_batch(manifest, run_id, "audit-fail", "audit", "failed")
    manifest.supersede_batch(run_id, "old-ts", superseded_by="new-ts", prior_error="x")

    allowed_ts, _ = compact_allowed(manifest, run_id, "trading_status")
    allowed_audit, _ = compact_allowed(manifest, run_id, "audit")
    assert allowed_ts is True  # superseded attempt no longer blocks compact
    assert allowed_audit is False  # genuinely unresolved batch still blocks


# --- real run read-only replay ----------------------------------------------

REAL_MANIFEST = "/Users/luke808/AI/asl-shared/meta/manifest.db"
REAL_RUN_ID = "0280a169-e73d-4d0c-a358-d0637ed8ca99"


def _would_be_superseded(batches: list[dict]) -> set[str]:
    """Read-only estimate: failed/stale batch with a LATER same-dataset success."""
    resolved: set[str] = set()
    by_dataset: dict[str, list[dict]] = {}
    for b in batches:
        by_dataset.setdefault(b["dataset"], []).append(b)
    for rows in by_dataset.values():
        successes = [r for r in rows if r["status"] == "success" and r["started_at"]]
        if not successes:
            continue
        earliest = min(r["started_at"] for r in successes)
        for r in rows:
            if (
                r["status"] in ("failed", "stale")
                and r["started_at"]
                and r["started_at"] < earliest
            ):
                resolved.add(r["batch_id"])
    return resolved


def test_real_run_readonly_ledger_replay(tmp_path):
    """Read-only replay of run 0280a169 with the patched ledger semantics."""
    if not __import__("os").path.exists(REAL_MANIFEST):
        pytest.skip("real shared manifest not present")
    conn = sqlite3.connect(f"file:{REAL_MANIFEST}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        run = conn.execute(
            "SELECT * FROM ingestion_runs WHERE run_id = ?", (REAL_RUN_ID,)
        ).fetchone()
        assert run is not None
        rows = conn.execute(
            "SELECT * FROM ingestion_batches WHERE run_id = ?", (REAL_RUN_ID,)
        ).fetchall()
    finally:
        conn.close()

    batches = [dict(r) for r in rows]
    superseded = _would_be_superseded(batches)
    old_incomplete = sum(1 for b in batches if b["status"] not in ("success", "superseded"))
    true_unresolved = old_incomplete - len(superseded)
    assert old_incomplete == 1, batches
    assert true_unresolved == 0, batches

    expected_steps = [
        "instruments",
        "trading_calendar",
        "corporate_actions",
        "daily_bars",
        "index_bars",
        "trading_status",
        "compact",
        "derive_adj_factors",
        "derive_industry_index",
        "audit",
    ]
    current: dict[str, str] = {}
    for step in expected_steps:
        step_rows = [b for b in batches if b["dataset"] == step]
        unresolved = [
            b
            for b in step_rows
            if b["status"] not in ("success", "superseded") and b["batch_id"] not in superseded
        ]
        current[step] = "CURRENT_FAILED" if unresolved else "CURRENT_SUCCESS"
    assert current["daily_bars"] == "CURRENT_SUCCESS"
    assert current["trading_status"] == "CURRENT_SUCCESS"
    assert current["compact"] == "CURRENT_SUCCESS"
    assert current["derive_adj_factors"] == "CURRENT_SUCCESS"
    assert all(v == "CURRENT_SUCCESS" for v in current.values())
