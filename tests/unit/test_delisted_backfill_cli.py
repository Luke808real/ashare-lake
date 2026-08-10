"""`asl delisted backfill` run-status truthfulness (fail-closed ledger)."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

import ashare_lake.steps.delisted as delisted_mod
from ashare_lake.cli.main import cli
from ashare_lake.config import load_config
from ashare_lake.config.bootstrap import path_for_toml
from ashare_lake.orchestrator.engine import JobEngine
from ashare_lake.orchestrator.manifest import Manifest
from ashare_lake.storage.layout import init_data_layout


@pytest.fixture
def cfg_path(tmp_path) -> str:
    path = tmp_path / "test.toml"
    path.write_text(
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
    cfg = load_config(path)
    init_data_layout(cfg)
    return str(path)


def _result(*, failed=0, empty=0, recovered=0, rows=0, symbols=0, note=None) -> dict:
    out = {
        "rows_read": rows,
        "rows_written": rows,
        "symbols": symbols,
        "recovered": recovered,
        "ending_patterns": {},
    }
    if failed:
        out["failed_symbols"] = failed
    if empty:
        out["empty_symbols"] = empty
    if note:
        out["note"] = note
    return out


def _compact(status: str) -> dict:
    return {"status": status, "rows_written": 0}


def _invoke(cfg_path: str) -> tuple[int, dict]:
    runner = CliRunner()
    result = runner.invoke(
        cli, ["delisted", "backfill", "--config", cfg_path, "--since", "2023-08-07"]
    )
    # JSON is printed before the ClickException message on incomplete runs.
    payload = json.loads(result.output[: result.output.rindex("}") + 1])
    return result.exit_code, payload


def _run_status(cfg_path: str, run_id: str) -> str:
    manifest = Manifest(load_config(cfg_path).manifest_path)
    return manifest.get_run(run_id)["status"]


def _run_error(cfg_path: str, run_id: str) -> str:
    manifest = Manifest(load_config(cfg_path).manifest_path)
    return manifest.get_run(run_id)["error_message"] or ""


def test_all_success_is_success_and_exit_zero(cfg_path, monkeypatch):
    monkeypatch.setattr(
        delisted_mod,
        "backfill_delisted_bars",
        lambda *a, **k: _result(recovered=1, rows=10, symbols=1),
    )
    monkeypatch.setattr(
        JobEngine,
        "run_step",
        lambda self, name, trade_date, run_id, context=None: _compact("success"),
    )

    exit_code, payload = _invoke(cfg_path)

    assert exit_code == 0
    assert payload["status"] == "success"
    assert _run_status(cfg_path, payload["run_id"]) == "success"


def test_failed_symbol_is_failed_and_nonzero(cfg_path, monkeypatch):
    monkeypatch.setattr(
        delisted_mod,
        "backfill_delisted_bars",
        lambda *a, **k: _result(failed=2, rows=3, symbols=2),
    )
    monkeypatch.setattr(
        JobEngine,
        "run_step",
        lambda self, name, trade_date, run_id, context=None: _compact("success"),
    )

    exit_code, payload = _invoke(cfg_path)

    assert exit_code != 0
    assert payload["status"] == "failed"
    assert "failed=2" in _run_error(cfg_path, payload["run_id"])
    assert _run_status(cfg_path, payload["run_id"]) == "failed"


def test_empty_symbol_is_failed_and_nonzero(cfg_path, monkeypatch):
    monkeypatch.setattr(
        delisted_mod,
        "backfill_delisted_bars",
        lambda *a, **k: _result(empty=1, rows=0, symbols=1),
    )
    monkeypatch.setattr(
        JobEngine,
        "run_step",
        lambda self, name, trade_date, run_id, context=None: _compact("success"),
    )

    exit_code, payload = _invoke(cfg_path)

    assert exit_code != 0
    assert payload["status"] == "failed"
    assert "empty=1" in _run_error(cfg_path, payload["run_id"])
    assert _run_status(cfg_path, payload["run_id"]) == "failed"


def test_mixed_partial_preserved_and_failed(cfg_path, monkeypatch):
    monkeypatch.setattr(
        delisted_mod,
        "backfill_delisted_bars",
        lambda *a, **k: _result(failed=1, empty=1, recovered=1, rows=5, symbols=3),
    )
    monkeypatch.setattr(
        JobEngine,
        "run_step",
        lambda self, name, trade_date, run_id, context=None: _compact("success"),
    )

    exit_code, payload = _invoke(cfg_path)

    assert exit_code != 0
    assert payload["status"] == "failed"
    assert payload["rows_written"] == 5  # successful symbols preserved
    assert "failed=1, empty=1" in _run_error(cfg_path, payload["run_id"])
    assert _run_status(cfg_path, payload["run_id"]) == "failed"


def test_second_retry_finishes_success(cfg_path, monkeypatch):
    state = {"calls": 0}

    def flaky(*a, **k):
        state["calls"] += 1
        if state["calls"] == 1:
            return _result(failed=1, empty=1, recovered=1, rows=5, symbols=3)
        return _result(recovered=2, rows=10, symbols=2)

    monkeypatch.setattr(delisted_mod, "backfill_delisted_bars", flaky)
    monkeypatch.setattr(
        JobEngine,
        "run_step",
        lambda self, name, trade_date, run_id, context=None: _compact("success"),
    )

    first_exit, first = _invoke(cfg_path)
    second_exit, second = _invoke(cfg_path)

    assert first_exit != 0 and first["status"] == "failed"
    assert _run_status(cfg_path, first["run_id"]) == "failed"
    assert second_exit == 0 and second["status"] == "success"
    assert _run_status(cfg_path, second["run_id"]) == "success"


def test_compact_failure_fails_run(cfg_path, monkeypatch):
    monkeypatch.setattr(
        delisted_mod,
        "backfill_delisted_bars",
        lambda *a, **k: _result(recovered=1, rows=10, symbols=1),
    )
    monkeypatch.setattr(
        JobEngine,
        "run_step",
        lambda self, name, trade_date, run_id, context=None: _compact("failed"),
    )

    exit_code, payload = _invoke(cfg_path)

    assert exit_code != 0
    assert payload["status"] == "failed"
    assert "compact=failed" in _run_error(cfg_path, payload["run_id"])
    assert _run_status(cfg_path, payload["run_id"]) == "failed"


def test_zero_targets_is_success(cfg_path, monkeypatch):
    monkeypatch.setattr(
        delisted_mod,
        "backfill_delisted_bars",
        lambda *a, **k: _result(note="no delisted recovery targets to ingest"),
    )
    monkeypatch.setattr(
        JobEngine,
        "run_step",
        lambda self, name, trade_date, run_id, context=None: _compact("success"),
    )

    exit_code, payload = _invoke(cfg_path)

    assert exit_code == 0
    assert payload["status"] == "success"
    assert _run_status(cfg_path, payload["run_id"]) == "success"
