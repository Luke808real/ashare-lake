"""Reconstruct historical suspension status from daily_bars gaps.

Exchanges publish no daily bar for a suspended stock, so a listed symbol with
no bar on a trading day was suspended that day. This is authoritative and
covers the whole bar history — filling the trading_status gap that free ST
feeds (EastMoney's current-snapshot ST board) cannot reach.

Only sparse ``suspended`` rows are written; real daily rows (EastMoney) win on
any primary-key overlap.

Optional ``start`` / ``end`` bound the calendar cross-join so a 2001→today
rebuild can run year-by-year without OOM.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl

from ashare_lake.config import Config
from ashare_lake.domain.datasets import DATASETS
from ashare_lake.domain.schemas import validate_dataframe, with_provenance
from ashare_lake.query.parquet_scan import dataset_has_parquet, scan_parquet_root
from ashare_lake.storage.parquet import CuratedWriter

logger = logging.getLogger(__name__)

_DERIVED_SOURCE = "derived_bar_gap"
_STATUS_SPEC = DATASETS["trading_status"]
_DAILY_STATUS_SOURCES = frozenset({"eastmoney", "tdx_protocol"})
_SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")


def _shanghai_date(value: Any) -> date | None:
    """Wall-clock date in Asia/Shanghai for *value* (UTC-aware or naive)."""
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        # Naive timestamps are treated as already-UTC (ASL storage norm).
        value = value.replace(tzinfo=ZoneInfo("UTC"))
    return value.astimezone(_SHANGHAI_TZ).date()


def status_row_precedence_class(row: Mapping[str, Any]) -> str:
    """Classify one existing ``trading_status`` row for a bar-gap collision.

    Returns one of:

    * ``TRUSTED_SAME_SESSION_DAILY`` — an EastMoney/TDX daily row whose fetch
      happened on the same Asia/Shanghai session date as its ``trade_date``;
      it is real same-session evidence and must keep precedence.
    * ``NON_PIT_DAILY_SNAPSHOT`` — an EastMoney/TDX current-state row fetched
      on a LATER session than ``trade_date`` (or with unknown fetch time);
      it is NOT point-in-time evidence for that historical date, so a proven
      bar-gap suspension wins the PK.
    * ``BAOSTOCK`` — historical ST rows; preserved, never reclassified.
    * ``DERIVED_BAR_GAP`` — the generated suspension row itself.
    * ``OTHER`` — unknown source; preserved, fail closed (no derived override).
    """
    source = str(row.get("source") or "")
    if source == _DERIVED_SOURCE:
        return "DERIVED_BAR_GAP"
    if source == "baostock":
        return "BAOSTOCK"
    if source in _DAILY_STATUS_SOURCES:
        trade_date = row.get("trade_date")
        fetched = _shanghai_date(row.get("fetched_at"))
        if fetched is not None and isinstance(trade_date, date) and fetched == trade_date:
            return "TRUSTED_SAME_SESSION_DAILY"
        return "NON_PIT_DAILY_SNAPSHOT"
    return "OTHER"


def _suspended_pairs(
    config: Config,
    *,
    start: date | None = None,
    end: date | None = None,
) -> pl.DataFrame:
    """(symbol, trade_date) that were trading days in a symbol's active range but have no bar."""
    bars_root = config.curated_root / "daily_bars"
    cal_root = config.curated_root / "trading_calendar"
    inst_path = config.curated_root / "instruments" / "part-merged.parquet"
    if not (
        dataset_has_parquet(bars_root) and dataset_has_parquet(cal_root) and inst_path.exists()
    ):
        return pl.DataFrame(schema={"symbol": pl.Utf8, "trade_date": pl.Date})

    # Lifetime bounds from the full bar history (cheap scan aggregate).
    sym_range = (
        scan_parquet_root(bars_root, partition_col="trade_date")
        .group_by("symbol")
        .agg(
            pl.col("trade_date").min().alias("bmin"),
            pl.col("trade_date").max().alias("bmax"),
        )
        .collect()
    )
    if sym_range.is_empty():
        return pl.DataFrame(schema={"symbol": pl.Utf8, "trade_date": pl.Date})

    # Anti-join only needs bars inside the derive window.
    bars_lf = scan_parquet_root(bars_root, partition_col="trade_date").select(
        ["symbol", "trade_date"]
    )
    if start is not None:
        bars_lf = bars_lf.filter(pl.col("trade_date") >= start)
    if end is not None:
        bars_lf = bars_lf.filter(pl.col("trade_date") <= end)
    bars = bars_lf.unique().collect()

    cal_lf = (
        scan_parquet_root(cal_root, partition_col="trade_date")
        .filter(pl.col("is_trading"))
        .select("trade_date")
    )
    if start is not None:
        cal_lf = cal_lf.filter(pl.col("trade_date") >= start)
    if end is not None:
        cal_lf = cal_lf.filter(pl.col("trade_date") <= end)
    cal = cal_lf.collect()
    if cal.is_empty():
        return pl.DataFrame(schema={"symbol": pl.Utf8, "trade_date": pl.Date})

    inst = pl.read_parquet(inst_path).select(["symbol", "list_date", "delist_date"])

    active = inst.join(sym_range, on="symbol", how="inner").with_columns(
        pl.max_horizontal(pl.col("list_date").fill_null(pl.col("bmin")), pl.col("bmin")).alias(
            "astart"
        ),
        pl.min_horizontal(pl.col("delist_date").fill_null(pl.col("bmax")), pl.col("bmax")).alias(
            "aend"
        ),
    )
    if start is not None:
        active = active.with_columns(
            pl.max_horizontal(pl.col("astart"), pl.lit(start)).alias("astart")
        )
    if end is not None:
        active = active.with_columns(pl.min_horizontal(pl.col("aend"), pl.lit(end)).alias("aend"))
    active = active.filter(pl.col("astart") <= pl.col("aend"))
    if active.is_empty():
        return pl.DataFrame(schema={"symbol": pl.Utf8, "trade_date": pl.Date})

    expected = (
        active.select(["symbol", "astart", "aend"])
        .join(cal, how="cross")
        .filter(
            (pl.col("trade_date") >= pl.col("astart")) & (pl.col("trade_date") <= pl.col("aend"))
        )
        .select(["symbol", "trade_date"])
    )
    return expected.join(bars, on=["symbol", "trade_date"], how="anti")


def derive_suspension_history(
    config: Config,
    *,
    start: date | None = None,
    end: date | None = None,
) -> int:
    """Write derived ``suspended`` rows into curated trading_status. Returns row count.

    When *start* / *end* are set, only that calendar window is considered — use
    yearly chunks for a full-history rebuild to keep the cross-join bounded.
    """
    pairs = _suspended_pairs(config, start=start, end=end)
    if pairs.is_empty():
        return 0

    rows = pairs.with_columns(
        pl.lit(False).alias("is_trading"),
        pl.lit("suspended").alias("status"),
    )
    rows = with_provenance(rows, source=_DERIVED_SOURCE, data_version="v1")
    rows = validate_dataframe(rows, "trading_status")

    # trading_status is month-partitioned — never write day dirs that fight the
    # registry (audit: mixed_partition_granularity) and republish PKs.
    writer = CuratedWriter(config.curated_root)
    pcol = _STATUS_SPEC.partition_col or "trade_date"
    part_vals = [
        _STATUS_SPEC.partition_for(td) if isinstance(td, date) else str(td)
        for td in rows["trade_date"].to_list()
    ]
    rows = rows.with_columns(pl.Series("_part", part_vals))
    total = 0
    for key, group in rows.partition_by("_part", as_dict=True).items():
        val = str(key[0] if isinstance(key, tuple) else key)
        group = group.drop("_part")
        existing_dir = writer.partition_path("trading_status", pcol, val)
        frames = [group]
        stray_parts: list = []
        if existing_dir.exists():
            for f in existing_dir.glob("*.parquet"):
                frames.append(validate_dataframe(pl.read_parquet(f), "trading_status"))
                if f.name != "part-merged.parquet":
                    stray_parts.append(f)
        merged = pl.concat(frames, how="diagonal_relaxed")
        if "fetched_at" not in merged.columns:
            merged = merged.with_columns(pl.lit(None, dtype=pl.Datetime("us")).alias("fetched_at"))
        # PIT-aware precedence: derived_bar_gap wins ONLY over NON-PIT daily
        # current-state snapshots. Same-session daily rows, baostock ST rows,
        # and unknown sources keep precedence (fail closed).
        merged = merged.with_columns(
            pl.struct(["source", "trade_date", "fetched_at"])
            .map_elements(
                lambda row: status_row_precedence_class(row),
                return_dtype=pl.Utf8,
            )
            .alias("_prec")
        )
        _PRECEDENCE_RANK = {
            "TRUSTED_SAME_SESSION_DAILY": 0,
            "BAOSTOCK": 0,
            "OTHER": 0,
            "DERIVED_BAR_GAP": 1,
            "NON_PIT_DAILY_SNAPSHOT": 2,
        }
        merged = merged.with_columns(
            pl.col("_prec").replace_strict(_PRECEDENCE_RANK, default=0).alias("_prec_rank")
        ).sort("_prec_rank")
        merged = merged.unique(subset=["symbol", "trade_date"], keep="first").drop(
            ["_prec", "_prec_rank"]
        )
        # Reuse compact's filename so we overwrite the single canonical part
        # rather than adding a second file (which would double-count on read).
        writer.write_partition("trading_status", pcol, val, merged, "part-merged.parquet")
        for stray in stray_parts:
            stray.unlink(missing_ok=True)
        total += group.height
    logger.info("derived %d historical suspension rows into trading_status", total)
    return total
