#!/usr/bin/env python3
"""
Unified entrypoint to build training inputs from impressions + user actions.

====================================================================
USAGE EXAMPLES
====================================================================

1. Demo mode (no files needed)
--------------------------------------------------------------------
  python scripts/run.py --demo --out out/training_inputs

2. Local mock data (JSON arrays written by generate_mock_data.py)
--------------------------------------------------------------------
  python scripts/run.py --out out/training_inputs \
      --max-actions 1000 \
      --lookback-days 365

3. External raw data (Local FS, HDFS, or S3)
--------------------------------------------------------------------
  You can also run against your own datasets by specifying explicit paths:

  python scripts/run.py \
      --out out/training_inputs \
      --impressions /path/to/impressions/* \
      --clicks /path/to/clicks/* \
      --atc /path/to/atc/* \
      --orders /path/to/orders/*

  Supported path schemes:
    • Local filesystem: /path/to/impressions/*
    • HDFS:             hdfs:///user/data/impressions/*
    • S3:               s3://bucket/impressions/*

  This ensures the pipeline works equally well whether run with:
    - Local mock data (generated with make run / run-custom)
    - Reviewer-provided raw data on disk
    - Cluster storage (HDFS/S3)
====================================================================
"""

import argparse
import os
import sys
import time
from datetime import date, timedelta

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Ensure `src/` is importable when running this script directly
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from challenge.actions_union import build_actions_union
from challenge.impressions import normalize_impressions
from challenge.io import (
    read_atc,
    read_clicks,
    read_impressions,
    read_orders,
    write_parquet,
    write_quarantine,
)
from challenge.pipeline import build_training_inputs
from challenge.quality_checks import check_no_same_day_leak_strict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build training inputs for the recsys challenge")

    # Outputs
    p.add_argument(
        "--out",
        type=str,
        default="out/training_inputs",
        help="Base output directory; a timestamped subfolder will be created inside",
    )

    # Modes
    p.add_argument("--demo", action="store_true", help="Run with small in-memory demo data (no IO)")
    p.add_argument(
        "--explode",
        action="store_true",
        help="Produce exploded rows (one per impression item) instead of fixed arrays",
    )

    # Windowing parameters
    p.add_argument(
        "--max-actions",
        type=int,
        default=1000,
        help="Max actions per customer history (pad to 1000)",
    )
    p.add_argument(
        "--lookback-days",
        type=int,
        default=365,
        help="Lookback window in days for actions",
    )
    p.add_argument(
        "--train-days",
        type=int,
        default=14,
        help="Keep only impressions from last N days",
    )

    # External input paths
    p.add_argument(
        "--impressions-path",
        type=str,
        default=None,
        help="Path/pattern to impressions JSON/Parquet",
    )
    p.add_argument(
        "--clicks-path",
        type=str,
        default=None,
        help="Path/pattern to clicks JSON/Parquet",
    )
    p.add_argument(
        "--atc-path",
        type=str,
        default=None,
        help="Path/pattern to add-to-cart JSON/Parquet",
    )
    p.add_argument(
        "--orders-path",
        type=str,
        default=None,
        help="Path/pattern to orders JSON/Parquet",
    )

    # Spark knobs
    p.add_argument(
        "--shuffle-partitions",
        type=int,
        default=0,
        help="Override spark.sql.shuffle.partitions",
    )
    # DQ behavior
    p.add_argument(
        "--fail-on-same-day-leak",
        action="store_true",
        help="Fail the job if any same-day action leaks into histories",
    )
    p.add_argument(
        "--external-upstream-scan",
        action="store_true",
        help=(
            "EXTERNAL mode: also scan and report if inputs contain same-day actions "
            "(upstream signal only; histories still exclude them). Off by default."
        ),
    )

    return p.parse_args()


def demo_inputs(spark: SparkSession):
    """Small synthetic demo frames that mirror the real schemas."""
    today = date.today()
    d0, d1, d2, d3 = (
        today,
        today - timedelta(1),
        today - timedelta(2),
        today - timedelta(3),
    )

    impressions = spark.createDataFrame(
        [
            (
                d1.strftime("%Y-%m-%d"),
                "r1",
                101,
                [{"item_id": 10, "is_order": False}, {"item_id": 11, "is_order": True}],
            ),
            (d0.strftime("%Y-%m-%d"), "r2", 101, [{"item_id": 12, "is_order": False}]),
        ],
        "dt string, ranking_id string, customer_id long, impressions array<struct<item_id:int,is_order:boolean>>",
    )

    clicks = spark.createDataFrame(
        [
            (f"dt={d2}", 101, 10, f"{d2} 12:00:00"),
            (f"dt={d1}", 101, 99, f"{d1} 08:00:00"),  # same-day → excluded
            (f"dt={d3}", 101, 11, f"{d3} 10:00:00"),
        ],
        "dt string, customer_id long, item_id long, click_time string",
    )

    atc = spark.createDataFrame(
        [(f"dt={d3}", 101, 12, None, f"{d3} 09:00:00")],
        "dt string, customer_id long, config_id long, simple_id long, occurred_at string",
    )

    orders = spark.createDataFrame(
        [(d3.strftime("%Y-%m-%d"), 101, 10)],
        "order_date string, customer_id long, config_id long",
    )

    return impressions, clicks, atc, orders


def resolve_out_dir(base_out: str) -> str:
    ts = time.strftime("%Y_%m_%d_%H_%M_%S")
    out_dir = os.path.join(base_out, f"{ts}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def slice_impressions_last_n_days(impressions, n_days: int):
    # Impressions dt is already a date-like string; keep last n_days
    max_date = impressions.agg(F.max(F.to_date("dt"))).first()[0]
    if max_date is None:
        return impressions.limit(0)
    cutoff = F.date_sub(F.lit(max_date), n_days - 1)
    return impressions.where(F.to_date("dt") >= cutoff)


def main():
    args = parse_args()
    job_start = time.time()

    # ---------------- Spark session ----------------
    spark_session_start = time.time()
    spark = (
        SparkSession.builder.appName("pyspark-coding-challenge-run")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.parquet.compression.codec", "zstd")
        .config("spark.sql.files.maxPartitionBytes", str(64 * 1024 * 1024))
        .config(
            "spark.driver.extraJavaOptions",
            f"-Dlog4j.configurationFile={os.path.join(os.path.dirname(__file__), '..', 'conf', 'log4j2.properties')}",
        )
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")  # belt & suspenders

    if args.shuffle_partitions and args.shuffle_partitions > 0:
        spark.conf.set("spark.sql.shuffle.partitions", str(args.shuffle_partitions))
    spark_session_end = time.time()

    out_dir = resolve_out_dir(args.out)
    print(f"▶ Running job. Outputs will be written to: {out_dir}")

    # ---------------- Load inputs ----------------
    t_load0 = time.time()
    if args.demo:
        mode = "DEMO"
        print("🔹 Mode: DEMO (synthetic in-memory data)")
        impressions, clicks, atc, orders = demo_inputs(spark)
        impressions = slice_impressions_last_n_days(impressions, args.train_days)
    elif args.impressions_path or args.clicks_path or args.atc_path or args.orders_path:
        mode = "EXTERNAL"
        print("🔹 Mode: EXTERNAL (using provided dataset paths)")
        impressions, bad_impr = read_impressions(spark, args.impressions_path)
        clicks, bad_clicks = read_clicks(spark, args.clicks_path)
        atc, bad_atc = read_atc(spark, args.atc_path)
        orders, bad_orders = read_orders(spark, args.orders_path)

        write_quarantine(bad_impr, out_dir, "impressions")
        write_quarantine(bad_clicks, out_dir, "clicks")
        write_quarantine(bad_atc, out_dir, "atc")
        write_quarantine(bad_orders, out_dir, "orders")

        impressions = slice_impressions_last_n_days(impressions, args.train_days)
    else:
        mode = "LOCAL"
        print("🔹 Mode: LOCAL MOCK DATA (./data/*)")
        impr_path, clk_path, atc_path, ord_path = (
            "data/impressions",
            "data/clicks",
            "data/atc",
            "data/orders",
        )

        impressions, bad_impr = read_impressions(spark, impr_path)
        clicks, bad_clicks = read_clicks(spark, clk_path)
        atc, bad_atc = read_atc(spark, atc_path)
        orders, bad_orders = read_orders(spark, ord_path)

        write_quarantine(bad_impr, out_dir, "impressions")
        write_quarantine(bad_clicks, out_dir, "clicks")
        write_quarantine(bad_atc, out_dir, "atc")
        write_quarantine(bad_orders, out_dir, "orders")

        impressions = slice_impressions_last_n_days(impressions, args.train_days)
    t_load1 = time.time()

    # ---------------- Transform ----------------
    t_tr0 = time.time()
    training_df = build_training_inputs(
        impressions_df=impressions,
        clicks_df=clicks,
        atc_df=atc,
        orders_df=orders,
        max_actions=args.max_actions,
        lookback_days=args.lookback_days,
        explode=args.explode,
    )
    t_tr1 = time.time()

    # ---------------- DQ: no same-day action leak ----------------
    leaks = None
    try:
        actions_union = build_actions_union(clicks, atc, orders)
        normalized_impr = normalize_impressions(impressions)

        # Strict check that matches the pipeline’s own filters
        leaks = check_no_same_day_leak_strict(actions_union, normalized_impr, lookback_days=args.lookback_days)

        # Conservative fallback ONLY for EXTERNAL runs.
        # This can over-warn on synthetic/local data, so we avoid it in LOCAL/DEMO.
        if leaks == 0 and mode == "EXTERNAL" and args.external_upstream_scan:
            ctx = normalized_impr.select(
                F.col("customer_id").alias("ctx_cid"),
                F.col("as_of_date").alias("ctx_date"),
            ).distinct()

            leaks_simple = (
                actions_union.join(
                    ctx,
                    (F.col("customer_id") == F.col("ctx_cid")) & (F.col("action_date") == F.col("ctx_date")),
                    "inner",
                )
                .limit(1)  # short-circuit quickly
                .count()
            )
            if leaks_simple > 0:
                print(
                    "ℹ️  Inputs contain at least one same-day action for a (customer_id, as_of_date). "
                    "Histories remain clean because the pipeline excludes same-day actions by design."
                )

    except Exception as e:
        print(f"⚠️  DQ check failed to run: {e}")
        leaks = None

    if leaks and leaks > 0:
        msg = f"[DQ] Same-day leak detected after applying pipeline filters: {leaks}"
        print("⚠️  " + msg)
        if args.fail_on_same_day_leak:
            sys.stdout.flush()
            sys.stderr.flush()
            raise SystemExit(2)
    elif leaks == 0:
        print("✅ No same-day leakage into histories (strict check).")

    # ---------------- Write outputs ----------------
    t_wr0 = time.time()
    write_parquet(training_df, out_dir, mode="overwrite", partition_by=["as_of_date"])
    t_wr1 = time.time()

    cnt = training_df.count()
    print(f"\n✅ Finished. Rows out: {cnt} → {out_dir}\n")

    # ---------------- Timing summary ----------------
    job_end = time.time()
    print("⏱️ Execution time breakdown:")
    print(f"   • Spark session setup                       {spark_session_end - spark_session_start:.2f}s")
    print(f"   • Data loading                              {t_load1 - t_load0:.2f}s")
    print(f"   • Transform (build_training_inputs):        {t_tr1 - t_tr0:.2f}s")
    print(f"   • Write outputs (Parquet):                  {t_wr1 - t_wr0:.2f}s")
    print(f"   • TOTAL runtime:                            {job_end - job_start:.2f}s")

    spark.stop()


if __name__ == "__main__":
    main()
