#!/usr/bin/env python3
# scripts/show_output.py
import argparse
import os
from glob import glob

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F


def parse_args():
    p = argparse.ArgumentParser("Show latest training output (Parquet)")
    p.add_argument("--base", type=str, default="out", help="Base output folder")
    p.add_argument(
        "--subdir",
        type=str,
        default=None,
        help=(
            "Subdir under base (e.g., custom_training_outputs/2025_08_16_21_38_15). "
            "If omitted, the newest run is auto-detected."
        ),
    )
    p.add_argument("--limit", type=int, default=10, help="Number of sample rows to show")
    p.add_argument(
        "--actions-head",
        type=int,
        default=5,  # tidy by default
        help="How many humanized actions to show per row",
    )
    return p.parse_args()


def newest_leaf_dir(root: str) -> str:
    """Find newest timestamped run directory under base/*/*."""
    candidates = []
    for top in glob(os.path.join(root, "*")):
        if not os.path.isdir(top):
            continue
        for leaf in glob(os.path.join(top, "*")):
            if os.path.isdir(leaf):
                candidates.append(leaf)
    if not candidates:
        raise FileNotFoundError(f"No output directories found under {root}")
    return max(candidates, key=os.path.getmtime)


def action_type_name(col):
    # 1=click, 2=ATC, 3=order, 0=pad, else=unk
    return (
        F.when(col == F.lit(1), F.lit("click"))
        .when(col == F.lit(2), F.lit("atc"))
        .when(col == F.lit(3), F.lit("order"))
        .when(col == F.lit(0), F.lit("pad"))
        .otherwise(F.lit("unk"))
    )


def main():
    args = parse_args()

    if args.subdir:
        out_dir = os.path.join(args.base, args.subdir)
    else:
        out_dir = newest_leaf_dir(args.base)

    spark = SparkSession.builder.appName("show-output").config("spark.sql.session.timeZone", "UTC").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    df = spark.read.parquet(out_dir)

    # ---------- Schema ----------
    print("\n=== SCHEMA ===")
    df.printSchema()

    # ---------- Per-row sizes preview ----------
    df_sizes = df.select(
        "ranking_id",
        "customer_id",
        "as_of_date",
        F.size("impressions").alias("n_impr"),
        F.size("actions").alias("n_actions"),
        F.size("action_types").alias("n_action_types"),
    )
    df_sizes.show(args.limit, truncate=False)

    # ---------- Aggregates (safe) ----------
    row_feats = df.select(
        "as_of_date",
        F.size("impressions").alias("n_impr"),
        F.size("actions").alias("n_actions"),
        # count of zeros in action_types (padding per row)
        F.expr("aggregate(transform(action_types, x -> int(x = 0)), 0, (acc, x) -> acc + x)").alias("pad_zeros"),
    )

    rows_total = row_feats.count()

    print("\n=== AGGREGATES ===")
    row_feats.select(
        F.lit(rows_total).alias("rows"),
        F.min("as_of_date").alias("min_as_of_date"),
        F.max("as_of_date").alias("max_as_of_date"),
        F.avg("n_impr").alias("avg_impressions_per_row"),
        F.avg("n_actions").alias("avg_actions_array_len"),
    ).show(truncate=False)

    row_feats.select(
        F.avg("pad_zeros").alias("avg_zero_pads_per_row"),
        F.min("pad_zeros").alias("min_zero_pads_in_row"),
        F.max("pad_zeros").alias("max_zero_pads_in_row"),
    ).show(truncate=False)

    # ---------- Per-customer snapshot (random subset) ----------
    print(f"\n=== PER-CUSTOMER SNAPSHOT (random {args.limit} customers) ===")
    cust_feats = df.select(
        "customer_id",
        "as_of_date",
        F.size("impressions").alias("n_impr"),
        F.size("actions").alias("n_actions"),
    )
    cust_ids = cust_feats.select("customer_id").distinct().orderBy(F.rand()).limit(args.limit)
    cust_summary = (
        cust_feats.join(cust_ids, "customer_id")
        .groupBy("customer_id")
        .agg(
            F.count("*").alias("rows"),
            F.min("as_of_date").alias("min_as_of_date"),
            F.max("as_of_date").alias("max_as_of_date"),
            F.avg("n_impr").alias("avg_impr"),
            F.avg("n_actions").alias("avg_actions"),
        )
        .orderBy("customer_id")
    )
    cust_summary.show(args.limit, truncate=False)

    # ---------- Sample actions (humanized, single column, left-aligned) ----------
    print(f"\n=== SAMPLE ACTIONS (humanized, top {args.actions_head}) ===")

    pairs = F.arrays_zip("actions", "action_types")

    # Build tokens like "1585:click" (left-aligned: no padding)
    token_width = 12  # adjust as you like
    tokens = F.transform(
        pairs,
        lambda x: F.rpad(
            F.concat(
                x.getField("actions").cast("string"),
                F.lit(":"),
                action_type_name(x.getField("action_types")),
            ),
            token_width,
            " ",
        ),
    )

    # take the first N and join with " | " for readability
    head_n = F.slice(tokens, 1, args.actions_head)
    actions_head_str = F.concat(F.concat_ws(" | ", head_n))

    (
        df.orderBy(F.rand())
        .select(
            "as_of_date",
            "ranking_id",
            "customer_id",
            F.size("actions").alias("n_actions"),
            actions_head_str.alias("actions_head"),
        )
        .limit(args.limit)
        .show(args.limit, truncate=False)
    )

    # ---------- Sample impressions (exploded & diversified) ----------
    print("\n=== SAMPLE IMPRESSIONS (exploded, diversified) ===")
    picks = df.select("as_of_date", "ranking_id", "customer_id").orderBy(F.rand()).limit(args.limit)
    exploded = df.join(picks, ["as_of_date", "ranking_id", "customer_id"]).select(
        "as_of_date",
        "ranking_id",
        "customer_id",
        F.explode("impressions").alias("impr"),
    )
    w = Window.partitionBy("as_of_date", "ranking_id", "customer_id").orderBy(F.rand())
    exploded_sample = (
        exploded.withColumn("rn", F.row_number().over(w))
        .where(F.col("rn") == 1)
        .select(
            "as_of_date",
            "ranking_id",
            "customer_id",
            F.col("impr.item_id").alias("impression_item_id"),
            F.col("impr.is_order").alias("is_order"),
        )
    )
    exploded_sample.show(args.limit, truncate=False)

    spark.stop()


if __name__ == "__main__":
    main()
