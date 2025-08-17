from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def normalize_impressions(impressions_df: DataFrame) -> DataFrame:
    """
    Normalize impressions to:
      as_of_date: date
      ranking_id: string
      customer_id: bigint
      impressions: array<struct<item_id:int,is_order:boolean>>

    Idempotent:
      - If a raw 'dt' column exists, parse it.
      - If 'as_of_date' already exists, just coerce/cast and pass through.
    """
    cols = set(impressions_df.columns)

    if "as_of_date" in cols:
        # Already normalized once; just ensure types
        return impressions_df.select(
            F.to_date("as_of_date").alias("as_of_date"),
            F.col("ranking_id").cast("string").alias("ranking_id"),
            F.col("customer_id").cast("bigint").alias("customer_id"),
            F.col("impressions"),
        )

    if "dt" in cols:
        # Raw input case: parse 'dt' (supports either 'YYYY-MM-DD' or 'dt=YYYY-MM-DD')
        # regexp_extract grabs the date if 'dt' is like 'dt=2025-08-15'
        return impressions_df.select(
            F.to_date(F.regexp_extract(F.col("dt"), r"(\d{4}-\d{2}-\d{2})", 1)).alias("as_of_date"),
            F.col("ranking_id").cast("string").alias("ranking_id"),
            F.col("customer_id").cast("bigint").alias("customer_id"),
            F.col("impressions"),
        )

    # Defensive: neither dt nor as_of_date exists
    raise ValueError("normalize_impressions: expected either 'dt' or 'as_of_date' column in impressions_df")


def explode_impressions(impressions_df: DataFrame) -> DataFrame:
    """
    Return one row per impression item with:
      as_of_date, ranking_id, customer_id, impression_item_id, is_order
    """
    imp = normalize_impressions(impressions_df)
    return imp.select(
        "as_of_date",
        "ranking_id",
        "customer_id",
        F.explode("impressions").alias("impr"),
    ).select(
        "as_of_date",
        "ranking_id",
        "customer_id",
        F.col("impr.item_id").cast("int").alias("impression_item_id"),
        F.col("impr.is_order").cast("boolean").alias("is_order"),
    )
