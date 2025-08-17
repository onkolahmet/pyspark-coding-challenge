# src/challenge/actions_union.py
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

# convention: 1=click, 2=ATC, 3=order, 0=missing
ACTION_CLICK = 1
ACTION_ATC = 2
ACTION_ORDER = 3


def build_actions_union(clicks_df: DataFrame, atc_df: DataFrame, orders_df: DataFrame) -> DataFrame:
    """
    Standardize actions into:
      customer_id (bigint), item_id (int), action_type (int),
      occurred_at (timestamp), action_date (date), source (string)

    IMPORTANT:
      - action_date is always derived from occurred_at to keep
        same-day logic consistent across the repo.
      - We do NOT rely on any partition 'dt' column here anymore.
    """

    # ---- Clicks ----
    # Mock schema typically: dt, customer_id, item_id, click_time
    clicks_norm = clicks_df.select(
        F.col("customer_id").cast("bigint").alias("customer_id"),
        F.col("item_id").cast("int").alias("item_id"),
        F.to_timestamp("click_time").alias("occurred_at"),
        F.lit(ACTION_CLICK).alias("action_type"),
        F.lit("clicks").alias("source"),
    ).withColumn("action_date", F.to_date("occurred_at"))

    # ---- Add-To-Cart ----
    # Mock schema typically: dt, customer_id, config_id, simple_id, occurred_at
    # If you map config_id->item_id elsewhere, keep it consistent here
    atc_norm = atc_df.select(
        F.col("customer_id").cast("bigint").alias("customer_id"),
        F.col("config_id").cast("int").alias("item_id"),
        F.to_timestamp("occurred_at").alias("occurred_at"),
        F.lit(ACTION_ATC).alias("action_type"),
        F.lit("atc").alias("source"),
    ).withColumn("action_date", F.to_date("occurred_at"))

    # ---- Orders ----
    # Mock schema typically: order_date (YYYY-MM-DD), customer_id, config_id
    # If you don`t have per-order time, use midnight so date math stays correct.
    orders_norm = orders_df.select(
        F.col("customer_id").cast("bigint").alias("customer_id"),
        F.col("config_id").cast("int").alias("item_id"),
        F.to_timestamp(F.concat_ws(" ", F.col("order_date").cast("string"), F.lit("00:00:00"))).alias("occurred_at"),
        F.lit(ACTION_ORDER).alias("action_type"),
        F.lit("orders").alias("source"),
    ).withColumn("action_date", F.to_date("occurred_at"))

    return clicks_norm.unionByName(atc_norm).unionByName(orders_norm)
