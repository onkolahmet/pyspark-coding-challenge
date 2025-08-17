from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from .utils import ACTION_MISSING, pad_array


def build_histories_for_impressions(
    actions_union_df: DataFrame,
    impressions_ctx_df: DataFrame,
    *,
    max_actions: int = 1000,
    lookback_days: int = 365,
) -> DataFrame:
    ctx = impressions_ctx_df.select("as_of_date", "customer_id").dropDuplicates()

    # Pre-join and strictly filter by the *timestamp* date
    joined = (
        actions_union_df.alias("a")
        .repartition("customer_id")
        .join(ctx.alias("c"), on="customer_id", how="inner")
        .where(
            # strictly before the impression day (no same-day)
            F.to_date(F.col("a.occurred_at"))
            < F.col("c.as_of_date")
        )
        .where(
            # within lookback window by actual timestamp date
            F.to_date(F.col("a.occurred_at"))
            >= F.date_sub(F.col("c.as_of_date"), lookback_days)
        )
        .select(
            F.col("c.as_of_date"),
            F.col("a.customer_id"),
            F.col("a.item_id"),
            F.col("a.action_type"),
            F.col("a.occurred_at"),
        )
        .repartition("customer_id", "as_of_date")
    )

    w = Window.partitionBy("as_of_date", "customer_id").orderBy(
        F.col("occurred_at").desc(),
        F.col("action_type").desc(),
        F.col("item_id").desc(),
    )

    ranked = joined.withColumn("rn", F.row_number().over(w)).where(F.col("rn") <= max_actions)

    structs = ranked.groupBy("as_of_date", "customer_id").agg(
        F.collect_list(F.struct("rn", "item_id", "action_type")).alias("lst")
    )
    sorted_structs = structs.withColumn("lst", F.array_sort("lst"))

    actions_arr = F.transform(F.col("lst"), lambda x: x.getField("item_id"))
    types_arr = F.transform(F.col("lst"), lambda x: x.getField("action_type"))

    hist = (
        sorted_structs.withColumn("actions", pad_array(actions_arr, max_actions, 0))
        .withColumn("action_types", pad_array(types_arr, max_actions, ACTION_MISSING))
        .select("as_of_date", "customer_id", "actions", "action_types")
    )

    # Left join to fill missing customers with fully padded arrays
    hist_full = (
        ctx.alias("c")
        .join(hist.alias("h"), on=["as_of_date", "customer_id"], how="left")
        .withColumn(
            "actions",
            F.when(F.col("h.actions").isNull(), F.array_repeat(F.lit(0), max_actions)).otherwise(F.col("h.actions")),
        )
        .withColumn(
            "action_types",
            F.when(F.col("h.action_types").isNull(), F.array_repeat(F.lit(0), max_actions)).otherwise(
                F.col("h.action_types")
            ),
        )
        .select("as_of_date", "customer_id", "actions", "action_types")
    )

    return hist_full
