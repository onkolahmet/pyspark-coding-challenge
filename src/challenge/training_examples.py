from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from .impressions import normalize_impressions


def join_impressions_with_histories(impressions_df: DataFrame, histories_df: DataFrame) -> DataFrame:
    """
    Join normalized impressions with histories on (as_of_date, customer_id).
    If impressions are not normalized yet, normalize; otherwise pass through.
    """
    cols = set(impressions_df.columns)
    if "as_of_date" in cols and "dt" not in cols:
        imp = impressions_df
    else:
        imp = normalize_impressions(impressions_df)

    return (
        imp.alias("i")
        .join(
            histories_df.alias("h"),
            on=["as_of_date", "customer_id"],
            how="left",
        )
        .select(
            "i.as_of_date",
            "i.ranking_id",
            "i.customer_id",
            "i.impressions",
            "h.actions",
            "h.action_types",
        )
    )


def build_training_examples(impressions_df: DataFrame, histories_df: DataFrame, *, explode: bool = False) -> DataFrame:
    base = join_impressions_with_histories(impressions_df, histories_df)

    if not explode:
        # one row per (as_of_date, customer_id, ranking_id), with arrays
        return base

    # explode to one row per impression item
    exploded = base.select(
        "as_of_date",
        "ranking_id",
        "customer_id",
        F.explode("impressions").alias("impr"),
        "actions",
        "action_types",
    ).select(
        "as_of_date",
        "ranking_id",
        "customer_id",
        F.col("impr.item_id").cast("int").alias("impression_item_id"),
        F.col("impr.is_order").cast("boolean").alias("is_order"),
        "actions",
        "action_types",
    )
    return exploded
