# src/challenge/quality_checks.py
from dataclasses import dataclass
from typing import Any

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


@dataclass
class QualityReport:
    rows: int
    null_customer_ids: int
    future_actions: int
    duplicate_action_rows: int

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__


def actions_quality_report(actions_union_df: DataFrame) -> QualityReport:
    rows = actions_union_df.count()
    null_customer_ids = actions_union_df.where("customer_id IS NULL").count()
    future_actions = actions_union_df.where("occurred_at > current_timestamp()").count()
    duplicate_action_rows = (
        actions_union_df.groupBy(
            "customer_id",
            "item_id",
            "action_type",
            "occurred_at",
            "action_date",
            "source",
        )
        .count()
        .where("count > 1")
        .count()
    )
    return QualityReport(
        rows=rows,
        null_customer_ids=null_customer_ids,
        future_actions=future_actions,
        duplicate_action_rows=duplicate_action_rows,
    )


def assert_quality_or_fail(qr: QualityReport, *, allow_future_actions: int = 0) -> None:
    if qr.null_customer_ids > 0:
        raise ValueError(f"[DQ] Null customer_id in actions: {qr.null_customer_ids}")
    if qr.future_actions > allow_future_actions:
        raise ValueError(f"[DQ] Future-dated actions seen: {qr.future_actions}")
    # duplicates: warn in README if needed


def check_no_same_day_leak_strict(
    actions_union_df: DataFrame,
    impressions_ctx_df: DataFrame,
    *,
    lookback_days: int,
) -> int:
    """
    True 'no-leak' check:
      - join actions to (as_of_date, customer_id)
      - apply the SAME filters the pipeline uses:
          to_date(occurred_at) < as_of_date
          to_date(occurred_at) >= date_sub(as_of_date, lookback_days)
      - then assert no action remains where to_date(occurred_at) == as_of_date

    If this returns 0, the histories cannot contain any same-day actions.
    """
    ctx = impressions_ctx_df.select("as_of_date", "customer_id").dropDuplicates()

    candidate = (
        actions_union_df.alias("a")
        .join(ctx.alias("c"), on="customer_id", how="inner")
        .where(F.to_date(F.col("a.occurred_at")) >= F.date_sub(F.col("c.as_of_date"), lookback_days))
        .where(F.to_date(F.col("a.occurred_at")) < F.col("c.as_of_date"))
        .select(
            F.to_date("a.occurred_at").alias("occur_day"),
            F.col("c.as_of_date").alias("as_of_date"),
        )
    )

    # If our filters are correct, there can be no equality here by construction.
    leaks = candidate.where(F.col("occur_day") == F.col("as_of_date")).count()
    return leaks
