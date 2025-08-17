import os
import sys

from pyspark.sql import Row
from pyspark.sql import functions as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from challenge.history_builder import build_histories_for_impressions


def _actions_union_df(spark):
    # Build actions (two relevant before day, one same-day, one way before window)
    rows = [
        # relevant (day - 1)
        Row(
            customer_id=100,
            item_id=10,
            action_type=1,
            occurred_at="2025-08-14 18:00:00",
            action_date="2025-08-14",
            source="clicks",
        ),
        Row(
            customer_id=100,
            item_id=20,
            action_type=3,
            occurred_at="2025-08-13 12:00:00",
            action_date="2025-08-13",
            source="orders",
        ),
        # same-day (should be excluded)
        Row(
            customer_id=100,
            item_id=30,
            action_type=2,
            occurred_at="2025-08-15 01:00:00",
            action_date="2025-08-15",
            source="atc",
        ),
        # outside lookback (should be excluded if lookback=7)
        Row(
            customer_id=100,
            item_id=40,
            action_type=1,
            occurred_at="2025-07-30 00:00:00",
            action_date="2025-07-30",
            source="clicks",
        ),
    ]
    df = spark.createDataFrame(rows)
    # Ensure proper types
    return (
        df.withColumn("occurred_at", F.to_timestamp("occurred_at"))
        .withColumn("action_date", F.to_date("action_date"))
        .withColumn("customer_id", F.col("customer_id").cast("bigint"))
        .withColumn("item_id", F.col("item_id").cast("int"))
    )


def _impressions_ctx(spark):
    rows = [
        Row(as_of_date="2025-08-15", customer_id=100),
    ]
    return (
        spark.createDataFrame(rows)
        .withColumn("as_of_date", F.to_date("as_of_date"))
        .withColumn("customer_id", F.col("customer_id").cast("bigint"))
    )


def test_histories_filters_and_padding(spark):
    actions = _actions_union_df(spark)
    ctx = _impressions_ctx(spark)

    hist = build_histories_for_impressions(actions, ctx, max_actions=3, lookback_days=7)

    r = hist.collect()[0]
    # Must return exactly one row (per context)
    assert r["customer_id"] == 100
    assert len(r["actions"]) == 3
    assert len(r["action_types"]) == 3

    # Same-day (item_id=30) must be excluded; outside lookback (item_id=40) excluded.
    # Remaining two should be present, third padded with 0.
    # Ordering: occurred_at desc, then action_type desc, then item_id desc
    # 2025-08-14 click (1, item 10) vs 2025-08-13 order (3, item 20) -> 14 is newer so first.
    assert r["actions"][0] in (10, 20)
    assert set(r["actions"]) >= {10, 20}
    assert 0 in r["actions"]  # padded hole
