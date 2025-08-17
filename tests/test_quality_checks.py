import os
import sys

from pyspark.sql import Row
from pyspark.sql import functions as F
from pyspark.sql import types as T

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import pytest

from challenge.quality_checks import (
    QualityReport,
    actions_quality_report,
    assert_quality_or_fail,
    check_no_same_day_leak_strict,
)
from challenge.schemas import actions_union_schema


def _actions_union(spark):
    rows = [
        Row(
            customer_id=1,
            item_id=10,
            action_type=1,
            occurred_at="2025-08-09 10:00:00",
            action_date="2025-08-09",
            source="clicks",
        ),
        Row(
            customer_id=1,
            item_id=11,
            action_type=2,
            occurred_at="2025-08-08 09:00:00",
            action_date="2025-08-08",
            source="atc",
        ),
    ]
    df = spark.createDataFrame(rows)
    return (
        df.withColumn("customer_id", F.col("customer_id").cast("bigint"))
        .withColumn("item_id", F.col("item_id").cast("int"))
        .withColumn("occurred_at", F.to_timestamp("occurred_at"))
        .withColumn("action_date", F.to_date("action_date"))
    )


def _impr_ctx(spark):
    rows = [Row(as_of_date="2025-08-10", customer_id=1)]
    return (
        spark.createDataFrame(rows)
        .withColumn("as_of_date", F.to_date("as_of_date"))
        .withColumn("customer_id", F.col("customer_id").cast("bigint"))
    )


def test_actions_quality_report_and_assert(spark):
    au = _actions_union(spark)
    qr = actions_quality_report(au)
    assert qr.rows == 2
    assert qr.null_customer_ids == 0
    assert qr.future_actions == 0
    assert qr.duplicate_action_rows == 0
    # Should not raise
    assert_quality_or_fail(qr, allow_future_actions=0)


def test_strict_no_same_day_leak(spark):
    au = _actions_union(spark)
    ctx = _impr_ctx(spark)
    leaks = check_no_same_day_leak_strict(au, ctx, lookback_days=365)
    assert leaks == 0


def test_actions_quality_report_empty_df(spark):
    """
    Covers the empty-input branch (line ~17).
    Ensures we can build a report on an empty DataFrame without blowing up,
    and assertions pass when all counts are zero.
    """
    empty = spark.createDataFrame([], schema=actions_union_schema)
    qr = actions_quality_report(empty)

    assert qr.rows == 0
    # these attrs may vary depending on your QR dataclass; include the ones you have
    assert getattr(qr, "null_customer_ids", 0) == 0
    assert getattr(qr, "future_actions", 0) == 0
    assert getattr(qr, "duplicate_action_rows", 0) == 0

    # should NOT raise when everything is zero
    assert_quality_or_fail(qr, allow_future_actions=0)


def test_null_customer_ids_raise(spark):
    """
    Covers the bad-data assert branch by introducing a null customer_id.
    Use an explicit schema so Spark doesn't have to infer types from None.
    """
    schema = T.StructType(
        [
            T.StructField("customer_id", T.LongType(), True),
            T.StructField("item_id", T.IntegerType(), True),
            T.StructField("action_type", T.IntegerType(), True),
            T.StructField("occurred_at", T.StringType(), True),  # we'll cast
            T.StructField("action_date", T.StringType(), True),  # we'll cast
            T.StructField("source", T.StringType(), True),
        ]
    )
    rows = [(None, 42, 1, "2025-08-10 00:00:00", "2025-08-10", "clicks")]

    df = (
        spark.createDataFrame(rows, schema=schema)
        .withColumn("occurred_at", F.to_timestamp("occurred_at"))
        .withColumn("action_date", F.to_date("action_date"))
    )

    qr = actions_quality_report(df)
    assert getattr(qr, "null_customer_ids", 0) >= 1

    with pytest.raises(ValueError):
        assert_quality_or_fail(qr, allow_future_actions=0)


def test_future_actions_raise(spark):
    rows = [
        Row(
            customer_id=1,
            item_id=1,
            action_type=1,
            occurred_at="2099-01-01 00:00:00",
            action_date="2099-01-01",
            source="clicks",
        ),
    ]
    df = spark.createDataFrame(rows)
    df = (
        df.withColumn("customer_id", F.col("customer_id").cast("bigint"))
        .withColumn("item_id", F.col("item_id").cast("int"))
        .withColumn("occurred_at", F.to_timestamp("occurred_at"))
        .withColumn("action_date", F.to_date("action_date"))
    )
    qr = actions_quality_report(df)
    with pytest.raises(ValueError):
        assert_quality_or_fail(qr, allow_future_actions=0)


def test_quality_report_as_dict():
    qr = QualityReport(
        rows=10,
        null_customer_ids=2,
        future_actions=1,
        duplicate_action_rows=0,
    )

    d = qr.as_dict()

    # contents match fields
    assert d == {
        "rows": 10,
        "null_customer_ids": 2,
        "future_actions": 1,
        "duplicate_action_rows": 0,
    }

    assert d is qr.__dict__
